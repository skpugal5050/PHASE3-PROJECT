import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import plotly.express as px
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF
import tempfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# ✅ File Paths
model_path = "disease_model.pkl"
symptom_path = "symptoms_list.pkl"

# ✅ Session State Initialization
if "username" not in st.session_state:
    st.session_state.username = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "predicted_history" not in st.session_state:
    st.session_state.predicted_history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "symptoms" not in st.session_state:
    st.session_state.symptoms = None
if "predicted_result" not in st.session_state:
    st.session_state.predicted_result = None

# ✅ Load Model & Symptoms
if not st.session_state.model:
    try:
        st.session_state.model = joblib.load(model_path)
        st.session_state.symptoms = joblib.load(symptom_path)
    except Exception as e:
        st.error(f"Error loading model or symptoms list: {e}")
        st.stop()

# ✅ Solutions & Advice
solutions = {
    'Flu': 'Rest, hydration, and antiviral medications if severe.',
    'Covid-19': 'Isolation, rest, and consult a doctor if symptoms worsen.',
    'Migraine': 'Pain relievers and avoid triggers like stress or noise.',
    'Food Poisoning': 'Stay hydrated and eat bland food. See a doctor if it persists.',
    'Dengue': 'Bed rest, hydration, and monitor platelet count.'
}

telemedicine_links = {
    'Flu': "https://www.teladoc.com/",
    'Covid-19': "https://www.practo.com/video-consult/",
    'Migraine': "https://www.apollo247.com/teleconsultation/",
}

# ✅ UI Styling
st.markdown("""
    <style>
        .big-title { font-size: 32px; font-weight: bold; text-align: center; color: #007BFF; }
        .sub-title { font-size: 20px; text-align: center; color: #555; }
        .prediction-box { border: 2px solid #007BFF; padding: 20px; border-radius: 10px; background: #F8F9FA; }
    </style>
""", unsafe_allow_html=True)

# ✅ User Login System
st.title("🔐 Login System ")
username = st.text_input("Enter your username")

if st.button("Login"):
    if username.strip():
        st.session_state.username = username.strip()
        st.session_state.logged_in = True
        st.success(f"Welcome, {st.session_state.username}!")
    else:
        st.error("Please enter a valid username.")

if st.session_state.logged_in:
    st.markdown('<h1 class="big-title">AI Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Select symptoms to predict your health condition.</p>', unsafe_allow_html=True)

    selected_symptoms = st.multiselect("Choose Symptoms", st.session_state.symptoms)

    # ✅ Prediction Step
    if st.button("Predict Disease"):
        if selected_symptoms:
            input_array = np.array([[int(sym in selected_symptoms) for sym in st.session_state.symptoms]])
            probabilities = st.session_state.model.predict_proba(input_array)[0]
            predicted_index = np.argmax(probabilities)
            predicted_disease = st.session_state.model.classes_[predicted_index]
            solution = solutions.get(predicted_disease, "Please consult a doctor.")

            st.session_state.predicted_result = {
                "disease": predicted_disease,
                "solution": solution,
                "probabilities": probabilities,
                "selected_symptoms": selected_symptoms
            }

            prediction_record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disease": predicted_disease,
                "symptoms": selected_symptoms,
                "solution": solution
            }
            st.session_state.predicted_history.append(prediction_record)

        else:
            st.warning("Please select at least one symptom.")

    # ✅ Display Prediction Results First
    if st.session_state.predicted_result:
        result = st.session_state.predicted_result
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.success(f"You may have: {result['disease']}")
        st.markdown('</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({"Disease": st.session_state.model.classes_, "Probability (%)": np.round(result["probabilities"] * 100, 2)})
        fig = px.bar(prob_df, x="Disease", y="Probability (%)", title="Disease Probability", text_auto=True)
        st.plotly_chart(fig)

    # ✅ Advice Section
        st.info(f"*Recommended Action:* {result['solution']}")

        if result["disease"] in telemedicine_links:
            st.markdown(f"🔗 *Consult a doctor online:* [Click Here]({telemedicine_links[result['disease']]})")

    # ✅ PDF Report Generation
        def generate_pdf(disease, advice, symptoms_selected):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="AI Disease Prediction Report", ln=True, align='C')
                pdf.cell(200, 10, txt=f"Predicted Disease: {disease}", ln=True)
                pdf.cell(200, 10, txt=f"Advice: {advice}", ln=True)
                pdf.cell(200, 10, txt="Symptoms Selected:", ln=True)
                for s in symptoms_selected:
                    pdf.cell(200, 10, txt=f"- {s}", ln=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    return tmp.name
            except Exception as e:
                st.error("Error generating PDF report.")
                return None

        pdf_path = generate_pdf(result["disease"], result["solution"], result["selected_symptoms"])
        if pdf_path:
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download PDF Report", data=f, file_name="disease_report.pdf", mime="application/pdf")

    # ✅ Separate Hospital Maps
    st.markdown("### 🇮🇳 India's Best Hospitals")
    india_hospitals = [
        {"name": "AIIMS Delhi", "location": [28.5672, 77.2100], "address": "New Delhi"},
        {"name": "Apollo Hospitals", "location": [13.0604, 80.2496], "address": "Chennai"}
    ]
    map_india = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for hosp in india_hospitals:
        folium.Marker(location=hosp["location"], popup=hosp["name"]).add_to(map_india)
    st_folium(map_india, width=700, height=500)

    st.markdown("### 🏥 Tamil Nadu's Best Hospitals")
    map_tn = folium.Map(location=[11.1271, 78.6569], zoom_start=6)
    for hosp in india_hospitals:
        folium.Marker(location=hosp["location"], popup=hosp["name"]).add_to(map_tn)
    st_folium(map_tn, width=700, height=500)

    # ✅ Real-Time Prediction History
    if st.session_state.predicted_history:
        st.markdown("### Previous Predictions")
        history_df = pd.DataFrame(st.session_state.predicted_history)
        st.dataframe(history_df)

st.sidebar.title("🔐 Logout")
if st.sidebar.button("Logout"):
    st.session_state.clear()