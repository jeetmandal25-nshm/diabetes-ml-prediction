import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from utils.pdf_report import create_pdf

st.set_page_config(page_title="AI Diabetes Detection", layout="wide")

# ---------------- LOGIN STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.title("🩺 AI Diabetes Detection System")

    st.subheader("Doctor Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "doctor" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error("Invalid username or password")

    st.stop()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Prediction Dashboard")

st.write("Enter patient health data to analyze diabetes risk")

st.divider()

# ---------------- INPUT FORM ----------------
col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 140)

with col2:
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
scaled_data = scaler.transform(input_data)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Analyze Patient"):

    with st.spinner("Analyzing patient data..."):
        time.sleep(2)
        prediction = model.predict(scaled_data)
        prob = model.predict_proba(scaled_data)

    if prediction[0] == 1:
        result = "High Risk"
        st.error("⚠ High Risk of Diabetes")

    else:
        result = "Low Risk"
        st.success("✅ Low Risk of Diabetes")

    st.subheader("Prediction Probability")

    st.write(f"Diabetes Risk: {prob[0][1]*100:.2f}%")
    st.write(f"No Diabetes: {prob[0][0]*100:.2f}%")

    # ---------------- GRAPH ----------------
    fig, ax = plt.subplots()

    labels = ["No Diabetes", "Diabetes"]
    ax.bar(labels, prob[0])

    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # ---------------- PDF REPORT ----------------
    patient_data = {
        "Pregnancies": preg,
        "Glucose": glucose,
        "Blood Pressure": bp,
        "BMI": bmi,
        "Age": age
    }

    pdf_file = create_pdf(patient_data, result, prob[0][1]*100)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "Download Medical Report",
            f,
            file_name="diabetes_report.pdf"
        )