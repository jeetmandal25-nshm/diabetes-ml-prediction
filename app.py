import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from utils.pdf_report import create_pdf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Diabetes Detection System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOGIN ----------------
st.sidebar.title("Doctor Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != "doctor" or password != "1234":
    st.warning("Login to access dashboard")
    st.stop()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TITLE ----------------
st.title("🩺 AI Diabetes Early Detection System")
st.write("Clinical Decision Support Tool using Machine Learning")

st.divider()

# ---------------- INPUT ----------------
st.sidebar.header("Patient Health Details")

preg = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose", 0, 200)
bp = st.sidebar.number_input("Blood Pressure", 0, 140)
skin = st.sidebar.number_input("Skin Thickness", 0, 100)
insulin = st.sidebar.number_input("Insulin", 0, 900)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.sidebar.number_input("Age", 1, 120)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
scaled_data = scaler.transform(input_data)

# ---------------- DASHBOARD METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Glucose Level", glucose)
col2.metric("BMI", bmi)
col3.metric("Age", age)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Analyze Patient"):

    with st.spinner("Analyzing patient health data..."):
        time.sleep(2)
        prediction = model.predict(scaled_data)
        prob = model.predict_proba(scaled_data)

    result = "High Risk" if prediction[0] == 1 else "Low Risk"

    if prediction[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
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

    # ---------------- SAVE HISTORY ----------------
    st.session_state.history.append({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Result": result
    })

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

st.divider()

# ---------------- HISTORY ----------------
st.subheader("Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

st.caption("AI Medical Dashboard - Machine Learning Project")