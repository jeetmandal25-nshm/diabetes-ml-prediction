import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Diabetes AI System")

st.sidebar.info("""
Algorithm : Random Forest  
Dataset : PIMA Indians Diabetes Dataset  
Purpose : Early Diabetes Detection
""")

# ---------------- TITLE ----------------
st.title("🏥 AI Powered Diabetes Detection Dashboard")

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    skin = st.number_input("Skin Thickness", 0, 100)

with col2:
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Patient"):

    input_data = np.array([[pregnancies, glucose, bp,
                            skin, insulin, bmi, dpf, age]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)
    prob = model.predict_proba(scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ HIGH RISK OF DIABETES")
        result_text = "High Risk"
    else:
        st.success("✅ LOW RISK OF DIABETES")
        result_text = "Low Risk"

    # ---------------- PROBABILITY CHART ----------------
    st.subheader("Risk Probability")

    labels = ["No Diabetes", "Diabetes"]
    values = prob[0]

    fig = plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Probability")
    st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Feature Importance")

    features = [
        "Pregnancies","Glucose","BP","Skin",
        "Insulin","BMI","DPF","Age"
    ]

    importance = model.feature_importances_

    fig2 = plt.figure()
    plt.barh(features, importance)
    st.pyplot(fig2)

    # ---------------- PDF REPORT ----------------
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200,10,txt="Diabetes Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200,10,txt=f"Result: {result_text}", ln=True)
    pdf.cell(200,10,txt=f"Diabetes Probability: {prob[0][1]*100:.2f}%", ln=True)

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as file:
        st.download_button(
            label="📄 Download Medical Report",
            data=file,
            file_name="Diabetes_Report.pdf"
        )

st.divider()

st.caption("Developed using Machine Learning & Streamlit")