import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Diabetes Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/diabetes_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TITLE ----------------
st.title("🩺 AI Diabetes Early Detection Dashboard")
st.write("Machine Learning based clinical decision support system")

st.divider()

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Patient Health Details")

preg = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose Level", 0, 200)
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
col1.metric("Glucose", glucose)
col2.metric("BMI", bmi)
col3.metric("Age", age)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Patient"):

    with st.spinner("Analyzing patient health data..."):
        time.sleep(2)
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)

    result = "High Risk" if prediction[0] == 1 else "Low Risk"

    if prediction[0] == 1:
        st.error("⚠ HIGH RISK OF DIABETES")
    else:
        st.success("✅ LOW RISK OF DIABETES")

    st.subheader("Prediction Probability")

    st.write(f"Diabetes Risk: {probability[0][1]*100:.2f}%")
    st.write(f"No Diabetes: {probability[0][0]*100:.2f}%")

    # ---------------- GRAPH ----------------
    fig, ax = plt.subplots()
    labels = ["No Diabetes", "Diabetes"]
    ax.bar(labels, probability[0])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # ---------------- SAVE HISTORY ----------------
    st.session_state.history.append({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Result": result
    })

    # ---------------- HEALTH ADVICE ----------------
    st.subheader("Doctor Recommendation")

    if prediction[0] == 1:
        st.warning("""
        • Reduce sugar intake  
        • Daily exercise recommended  
        • Maintain healthy BMI  
        • Consult healthcare professional
        """)
    else:
        st.info("""
        • Maintain balanced diet  
        • Regular health checkups  
        • Continue healthy lifestyle
        """)

st.divider()

# ---------------- HISTORY TABLE ----------------
st.subheader("📊 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

st.caption("Developed using Machine Learning & Streamlit")