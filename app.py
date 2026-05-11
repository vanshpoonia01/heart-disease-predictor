import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Risk Predictor")
st.caption("AI-powered prediction system")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Gender", ["Female", "Male"])
    chest = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 0, 600, 200)
    fasting = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = st.number_input("Maximum Heart Rate", 60, 250, 150)
    angina = st.selectbox("Exercise Angina", ["N", "Y"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.5, 1.0)
    slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("🔍 Predict"):

    columns = scaler.feature_names_in_
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # numeric
    if "Age" in columns:
        input_df["Age"] = age
    if "RestingBP" in columns:
        input_df["RestingBP"] = resting_bp
    if "Cholesterol" in columns:
        input_df["Cholesterol"] = chol
    if "FastingBS" in columns:
        input_df["FastingBS"] = fasting
    if "MaxHR" in columns:
        input_df["MaxHR"] = maxhr
    if "Oldpeak" in columns:
        input_df["Oldpeak"] = oldpeak

    # one-hot encoding
    if sex == "Male" and "Sex_M" in columns:
        input_df["Sex_M"] = 1

    chest_col = f"ChestPainType_{chest}"
    if chest_col in columns:
        input_df[chest_col] = 1

    ecg_col = f"RestingECG_{ecg}"
    if ecg_col in columns:
        input_df[ecg_col] = 1

    angina_col = f"ExerciseAngina_{angina}"
    if angina_col in columns:
        input_df[angina_col] = 1

    slope_col = f"ST_Slope_{slope}"
    if slope_col in columns:
        input_df[slope_col] = 1

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
        st.progress(85)
    else:
        st.success("✅ No Heart Disease Detected")
        st.progress(20)
