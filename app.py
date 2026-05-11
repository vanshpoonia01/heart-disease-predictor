import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load model
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #dc2626, #b91c1c);
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">❤️ Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered heart disease prediction system</div>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    st.subheader("🩺 Clinical Parameters")
    restecg = st.selectbox("Rest ECG", [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ])
    thalach = st.number_input("Maximum Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encoding
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_map = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

cp = cp_map[cp]
restecg = restecg_map[restecg]
slope = slope_map[slope]
thal = thal_map[thal]

# Prediction
if st.button("🔍 Predict Heart Disease Risk"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Heart Disease Detected")
        st.progress(85)
    else:
        st.success("✅ Low Risk: No Heart Disease Detected")
        st.progress(25)

st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Project")
