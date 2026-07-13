import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Pickle Files
# -----------------------------
with open("heartdisease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("expected_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")

st.markdown(
    """
    Enter your medical information below.
    The Machine Learning model will estimate whether you are at risk of heart disease.
    """
)

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200
    )

with col2:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Risk"):

    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1
    }

    input_df = pd.DataFrame([input_data])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in training order
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0][1]
        st.metric(
            "Risk Probability",
            f"{probability*100:.2f}%"
        )

    # Result
    st.markdown("---")

    if prediction == 1:
        st.error(
            "⚠️ High Risk of Heart Disease Detected"
        )

        st.info(
            """
            Recommendations:
            - Consult a cardiologist
            - Monitor blood pressure regularly
            - Maintain a healthy diet
            - Exercise regularly
            """
        )

    else:
        st.success(
            "✅ Low Risk of Heart Disease"
        )

        st.info(
            """
            Recommendations:
            - Continue a healthy lifestyle
            - Maintain regular exercise
            - Schedule routine health checkups
            """
        )