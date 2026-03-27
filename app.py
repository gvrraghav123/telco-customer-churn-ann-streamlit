import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("📊 Telco Customer Churn Prediction (ANN Model)")
st.write("Enter customer details to predict whether the customer will churn or not.")

# -----------------------------
# Load Model & Artifacts
# -----------------------------
@st.cache_resource
def load_assets():
    model = load_model("churn_ann_model.keras")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    target_encoder = pickle.load(open("target_encoders.pkl", "rb"))
    return model, scaler, encoders, target_encoder

model, scaler, encoders, target_encoder = load_assets()

# -----------------------------
# Input UI
# -----------------------------
def user_input():
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input("Tenure (months)", 0, 100, 1)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    monthly = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

    data = pd.DataFrame([[
        gender, senior, partner, dependents, tenure,
        phone, multiple, internet, online_sec, online_backup,
        device_protect, tech_support, streaming_tv, streaming_movies,
        contract, paperless, payment, monthly, total
    ]])

    data.columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    return data


input_df = user_input()

# -----------------------------
# Preprocessing (same as training)
# -----------------------------
def preprocess(df):
    df = df.copy()

    # Apply label encoders safely
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    return df


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Churn"):
    processed = preprocess(input_df)

    scaled = scaler.transform(processed)
    prediction = model.predict(scaled)[0][0]

    result = "YES (Customer will churn)" if prediction > 0.5 else "NO (Customer will stay)"

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{prediction:.2f}**")
    st.success(result)