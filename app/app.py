# app/app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# load pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_pipeline.pkl")
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
trained_cols = bundle["feature_order"]  # the raw column names expected by the preprocessor

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")
st.title("📉 Telco Customer Churn Prediction")

st.markdown("Fill the fields and click **Predict**. (This app uses the same preprocessing as training.)")

# --- Inputs matching raw training columns ---
# Numeric
tenure = st.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=10.0)

# Categorical (Telco columns)
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"])  # stayed numeric in CSV but we’ll pass as string safely
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Build a raw single-row DataFrame with the exact training column names
raw_input = {
    "gender": gender,
    "SeniorCitizen": int(SeniorCitizen),
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# Ensure all trained columns exist (any missing will be added as NaN then handled by the pipeline correctly)
row = pd.DataFrame([raw_input])
# Reorder/add columns to match training DataFrame (preprocessor will select what it needs)
for col in trained_cols:
    if col not in row.columns:
        row[col] = np.nan
row = row[trained_cols]

if st.button("🔮 Predict"):
    proba = pipeline.predict_proba(row)[0][1]
    pred = int(proba >= 0.5)
    if pred == 1:
        st.error(f"⚠️ Likely to churn (probability: {proba:.2f})")
    else:
        st.success(f"✅ Not likely to churn (probability: {proba:.2f})")
