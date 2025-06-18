import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model, scaler, and feature names
model, scaler, feature_names = joblib.load(
    "/Users/yevrud/redi_school_ML_AI/final_project/rf_model.pkl"
)

# Load feature means separately from a CSV (optional, if not in the model file)
feature_means = (
    pd.read_csv("/Users/yevrud/redi_school_ML_AI/final_project/feature_means.csv")
    .set_index("Feature")["Mean"]
    .to_dict()
)

# Set up Streamlit app
st.set_page_config(page_title="🏖️ Hotel Booking Cancellation Predictor", layout="wide")

st.title("🌴 Hotel Booking Cancellation Predictor")
st.subheader("🏨 Let's check if your booking is **safe or at risk**!")

# User inputs for 10 key features
lead_time = st.slider("⏳ Lead Time (days before arrival)", 0, 365, 100)
stays_weekend = st.slider("🌅 Nights on weekend", 0, 10, 2)
stays_week = st.slider("🏠 Nights during the week", 0, 20, 3)
adults = st.slider("👨‍👩‍👧 Number of adults", 0, 5, 2)
children = st.slider("🧒 Number of children", 0, 5, 0)
previous_cancellations = st.slider("❌ Previous cancellations", 0, 5, 0)
deposit_type = st.selectbox(
    "💳 Deposit type", ["No Deposit", "Non Refund", "Refundable"]
)
booking_changes = st.slider("🔄 Booking changes", 0, 5, 0)
special_requests = st.slider("✨ Number of special requests", 0, 5, 0)
booking_id = st.text_input("🆔 Booking ID (for saving results)")

# Create DataFrame with user inputs
input_df = pd.DataFrame(
    {
        "lead_time": [lead_time],
        "stays_in_weekend_nights": [stays_weekend],
        "stays_in_week_nights": [stays_week],
        "adults": [adults],
        "children": [children],
        "previous_cancellations": [previous_cancellations],
        "booking_changes": [booking_changes],
        "total_of_special_requests": [special_requests],
    }
)

# One-hot encode deposit type
deposit_dummies = pd.get_dummies([deposit_type], prefix="deposit_type")
for col in [
    "deposit_type_No Deposit",
    "deposit_type_Non Refund",
    "deposit_type_Refundable",
]:
    if col not in deposit_dummies.columns:
        deposit_dummies[col] = 0
input_df = pd.concat([input_df, deposit_dummies], axis=1)

# Fill missing features with dataset averages
missing_cols = [col for col in feature_names if col not in input_df.columns]
for col in missing_cols:
    input_df[col] = feature_means.get(col, 0)  # Assign mean value, or 0 if not found

# Ensure correct feature alignment
input_df = input_df.reindex(columns=feature_names)

# Scale features before prediction
input_scaled = scaler.transform(input_df)

# Generate predictions
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Display results
st.subheader("🎯 Prediction Result")
if prediction == 1:
    st.error(
        f"⚠️ **This booking is likely to be CANCELED**.\n\n🚨 Probability: **{probability:.2%}**"
    )
else:
    st.success(
        f"✅ **The booking is likely to be KEPT**.\n\n☀️ Probability of cancellation: **{probability:.2%}**"
    )

# Save results with Booking ID
if booking_id:
    result_df = pd.DataFrame({"id": [booking_id], "is_canceled": [prediction]})
    result_df.to_csv("submission_results.csv", index=False)
    st.markdown(f"📁 **Results saved as:** `submission_results.csv` ✅")

st.markdown("🏝️ **Enjoy your trip planning, stress-free!**")
