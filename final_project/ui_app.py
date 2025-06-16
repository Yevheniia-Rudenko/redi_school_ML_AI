import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load("/Users/yevrud/redi_school_ML_AI/final_project/rf_model.pkl")
scaler = joblib.load("/Users/yevrud/redi_school_ML_AI/final_project/scaler.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="centered")
st.title("🏨 Hotel Booking Cancellation Predictor")

st.markdown(
    "Fill in the booking details below to predict whether the booking will be **canceled**."
)

# Input widgets
lead_time = st.slider("Lead Time (days before arrival)", 0, 365, 100)
stays_weekend = st.slider("Nights on weekend", 0, 10, 2)
stays_week = st.slider("Nights during the week", 0, 20, 3)
adults = st.slider("Number of adults", 0, 5, 2)
children = st.slider("Number of children", 0, 5, 0)
is_repeated_guest = st.selectbox("Is a repeated guest?", [0, 1])
previous_cancellations = st.slider("Previous cancellations", 0, 5, 0)
deposit_type = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
booking_changes = st.slider("Booking changes", 0, 5, 0)
special_requests = st.slider("Number of special requests", 0, 5, 0)

# Encode deposit type
deposit_map = {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}
deposit_encoded = deposit_map[deposit_type]

# Prepare input data
input_features = pd.DataFrame(
    [
        [
            lead_time,
            stays_weekend,
            stays_week,
            adults,
            children,
            is_repeated_guest,
            previous_cancellations,
            deposit_encoded,
            booking_changes,
            special_requests,
        ]
    ],
    columns=[
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "is_repeated_guest",
        "previous_cancellations",
        "deposit_type",
        "booking_changes",
        "total_of_special_requests",
    ],
)

# Pad missing features if needed
n_model_features = model.n_features_in_
while input_features.shape[1] < n_model_features:
    input_features[f"missing_{input_features.shape[1]}"] = 0

# Scale and predict
input_scaled = scaler.transform(input_features)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Show result
st.markdown("---")
st.subheader("Prediction Result")

if prediction == 1:
    st.error(
        f"⚠️ The booking is likely to be **CANCELED**.\n\nProbability: **{probability:.2%}**"
    )
else:
    st.success(
        f"✅ The booking is likely to be **KEPT**.\n\nProbability of cancellation: **{probability:.2%}**"
    )
