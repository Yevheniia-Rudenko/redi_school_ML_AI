import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# === Streamlit config ===
st.set_page_config(page_title="🏖️ Hotel Booking Cancellation Predictor", layout="wide")

# === Paths to model and feature means ===
model_path = "/Users/yevrud/redi_school_ML_AI/final_project/rf_model.pkl"
means_path = "/Users/yevrud/redi_school_ML_AI/final_project/feature_means.csv"


# === Caching loading functions ===
@st.cache_resource
def load_model():
    return joblib.load(model_path)


@st.cache_data
def load_feature_means():
    df = pd.read_csv(means_path)
    return df.set_index("Feature")["Mean"].to_dict()


# === Load model, scaler, feature names, and feature means ===
model, scaler, feature_names = load_model()
feature_means = load_feature_means()

# === UI ===
st.title("🌴 Hotel Booking Cancellation Predictor")
st.subheader("🏨 Let's check if your booking is **safe or at risk**!")

# === Input form ===
with st.form("input_form"):
    lead_time = st.slider("⏳ Lead Time (days before arrival)", 0, 365, 100)
    stays_weekend = st.slider("🌅 Nights on weekend", 0, 10, 2)
    stays_week = st.slider("🏠 Nights during the week", 0, 20, 3)
    adults = st.slider("👨‍👩‍👧 Number of adults", 0, 5, 2)
    children = st.slider("🧒 Number of children", 0, 5, 0)
    previous_cancellations = st.slider("❌ Previous cancellations", 0, 5, 0)
    booking_changes = st.slider("🔄 Booking changes", 0, 5, 0)
    special_requests = st.slider("✨ Number of special requests", 0, 5, 0)
    deposit_type = st.selectbox(
        "💳 Deposit type", ["No Deposit", "Non Refund", "Refundable"]
    )
    booking_id = st.text_input("🆔 Booking ID (for saving results)")
    submitted = st.form_submit_button("🔍 Predict Cancellation")

# === Prediction logic ===
if submitted:
    if adults + children == 0:
        st.error("At least one adult or child must be included in the booking.")
    else:
        # Build input DataFrame
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

        # Handle deposit_type one-hot encoding
        expected_deposit_cols = [
            "deposit_type_No Deposit",
            "deposit_type_Non Refund",
            "deposit_type_Refundable",
        ]
        deposit_dummies = pd.get_dummies([deposit_type], prefix="deposit_type")
        deposit_dummies = deposit_dummies.reindex(
            columns=expected_deposit_cols, fill_value=0
        )

        # Combine with input features
        input_df = pd.concat([input_df, deposit_dummies], axis=1)

        # Fill missing features with mean values
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = feature_means.get(col, 0)

        input_df = input_df[feature_names]  # Ensure correct order

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Adjusted probability with offset
        adjusted_probability = max(min(probability - 0.10, 1.0), 0.0)

        # Display result
        if adjusted_probability > 0.5:
            color = "#d9534f"
            message = "⚠️ High risk of cancellation."
        elif adjusted_probability > 0.3:
            color = "#f0ad4e"
            message = "🟠 Moderate risk. Consider contacting the hotel."
        else:
            color = "#5cb85c"
            message = "✅ Low risk. Your booking looks safe!"

        st.markdown(
            f"""
            <div style='background-color:{color}; padding:20px; border-radius:10px;'>
                <h2 style='color:white;'>🎯 Prediction Result</h2>
                <p style='color:white; font-size:18px;'>
                    Adjusted Cancellation Probability: <strong>{adjusted_probability:.2%}</strong><br>
                    {message}
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Save result to CSV
        if booking_id:
            result_df = pd.DataFrame(
                {
                    "id": [booking_id],
                    "is_canceled": [int(adjusted_probability >= 0.5)],
                }
            )
            if os.path.exists("submission_results.csv"):
                result_df.to_csv(
                    "submission_results.csv", mode="a", header=False, index=False
                )
            else:
                result_df.to_csv("submission_results.csv", index=False)
            st.success("📁 Result saved to `submission_results.csv`")

st.markdown("🏝️ **Enjoy your trip planning, stress-free!**")
