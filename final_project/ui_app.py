# ui_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загружаем модель, scaler и названия фичей
model, scaler, feature_names = joblib.load(
    "/Users/yevrud/redi_school_ML_AI/final_project/rf_model.pkl"
)

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

# One-hot encode deposit type
deposit_dummies = pd.get_dummies([deposit_type], prefix="deposit_type")
for col in [
    "deposit_type_No Deposit",
    "deposit_type_Non Refund",
    "deposit_type_Refundable",
]:
    if col not in deposit_dummies.columns:
        deposit_dummies[col] = 0

# Основные признаки
main_features = pd.DataFrame(
    {
        "lead_time": [lead_time],
        "stays_in_weekend_nights": [stays_weekend],
        "stays_in_week_nights": [stays_week],
        "adults": [adults],
        "children": [children],
        "is_repeated_guest": [is_repeated_guest],
        "previous_cancellations": [previous_cancellations],
        "booking_changes": [booking_changes],
        "total_of_special_requests": [special_requests],
    }
)

# Объединяем всё
input_df = pd.concat([main_features, deposit_dummies], axis=1)

# Добавляем отсутствующие признаки
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Упорядочим колонки
input_df = input_df[feature_names]

# Масштабируем
input_scaled = scaler.transform(input_df)

# Предсказание
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Результат
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
