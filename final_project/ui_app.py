import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    layout="centered",
    initial_sidebar_state="auto",
)


# Load model and resources with caching
@st.cache_resource
def load_resources():
    model, scaler, feature_names = joblib.load(
        "/Users/yevrud/redi_school_ML_AI/final_project/ui_app.py"
    )
    return model, scaler, feature_names


model, scaler, feature_names = load_resources()

# Define the 10 most important and understandable features for UI input
# This list determines what the user sees and interacts with.
UI_TOP_FEATURES = [
    "lead_time",
    "total_of_special_requests",
    "booking_changes",
    "adr",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "deposit_type",
    "is_repeated_guest",
]

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("### Predict if a booking will be canceled by adjusting key factors.")
st.markdown("---")

# Initialize input_data dictionary
input_data = {}

# Initialize variables for categorical defaults (if not in UI_TOP_FEATURES)
# These values should reflect the most common/mode category from your training data.
hotel_type_selected = "City Hotel"
arrival_date_month = "July"
meal_type = "BB"
deposit_type_selected = (
    "No Deposit"  # Will be overwritten if 'deposit_type' is in UI_TOP_FEATURES
)
customer_type = "Transient"
market_segment = "Online TA"
distribution_channel = "TA/TO"
reserved_room_type = "A"
babies = 0  # Default, if 'babies' not in UI_TOP_FEATURES

# --- User Input Section (Single Column) ---

st.subheader("Booking Details")

if "lead_time" in UI_TOP_FEATURES:
    input_data["lead_time"] = st.slider("Lead Time (days):", 0, 700, 90)

if "total_of_special_requests" in UI_TOP_FEATURES:
    input_data["total_of_special_requests"] = st.slider("Special Requests:", 0, 5, 0)

if "booking_changes" in UI_TOP_FEATURES:
    input_data["booking_changes"] = st.slider("Booking Changes:", 0, 5, 0)

if "adr" in UI_TOP_FEATURES:
    input_data["adr"] = st.number_input("Average Daily Rate (ADR):", 0.0, 600.0, 100.0)

if "deposit_type" in UI_TOP_FEATURES:
    deposit_type_selected = st.selectbox(
        "Deposit Type:", ["No Deposit", "Non Refund", "Refundable"]
    )
else:
    deposit_type_selected = "No Deposit"  # Default to mode

st.subheader("Guest & Stay Information")

if "adults" in UI_TOP_FEATURES:
    input_data["adults"] = st.slider("Adults:", 1, 10, 2)
else:
    input_data["adults"] = 1  # Minimum 1 adult

if "children" in UI_TOP_FEATURES:
    input_data["children"] = st.slider("Children:", 0, 5, 0)
else:
    input_data["children"] = 0

if (
    "babies" in UI_TOP_FEATURES
):  # Added babies as a UI option if deemed important enough
    input_data["babies"] = st.slider("Babies:", 0, 5, 0)
else:
    input_data["babies"] = babies  # Use default from above

if "stays_in_weekend_nights" in UI_TOP_FEATURES:
    input_data["stays_in_weekend_nights"] = st.slider("Weekend Nights:", 0, 10, 2)
else:
    input_data["stays_in_weekend_nights"] = 0

if "stays_in_week_nights" in UI_TOP_FEATURES:
    input_data["stays_in_week_nights"] = st.slider("Week Nights:", 0, 20, 3)
else:
    input_data["stays_in_week_nights"] = 0

if "is_repeated_guest" in UI_TOP_FEATURES:
    input_data["is_repeated_guest"] = st.selectbox(
        "Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.markdown("---")

# --- Prepare Data for Prediction ---

processed_input = {
    feature: 0 for feature in feature_names
}  # Initialize all features to 0

# Apply default values for features not exposed in the UI or engineered
processed_input.update(
    {
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "required_car_parking_spaces": 0,
        "agent": 0,
        "company": 0,
        "booking_difficulty_score": 1.0,
        "estimated_check_in_duration": 50.0,
        "arrival_date_year": 2017,
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 15,
        # Assuming 'country_PRT' was the mode country for one-hot encoding
        # If your model uses specific country OHE features, uncomment and set the mode country to 1
        # 'country_PRT': 1,
    }
)

# Engineered features based on UI inputs or defaults
input_data["total_nights"] = input_data.get(
    "stays_in_weekend_nights", 0
) + input_data.get("stays_in_week_nights", 0)
input_data["total_guests"] = (
    input_data.get("adults", 0)
    + input_data.get("children", 0)
    + input_data.get("babies", 0)
)
if input_data["total_guests"] == 0:
    input_data["total_guests"] = 1  # Avoid division by zero
input_data["adr_per_person"] = input_data.get("adr", 0.0) / input_data["total_guests"]

total_prev_bookings = input_data.get("previous_cancellations", 0) + input_data.get(
    "previous_bookings_not_canceled", 0
)
input_data["cancellation_ratio_prev_bookings"] = (
    input_data.get("previous_cancellations", 0) / total_prev_bookings
    if total_prev_bookings > 0
    else 0
)
input_data["is_direct_booking"] = (
    1 if (input_data.get("agent", 0) == 0 and input_data.get("company", 0) == 0) else 0
)

# Overwrite defaults with actual user inputs and engineered values
processed_input.update(input_data)

# Handle one-hot encoding for categorical features based on selected values (UI or default)
if "hotel_City Hotel" in feature_names:
    processed_input["hotel_City Hotel"] = (
        1 if hotel_type_selected == "City Hotel" else 0
    )
if "hotel_Resort Hotel" in feature_names:
    processed_input["hotel_Resort Hotel"] = (
        1 if hotel_type_selected == "Resort Hotel" else 0
    )

months_list = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
for month in months_list:
    col_name = f"arrival_date_month_{month}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if arrival_date_month == month else 0

meal_options = ["BB", "FB", "HB", "SC", "Undefined"]
for m in meal_options:
    col_name = f"meal_{m}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if meal_type == m else 0

deposit_options = ["No Deposit", "Non Refund", "Refundable"]
for dt in deposit_options:
    col_name = f"deposit_type_{dt}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if deposit_type_selected == dt else 0

customer_type_options = ["Transient", "Contract", "Group", "Transient-Party"]
for ct in customer_type_options:
    col_name = f"customer_type_{ct}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if customer_type == ct else 0

market_segment_options = [
    "Online TA",
    "Offline TA/TO",
    "Groups",
    "Direct",
    "Corporate",
    "Complementary",
    "Aviation",
    "Undefined",
]
for ms in market_segment_options:
    col_name = f"market_segment_{ms}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if market_segment == ms else 0

distribution_channel_options = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
for dc in distribution_channel_options:
    col_name = f"distribution_channel_{dc}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if distribution_channel == dc else 0

room_type_options = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
for rt in room_type_options:
    col_name = f"reserved_room_type_{rt}"
    if col_name in feature_names:
        processed_input[col_name] = 1 if reserved_room_type == rt else 0

final_input_df = pd.DataFrame([processed_input])[feature_names]
input_scaled = scaler.transform(final_input_df)

# --- Prediction and Display Results ---
st.markdown("---")
st.subheader("📊 Prediction Result")

if st.button("Predict Cancellation"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"### ⚠️ High probability of **CANCELLATION**!")
        st.markdown(
            f"**Cancellation Probability: <span style='color:red; font-size: 28px;'>{probability:.1%}</span>**",
            unsafe_allow_html=True,
        )
        st.info("💡 Action: Consider proactive communication or offering incentives.")
    else:
        st.success(f"### ✅ The booking is likely to be **KEPT**.")
        st.markdown(
            f"**Cancellation Probability: <span style='color:green; font-size: 28px;'>{probability:.1%}</span>**",
            unsafe_allow_html=True,
        )
        st.info("👍 Great! This booking appears stable.")

st.markdown("---")
st.caption("Developed by Yevheniia Rudenko for Redi School ML/AI Final Project.")
