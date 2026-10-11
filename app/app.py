import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# --------------------------------------------------
# Model paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHURN_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
CLUSTER_MODEL_PATH = PROJECT_ROOT / "models" / "cluster_model.joblib"


# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():
    churn_model = joblib.load(CHURN_MODEL_PATH)
    cluster_model = joblib.load(CLUSTER_MODEL_PATH)

    return churn_model, cluster_model


churn_model, cluster_model = load_models()


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("Customer Churn Prediction")

st.write(
    "Enter the customer's information to predict "
    "their churn risk and customer profile."
)


# --------------------------------------------------
# Customer information
# --------------------------------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=700.0
)


# --------------------------------------------------
# Services
# --------------------------------------------------

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# --------------------------------------------------
# Contract and payment
# --------------------------------------------------

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


# --------------------------------------------------
# Feature engineering: NumServices
# --------------------------------------------------

service_values = [
    phone_service,
    multiple_lines,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies
]

num_services = sum(
    service == "Yes"
    for service in service_values
)


# --------------------------------------------------
# Build one customer row
# --------------------------------------------------

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "NumServices": [num_services]
})


# --------------------------------------------------
# Cluster descriptions
# --------------------------------------------------

cluster_profiles = {
    0: "Long-term / High-value customer",
    1: "Basic / Low-cost customer",
    2: "Newer / Higher-risk customer"
}


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Churn"):

    cluster = int(
        cluster_model.predict(customer_data)[0]
    )

    churn_prediction = int(
        churn_model.predict(customer_data)[0]
    )

    churn_probability = churn_model.predict_proba(
        customer_data
    )[0, 1]

    st.subheader("Prediction Results")

    st.write(
        f"Customer Profile: **Cluster {cluster} - "
        f"{cluster_profiles[cluster]}**"
    )

    if churn_prediction == 1:
        st.error("Churn Prediction: YES")
    else:
        st.success("Churn Prediction: NO")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )