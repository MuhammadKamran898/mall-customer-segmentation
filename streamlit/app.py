import requests
import streamlit as st


API_URL = "http://127.0.0.1:8001"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mall Customer Segmentation", page_icon="👥", layout="centered"
)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("👥 Mall Customer Segmentation")

st.write(
    "Enter the customer's annual income and spending score "
    "to predict their customer cluster."
)


annual_income = st.number_input(
    "Annual Income (k$)", min_value=0.0, value=60.0, step=1.0
)


spending_score = st.slider("Spending Score", min_value=1, max_value=100, value=50)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Customer Cluster", type="primary"):
    payload = {"annual_income": annual_income, "spending_score": spending_score}

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        response.raise_for_status()

        result = response.json()

        st.success("Prediction completed!")

        st.metric("Predicted Cluster", f"Cluster {result['cluster'] + 1}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Annual Income", f"{result['annual_income']}k")

        with col2:
            st.metric("Spending Score", result["spending_score"])

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to the FastAPI server: {error}")
