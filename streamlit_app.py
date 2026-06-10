import streamlit as st
import requests

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 California House Price Predictor")
st.markdown("Fill in the details below to predict the median house value.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Housing Median Age", value=41)
    total_rooms = st.number_input("Total Rooms", value=880)
    total_bedrooms = st.number_input("Total Bedrooms", value=129)

with col2:
    population = st.number_input("Population", value=322)
    households = st.number_input("Households", value=126)
    median_income = st.number_input("Median Income (in tens of thousands)", value=8.32)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    )

# Predict button
if st.button("Predict House Price"):
    with st.spinner("Predicting..."):
        payload = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }

        try:
            response = requests.post(
                "https://house-price-prediction-2-4kv9.onrender.com/predict",
                json=payload
            )
            result = response.json()
            predicted_price = result["predicted_price"]

            st.success(f"### Predicted House Price: ${predicted_price:,.2f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")