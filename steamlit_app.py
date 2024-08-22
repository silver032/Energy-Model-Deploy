import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from io import BytesIO

# Function to load pickle from URL
def load_pickle_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        return pickle.load(BytesIO(response.content))  # Load pickle from bytes
    except Exception as e:
        st.error(f"Error loading file from {url}: {e}")
        return None

# URLs for model and scaler
model_url = 'https://github.com/silver032/Energy-Model-Deploy/raw/main/random_forest_model.pkl'
scaler_url = 'https://github.com/silver032/Energy-Model-Deploy/raw/main/scaler.pkl'

# Load model and scaler
model = load_pickle_from_url(model_url)
scaler = load_pickle_from_url(scaler_url)

# Ensure model and scaler are loaded properly
if model is not None and scaler is not None:
    st.write(f"Loaded model type: {type(model)}")
    st.write(f"Model has predict method: {hasattr(model, 'predict')}")
    st.write(f"Loaded scaler type: {type(scaler)}")

    # App title
    st.title("Energy Generation Classification App")

    # Input fields for the features
    st.header("Input the Features")

    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
    plant_age = st.number_input("Plant Age", min_value=0, value=10)
    capacity_mw = st.number_input("Capacity (MW)", min_value=0.0, value=50.0)

    # Fuel encoding
    fuel_encoding_map = {
        'Coal': 0, 'Oil': 1, 'Gas': 2, 'Hydro': 3,
        'Solar': 4, 'Wind': 5, 'Nuclear': 6
    }
    primary_fuel = st.selectbox("Primary Fuel Type", options=list(fuel_encoding_map.keys()))
    primary_fuel_encoded = fuel_encoding_map[primary_fuel]

    # Class labels for prediction output
    class_labels = ['very_low', 'low', 'mid', 'high']

    # Button to make predictions
    if st.button("Classify Power Plant"):
        # Prepare input data for prediction
        input_data = pd.DataFrame([[latitude, longitude, plant_age, capacity_mw, primary_fuel_encoded]],
                                  columns=['latitude', 'longitude', 'plant_age', 'capacity_mw', 'primary_fuel_encoded'])

        # Debugging: Print input shape before scaling
        st.write(f"Input data shape: {input_data.shape}")

        # Scale the input features and predict
        input_features_scaled = scaler.transform(input_data)

        # Debugging: Print scaled input shape
        st.write(f"Scaled input shape: {input_features_scaled.shape}")

        # Make the prediction
        try:
            prediction = model.predict(input_features_scaled)

            # Check if prediction is an integer or string
            if isinstance(prediction[0], int):
                predicted_class = class_labels[prediction[0]]
            else:
                predicted_class = prediction[0]

            st.success(f"The predicted generation class is: {predicted_class}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.error("Failed to load model or scaler.")
