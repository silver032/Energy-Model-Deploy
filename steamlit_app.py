import streamlit as st
import pickle
import numpy as np
import requests
from io import BytesIO

# Define function to load pickle from URL
def load_pickle_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors in the request
        return pickle.load(BytesIO(response.content))  # Load as BytesIO
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching file from {url}: {e}")
        return None
    except (pickle.UnpicklingError, EOFError, ValueError) as e:
        st.error(f"Error unpickling the file from {url}: {e}")
        return None

# Use the updated raw URLs
model_url = 'https://github.com/silver032/Energy-Generation-Deployment/raw/main/random_forest_model.pkl'
scaler_url = 'https://github.com/silver032/Energy-Generation-Deployment/raw/main/scaler.pkl'

# Load model and scaler using pickle
model = load_pickle_from_url(model_url)
scaler = load_pickle_from_url(scaler_url)

# Check if model and scaler are loaded correctly and are of the expected type
if model is None or scaler is None:
    st.error("Failed to load model or scaler. Please check the file paths.")
elif not hasattr(scaler, 'transform'):
    st.error("The scaler object does not have the 'transform' method. Please ensure it is a valid scaler instance.")
else:
    # Title of the app
    st.title("Energy Generation Classification App")

    # Input fields for the features
    st.header("Input the Features")

    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)
    plant_age = st.number_input("Plant Age", min_value=0, value=10)
    capacity_mw = st.number_input("Capacity (MW)", min_value=0.0, value=50.0)

    # Define the primary fuel encoding map
    fuel_encoding_map = {
        'Coal': 0,
        'Oil': 1,
        'Gas': 2,
        'Hydro': 3,
        'Solar': 4,
        'Wind': 5,
        'Nuclear': 6
    }

    # Create the selectbox with descriptive labels
    primary_fuel = st.selectbox(
        "Primary Fuel Type",
        options=list(fuel_encoding_map.keys())  # Display the fuel types
    )

    # Get the encoded value of the selected fuel type
    primary_fuel_encoded = fuel_encoding_map[primary_fuel]

    # Class labels for prediction output
    class_labels = ['very_low', 'low', 'mid', 'high']

    # Button to make predictions
    if st.button("Classify Power Plant"):
        # Prepare input data for prediction
        input_features = np.array([[latitude, longitude, plant_age, capacity_mw, primary_fuel_encoded]])

        try:
            # Ensure scaler is working correctly
            input_features_scaled = scaler.transform(input_features)
            # Make the prediction
            prediction = model.predict(input_features_scaled)
            
            # Map the prediction to the class label
            predicted_class = class_labels[int(prediction[0])]
            st.success(f"The predicted generation class is: {predicted_class}")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    st.success("Model and Scaler loaded successfully!")
