import streamlit as st
import joblib
import numpy as np
from gtts import gTTS
import os
from streamlit_chat import message

# Initialize session state for chat messages at the start
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I can help you recommend the best organic fertilizer. Please provide the required inputs."}]

# Load the saved model and transformers
model = joblib.load('fertilizer_model.pkl')
mx = joblib.load('minmaxscaler.pkl')
scaler = joblib.load('standardscaler.pkl')
label_encoder = joblib.load('fertilizer_label_encoder.pkl')

# Streamlit App UI
st.title("Organic Fertilizer Recommendation System")

st.write("""
         This system predicts the best organic fertilizer based on the following parameters:
         - Temperature
         - Moisture
         - Humidity
         - Soil Type
         - Crop Type
         - Nitrogen
         - Potassium
         - Phosphorus
         """)

# Function to generate and play the speech
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")

# Display chat messages from history
for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"))

# Collect user inputs in a form
with st.form("user_inputs"):
    Temperature = st.slider("Enter Temperature (°C)", min_value=-50, max_value=50)
    Humidity = st.slider("Enter Humidity (%)", min_value=0, max_value=100)
    Moisture = st.slider("Enter Moisture (%)", min_value=0, max_value=100)
    Soil_Type = st.selectbox("Select Soil Type", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
    Crop_Type = st.selectbox("Select Crop Type", ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
    Nitrogen = st.slider("Enter Nitrogen Level (kg/ha)", min_value=0)
    Potassium = st.slider("Enter Potassium Level (kg/ha)", min_value=0)
    Phosphorous = st.slider("Enter Phosphorous Level (kg/ha)", min_value=0)

    submit_button = st.form_submit_button(label="Predict Organic Fertilizer")

# Soil and crop mappings
soil_mapping = {'Sandy': 0.0, 'Loamy': 1.0, 'Black': 2.0, 'Red': 3.0, 'Clayey': 4.0}
crop_mapping = {"Maize": 0.0, "Sugarcane": 1.0, "Cotton": 2.0, "Tobacco": 3.0, "Paddy": 4.0, "Barley": 5.0, "Wheat": 6.0, "Millets": 7.0, "Oil seeds": 8.0, 'Pulses': 9.0, 'Ground Nuts': 10.0}

if submit_button:
    # Add user input to chat history
    user_message = f"Inputs provided - Temperature: {Temperature}, Humidity: {Humidity}, Moisture: {Moisture}, Soil Type: {Soil_Type}, Crop Type: {Crop_Type}, Nitrogen: {Nitrogen}, Potassium: {Potassium}, Phosphorous: {Phosphorous}"
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Convert inputs to numerical values
    Soil_Type_numeric = soil_mapping.get(Soil_Type, 0.0)
    Crop_Type_numeric = crop_mapping.get(Crop_Type, 0.0)

    # Prepare input data for prediction
    input_data = np.array([[Temperature, Humidity, Moisture, Soil_Type_numeric, Crop_Type_numeric, Nitrogen, Potassium, Phosphorous]])

    # Scale input data
    scaled_input = mx.transform(input_data)
    scaled_input = scaler.transform(scaled_input)

    # Predict using the model
    encoded_prediction = model.predict(scaled_input)

    # Decode the predicted fertilizer
    fertilizer_name = label_encoder.inverse_transform(encoded_prediction)[0]

    # Assistant response
    response = f"The recommended organic fertilizer is: {fertilizer_name}."
    st.session_state.messages.append({"role": "assistant", "content": response})
    message(response)

    # Speak the response
    speak_text(response)
