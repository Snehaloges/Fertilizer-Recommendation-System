#Importing the  required packages
import streamlit as st
import joblib
import numpy as np
from gtts import gTTS
import os
from streamlit_chat import message


# Load the saved model and transformers
model = joblib.load('fertilizer_model.pkl')
mx = joblib.load('minmaxscaler.pkl')
scaler = joblib.load('standardscaler.pkl')
label_encoder = joblib.load('fertilizer_label_encoder.pkl')

if "messages" not in st.session_state:
    st.session_state.messages =[{"role":"HelpingBot","content":"Welcome! I can help you recommend the best organic fertilizer. Please provide the required inputs."}]

# Streamlit App UI
st.title("Organic Fertilizer Recommendation system")

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

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("Start response.mp3")
 

for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"]=="user"))


# Collect user inputs

with st.form("user_inputs"):
    Temperature = st.slider("Enter Temperature (°C)", min_value=0, max_value=50)
    Humidity = st.slider("Enter Humidity (%)", min_value=0, max_value=100)
    Moisture = st.slider("Enter Moisture (%)", min_value=0, max_value=100)
    Soil_Type = st.selectbox("Select Soil Type", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
    Crop_Type = st.selectbox("Select Crop Type", ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley','Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
    Nitrogen = st.slider("Enter Nitrogen Level (kg/ha)", min_value=0)
    Potassium = st.slider("Enter Potassium Level (kg/ha)", min_value=0)
    Phosphorous = st.slider("Enter Phosphorous Level (kg/ha)", min_value=0)

    submit_button = st.form_submit_button(label="Perdict Organic Fertilizer")

# Convert Soil_Type and Crop_Type to numerical values using the encoder
soil_mapping = {'Sandy':0.0, 'Loamy':1.0, 'Black':2.0, 'Red':3.0, 'Clayey':4.0}
crop_mapping = {"Maize": 0.0, "Sugarcane": 1.0, "Cotton": 2.0, "Tobacco": 3.0, "Paddy": 4.0, "Barley": 5.0, "Wheat": 6.0, "Millets": 7.0, "Oil seeds": 8.0,'Pulses':9.0,'Ground Nuts':10.0}

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"message_{i}")

if submit_button:

    st.session_state.messages.append({"role": "user", "content": f"I entered: Temperature={Temperature}, Humidity={Humidity}, Moisture={Moisture}, Soil Type={Soil_Type}, Crop Type={Crop_Type}, Nitrogen={Nitrogen}, Potassium={Potassium}, Phosphorous={Phosphorous}"})

    # Map the inputs to numerical values
    Soil_Type_numeric = soil_mapping.get(Soil_Type,0.0)
    Crop_Type_numeric = crop_mapping.get(Crop_Type,0.0)


    # Prepare the input for prediction
    input_data = np.array([[Temperature,Humidity, Moisture, Soil_Type_numeric, Crop_Type_numeric, Nitrogen, Potassium, Phosphorous]])

    # Apply scaling for scaler and minmax transformation
    scaled_input= mx.transform(input_data)
    scaled_input = scaler.transform(scaled_input)
   
    # Predict using the model
    encoded_prediction = model.predict(scaled_input)


    # Decode the predicted fertilizer
    fertilizer_name = label_encoder.inverse_transform(encoded_prediction)[0]

    #Tell the recommended fertilizer to user
    response = f"The Recommended Organic Fertilizer is :{fertilizer_name}"
    st.session_state.messages.append({"role": "HelpingBot", "content": response})
    message(response, key =f"message_{len(st.session_state.messages)}")


    speak_text(response)


