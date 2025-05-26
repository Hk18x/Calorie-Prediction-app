import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("elasticnetcv_model.pkl", "rb"))
scaler = pickle.load(open("scaler (1).pkl", "rb"))

# Title
st.title("Calorie Prediction App")

# User Inputs
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=10, max_value=100, value=30)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65)
duration = st.number_input("Exercise Duration (min)", min_value=1, max_value=180, value=30)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=200, value=100)
body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=45.0, value=40.0)

# Predict button
if st.button("Predict Calories Burned"):
    sex_encoded = 1 if sex == "male" else 0
    features = np.array([[sex_encoded, age, height, weight, duration, heart_rate, body_temp]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"Current Calories: {prediction:.2f}")
