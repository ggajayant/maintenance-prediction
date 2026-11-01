from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

project_folder = Path(__file__).resolve().parent
model = joblib.load(project_folder / "artifacts/models/best_model.joblib")
st.title("Engine Condition Prediction")
st.write("Enter the six sensor readings to predict the recorded engine condition.")
st.info("The meaning of class 0/1 and the pressure units need to be confirmed with the data owner.")

engine_rpm = st.number_input("Engine rpm", value=746.0, min_value=0.0)
lub_oil_pressure = st.number_input("Lub oil pressure", value=3.16, min_value=0.0)
fuel_pressure = st.number_input("Fuel pressure", value=6.20, min_value=0.0)
coolant_pressure = st.number_input("Coolant pressure", value=2.17, min_value=0.0)
lub_oil_temp = st.number_input("Lub oil temp (Celsius)", value=76.82)
coolant_temp = st.number_input("Coolant temp (Celsius)", value=78.35)

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])
if st.button("Predict"):
    st.write("Input readings:")
    st.dataframe(input_data)
    prediction = model.predict(input_data)[0]
    score = model.predict_proba(input_data)[0, 1]
    st.write("Predicted recorded condition: Class", int(prediction))
    st.write("Model score for class 1:", round(float(score), 2))
