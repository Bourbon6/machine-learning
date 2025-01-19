import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Streamlit application interface
st.title('Mortality Prediction for Patients with Chronic Kidney Disease and Heart Failure')
st.write('Please enter the following information to predict mortality risk:')

# User input
gender = st.selectbox("Gender (0=Female, 1=Male)", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
admission_age = st.number_input("Admission Age (years)", min_value=0)
race = st.selectbox("Race (0=Non-White, 1=White)", options=[0, 1], format_func=lambda x: 'Non-White' if x == 0 else 'White')
ventilator = st.selectbox("Ventilator Used (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
noninvasive_ventilator = st.selectbox("Non-Invasive Ventilator Used (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
crrt = st.selectbox("CRRT Used (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
sodium = st.number_input("Sodium (mmol/L)", format="%.2f")
glucose = st.number_input("Glucose (mg/dL)", format="%.2f")
creatinine = st.number_input("Creatinine (mg/dL)", format="%.2f")
bilirubin_total = st.number_input("Total Bilirubin (mg/dL)", format="%.2f")
wbc = st.number_input("White Blood Cell Count (10^9/L)", format="%.2f")
pco2 = st.number_input("PCO2 (mmHg)", format="%.2f")
bun = st.number_input("Blood Urea Nitrogen (mg/dL)", format="%.2f")
pt = st.number_input("PT (seconds)", format="%.2f")
chloride = st.number_input("Chloride (mmol/L)", format="%.2f")
anion_gap = st.number_input("Anion Gap (mmol/L)", format="%.2f")
calcium = st.number_input("Calcium (mmol/L)", format="%.2f")
potassium = st.number_input("Potassium (mmol/L)", format="%.2f")
heart_rate = st.number_input("Heart Rate (beats/min)", format="%.2f")
sbp = st.number_input("Systolic Blood Pressure (mmHg)", format="%.2f")
temperature = st.number_input("Temperature (°C)", format="%.2f")
myocardial_infarct = st.selectbox("Myocardial Infarction (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
rheumatic_disease = st.selectbox("Rheumatic Disease (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
diabetes = st.selectbox("Diabetes (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
malignant_cancer = st.selectbox("Malignant Cancer (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
copd = st.selectbox("Chronic Obstructive Pulmonary Disease (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
sapsii = st.number_input("SAPSII Score", format="%.2f")
steroids_icu_used = st.selectbox("Steroids Used in ICU (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
rar = st.number_input("Red Cell Distribution Width to Albumin Ratio (RAR)", format="%.2f")

# Construct feature array
features = np.array([[gender, admission_age, race, ventilator, noninvasive_ventilator, crrt, sodium, glucose,
                      creatinine, bilirubin_total, wbc, pco2, bun, pt, chloride, anion_gap, calcium,
                      potassium, heart_rate, sbp, temperature, myocardial_infarct, rheumatic_disease,
                      diabetes, malignant_cancer, copd, sapsii, steroids_icu_used, rar]])

# Standardize data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features_scaled)
    st.write(f"Prediction Result: {'High Mortality Risk' if prediction[0] == 1 else 'Low Mortality Risk'}")