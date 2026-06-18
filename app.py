import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Diabetes Prediction System")

# Numerical Inputs
age = st.number_input("Age", min_value=0)
alcohol_consumption_per_week = st.number_input("Alcohol Consumption Per Week", min_value=0)
physical_activity_minutes_per_week = st.number_input("Physical Activity Minutes Per Week", min_value=0)
diet_score = st.number_input("Diet Score")
sleep_hours_per_day = st.number_input("Sleep Hours Per Day")
screen_time_hours_per_day = st.number_input("Screen Time Hours Per Day")
bmi = st.number_input("BMI")
waist_to_hip_ratio = st.number_input("Waist to Hip Ratio")
systolic_bp = st.number_input("Systolic BP")
diastolic_bp = st.number_input("Diastolic BP")
heart_rate = st.number_input("Heart Rate")
cholesterol_total = st.number_input("Total Cholesterol")
hdl_cholesterol = st.number_input("HDL Cholesterol")
ldl_cholesterol = st.number_input("LDL Cholesterol")
triglycerides = st.number_input("Triglycerides")

# Categorical Inputs (encoded)
gender = st.selectbox("Gender", [0, 1])
ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])
education_level = st.selectbox("Education Level", [0, 1, 2, 3])
income_level = st.selectbox("Income Level", [0, 1, 2, 3])
smoking_status = st.selectbox("Smoking Status", [0, 1, 2])
employment_status = st.selectbox("Employment Status", [0, 1, 2])

family_history_diabetes = st.selectbox("Family History Diabetes", [0, 1])
hypertension_history = st.selectbox("Hypertension History", [0, 1])
cardiovascular_history = st.selectbox("Cardiovascular History", [0, 1])

# Predict Button
if st.button("Predict"):

    data = [[
        age,
        alcohol_consumption_per_week,
        physical_activity_minutes_per_week,
        diet_score,
        sleep_hours_per_day,
        screen_time_hours_per_day,
        bmi,
        waist_to_hip_ratio,
        systolic_bp,
        diastolic_bp,
        heart_rate,
        cholesterol_total,
        hdl_cholesterol,
        ldl_cholesterol,
        triglycerides,
        gender,
        ethnicity,
        education_level,
        income_level,
        smoking_status,
        employment_status,
        family_history_diabetes,
        hypertension_history,
        cardiovascular_history
    ]]

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Person is likely Diabetic")
    else:
        st.success("✅ Person is likely Not Diabetic")