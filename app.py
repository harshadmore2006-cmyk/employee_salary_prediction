import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Load Model
# -----------------------------------

model = joblib.load("employee_salary_model.pkl")
feature_columns = joblib.load("employee_features.pkl")

# -----------------------------------
# Page Settings
# -----------------------------------

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼"
)

st.title("💼 Employee Salary Prediction")
st.write("Predict employee salary based on experience, education, role and performance.")

st.divider()

# -----------------------------------
# User Inputs
# -----------------------------------

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=30,
    value=5
)

performance = st.number_input(
    "Performance Score",
    min_value=50.0,
    max_value=100.0,
    value=80.0
)

working_hours = st.number_input(
    "Working Hours per Week",
    min_value=35.0,
    max_value=60.0,
    value=45.0
)

education = st.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Software Developer",
        "Data Analyst",
        "Project Manager",
        "Business Analyst",
        "HR Executive"
    ]
)

# -----------------------------------
# Prediction Button
# -----------------------------------

if st.button("Predict Salary"):

    # Create empty input
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical values
    input_data["experience_years"] = experience
    input_data["performance_score"] = performance
    input_data["working_hours"] = working_hours

    # Education
    education_column = "education_level_" + education

    if education_column in input_data.columns:
        input_data[education_column] = 1

    # Job Role
    job_column = "job_role_" + job_role

    if job_column in input_data.columns:
        input_data[job_column] = 1

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Salary: ₹{prediction:.2f} Lakh per Year"
    )