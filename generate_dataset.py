import pandas as pd
import numpy as np


# -----------------------------------
# Generate Employee Dataset
# -----------------------------------

np.random.seed(42)

number_of_employees = 500


# -----------------------------------
# Generate Employee Features
# -----------------------------------

experience_years = np.random.randint(
    0, 21, number_of_employees
)

education_level = np.random.choice(
    ["Bachelor", "Master", "PhD"],
    number_of_employees
)

job_role = np.random.choice(
    [
        "Software Developer",
        "Data Analyst",
        "Project Manager",
        "Business Analyst",
        "HR Executive"
    ],
    number_of_employees
)

performance_score = np.random.uniform(
    50, 100, number_of_employees
)

working_hours = np.random.uniform(
    35, 60, number_of_employees
)


# -----------------------------------
# Education Effect
# -----------------------------------

education_effect = np.where(
    education_level == "Bachelor",
    0,
    np.where(
        education_level == "Master",
        5,
        10
    )
)


# -----------------------------------
# Job Role Effect
# -----------------------------------

job_effect = np.select(
    [
        job_role == "Software Developer",
        job_role == "Data Analyst",
        job_role == "Project Manager",
        job_role == "Business Analyst",
        job_role == "HR Executive"
    ],
    [
        12,
        10,
        18,
        8,
        5
    ]
)


# -----------------------------------
# Calculate Salary
# Salary is in Lakh per Year
# -----------------------------------

salary = (
    2.5
    + experience_years * 0.8
    + performance_score * 0.06
    + working_hours * 0.05
    + education_effect
    + job_effect
)


# Add natural variation

noise = np.random.normal(
    0, 2.5, number_of_employees
)

salary = salary + noise


# Minimum salary

salary = np.maximum(
    salary,
    2.5
)


# -----------------------------------
# Create DataFrame
# -----------------------------------

data = pd.DataFrame({
    "experience_years": experience_years,
    "education_level": education_level,
    "job_role": job_role,
    "performance_score": performance_score.round(1),
    "working_hours": working_hours.round(1),
    "salary_lakh": salary.round(2)
})


# -----------------------------------
# Save Dataset
# -----------------------------------

data.to_csv(
    "employee_data.csv",
    index=False
)


print("Employee dataset created successfully!")

print("\nNumber of employees:", len(data))

print("\nFirst 5 records:")
print(data.head())

print("\nDataset saved as employee_data.csv")