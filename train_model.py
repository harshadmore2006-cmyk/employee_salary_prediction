import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------
# Load Dataset
# -----------------------------------

data = pd.read_csv("employee_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# -----------------------------------
# Data Visualization
# -----------------------------------

sns.scatterplot(
    x="experience_years",
    y="salary_lakh",
    data=data
)

plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (Lakh)")
plt.show()

sns.scatterplot(
    x="performance_score",
    y="salary_lakh",
    data=data
)

plt.title("Performance vs Salary")
plt.xlabel("Performance Score")
plt.ylabel("Salary (Lakh)")
plt.show()

# -----------------------------------
# Convert Categorical Data
# -----------------------------------

data = pd.get_dummies(
    data,
    columns=["education_level", "job_role"]
)

# -----------------------------------
# Separate Features and Target
# -----------------------------------

X = data.drop("salary_lakh", axis=1)
y = data["salary_lakh"]

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Create Random Forest Model
# -----------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# -----------------------------------
# Train Model
# -----------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# -----------------------------------
# Make Predictions
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Model Evaluation
# -----------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------------------
# Save Model
# -----------------------------------

joblib.dump(
    model,
    "employee_salary_model.pkl"
)

# Save feature columns for Streamlit
joblib.dump(
    list(X.columns),
    "employee_features.pkl"
)

print("\nModel saved as employee_salary_model.pkl")
print("Features saved as employee_features.pkl")

# -----------------------------------
# Test Prediction
# -----------------------------------

sample_employee = pd.DataFrame({
    "experience_years": [5],
    "performance_score": [80],
    "working_hours": [45],
    "education_level_Bachelor": [0],
    "education_level_Master": [1],
    "education_level_PhD": [0],
    "job_role_Business Analyst": [0],
    "job_role_Data Analyst": [0],
    "job_role_HR Executive": [0],
    "job_role_Project Manager": [0],
    "job_role_Software Developer": [1]
})

prediction = model.predict(sample_employee)

print(
    "\nPredicted Salary:",
    round(prediction[0], 2),
    "Lakh per Year"
)