# 💼 Employee Salary Prediction

## 📌 Project Overview

Employee Salary Prediction is a Machine Learning project that predicts an employee's estimated annual salary based on different employee-related factors.

The project uses the **Random Forest Regression** algorithm and provides a simple **Streamlit web interface** for salary prediction.

## 🎯 Objective

The main objective of this project is to predict an employee's annual salary based on:

* Years of Experience
* Education Level
* Job Role
* Performance Score
* Working Hours

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn

## 📊 Dataset

The project uses a dataset containing **500 employee records**.

Features:

* Experience Years
* Education Level
* Job Role
* Performance Score
* Working Hours

Target:

* Salary (Lakh per Year)

The dataset is synthetically generated for educational and demonstration purposes.

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Regressor

The dataset is divided into training and testing data.

* Training Data: 80%
* Testing Data: 20%

Categorical features such as education level and job role are converted into numerical values using one-hot encoding.

## 📈 Model Performance

The trained model achieved the following results:

* **Mean Absolute Error (MAE):** 2.37 Lakh
* **Mean Squared Error (MSE):** 10.28
* **R² Score:** 0.861

## 💻 Streamlit Application

The application allows the user to enter:

* Years of Experience
* Performance Score
* Working Hours
* Education Level
* Job Role

After clicking **Predict Salary**, the application displays the estimated annual salary.

Example:

```text
Predicted Salary: ₹30.89 Lakh per Year
```

## 📁 Project Structure

```text
Employee_Salary_Prediction/
│
├── employee_data.csv
├── generate_dataset.py
├── train_model.py
├── employee_salary_model.pkl
├── employee_features.pkl
├── app.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ⚠️ Note

This project is created for educational and demonstration purposes. The employee dataset is synthetically generated and the predicted salary should not be considered an actual salary recommendation.
