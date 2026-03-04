import streamlit as st
import joblib
import pandas as pd

st.title("Employee Attrition Prediction")

st.write("Enter employee details to predict attrition risk.")

model = joblib.load("outputs/attrition_model.pkl")

age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
distance_from_home = st.slider("Distance From Home", 1, 30, 5)
total_working_years = st.slider("Total Working Years", 0, 40, 5)
years_at_company = st.slider("Years At Company", 0, 40, 3)
num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
overtime = st.selectbox("Overtime", ["Yes", "No"])

overtime_yes = 1 if overtime == "Yes" else 0

input_data = pd.DataFrame({
    "Age":[age],
    "MonthlyIncome":[monthly_income],
    "DistanceFromHome":[distance_from_home],
    "TotalWorkingYears":[total_working_years],
    "YearsAtCompany":[years_at_company],
    "NumCompaniesWorked":[num_companies_worked],
    "OverTime_Yes":[overtime_yes]
})

# Fill missing features with 0
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]

if st.button("Predict Attrition"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee likely to stay")