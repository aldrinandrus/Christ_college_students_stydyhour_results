import streamlit as st
import joblib
model=joblib.load("logistic_regression_StudyHrs_model.pkl")
st.title("Student Pass/Fail based on Study Hours")
hours=st.number_input("Enter Study Hours",min_value=0.0,max_value=15.0,value=5.0)
if st.button("Predict"):
  prediction=model.predict([[hours]])
  if prediction[0]==1:
    st.success("Student will PASS")
  else:
    st.error("Student will FAIL")
[10:49, 17/09/2026] Abhay: import streamlit as st
import joblib
import pandas as pd

model = joblib.load("logistic_regression_StudyHrs_model.pkl")

st.title("Student Pass/Fail Prediction")

st.write("Enter student details to predict the result.")

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.number_input(
    "Enter Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

if st.button("Predict"):

    student = pd.DataFrame(
        [[hours, attendance]],
        columns=["StudyHours", "Attendance"]
    )

    prediction = model.predict(student)

    probability = model.predict_proba(student)

    if prediction[0] == 1:
        st.success("Student will PASS")
    else:
        st.error("Student will FAIL")

    st.write(
        "Probability of Pass:",
        round(probability[0][1] * 100, 2),
        "%"
    )
