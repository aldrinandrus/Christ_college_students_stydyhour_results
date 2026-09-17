import streamlit as st
import joblib
import pandas as pd


# Page configuration
st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="centered"
)


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_StudyHrs_model.pkl")


model = load_model()


# Title
st.title("🎓 Student Pass/Fail Prediction")

st.write(
    "Enter the student's study hours and attendance "
    "to predict whether the student will PASS or FAIL."
)


# Student inputs
hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Enter Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)


# Prediction
if st.button("Predict Result"):

    student = pd.DataFrame(
        {
            "StudyHours": [hours],
            "Attendance": [attendance]
        }
    )

    prediction = model.predict(student)
    probability = model.predict_proba(student)

    pass_probability = probability[0][1] * 100
    fail_probability = probability[0][0] * 100

    # Display result
    if prediction[0] == 1:
        st.success("🎉 Student will PASS")
    else:
        st.error("❌ Student will FAIL")

    # Display probabilities
    st.write(f"**Probability of Pass:** {pass_probability:.2f}%")
    st.write(f"**Probability of Fail:** {fail_probability:.2f}%")

    # Display entered details
    st.subheader("Student Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Study Hours", f"{hours:g} hours")

    with col2:
        st.metric("Attendance", f"{attendance:g}%")
