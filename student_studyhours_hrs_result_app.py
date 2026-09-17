```python
import streamlit as st
import joblib
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_StudyHrs_model.pkl")


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Pass/Fail Prediction")

st.write(
    "Enter the student's study hours and attendance "
    "to predict whether the student will PASS or FAIL."
)


# ============================================================
# INPUTS
# ============================================================

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


# ============================================================
# PREDICT
# ============================================================

if st.button("Predict Result"):

    # Create input in the same format used during training
    student = pd.DataFrame(
        {
            "StudyHours": [hours],
            "Attendance": [attendance]
        }
    )

    # Make prediction
    prediction = model.predict(student)

    # Get prediction probability
    probability = model.predict_proba(student)

    pass_probability = probability[0][1] * 100
    fail_probability = probability[0][0] * 100


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if prediction[0] == 1:
        st.success("🎉 Student will PASS")
    else:
        st.error("❌ Student will FAIL")


    # ========================================================
    # DISPLAY PROBABILITY
    # ========================================================

    st.write(
        f"**Probability of Pass:** {pass_probability:.2f}%"
    )

    st.write(
        f"**Probability of Fail:** {fail_probability:.2f}%"
    )


    # ========================================================
    # DISPLAY INPUT DETAILS
    # ========================================================

    st.subheader("Student Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Study Hours",
            f"{hours:g} hours"
        )

    with col2:
        st.metric(
            "Attendance",
            f"{attendance:g}%"
        )
```
