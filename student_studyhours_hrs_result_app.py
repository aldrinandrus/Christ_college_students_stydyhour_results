import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# DATASET
# ============================================================

data = {
    "StudyHours": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 8,
        9, 9, 10, 10, 11
    ],

    "Attendance": [
        55, 60, 65, 60, 70,
        65, 75, 70, 80, 75,
        85, 80, 90, 85, 92,
        90, 95, 92, 96, 98
    ],

    "Result": [
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)


# ============================================================
# INPUT FEATURES AND TARGET
# ============================================================

X = df[["StudyHours", "Attendance"]]
y = df["Result"]


# ============================================================
# TRAIN MODEL
# ============================================================

model = LogisticRegression(
    random_state=42
)

model.fit(X, y)


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
# PREDICTION
# ============================================================

if st.button("Predict Result"):

    # Create student data
    student = pd.DataFrame(
        {
            "StudyHours": [hours],
            "Attendance": [attendance]
        }
    )

    # Get Logistic Regression prediction
    prediction = model.predict(student)
    probability = model.predict_proba(student)

    model_pass_probability = probability[0][1] * 100
    model_fail_probability = probability[0][0] * 100


    # ========================================================
    # MINIMUM REQUIREMENTS
    # ========================================================

    minimum_study_hours = 4.0
    minimum_attendance = 60.0


    # ========================================================
    # FINAL DECISION
    # ========================================================

    # Very low study hours = FAIL
    if hours < minimum_study_hours:

        final_prediction = 0

        # Override probability to reflect the final decision
        pass_probability = 0.0
        fail_probability = 100.0

        reason = (
            f"Study hours are below the minimum required "
            f"{minimum_study_hours:g} hours."
        )

    # Very low attendance = FAIL
    elif attendance < minimum_attendance:

        final_prediction = 0

        pass_probability = 0.0
        fail_probability = 100.0

        reason = (
            f"Attendance is below the minimum required "
            f"{minimum_attendance:g}%."
        )

    # Otherwise use Logistic Regression
    else:

        final_prediction = prediction[0]

        pass_probability = model_pass_probability
        fail_probability = model_fail_probability

        reason = "Prediction is based on the Logistic Regression model."


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")

    if final_prediction == 1:
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
    # DISPLAY REASON
    # ========================================================

    st.info(reason)


    # ========================================================
    # STUDENT DETAILS
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

