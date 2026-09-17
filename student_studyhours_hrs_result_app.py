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
# PREPARE FEATURES AND TARGET
# ============================================================

# Input features
X = df[["StudyHours", "Attendance"]]

# Output
y = df["Result"]


# ============================================================
# TRAIN LOGISTIC REGRESSION MODEL
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
# INPUT: STUDY HOURS
# ============================================================

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)


# ============================================================
# INPUT: ATTENDANCE
# ============================================================

attendance = st.number_input(
    "Enter Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Predict Result"):

    # Create student input
    student = pd.DataFrame(
        {
            "StudyHours": [hours],
            "Attendance": [attendance]
        }
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(student)

    probability = model.predict_proba(student)


    # Probability values
    fail_probability = probability[0][0] * 100
    pass_probability = probability[0][1] * 100


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")

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
