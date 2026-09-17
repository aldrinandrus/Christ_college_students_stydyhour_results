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
# PREPARE INPUT AND OUTPUT
# ============================================================

X = df[["StudyHours"]]
y = df["Result"]


# ============================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(random_state=42)

model.fit(X, y)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Pass/Fail Prediction")

st.write(
    "Enter the student's study hours to predict "
    "whether the student will PASS or FAIL."
)


# ============================================================
# STUDY HOURS INPUT
# ============================================================

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0,
    step=0.5
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Result"):

    student = pd.DataFrame(
        {
            "StudyHours": [hours]
        }
    )

    prediction = model.predict(student)

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
    # DISPLAY STUDENT DETAILS
    # ========================================================

    st.subheader("Student Details")

    st.metric(
        "Study Hours",
        f"{hours:g} hours"
    )
