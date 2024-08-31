import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("admission_ridge_model_done.pkl", "rb") as file:
    model = pickle.load(file)

# Define the input features

def get_user_data():

    st.header("ADMISSION PREDICTION")

    GRE_Score = st.slider("GRE Score", 260, 340, 300)
    TOEFL_Score = st.slider("GRE Score", 0, 120, 100)
    University_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    SOP = st.slider("SOP", 1.0, 5.0, 3.0)
    LOR = st.slider("LOR", 1.0, 5.0, 3.0)
    CGPA = st.slider("CGPA", 0.0, 10.0, 8.0)
    Research = st.selectbox("Research", [0, 1])

    user_data = pd.DataFrame({
        "GRE Score": GRE_Score,
        "TOEFL Score": TOEFL_Score,
        "University Rating": University_rating,
        "SOP": SOP,
        "LOR": LOR,
        "CGPA": CGPA,
        "Research": Research,
    }, index = [0])

    return user_data


# User Input

input_df = get_user_data()

# Display User input

st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
if st.button("Submit"):
    prediction = model.predict(input_df)

# Display Prediction

    st.subheader("Admission Prediction Probability")
    st.write(prediction[0] * 100)