import streamlit as st
import pandas as pd
import joblib
import os
import torch
from patient_info import get_patient_info


# Function to load the model and features
def load_model_and_features(folder):
    # Correct the path to look for the models in the 'csv' directory
    model_path = os.path.join('csv', folder, 'trained_ensemble_model.pkl')
    features_path = os.path.join('csv', folder, 'selected_features_info.csv')

    # Load model
    model = joblib.load(model_path)

    # Load feature names and check the columns to avoid KeyError
    features_df = pd.read_csv(features_path)
    st.write("Available columns in the CSV file:", features_df.columns.tolist())  # Debugging: Check column names
    # Assuming the feature names are stored in the first column
    features = features_df.iloc[:, 0].tolist()

    return model, features


# Predict function
def predict(model, input_data):
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0]
    return prediction, probability


# Streamlit App
def model_main():
    st.title("Multi-Dataset Model Predictor")

    # Get patient details
    patient_name, patient_age, had_cancer = get_patient_info()

    # Choose model folder from subfolders inside 'csv'
    folder_choice = st.selectbox("Choose a Dataset Model", ["Wisconsin","Risk-Factor", "COIMBRA" ])

    # Load the corresponding model and features
    model, features = load_model_and_features(folder_choice)

    # Show the required input features
    st.subheader(f"Enter values for the following features:")

    input_values = []
    for feature in features:
        value = st.number_input(f"Enter {feature}:", value=0.0)
        input_values.append(value)

    # Perform prediction when the button is clicked
    if st.button("Predict"):
        prediction, probability = predict(model, input_values)
        st.success(f"Predicted Class: {prediction}")
        st.info(f"Prediction Probabilities: {probability}")


if __name__ == "__main__":
    model_main()
 