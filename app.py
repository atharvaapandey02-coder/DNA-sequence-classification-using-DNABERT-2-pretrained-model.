import streamlit as st
from model1_app import model1_main
from model2_app import model2_main
from model_app import model_main
from model3_app import model3_main
from model4_app import model4_main

from patient_info import get_patient_info


def main():
    st.title("Breast Cancer Prediction App")


    # Create a sidebar menu for model selection
    st.sidebar.title("Select Model")
    model_choice = st.sidebar.radio("Choose Model", ["BI-RADS Prediction", "Multi-Dataset Model Predictor","BI-RADS Prediction App by DICOM","Mammograms calcification","FNAC Feature Extraction"])


    # Display selected model's page
    if model_choice == "Multi-Dataset Model Predictor":
        st.write("## Multi-Dataset Model Predictor")
        model_main()  # Call the main function of Model 1
    elif model_choice == "BI-RADS Prediction":
        st.write("## BI-RADS Prediction Model")
        model1_main()  # Call the main function of Model 2
    elif model_choice == "BI-RADS Prediction App by DICOM":
        st.write("## BI-RADS Prediction App by DICOM")
        model2_main()  # Call the main function of Model 3
    elif model_choice == "Mammograms calcification":
        st.write("## Mammograms calcification")
        model3_main()  # Call the main function of Model 4
    elif model_choice == "FNAC Feature Extraction":
        st.write("## FNAC Feature Extraction")
        model4_main()  # Call the main function of Model 5

if __name__ == "__main__":
    main()
