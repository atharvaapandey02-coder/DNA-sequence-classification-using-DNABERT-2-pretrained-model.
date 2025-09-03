
import streamlit as st

def get_patient_info():
    st.sidebar.header("Patient Information")
    patient_name = st.sidebar.text_input("Patient Name:")
    patient_age = st.sidebar.number_input("Age:", min_value=0, max_value=120, value=30)
    had_cancer = st.sidebar.selectbox("Previous Cancer History?", ["Yes", "No"])
    return patient_name, patient_age, had_cancer
