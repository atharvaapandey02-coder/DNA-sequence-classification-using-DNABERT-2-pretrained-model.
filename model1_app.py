import os
import streamlit as st
from Birads.birads import inference
from patient_info import get_patient_info

def model1_main():
    st.title("BI-RADS Prediction App")


    image_inputs = {
        "L-CC": st.file_uploader("Upload Left CC Image", type=["png"]),
        "L-MLO": st.file_uploader("Upload Left MLO Image", type=["png"]),
        "R-CC": st.file_uploader("Upload Right CC Image", type=["png"]),
        "R-MLO": st.file_uploader("Upload Right MLO Image", type=["png"]),
    }

    # Get patient details
    patient_name, patient_age, had_cancer = get_patient_info()

    if patient_name and all(image_inputs.values()):
        user_name = patient_name.strip().replace(" ", "_")
        image_folder = os.path.join("Birads-Prediction_Images", user_name)
        os.makedirs(image_folder, exist_ok=True)

        # Save uploaded images
        for view, image in image_inputs.items():
            with open(os.path.join(image_folder, f"{view}.png"), "wb") as f:
                f.write(image.getbuffer())

        # Display images for confirmation
        st.image([image_inputs["L-CC"], image_inputs["L-MLO"],
                  image_inputs["R-CC"], image_inputs["R-MLO"]],
                 caption=["L-CC", "L-MLO", "R-CC", "R-MLO"])

        if st.button("Predict BI-RADS Rating"):
            parameters = {
                "model_path": r"D:\LIVE\models\model.p",
                "device_type": "cpu",
                "gpu_number": 0,
                "image_path": image_folder + "/",
                "input_size": (2600, 2000),
            }

            prediction = inference(parameters)

            st.subheader("BI-RADS Prediction Probabilities")
            st.write(f"BI-RADS 0: {prediction[0] * 100:.2f}%")
            st.write(f"BI-RADS 1: {prediction[1] * 100:.2f}%")
            st.write(f"BI-RADS 2: {prediction[2] * 100:.2f}%")

    else:
        st.warning("Please enter a name and upload all 4 images to proceed.")
