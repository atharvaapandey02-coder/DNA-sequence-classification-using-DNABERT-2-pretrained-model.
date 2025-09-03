import os
import time
import streamlit as st
from PIL import Image
import torch
from processing import load_mamm, predict, save_mamm_w_boxes, NetC
from patient_info import get_patient_info

# Constants
UPLOAD_FOLDER = "Mammogram_calcification_Images"
MODEL_WEIGHTS = r"D:\LIVE\models\C__00900.weights"

def model3_main():
    # Streamlit app title
    st.title("Mammogram Calcification App")

    # Upload section for mammogram image
    uploaded_file = st.file_uploader("Upload Mammogram Image", type=["jpg", "png", "jpeg"])

    # Threshold adjustment slider
    threshold = st.slider("Set Calcification Detection Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Get patient details
    patient_name, patient_age, had_cancer = get_patient_info()

    if uploaded_file:
        # Create folder to save user data
        patient_folder = patient_name.strip().replace(" ", "_") or "Anonymous"
        image_folder = os.path.join(UPLOAD_FOLDER, patient_folder)
        os.makedirs(image_folder, exist_ok=True)

        # Save the uploaded image
        image_path = os.path.join(image_folder, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        image = Image.open(uploaded_file)

        # Convert image to an appropriate format if necessary
        if image.mode == "I;16":
            image = image.convert("L")  # Convert 16-bit image to 8-bit grayscale

        st.image(image, caption="Uploaded Mammogram", use_container_width=True)

        # Start processing timer
        start_time = time.time()

        # Preprocess the mammogram image
        processed_image = load_mamm(image_path)

        # Load the pre-trained model
        model = NetC()
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
        model.eval()

        # Predict calcifications
        prediction = predict(model, processed_image)

        # Save the image with bounding boxes
        result_image_path = os.path.join(image_folder, f"result_{uploaded_file.name}")
        save_mamm_w_boxes(processed_image, prediction, result_image_path, th=threshold)

        # Display the result image
        result_image = Image.open(result_image_path)

        # Convert result image to displayable format
        if result_image.mode == "I;16":
            result_image = result_image.convert("L")

        st.image(result_image, caption="Processed Image with Detected Calcifications", use_container_width=True)

        # Stop the processing timer
        processing_time = round(time.time() - start_time, 2)

        # Display results
        st.subheader("Detection Results")
        st.write(f"Highest Prediction Score: {prediction.max():.4f}")
        st.write(f"Processing Time: {processing_time} seconds")
    else:
        st.warning("Please upload a mammogram image to proceed.")

if __name__ == "__main__":
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Run the app
    model3_main()
