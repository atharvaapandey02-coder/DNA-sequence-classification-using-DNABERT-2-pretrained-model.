import os
import streamlit as st
import numpy as np
import pydicom
import pickle
import pandas as pd
from PIL import Image
from skimage.transform import resize
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from patient_info import get_patient_info


def preprocess_dicom_image(dicom_path):
    """
    Preprocess a DICOM image for prediction.

    Parameters:
        dicom_path (str): Path to the DICOM image file.

    Returns:
        numpy.ndarray: Preprocessed RGB image suitable for saving as JPG.
    """
    # Create an empty array for RGB image
    img_zeros = np.zeros(shape=(1152, 896, 3), dtype=np.uint8)

    # Read DICOM file and extract pixel array
    dicom_img = pydicom.dcmread(dicom_path)
    grayscale_img = dicom_img.pixel_array

    # Normalize the grayscale image to the 0-255 range
    normalized_img = ((grayscale_img - np.min(grayscale_img)) /
                      (np.max(grayscale_img) - np.min(grayscale_img)) * 255).astype(np.uint8)

    # Resize the normalized image
    resized_img = resize(normalized_img, (1152, 896), anti_aliasing=True)
    resized_img = (resized_img * 255).astype(np.uint8)  # Scale back to 0-255 after resizing

    # Fill the RGB channels with the resized grayscale image
    img_zeros[:, :, 0] = resized_img
    img_zeros[:, :, 1] = resized_img
    img_zeros[:, :, 2] = resized_img

    return img_zeros


def dicom_to_jpg(dicom_file, output_folder, output_name=None):
    """
    Convert a DICOM file to JPG format.

    Parameters:
        dicom_file (str): Path to the DICOM file.
        output_folder (str): Path to save the JPG file.
        output_name (str): Name of the output JPG file (optional).

    Returns:
        str: Path to the saved JPG file.
    """
    # Preprocess the DICOM image
    preprocessed_img = preprocess_dicom_image(dicom_file)

    # Convert to a PIL Image
    img = Image.fromarray(preprocessed_img, 'RGB')

    # Define output name
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(dicom_file))[0] + '.jpg'

    # Save the image
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)
    img.save(output_path)
    print(f"Saved JPG file: {output_path}")
    return output_path


def extract_features(image, model):
    """
    Extract features from the preprocessed image using the VGG16 feature extractor.
    """
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = model.predict(image)  # Extract features
    flattened_features = features.flatten()
    feature_names = [f"feature_{i}" for i in range(flattened_features.shape[0])]
    return pd.DataFrame([flattened_features], columns=feature_names)


def inference(parameters):
    """
    Perform inference on a DICOM image using the specified model.
    """
    model_path = parameters["model_path"]
    dicom_path = parameters["dicom_path"]

    # Load the VGG16 feature extractor
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(1152, 896, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    ft_ext_vgg = Model(inputs=base_model.input, outputs=x)
    ft_ext_vgg.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    # Preprocess image and extract features
    preprocessed_image = preprocess_dicom_image(dicom_path)
    features_df = extract_features(preprocessed_image, ft_ext_vgg)

    # Load saved classification model
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)

    # Predict
    predictions = loaded_model.predict(features_df)
    if predictions.ndim == 1:
        return np.argmax(predictions)
    return np.argmax(predictions, axis=1)[0]


def model2_main():
    """
    Streamlit-based app for predicting BI-RADS ratings from DICOM images.
    """
    st.title("BI-RADS Prediction App by DICOM")

    # Get patient details
    patient_name, patient_age, had_cancer = get_patient_info()

    dicom_file = st.file_uploader("Upload DICOM Image", type=["dcm"])

    if patient_name and dicom_file:
        user_name = patient_name.replace(" ", "_")
        dicom_path = os.path.join("dicom_images", f"{user_name}.dcm")
        jpg_path = os.path.join("dicom_images", f"{user_name}.jpg")
        os.makedirs("dicom_images", exist_ok=True)

        with open(dicom_path, "wb") as f:
            f.write(dicom_file.getbuffer())

        dicom_to_jpg(dicom_path, "dicom_images", f"{user_name}.jpg")
        st.image(jpg_path, caption="Uploaded DICOM Image", use_container_width=True)

        if st.button("Predict BI-RADS Rating"):
            parameters = {
                "model_path": r"D:\LIVE\models\final_model.pkl",
                "dicom_path": dicom_path,
            }

            try:
                prediction = inference(parameters)
                st.subheader(f"Predicted BI-RADS Rating: {prediction}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a DICOM image to proceed.")


if __name__ == "__main__":
    model2_main()
