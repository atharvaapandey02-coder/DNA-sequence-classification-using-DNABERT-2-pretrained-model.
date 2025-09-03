import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import cv2
import math
import tkinter as tk
from tkinter import ttk
from skimage.segmentation import active_contour
from skimage.filters import sobel
from PIL import Image, ImageTk
import threading
import queue

from patient_info import get_patient_info



class CellBoundaryDetector:
    def __init__(self, image_path, result_queue):
        self.boundaries = []
        self.manual_boundary_points = []
        self.refined_boundaries = []
        self.refined_boundary_ids = []
        self.drawing = False
        self.image_path = image_path
        self.img = cv2.imread(self.image_path)
        self.features = None
        self.result_queue = result_queue

        # Extract filename for Serial no.
        self.serial_no = os.path.splitext(os.path.basename(self.image_path))[0]

        if self.img is None:
            raise ValueError(f"Error: Unable to load image at {self.image_path}")

        self.original_img = self.img.copy()
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.features_list = []
        self.zoom_factor = 0.1938
        self.magnification_adjustment_factor = 0.7875

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Cell Boundary Detector")
        img_height, img_width = self.original_img.shape[:2]

        control_frame = ttk.Frame(self.root)
        control_frame.pack()

        ttk.Button(control_frame, text="Zoom In", command=self.zoom_in).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Draw Manual Boundary", command=lambda: None).grid(row=0, column=2, padx=5,
                                                                                          pady=5)
        ttk.Button(control_frame, text="Add Snake", command=self.refine_boundary_with_snake).grid(row=0, column=3,
                                                                                                  padx=5, pady=5)
        ttk.Button(control_frame, text="Delete", command=self.delete_last_boundary).grid(row=0, column=4, padx=5,
                                                                                         pady=5)
        ttk.Button(control_frame, text="Finalize", command=self.finalize_processing).grid(row=0, column=5, padx=5)

        self.canvas = tk.Canvas(self.root, width=img_width, height=img_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        self.update_canvas()
        self.root.mainloop()

    def update_canvas(self):
        zoomed_img = cv2.resize(
            self.original_img,
            None,
            fx=self.zoom_factor,
            fy=self.zoom_factor,
            interpolation=cv2.INTER_LINEAR,
        )

        self.display_img = cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB)
        self.display_img = Image.fromarray(self.display_img)
        self.photo = ImageTk.PhotoImage(self.display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(width=self.display_img.width, height=self.display_img.height)
        self.canvas.image = self.photo

    def zoom_in(self):
        self.zoom_factor *= 1/0.1938
        self.update_canvas()

    def zoom_out(self):
        self.zoom_factor /= 1/0.1938
        self.update_canvas()

    def start_draw(self, event):
        self.drawing = True
        self.manual_boundary_points = [(event.x, event.y)]

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.manual_boundary_points.append((x, y))
            self.canvas.create_line(self.manual_boundary_points[-2], (x, y), fill="green", width=2)

    def end_draw(self, event):
        self.drawing = False
        self.boundaries.append(self.manual_boundary_points)
        self.canvas.create_polygon(self.manual_boundary_points, outline="red", fill="", width=2)

    def refine_boundary_with_snake(self):
        if self.boundaries:
            initial_snake = np.array(self.boundaries[-1], dtype=np.float32)
            refined_snake = active_contour(
                sobel(self.gray_img),
                initial_snake,
                alpha=0.006,
                beta=0.25,
                gamma=0.01,
                max_num_iter=2550,
                boundary_condition="periodic",
            )
            boundary_id = self.draw_refined_boundary(refined_snake)
            self.refined_boundaries.append(refined_snake)
            self.refined_boundary_ids.append(boundary_id)
            self.calculate_features(refined_snake)

    def draw_refined_boundary(self, refined_boundary):
        refined_points = [(int(x), int(y)) for x, y in refined_boundary]
        return self.canvas.create_polygon(refined_points, outline="blue", fill="", width=2)

    def calculate_features(self, boundary_points):
        contour = np.array(boundary_points, dtype=np.int32)

        # Calculate area
        area = cv2.contourArea(contour) if len(contour) > 2 else 0

        # Calculate perimeter
        perimeter = cv2.arcLength(contour, closed=True) if len(contour) > 2 else 0

        # Calculate radius
        radius = math.sqrt(area / math.pi) if area > 0 else 0

        # Calculate smoothness
        center = np.mean(contour, axis=0)  # Approximate center
        distances = np.linalg.norm(contour - center, axis=1)  # Distances from the center
        r_mean = np.mean(distances)
        smoothness = np.mean(np.abs(distances - r_mean) / r_mean) if r_mean > 0 else 0

        # Adjust for magnification
        area *= self.magnification_adjustment_factor ** 2
        perimeter *= self.magnification_adjustment_factor
        radius *= self.magnification_adjustment_factor
        smoothness *= self.magnification_adjustment_factor

        self.features_list.append((area, perimeter, radius, smoothness))

    def delete_last_boundary(self):
        if self.refined_boundaries:
            self.refined_boundaries.pop()
            last_refined_id = self.refined_boundary_ids.pop()
            self.canvas.delete(last_refined_id)

    def finalize_processing(self):
        refined_count = len(self.refined_boundaries)
        if refined_count == 0:
            return None

        self.features = {
            "Serial no.": self.serial_no,
            "area_mean": np.mean([f[0] for f in self.features_list]),
            "area_se": np.std([f[0] for f in self.features_list]) / math.sqrt(refined_count),
            "area_worst": np.max([f[0] for f in self.features_list]),
            "perimeter_mean": np.mean([f[1] for f in self.features_list]),
            "perimeter_se": np.std([f[1] for f in self.features_list]) / math.sqrt(refined_count),
            "perimeter_worst": np.max([f[1] for f in self.features_list]),
            "radius_mean": np.mean([f[2] for f in self.features_list]),
            "radius_se": np.std([f[2] for f in self.features_list]) / math.sqrt(refined_count),
            "radius_worst": np.max([f[2] for f in self.features_list]),
            "smoothness_mean": np.mean([f[3] for f in self.features_list]),
            "smoothness_worst": np.max([f[3] for f in self.features_list]),
            "snake_refined_count": refined_count
        }
        self.result_queue.put(self.features)
        self.root.destroy()


def load_model_and_features(folder):
    """
    Load the trained model and its associated features

    Args:
        folder (str): Folder containing the model and features

    Returns:
        tuple: Loaded model and list of features
    """
    model_path = os.path.join('csv', folder, 'trained_ensemble_model.pkl')
    features_path = os.path.join('csv', folder, 'selected_features_info.csv')

    # Load model
    model = joblib.load(model_path)

    # Load feature names
    features_df = pd.read_csv(features_path)
    features = features_df.iloc[:, 0].tolist()

    return model, features


def predict(model, input_data):
    """
    Predict using the trained model

    Args:
        model: Trained machine learning model
        input_data (list): Input features for prediction

    Returns:
        tuple: Prediction class and probabilities
    """
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0]
    return prediction, probability


def model4_main():
    """
    Main Streamlit application for cell boundary detection and prediction
    """
    st.title("Cell Boundary Detector and Cancer Prediction")
    temp_image_folder = "FNAC_Images"
    os.makedirs(temp_image_folder, exist_ok=True)

    # Get patient details
    patient_name, patient_age, had_cancer = get_patient_info()

    # Specific features for Wisconsin model prediction
    wisconsin_features = [
        'radius_mean', 'perimeter_mean', 'area_mean',
        'radius_se', 'perimeter_se', 'area_se',
        'radius_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst'
    ]

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if patient_name and uploaded_file:
        # Generate filename using patient name and original filename
        original_filename = uploaded_file.name
        temp_image_filename = f"{patient_name}_{original_filename}"
        temp_image_path = os.path.join(temp_image_folder, temp_image_filename)

        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Cell Boundaries and Predict"):
            result_queue = queue.Queue()

            def run_gui():
                detector = CellBoundaryDetector(temp_image_path, result_queue)
                detector.run_gui()

            # Start the GUI in the main thread
            gui_thread = threading.Thread(target=run_gui)
            gui_thread.start()
            gui_thread.join()

            # Handle the results from the queue
            if not result_queue.empty():
                features = result_queue.get()
                st.write("Extracted Cell Features:")
                features_df = pd.DataFrame([features])
                st.write(features_df)

                try:
                    # Load Wisconsin model
                    model, feature_list = load_model_and_features("Wisconsin")

                    # Prepare input values for prediction
                    # Carefully select only the specific features mentioned
                    input_values = [
                        features['radius_mean'],
                        features['perimeter_mean'],
                        features['area_mean'],
                        features['radius_se'],
                        features['perimeter_se'],
                        features['area_se'],
                        features['radius_worst'],
                        features['perimeter_worst'],
                        features['area_worst'],
                        features['smoothness_worst']
                    ]

                    # Perform prediction
                    prediction, probability = predict(model, input_values)

                    st.header("Cancer Prediction Results")

                    # Color-code prediction
                    if prediction == 'Benign':
                        st.success(f"Predicted Class: {prediction}")
                    else:
                        st.error(f"Predicted Class: {prediction}")

                    # Display probabilities
                    st.info(f"Prediction Probabilities: Benign: {probability[0]:.2%}, Malignant: {probability[1]:.2%}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")


# This would typically be called in another part of your Streamlit app
if __name__ == "__main__":
    model4_main()