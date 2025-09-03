import numpy as np
from PIL import Image
import imageio

def load_images(image_path, view, input_size):
    """Load and preprocess mammogram images"""
    image = imageio.imread(image_path + view + '.png')
    image = Image.fromarray(image).convert("L")  # Convert to grayscale
    image = image.resize(input_size, Image.LANCZOS)
    image = np.array(image).astype(np.float32)
    normalize_single_image(image)
    return np.expand_dims(np.expand_dims(image, 0), 3)

def normalize_single_image(image):
    """Normalize image in-place"""
    image -= np.mean(image)
    image /= np.std(image)