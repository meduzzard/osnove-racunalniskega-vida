import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import random

# Determine the root directory of the project
root_dir = os.path.dirname(os.path.abspath(__file__))

# Directory for user faces dataset
user_faces_dir = os.path.join(root_dir, 'captured_faces', 'user', 'user')
lfw_dir = os.path.join(root_dir, 'lfw')

# Load the trained model
model = load_model('face_recognition_model_user.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to select a random image from a directory
def select_random_image(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    if not all_files:
        raise ValueError(f"No images found in directory: {directory}")
    selected_file = random.choice(all_files)
    return selected_file

# Select a random user image (as input)
user_input_image_path = select_random_image(user_faces_dir)
user_input_image = preprocess_image(user_input_image_path)

# Select another random user image (for comparison)
user_compare_image_path = select_random_image(user_faces_dir)
user_compare_image = preprocess_image(user_compare_image_path)

# Predict using the model
user_input_prediction = model.predict(user_input_image)
user_compare_prediction = model.predict(user_compare_image)

# Determine labels
user_input_label = 'User' if user_input_prediction > 0.5 else 'Not User'
user_compare_label = 'User' if user_compare_prediction > 0.5 else 'Not User'

# Load the original images for display
user_input_display_image = cv2.imread(user_input_image_path)
user_compare_display_image = cv2.imread(user_compare_image_path)

# Display the images and predictions
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(user_input_display_image, cv2.COLOR_BGR2RGB))
plt.title(f'User Input Image\nPrediction: {user_input_label}')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(user_compare_display_image, cv2.COLOR_BGR2RGB))
plt.title(f'User Comparison Image\nPrediction: {user_compare_label}')

plt.show()
