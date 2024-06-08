import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Get the image path from the command-line arguments
if len(sys.argv) != 3:
    print("Usage: python predict.py <model_path> <image_path>")
    sys.exit(1)

model_path = sys.argv[1]
image_path = sys.argv[2]

# Load the trained model
model = load_model(model_path)

# Preprocess the input image
preprocessed_image = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_image)
prediction_value = prediction[0][0]  # Assuming the model returns a single value

# Print the prediction value
print(f'Prediction: {prediction_value}')

# Define a threshold for classification (example threshold)
threshold = 0.5
if prediction_value >= threshold:
    verification_message = "User verified successfully."
else:
    verification_message = "User verification failed."

# Print the verification message
print(verification_message)
