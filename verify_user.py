import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Get the image path from the command-line arguments
image_path = sys.argv[1]
user_id = sys.argv[2]

# Directory where user models are stored
models_dir = 'models'
model_path = os.path.join(models_dir, f'{user_id}_model.h5')

# Load the trained model
model = load_model(model_path)

# Preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

preprocessed_image = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_image)
print(f'Prediction: {prediction}')

# Define a threshold for classification
threshold = 0.5
if prediction >= threshold:
    print(f'User {user_id} verified successfully.')
else:
    print(f'User {user_id} verification failed.')
