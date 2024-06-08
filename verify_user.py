import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

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
prediction_value = prediction[0][0]  # Assuming the model returns a single value

# Print the prediction value
print(f'Prediction: {prediction_value}')

# Define a threshold for classification
threshold = 0.0
if prediction_value >= threshold:
    verification_message = f'User {user_id} verified successfully.'
else:
    verification_message = f'User {user_id} verification failed.'

# Print the verification message
print(verification_message.encode('utf-8').decode('utf-8'))
