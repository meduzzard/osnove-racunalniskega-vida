import os
import cv2
import numpy as np
import time
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Determine the root directory of the project
root_dir = os.path.dirname(os.path.abspath(__file__))

# Directories for datasets
user_faces_dir = os.path.join(root_dir, 'captured_faces', 'user')
public_faces_dir = os.path.join(root_dir, 'captured_faces', 'public')
lfw_dir = os.path.join(root_dir, 'lfw')

# Create directories if they don't exist
os.makedirs(user_faces_dir, exist_ok=True)
os.makedirs(public_faces_dir, exist_ok=True)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for noise removal, color space conversion, and histogram equalization
def preprocess_face(face):
    face = cv2.GaussianBlur(face, (5, 5), 0)
    face = cv2.equalizeHist(face)
    return face

# Function for manual data augmentation
def augment_data(face):
    rows, cols = face.shape
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    face = cv2.warpAffine(face, M, (cols, rows))
    if np.random.choice([True, False]):
        face = np.flip(face, axis=1)
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.randint(-30, 30)
    face = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
    noise = np.random.normal(0, np.random.randint(10, 50), face.shape).astype(np.uint8)
    face = cv2.add(face, noise)
    shift_x = np.random.randint(-10, 10)
    shift_y = np.random.randint(-10, 10)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    face = cv2.warpAffine(face, M, (cols, rows))
    scale = np.random.uniform(0.5, 1.5)
    resized_face = cv2.resize(face, (int(cols * scale), int(rows * scale)))
    face = cv2.resize(resized_face, (cols, rows))
    return face

# Process public dataset images
def process_public_dataset(data_dir):
    image_counter = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            img_path = os.path.join(root, file)
            face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if face is not None:
                preprocessed_face = preprocess_face(face)
                face_path = f'{public_faces_dir}/face_{image_counter}_preprocessed.jpg'
                cv2.imwrite(face_path, preprocessed_face)
                for _ in range(20):
                    aug_face = augment_data(preprocessed_face)
                    aug_face_path = f'{public_faces_dir}/face_{image_counter}_aug.jpg'
                    cv2.imwrite(aug_face_path, aug_face)
                    image_counter += 1
    print(f"Processed {image_counter} images from public dataset.")

# Record user faces
def record_user_faces(duration=5):
    start_time = time.time()
    image_counter = 0
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Frame', frame)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                preprocessed_face = preprocess_face(face)
                face_path = f'{user_faces_dir}/face_{image_counter}_preprocessed.jpg'
                cv2.imwrite(face_path, preprocessed_face)
                for _ in range(20):
                    aug_face = augment_data(preprocessed_face)
                    aug_face_path = f'{user_faces_dir}/face_{image_counter}_aug.jpg'
                    cv2.imwrite(aug_face_path, aug_face)
                    image_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"Recorded {image_counter} images from user.")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Process the LFW dataset
process_public_dataset(lfw_dir)

# Record user faces
record_user_faces()

# Define the model
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
def load_data(user_dir, public_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    user_generator = datagen.flow_from_directory(
        user_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale'
    )
    public_generator = datagen.flow_from_directory(
        public_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale'
    )
    return user_generator, public_generator

user_data_dir = os.path.join(root_dir, 'captured_faces', 'user')
public_data_dir = os.path.join(root_dir, 'captured_faces', 'public')
user_data, public_data = load_data(user_data_dir, public_data_dir)

# Combine user and public data
combined_data = tf.data.Dataset.zip((user_data, public_data))

input_shape = (128, 128, 1)
model = create_model(input_shape)
model.fit(combined_data, epochs=10)
model.save('face_recognition_model.h5')
