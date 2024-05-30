import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from matplotlib import pyplot as plt
import time

# Determine the root directory of the project
root_dir = os.path.dirname(os.path.abspath(__file__))

# Directory for user faces dataset
user_faces_dir = os.path.join(root_dir, 'captured_faces', 'user')
user_faces_class_dir = os.path.join(user_faces_dir, 'user')  # Add a subdirectory for the user's class
lfw_dir = os.path.join(root_dir, 'lfw')

# Create directories if they don't exist
os.makedirs(user_faces_class_dir, exist_ok=True)


# Initialize the camera (try different indices if 0 does not work)
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(3)
    return cap


cap = initialize_camera()

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

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


# Record user faces
def record_user_faces(duration=5):
    start_time = time.time()
    image_counter = 0
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
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
                face_path = f'{user_faces_class_dir}/face_{image_counter}_preprocessed.jpg'
                cv2.imwrite(face_path, preprocessed_face)
                for _ in range(20):
                    aug_face = augment_data(preprocessed_face)
                    aug_face_path = f'{user_faces_class_dir}/face_{image_counter}_aug.jpg'
                    cv2.imwrite(aug_face_path, aug_face)
                    image_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"Recorded {image_counter} images from user.")


# Record user faces
record_user_faces()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Ensure images have been captured
if not os.listdir(user_faces_class_dir):
    print("No images captured. Exiting...")
    exit()


# Define the model
def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
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
def load_data(user_dir, lfw_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)

    user_generator = datagen.flow_from_directory(
        user_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale',
        classes=['user']
    )

    lfw_generator = datagen.flow_from_directory(
        lfw_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale'
    )

    combined_generator = tf.data.Dataset.zip((tf.data.Dataset.from_generator(
        lambda: user_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).map(lambda x, y: (x, tf.ones_like(y))),
                                              tf.data.Dataset.from_generator(
                                                  lambda: lfw_generator,
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(None,), dtype=tf.float32)
                                                  )
                                              ).map(lambda x, y: (x, tf.zeros_like(y)))))

    combined_data = combined_generator.flat_map(
        lambda user, lfw: tf.data.Dataset.from_tensors(user).concatenate(tf.data.Dataset.from_tensors(lfw)))

    return combined_data, user_generator.samples + lfw_generator.samples, user_generator.batch_size


user_data_dir = os.path.join(root_dir, 'captured_faces', 'user')
lfw_data_dir = lfw_dir
combined_data, samples, batch_size = load_data(user_data_dir, lfw_data_dir)

input_shape = (128, 128, 1)
model = create_model(input_shape)

# Add TensorBoard and EarlyStopping callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Calculate the number of steps per epoch
steps_per_epoch = samples // batch_size
validation_steps = samples // batch_size

# Train the model with hyperparameters
history = model.fit(
    combined_data.repeat(),  # Repeat the dataset to avoid running out of data
    epochs=5,  # Increased number of epochs for better training
    steps_per_epoch=steps_per_epoch,
    validation_data=combined_data.repeat(),  # Repeat the validation dataset as well
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# Save the model
model.save('face_recognition_model_user.h5')

# Plot the training history
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
