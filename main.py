import cv2
import numpy as np
import os
import time

# Create a directory to save captured faces if it doesn't exist
if not os.path.exists('captured_faces'):
    os.makedirs('captured_faces')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for noise removal, color space conversion, and histogram equalization
def preprocess_face(face):
    # Noise removal with Gaussian Blur
    face = cv2.GaussianBlur(face, (5, 5), 0)
    # Histogram equalization for contrast adjustment
    face = cv2.equalizeHist(face)
    return face


# Function for manual data augmentation
def augment_data(face):
    augmentations = []

    # Rotation
    rows, cols = face.shape
    angle = np.random.uniform(-15, 15)  # Random value between -15 and 15 degrees
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    face = cv2.warpAffine(face, M, (cols, rows))

    # Flip
    if np.random.choice([True, False]):
        face = cv2.flip(face, 1)

    # Brightness adjustment
    alpha = np.random.uniform(0.8, 1.2)  # Random value between 0.8 and 1.2
    beta = np.random.randint(-30, 30)    # Random value between -30 and 30
    face = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)

    # Add noise
    noise = np.random.normal(0, np.random.randint(10, 50), face.shape).astype(np.uint8)
    face = cv2.add(face, noise)

    # Translation
    shift_x = np.random.randint(-10, 10)
    shift_y = np.random.randint(-10, 10)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    face = cv2.warpAffine(face, M, (cols, rows))

    # Resize
    scale = np.random.uniform(0.5, 1.5)
    resized_face = cv2.resize(face, (int(cols * scale), int(rows * scale)))
    face = cv2.resize(resized_face, (cols, rows))

    return face


# Recording duration in seconds
record_duration = 5
# Counter for processed faces
image_counter = 0
# Start time of the recording
start_time = time.time()

while time.time() - start_time < record_duration:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces in the frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Save the face if detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            preprocessed_face = preprocess_face(face)
            face_path = f'captured_faces/face_{image_counter}_preprocessed.jpg'
            cv2.imwrite(face_path, preprocessed_face)
            print(f"Preprocessed image saved as {face_path}")

            # Augment data
            for _ in range(20):  # Perform augmentation 20 times
                aug_face = augment_data(preprocessed_face)
                aug_face_path = f'captured_faces/face_{image_counter}_aug.jpg'
                cv2.imwrite(aug_face_path, aug_face)
                print(f"Augmented image saved as {aug_face_path}")
                image_counter += 1

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

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
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
