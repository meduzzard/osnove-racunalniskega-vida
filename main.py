import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from matplotlib import pyplot as plt

# Get the username and image paths from the command-line arguments
username = sys.argv[-1]
image_paths = sys.argv[1:-1]

# Directory for user faces dataset
user_faces_dir = os.path.join('captured_faces', username)
os.makedirs(user_faces_dir, exist_ok=True)

# Save the images to the user's directory
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_image_path = os.path.join(user_faces_dir, os.path.basename(image_path))
    cv2.imwrite(new_image_path, image)

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


user_data_dir = os.path.join('captured_faces', username)
lfw_data_dir = 'lfw'  # Change this to your actual directory
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
model_path = os.path.join('models', f'{username}_model.h5')
model.save(model_path)

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
