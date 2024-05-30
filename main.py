import cv2
import numpy as np
import os
import random

# Ustvari mapo za shranjevanje obrazov, če ne obstaja
if not os.path.exists('captured_faces'):
    os.makedirs('captured_faces')

# Inicializiramo kamero
cap = cv2.VideoCapture(0)

# Naložimo Haar Cascade za detekcijo obraza
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Funkcija za odstranjevanje šuma, pretvorbo v barvne modele in linearizacijo sivin
def preprocess_face(face):
    # Odstranjevanje šuma z Gaussian Blur
    face = cv2.GaussianBlur(face, (5, 5), 0)
    # Linearizacija sivin (uporaba prilagoditve kontrasta)
    face = cv2.equalizeHist(face)
    return face

# Funkcija za ročno augmentacijo podatkov
def augment_data(face):
    # Rotacija
    rows, cols = face.shape
    angle = random.uniform(-15, 15)  # Naključna vrednost med -15 in 15 stopinj
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    face = cv2.warpAffine(face, M, (cols, rows))

    # Zrcaljenje
    if random.choice([True, False]):
        face = cv2.flip(face, 1)

    # Spreminjanje svetlosti
    alpha = random.uniform(0.8, 1.2)  # Naključna vrednost med 0.8 in 1.2
    beta = random.randint(-30, 30)    # Naključna vrednost med -30 in 30
    face = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)

    # Dodajanje šuma
    noise = np.random.normal(0, random.randint(10, 50), face.shape).astype(np.uint8)
    face = cv2.add(face, noise)

    # Premikanje
    shift_x = random.randint(-10, 10)
    shift_y = random.randint(-10, 10)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    face = cv2.warpAffine(face, M, (cols, rows))

    # Spreminjanje velikosti
    scale = random.uniform(0.5, 1.5)
    resized_face = cv2.resize(face, (int(cols * scale), int(rows * scale)))
    face = cv2.resize(resized_face, (cols, rows))

    return face

image_counter = 0

while True:
    # Preberemo frame iz kamere
    ret, frame = cap.read()
    if not ret:
        break

    # Pretvorimo sliko v sivinsko
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcija obraza
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Označimo obraze v frameu
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Prikaz slike
    cv2.imshow('Frame', frame)

    # Shrani obraz, ko pritisnemo 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                preprocessed_face = preprocess_face(face)
                face_path = f'captured_faces/face_{image_counter}.jpg'
                cv2.imwrite(face_path, preprocessed_face)
                print(f"Obraz shranjen kot {face_path}")

                # Augmentacija podatkov
                aug_face = augment_data(preprocessed_face)
                aug_face_path = f'captured_faces/face_{image_counter}_aug.jpg'
                cv2.imwrite(aug_face_path, aug_face)
                print(f"Augmentirana slika shranjena kot {aug_face_path}")

                image_counter += 1

    # Izstopi, ko pritisnemo 'q'
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Osvobodimo kamero in zapremo vsa okna
cap.release()
cv2.destroyAllWindows()
