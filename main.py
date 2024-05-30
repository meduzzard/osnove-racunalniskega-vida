import cv2
import numpy as np
import os

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


                image_counter += 1

    # Izstopi, ko pritisnemo 'q'
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Osvobodimo kamero in zapremo vsa okna
cap.release()
cv2.destroyAllWindows()
