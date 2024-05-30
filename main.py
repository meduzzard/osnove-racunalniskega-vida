import cv2
import numpy as np

import cv2

# Inicializiramo kamero
cap = cv2.VideoCapture(0)

# Naložimo Haar Cascade za detekcijo obraza
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            cv2.imwrite('captured_face.jpg', face)
            print("Obraz shranjen kot captured_face.jpg")

    # Izstopi, ko pritisnemo 'q'
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Osvobodimo kamero in zapremo vsa okna
cap.release()
cv2.destroyAllWindows()
