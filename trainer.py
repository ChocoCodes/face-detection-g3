import cv2 as cv
import os
import numpy as np
import json 

ALLOWED_EXT = ["png", "jpg", "jpeg"]
LFW_PATH = "lfw/sample"

# Resize to 100x100 - uniform size for LBPH
IMG_SIZE = (100, 100)

faces = [] # Store grayscale face images
labels = [] # ID for each person
label_map = {} # Maps person to ids (labels)
current_id = 0

# Load Haar Feature (Front Face)
face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
# LBPH Recognizer model
print("[INFO] Initializing LBPH model...")
model = cv.face.LBPHFaceRecognizer_create()

all_persons = os.listdir(LFW_PATH)
for person in all_persons:
    person_path = os.path.join(LFW_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing person: {person}")
    # Add Person ID as Label
    if person not in label_map:
        label_map[person] = current_id
        current_id += 1

    for image in os.listdir(person_path):
        if image.split(".")[-1].lower() not in ALLOWED_EXT:
            continue

        image_path = os.path.join(person_path, image)

        img = cv.imread(image_path)
        if img is None:
            continue 

        print(f"  → Reading image: {image}")
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        detected_face = face_cascade.detectMultiScale(grayscale_img, 1.1, 5)

        if len(detected_face) > 0:
            (x, y, w, h) = detected_face[0]
            face_roi = grayscale_img[y: y + h, x: x + w]

            roi_resized = cv.resize(face_roi, IMG_SIZE)

            faces.append(roi_resized)
            labels.append(label_map[person])

# Labels dump
with open("labels.json", 'w') as file:
    json.dump(label_map, file)

model.train(faces, np.array(labels))
model.save("trainer.yml")
# Debugs
print(f"Extracted Faces: {len(faces)}")
print(f"Faces: {faces[0]}")
print(f"Labels: {len(labels)}")
print(f"Classes: {len(label_map)}")
