import cv2 as cv
import os
import numpy as np
import json 
import time 

ALLOWED_EXT = ["png", "jpg", "jpeg"]
LFW_PATH = "lfw/images"

# Resize to 70x70 - uniform size for LBPH
IMG_SIZE = (100, 100)
MAX_IMAGES = 15

faces = [] # Store grayscale face images
labels = [] # ID for each person
label_map = {} # Maps person to ids (labels)
current_id = 0

# Load Haar Feature (Front Face)
face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
# LBPH Recognizer model
print("[INFO] Initializing LBPH model...")
model = cv.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

total_start_time = time.time()

all_persons = os.listdir(LFW_PATH)
for person in all_persons:
    person_path = os.path.join(LFW_PATH, person)

    if not os.path.isdir(person_path):
        continue

    person_start_time = time.time()
    print(f"[INFO] Processing person: {person}")
    # Add Person ID as Label
    if person not in label_map:
        label_map[person] = current_id
        current_id += 1

    image_count = 0
    # Use at most 20 images only
    for image in os.listdir(person_path)[:MAX_IMAGES]:
        if image.split(".")[-1].lower() not in ALLOWED_EXT:
            continue

        image_path = os.path.join(person_path, image)

        img = cv.imread(image_path)
        if img is None:
            continue 
        
        image_count += 1
        print(f"  → Reading image: {image}")
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        detected_face = face_cascade.detectMultiScale(
            grayscale_img, 
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(30, 30)
        )

        for (x, y, w, h) in detected_face:
            if w < 60 or h < 60:
                continue

            face_roi = grayscale_img[y: y + h, x: x + w]
            face_roi = cv.equalizeHist(face_roi)
            roi_resized = cv.resize(face_roi, IMG_SIZE)

            faces.append(roi_resized)
            labels.append(label_map[person])
            
    person_end_time = time.time()
    person_time = person_end_time - person_start_time
    print(f"[TIME] {person} → {image_count} images processed in {person_time:.2f} seconds\n")

# Labels dump
with open("labels.json", 'w') as file:
    json.dump(label_map, file)

train_start_time = time.time()
model.train(faces, np.array(labels))
train_end_time = time.time()
train_time = train_end_time - train_start_time

model.save("trainer.yml")

total_end_time = time.time()
total_time = total_end_time - total_start_time

# Debugs
print(f"Extracted Faces: {len(faces)}")
print(f"Labels: {len(labels)}")
print(f"Classes: {len(label_map)}")

print(f"[TIME] Model training time: {train_time:.2f} seconds")
print(f"[TIME] Total execution time: {total_time:.2f} seconds")
