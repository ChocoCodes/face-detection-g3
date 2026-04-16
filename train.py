import os
import cv2
import time
import numpy as np
import json 

BASE_DIR = "LaSalleDB1"

train_folders = ["heavy", "medium", "light"]
base_folder = "original"

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

persons = sorted(os.listdir(f"{BASE_DIR}/{base_folder}"))
label_map = {name: idx for idx, name in enumerate(persons)}
reverse_map = {idx: name for name, idx in label_map.items()}

def load_train_dataset(img_size: tuple = (100, 100)) -> list:
    X_train = []
    y_train = []
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )
    for aug_type in train_folders:
        aug_path = os.path.join(BASE_DIR, aug_type)

        for person in os.listdir(aug_path):
            person_path = os.path.join(aug_path, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                enhanced = clahe.apply(img)
                img = cv2.resize(enhanced, img_size)

                X_train.append(img)
                y_train.append(label_map[person])

    return X_train, y_train


def main():
    X_train, y_train = load_train_dataset()
    print("Train samples:", len(X_train))
    print("X_train: ", X_train[0].shape)
    print("Y_train: ", list(set(y_train)))

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius = 1,
        neighbors = 8,
        grid_x = 8, 
        grid_y = 8
    )

    t0 = time.time()
    recognizer.train(X_train, np.array(y_train))
    recognizer.save("lbph.yml")
    print(f"Training finished in: {time.time() - t0:0.2f}s")
    with open("labels.json", "w") as f:
        json.dump(label_map, f)
    print("Labels saved in labels.json")    

if __name__ == "__main__":
    main()