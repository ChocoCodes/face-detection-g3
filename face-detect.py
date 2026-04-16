import cv2 as cv
import time 
import os
import json

IMG_SIZE = (100, 100)

# Load the pre-trained Viola-Jones XML model for detecting frontal faces
face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))

def load_model():
    print("[INFO] Initializing LBPH model...")
    t0 = time.time()
    model = cv.face.LBPHFaceRecognizer_create(
            radius = 1,
            neighbors = 8,
            grid_x = 8, 
            grid_y = 8
        )
    model.read("lbph.yml")

    print(f"Model loaded in: {time.time() - t0:0.2f}s")
    return model

def load_labels():
    with open("labels.json", "r") as f:
        label_map = json.load(f)

    label_map = {int(v): str(k) for k, v in label_map.items()}
    return label_map

def preprocess_img(grayscale_img):
    enhanced = clahe.apply(grayscale_img)
    resized = cv.resize(enhanced, IMG_SIZE)
    return resized

def main():
    model = load_model()
    label_map = load_labels()

    cap = cv.VideoCapture(0)

    prev_t = 0
    fps_display = 0
    frame_count = 0

    while True:
        # Capture the current frame from the webcam feed
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to Grayscale and display raw grayscale version in a separate window
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # FPS calculation
        new_time = time.time()
        fps = 1 / (new_time - prev_t) if prev_t else 0
        prev_t = new_time

        frame_count += 1
        if frame_count % 10 == 0:
            fps_display = int(fps)
        

        # Draw a bounding box when a front face is detected
        for (x, y, w, h) in faces:
            face_roi = grayscale[y: y + h, x: x + w]
            if face_roi.size == 0:
                continue
            
            processed = preprocess_img(face_roi)

            label_id, confidence = model.predict(processed)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 240), 2)

            if confidence < 100:
                print(f"ID: {label_id}, Confidence: {confidence:.2f}")
                cv.putText(
                    frame, 
                    f"{label_map.get(label_id, 'Unknown')} ({round(confidence, 2)})", 
                    (x, y - 10), 
                    cv.FONT_HERSHEY_COMPLEX, 
                    0.5,
                    (255, 255, 255), 
                    2, 
                    cv.LINE_AA
                )

        # FPS Text
        cv.putText(frame, f"FPS: {fps_display}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Final processed frame with bounding box
        cv.imshow('LBPH Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('d'):
            break   

    # Close camera and window to free memory resources
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRASH ERROR: {e}")