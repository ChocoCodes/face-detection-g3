import cv2 as cv
import json 
import time 

IMG_SIZE = (100, 100)

# Load the pre-trained Viola-Jones XML model for detecting frontal faces
face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

print("[INFO] Initializing LBPH model...")
model = cv.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")


with open("labels.json", "r") as file:
    labels = json.load(file)

label_map = {int(v): k for k,v in labels.items()}
print(len(label_map))

webcam = cv.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0
fps = 0
display_fps = 0
frame_count = 0

target_width = 300

while True:
    # Capture the current frame from the webcam feed
    ret, frame = webcam.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    ratio = target_width / float(width)
    small_frame = cv.resize(frame, (target_width, int(height * ratio)))

    # FPS Calculation
    new_frame_time = time.time()
    frame_time = new_frame_time - prev_frame_time
    fps = 1 / (new_frame_time - prev_frame_time) if frame_time > 0 else 0
    prev_frame_time = new_frame_time

    frame_count += 1
    if frame_count % 5 == 0:
        display_fps = int(fps)
    
    # Convert to Grayscale and display raw grayscale version in a separate window
    grayscale = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray Scale Video', grayscale) 
    # Spread out the pixel intensities to ensure the face has enough contrast for the algorithm to "see" faces properly
    hist_equalized = cv.equalizeHist(grayscale)
    faces = face_cascade.detectMultiScale(hist_equalized, 1.1, 5, minSize=(50,50))

    # Draw a bounding box when a front face is detected
    for (x, y, w, h) in faces:
        orig_x, orig_y = int(x / ratio), int(y / ratio)
        orig_w, orig_h = int(w / ratio), int(h / ratio)

        roi_grayscale = hist_equalized[y: y + h, x: x + w]
        if roi_grayscale.size == 0:
            continue
        
        roi_resized = cv.resize(roi_grayscale, IMG_SIZE)

        id, confidence = model.predict(roi_resized)

        cv.rectangle(frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 255, 240), 2)

        if confidence < 100:
            print(f"ID: {id}, Confidence: {confidence}")
            cv.putText(
                frame, 
                f"{label_map.get(id, 'Unknown')} ({round(confidence, 2)})", 
                (orig_x, orig_y - 10), 
                cv.FONT_HERSHEY_COMPLEX, 
                0.5,
                (255, 255, 255), 
                2, 
                cv.LINE_AA
            )

    # FPS Text
    cv.putText(frame, f"FPS: {display_fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Final processed frame with bounding box
    cv.imshow('Camera Output', frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break   

# Close camera and window to free memory resources
webcam.release()
cv.destroyAllWindows()
