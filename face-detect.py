import cv2 as cv
import json 

IMG_SIZE = (100, 100)

# Load the pre-trained Viola-Jones XML model for detecting frontal faces
face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

print("[INFO] Initializing LBPH model...")
model = cv.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")


with open("labels.json", "r") as file:
    labels = json.load(file)

label_map = {int(v): k for k,v in labels.items()}
print(label_map)

webcam = cv.VideoCapture(0)

while True:
    # Capture the current frame from the webcam feed
    ret, frame = webcam.read()
    
    # Convert to Grayscale and display raw grayscale version in a separate window
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray Scale Video', grayscale) 

    # Spread out the pixel intensities to ensure the face has enough contrast for the algorithm to "see" faces properly
    hist_equalized = cv.equalizeHist(grayscale)
    faces = face_cascade.detectMultiScale(hist_equalized, 1.1, 5)

    # Draw a bounding box when a front face is detected
    for (x, y, w, h) in faces:
        roi_grayscale = grayscale[y: y + h, x: x + w]
        roi_resized = cv.resize(roi_grayscale, IMG_SIZE)

        id, confidence = model.predict(roi_grayscale)

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 240), 2)

        if confidence >= 50 and confidence <= 90:
            print(f"ID: {id}, Confidence: {confidence}")
            cv.putText(
                frame, 
                f"{label_map.get(id, "Unknown")} ({round(confidence, 2)})", 
                (x, y), 
                cv.FONT_HERSHEY_COMPLEX, 
                0.5,
                (255, 255, 255), 
                2, 
                cv.LINE_AA
            )

    # Final processed frame with bounding box
    cv.imshow('Camera Output', frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break   

# Close camera and window to free memory resources
webcam.release()
cv.destroyAllWindows()
