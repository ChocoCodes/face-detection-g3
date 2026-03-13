import cv2 as cv

# Load the pre-trained Viola-Jones XML model for detecting frontal faces
face_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
webcam = cv.VideoCapture(0)

while True:
    # Capture the current frame from the webcam feed
    ret, frame = webcam.read()
    
    # Convert to Grayscale and display raw grayscale version in a separate window
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray Scale Video', grayscale) 

    # Spread out the pixel intensities to ensure the face has enough contrast for the algorithm to "see" faces properly
    hist_equalized = cv.equalizeHist(grayscale)
    faces = face_cascade.detectMultiScale(hist_equalized, 1.1, 5)

    # Draw a bounding box when a front face is detected
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 240), 2)

    # Final processed frame with bounding box
    cv.imshow('Camera Output', frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break   

# Close camera and window to free memory resources
webcam.release()
cv.destroyAllWindows()
