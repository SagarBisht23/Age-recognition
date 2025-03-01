import cv2
import numpy as np
import os

# Set the paths to the model files
prototxt_path = 'age_deploy.prototxt'
caffemodel_path = 'age_net.caffemodel'

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Age groups and model mean values
AGE_GROUPS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78, 87, 115)

# Predict age of detected face
def predict_age(frame, face):
    x, y, w, h = face
    face_img = frame[y:y+h, x:x+w]  # Crop the face region
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES)  # Preprocess the image
    age_net.setInput(blob)  # Set the image as input to the network
    age_predictions = age_net.forward()  # Get predictions
    return AGE_GROUPS[age_predictions[0].argmax()]  # Return the age group with the highest probability

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read frame from the webcam
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray)  # Detect faces

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Predict age for the face
        age = predict_age(frame, (x, y, w, h))
        # Display the predicted age on the frame
        cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Age Recognition', frame)  # Show the frame with age prediction
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Age Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
