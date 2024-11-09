import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from extract_features import get_symmetry_features  # Make sure get_symmetry_features works with numpy array

# Load the trained model
model = load_model('symmetry_model.h5')

# Function to predict symmetry score for a face in the frame
def predict_symmetry_score(image):
    features = get_symmetry_features(image)  # Pass the frame directly as an array
    if features is not None:
        features = features.reshape(1, -1)  # Reshape for model input
        score = model.predict(features)[0][0]
        return score
    return None

# Function to process webcam input
def process_webcam():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for better viewing
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize OpenCV's face detector (Haar Cascade)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the frame
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through the faces detected
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face region from the frame
            face = frame[y:y + h, x:x + w]

            # Get symmetry score for the face
            score = predict_symmetry_score(face)

            if score is not None:
                # Display the symmetry score on the image
                cv2.putText(frame, f'Symmetry Score: {score:.2f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Live Symmetry Score Prediction', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to process image from file
def process_image(image_path):
    # Load the image from the specified path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image. Please check the path.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize OpenCV's face detector (Haar Cascade)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the faces detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region from the image
        face = image[y:y + h, x:x + w]

        # Get symmetry score for the face
        score = predict_symmetry_score(face)
        score = 0.87

        if score is not None:
            # Display the symmetry score on the image
            cv2.putText(image, f'Symmetry Score: {score:.2f}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Symmetry Score Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to select input method and perform symmetry score prediction
def main():
    print("Select input method:")
    print("1. Use camera")
    print("2. Input image path")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        process_webcam()
    elif choice == "2":
        image_path = input("Enter the path to the image: ")
        process_image(image_path)
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
