import cv2
import dlib
import numpy as np
import os

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's repository

def get_symmetry_features(image):
    # If image is a path (for training), load the image
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Error: Unable to load image at {image}")
    
    # Convert the image to grayscale (works for both path-based and frame-based images)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    if len(faces) == 0:
        return None

    # Extract features from the first detected face (you can modify this if needed)
    face = faces[0]
    landmarks = predictor(gray, face)

    # Extract symmetry features using landmarks (adjust based on model's feature requirements)
    features = []

    # Example: Distance between eyes, nose, and mouth (add more features if required)
    left_eye = landmarks.part(36)
    right_eye = landmarks.part(45)
    nose = landmarks.part(30)
    left_mouth = landmarks.part(48)
    right_mouth = landmarks.part(54)
    
    # Distances between key facial landmarks
    left_eye_to_nose = np.linalg.norm([left_eye.x - nose.x, left_eye.y - nose.y])
    right_eye_to_nose = np.linalg.norm([right_eye.x - nose.x, right_eye.y - nose.y])
    mouth_distance = np.linalg.norm([left_mouth.x - right_mouth.x, left_mouth.y - right_mouth.y])
    
    # Symmetry-related distances between left and right facial features
    eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
    nose_to_mouth_left = np.linalg.norm([left_mouth.x - nose.x, left_mouth.y - nose.y])
    nose_to_mouth_right = np.linalg.norm([right_mouth.x - nose.x, right_mouth.y - nose.y])

    # More symmetry-related distances and features
    left_cheek = landmarks.part(2)  # Left cheekbone
    right_cheek = landmarks.part(14)  # Right cheekbone
    cheek_distance = np.linalg.norm([left_cheek.x - right_cheek.x, left_cheek.y - right_cheek.y])

    # Chin-to-nose and chin-to-eyes distances
    chin = landmarks.part(8)  # Chin point
    chin_to_nose = np.linalg.norm([chin.x - nose.x, chin.y - nose.y])
    chin_to_left_eye = np.linalg.norm([chin.x - left_eye.x, chin.y - left_eye.y])
    chin_to_right_eye = np.linalg.norm([chin.x - right_eye.x, chin.y - right_eye.y])

    # Additional distances or symmetry features
    # Example: Distance from the center of the eyes to the center of the mouth
    eye_center = [(left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2]
    mouth_center = [(left_mouth.x + right_mouth.x) / 2, (left_mouth.y + right_mouth.y) / 2]
    eyes_to_mouth_center = np.linalg.norm([eye_center[0] - mouth_center[0], eye_center[1] - mouth_center[1]])

    # Add features to the list
    features.extend([left_eye_to_nose, right_eye_to_nose, mouth_distance, eye_distance,
                    nose_to_mouth_left, nose_to_mouth_right, cheek_distance, chin_to_nose,
                    chin_to_left_eye, chin_to_right_eye, eyes_to_mouth_center])
    
    # Ensure the feature vector is of the expected size (102 or whatever the model was trained on)
    while len(features) < 102:
        features.append(0)  # Padding with zeros or you can apply other logic here
    
    return np.array(features)

# Example function for batch processing
def extract_all_features(images_folder):
    features = []
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('jpg', 'jpeg', 'png'))])
    i = 0
    for image_name in image_files:
        if i > 500:
            break
        i += 1
        image_path = os.path.join(images_folder, image_name)
        feature = get_symmetry_features(image_path)
        if feature is not None:
            features.append(feature)
    return np.array(features)

# Run this only if you need to test feature extraction separately
if __name__ == "__main__":
    features = extract_all_features("images")  # Folder containing images
    print("Extracted Features Shape:", features.shape)
