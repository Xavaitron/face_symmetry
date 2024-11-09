import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from extract_features import get_symmetry_features

# Folder paths
images_folder = 'Images'
scores_file = os.path.join(images_folder, 'scores.txt')

# Load scores from scores.txt
image_scores = {}
with open(scores_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            image_name, score = parts
            image_scores[image_name] = float(score)

# Load features and match with scores
features = []
scores = []
for image_name in sorted(os.listdir(images_folder)):
    if image_name in image_scores:
        image_path = os.path.join(images_folder, image_name)
        feature = get_symmetry_features(image_path)
        if feature is not None:
            features.append(feature)
            scores.append(image_scores[image_name])

# Convert lists to numpy arrays
features = np.array(features)
scores = np.array(scores)

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(features, scores, test_size=0.2, random_state=69)

# Define an enhanced model with additional layers and lower dropout rates
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')  # Sigmoid to ensure output is between 0 and 1
])

# Configure the optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model and capture the training history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=16, callbacks=[early_stopping])

# Save the model
model.save('symmetry_model.h5')
print("Model training complete and saved as 'symmetry_model.h5'")

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
