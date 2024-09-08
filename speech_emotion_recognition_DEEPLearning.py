import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Function to extract features from audio files
def extract_features(file_name):
    # Load the audio file
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return mfccs

# Function to load data from the dataset path
def load_data(dataset_path):
    features = []
    labels = []

    # Traverse through all files in the directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Extract the emotion label from the file name
                emotion = file.split(".")[0].split("_")[-1]  # Adjust this based on file naming convention
                data = extract_features(file_path)
                features.append(data)
                labels.append(emotion)
    
    # Find the maximum length of the MFCC feature arrays
    max_len = max([feature.shape[1] for feature in features])
    
    # Pad each MFCC array to the maximum length
    features_padded = np.array([np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant') for feature in features])
    
    labels = np.array(labels)
    return features_padded, labels

# Path to your dataset
dataset_path = r'C:\Users\Shrey\OneDrive\Desktop\data science projects\Speech Emotion Recognition using Deep Learning\data set\wav'

# Load the data
features, labels = load_data(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = to_categorical(label_encoder.fit_transform(labels))

# Reshape features for the TimeDistributed Conv2D input
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build the CNN + RNN Model
model = Sequential()

# TimeDistributed wrapper for Conv2D
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(None, X_train.shape[2], X_train.shape[3], 1)))

# Modify MaxPooling2D to pool only on height
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 1))))  # Only pool over the height dimension
model.add(TimeDistributed(Flatten()))

# RNN layers
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))

# Dense output layer
model.add(Dense(labels_encoded.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Visualize Training Results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()
