import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define the paths to the training and testing datasets
train_dataset_path = r'S:\Projects\Python-BasedProject\ASLDetectionProject\asl_alphabet_train'
test_dataset_path = r'S:\Projects\Python-BasedProject\ASLDetectionProject\asl_alphabet_test'

# Create a DataFrame for training data
train_data = []
train_labels = []

# Load training images and labels
for letter in os.listdir(train_dataset_path):
    letter_path = os.path.join(train_dataset_path, letter)
    if os.path.isdir(letter_path):
        for image_file in os.listdir(letter_path):
            image_path = os.path.join(letter_path, image_file)
            train_data.append(image_path)
            train_labels.append(letter)

# Create a DataFrame
train_df = pd.DataFrame({'image': train_data, 'label': train_labels})

# Shuffle the training DataFrame
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Load training images into a NumPy array
X_train = []
y_train = []

for index, row in train_df.iterrows():
    img = cv2.imread(row['image'])
    img = cv2.resize(img, (64, 64))  # Resize images to 64x64
    X_train.append(img)
    y_train.append(row['label'])

X_train = np.array(X_train) / 255.0  # Normalize pixel values to [0, 1]

# Convert labels to numerical format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # Automatically maps labels to numbers

# Step 2: Build the Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Number of classes dynamically determined
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Step 4: Train the Model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=10)

# Step 5: Save the Model
model.save('asl_alphabet_model.h5')

# Evaluate the model on the test dataset
test_data = []
test_labels = []

# Load test images and labels
for letter in os.listdir(test_dataset_path):
    letter_path = os.path.join(test_dataset_path, letter)
    if os.path.isdir(letter_path):
        for image_file in os.listdir(letter_path):
            image_path = os.path.join(letter_path, image_file)
            test_data.append(image_path)
            test_labels.append(letter)

# Create a DataFrame for test data
test_df = pd.DataFrame({'image': test_data, 'label': test_labels})

# Load test images into a NumPy array
X_test = []
y_test = []

for index, row in test_df.iterrows():
    img = cv2.imread(row['image'])
    img = cv2.resize(img, (64, 64))  # Resize images to 64x64
    X_test.append(img)
    y_test.append(row['label'])

X_test = np.array(X_test) / 255.0  # Normalize pixel values to [0, 1]
y_test = label_encoder.transform(y_test)  # Transform test labels to numerical format

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.2f}')
