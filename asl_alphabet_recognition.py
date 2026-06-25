import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('asl_alphabet_model.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the expected input shape for the model
    roi = cv2.resize(frame, (64, 64))
    roi = np.array(roi) / 255.0  # Normalize
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(roi)
    label_index = np.argmax(predictions)
    label = chr(label_index + ord('A'))  # Convert to corresponding letter

    # Display the prediction on the frame
    cv2.putText(frame, f'Sign: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Interpreter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
