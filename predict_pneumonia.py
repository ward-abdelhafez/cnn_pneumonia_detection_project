import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")

# Specify the image path (expand ~ to the full path)
img_path = os.path.expanduser("~/Desktop/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg")

# Load and preprocess the image
img = load_img(img_path, target_size=(128, 128))  # Match training image size
img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(img_array)
threshold = 0.5  # Default threshold; adjust as needed (e.g., 0.6 for higher PNEUMONIA recall)
label = "PNEUMONIA" if prediction[0] > threshold else "NORMAL"
print(f"Prediction: {label} (Confidence: {prediction[0][0]:.2f})")