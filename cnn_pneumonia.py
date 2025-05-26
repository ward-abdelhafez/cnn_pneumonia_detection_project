import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid plotting errors
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from build_model import build_pneumonia_model
from train_evaluate import compile_model, train_model, evaluate_model

# Paths
base_dir = os.path.expanduser("~/Desktop/chest_xray/chest_xray")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Image parameters (optimized for 8 GB memory)
img_size = (128, 128)  # Reduced from 224x224
batch_size = 16       # Reduced from 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)
val_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)
test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
)

# Compute class weights for imbalance
class_weights = {
    0: (1 / train_generator.classes.sum()) * (len(train_generator.classes) / 2.0),
    1: (1 / (len(train_generator.classes) - train_generator.classes.sum())) * (len(train_generator.classes) / 2.0)
}

# Build and fine-tune the model
model = build_pneumonia_model(img_size)
base_model = model.layers[0]  # Access VGG16 base
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compile the model
compile_model(model)

# Train the model (20 epochs specified here)
history = train_model(model, train_generator, val_generator, class_weights, epochs=20)

# Evaluate the model
evaluate_model(model, test_generator)

# Save the model
model.save("pneumonia_cnn_model.keras")  # Use .keras format

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig("training_plots.png")