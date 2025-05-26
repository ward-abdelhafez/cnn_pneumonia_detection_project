import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout


def build_pneumonia_model(img_size=(128, 128)):
    """
    Builds a CNN model for pneumonia detection using VGG16 as the base.

    Args:
        img_size (tuple): Image size (height, width) for input images.

    Returns:
        model: Compiled Keras Sequential model.
    """
    # Load VGG16 with pre-trained weights, excluding the top layers
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False  # Freeze VGG16 layers for transfer learning

    # Build the model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # Binary classification (NORMAL vs. PNEUMONIA)
    ])

    return model