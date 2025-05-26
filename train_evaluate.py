import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score


def compile_model(model):
    """
    Compiles the model with  legacy Adam optimizer (normal adam optimizer did not work), binary crossentropy loss, and accuracy metric.

    Args:
        model: Keras model to compile.

    Returns:
        None
    """
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


def train_model(model, train_generator, val_generator, class_weights, epochs=10):
    """
    Trains the model using the provided data generators.

    Args:
        model: Keras model to train.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        class_weights: Dictionary of class weights for imbalanced data.
        epochs (int): Number of training epochs.

    Returns:
        history: Training history object.
    """
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights
    )
    return history


def evaluate_model(model, test_generator):
    """
    Evaluates the model on the test set and prints metrics.

    Args:
        model: Trained Keras model.
        test_generator: Test data generator.

    Returns:
        None
    """
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Detailed metrics
    y_pred = model.predict(test_generator)
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true = test_generator.classes
    print(classification_report(y_true, y_pred_binary, target_names=["NORMAL", "PNEUMONIA"]))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred):.4f}")