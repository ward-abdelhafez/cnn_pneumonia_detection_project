# CNN Pneumonia Detection Project

A Convolutional Neural Network (CNN) for detecting pneumonia in chest X-ray images, built using TensorFlow and a pre-trained VGG16 model. This project achieves a test accuracy of 90.38% and an ROC-AUC of 0.9708 on the Kaggle Chest X-ray Pneumonia dataset.

## Project Structure
- `cnn_pneumonia.py`: Main script for training and evaluating the model.
- `build_model.py`: Defines the VGG16-based model architecture.
- `train_evaluate.py`: Functions for training and evaluation.
- `predict_pneumonia.py`: Script for making predictions on new images.
- `generate_report_pdf.py`: Generates a PDF report comparing training runs.
- `training_plots.png`: Training history plots (loss and accuracy).
- `training_report.pdf`: Combined report of training runs.

## Requirements
- Python 3.9
- TensorFlow 2.15.0 with `tensorflow-metal` (for M1 GPU support)
- Other dependencies: `numpy`, `matplotlib`, `reportlab`, `scikit-learn`

Install dependencies:
```bash
conda create -n medical_cnn python=3.9
conda activate medical_cnn
pip install tensorflow==2.15.0 tensorflow-metal numpy matplotlib reportlab scikit-learn


