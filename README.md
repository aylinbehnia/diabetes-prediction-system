👤 Gender Classification from Facial Images

An end-to-end Computer Vision and Machine Learning project for classifying facial images into two dataset-defined classes: man and woman.

The project implements a complete image classification pipeline, including face detection, image preprocessing, feature extraction, model training, evaluation, and inference.

The current version uses MTCNN for face detection and preprocessing, followed by SGDClassifier as a baseline machine learning model.

«Important: This model predicts the labels defined in the dataset based on visual features. It does not determine a person's actual gender identity.»

---

🎯 Project Overview

The main goal of this project is to explore a practical Computer Vision + Machine Learning pipeline for facial image classification.

Instead of directly feeding raw images into a classifier, the project first detects and extracts the face using MTCNN, preprocesses the image, and then converts it into a numerical feature vector that can be used by a traditional machine learning model.

Pipeline

Input Image
     ↓
Face Detection
     ↓
Face Cropping
     ↓
Resize to 32 × 32
     ↓
Pixel Normalization
     ↓
Feature Vector
     ↓
SGD Classifier
     ↓
Prediction

---

✨ Key Features

👁️ Face Detection

- Detects faces using MTCNN
- Extracts the detected face region
- Removes unnecessary image background before classification

🖼️ Image Preprocessing

Each detected face goes through the following preprocessing steps:

1. Face detection
2. Face cropping
3. Resizing to 32 × 32 pixels
4. Pixel normalization to the range [0, 1]
5. Flattening into a 3072-dimensional feature vector

🤖 Machine Learning

The current implementation uses:

SGDClassifier — Scikit-learn

The model is used as a baseline classifier before moving toward more advanced deep learning approaches.

📊 Model Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Evaluation results are automatically stored in the "outputs/" directory.

---

🗂️ Dataset Structure

The dataset is organized into training, validation, and testing subsets:

man-woman/
│
├── train/
│   ├── man/
│   └── woman/
│
├── val/
│   ├── man/
│   └── woman/
│
└── test/
    ├── man/
    └── woman/

The model is trained using the training data and evaluated on unseen data.

---

🛠️ Tech Stack

Technology| Purpose
Python| Core development
OpenCV| Image processing
MTCNN| Face detection
NumPy| Numerical operations
Scikit-learn| Machine Learning
TensorFlow| MTCNN dependency / computer vision pipeline
Matplotlib| Visualization
Seaborn| Evaluation visualization
Joblib| Model persistence

---

🏗️ Project Structure

gender-classification/
│
├── data/
│   └── man-woman/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   └── gender_classifier.joblib
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── metrics.txt
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

🔬 Machine Learning Workflow

1. Face Detection

MTCNN identifies the face region within the input image.

2. Face Preprocessing

The detected face is cropped, resized to 32 × 32, and normalized.

3. Feature Extraction

The RGB image is flattened into a numerical feature vector containing:

32 × 32 × 3 = 3072 features

4. Model Training

The extracted features are passed to an SGDClassifier to learn the classification boundary.

5. Evaluation

The trained model is evaluated using standard classification metrics and a confusion matrix.

6. Inference

The trained model can be loaded from:

models/gender_classifier.joblib

and used to classify new images.

---

📊 Evaluation

The project generates the following evaluation outputs:

outputs/
├── confusion_matrix.png
├── classification_report.txt
└── metrics.txt

These outputs provide information about:

- Accuracy
- Precision
- Recall
- F1-score
- Classification performance
- Prediction errors

---

🚀 Getting
