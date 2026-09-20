🩺 Diabetes Prediction AI System

An end-to-end Machine Learning project for diabetes risk prediction, developed using Python and a structured data science workflow.

The project covers the complete machine learning pipeline, including data preprocessing, exploratory data analysis, feature transformation, model training, evaluation, visualization, and deployment through an interactive desktop application.

The final application provides a user-friendly interface for entering patient information, generating a diabetes risk prediction, and viewing the prediction results through an interactive dashboard and PDF report.

---

🎯 Project Overview

The goal of this project is to develop a machine learning system capable of predicting the likelihood of diabetes based on patient health-related features.

The project follows an end-to-end workflow:

Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Transformation
     ↓
Train / Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Selection
     ↓
Prediction System
     ↓
Interactive GUI & Report

---

✨ Key Features

📊 Data Analysis & Preprocessing

- Data loading and exploration using Pandas and NumPy
- Detection and handling of invalid zero values
- Data preprocessing and normalization
- Exploratory data analysis
- Statistical analysis and visualization
- Feature distribution analysis

🤖 Machine Learning Models

The project implements and evaluates multiple classification algorithms:

- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)

Model performance is compared using standard classification metrics.

📈 Model Evaluation

The project includes:

- Accuracy
- Confusion Matrix
- Classification Report
- Cross-validation
- Model comparison

📉 Data Visualization

Visualizations are created using:

- Matplotlib
- Seaborn

Including:

- Feature distributions
- Boxplots
- Model evaluation visualizations
- Confusion matrix
- Prediction-related visualizations

🖥️ Interactive Prediction Application

A desktop GUI built with Tkinter allows users to:

- Enter patient information
- Run the trained machine learning model
- Generate a diabetes risk prediction
- View prediction results
- Access model-related information

📄 Medical Report Generation

The system also provides a report-generation workflow that allows prediction results to be exported as a PDF report.

«Note: This project is intended for educational and demonstration purposes and should not be used as a medical diagnostic tool.»

---

🗂️ Dataset

The project uses the Pima Indians Diabetes Dataset, a commonly used dataset for binary diabetes classification.

The dataset contains medical and demographic features such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The target variable represents the diabetes outcome.

---

🛠️ Tech Stack

Technology| Purpose
Python| Core development
NumPy| Numerical computation
Pandas| Data manipulation and analysis
Matplotlib| Data visualization
Seaborn| Statistical visualization
Scikit-learn| Machine Learning
Tkinter| Desktop GUI
Joblib / Pickle| Model persistence
PDF Generation| Prediction report generation

---

🏗️ Project Structure

diabetes-prediction-system/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── 01_training.ipynb
│
├── models/
│   └── saved_model.pkl
│
├── src/
│   ├── config.py
│   ├── evaluation.py
│   ├── impurity.py
│   ├── logger.py
│   ├── main.py
│   ├── models.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── outputs/
│   ├── figures/
│   └── reports/
│
└── README.md

---

🔬 Machine Learning Workflow

1. Data Preprocessing

The dataset is inspected for missing, invalid, and zero-valued entries.

Relevant preprocessing steps are then applied before model training.

2. Exploratory Data Analysis

The dataset is analyzed to understand:

- Feature distributions
- Relationships between variables
- Class distribution
- Potential outliers

3. Model Training

Multiple supervised learning algorithms are trained and evaluated on the processed dataset.

4. Model Evaluation

Models are evaluated using classification metrics such as:

Accuracy
Confusion Matrix
Precision
Recall
F1-Score
Cross-Validation

5. Model Selection

The trained models are compared based on their evaluation results, with the selected model saved for later inference.

6. Prediction

The saved model can then be used by the desktop application to generate predictions for new input data.

---

📊 Model Performance

The project evaluates the following models:

Model| Accuracy
Decision Tree| ~79.2%
Random Forest| ~74.7%
Logistic Regression| ~74.7%
KNN| ~66.2%

«Results depend on the train/test split and preprocessing configuration used during experimentation.»

---

🖥️ Application Screenshots

Main Dashboard

<!-- Add screenshot here -->Prediction Interface

<!-- Add screenshot here -->Model Evaluation

<!-- Add screenshot here -->Generated Report

<!-- Add screenshot here -->---

🚀 Getting Started

1. Clone the repository

git clone https://github.com/aylinbehnia/diabetes-prediction-system.git

2. Navigate to the project

cd diabetes-prediction-system

3. Install dependencies

pip install numpy pandas matplotlib seaborn scikit-learn

4. Run the project

Depending on the project entry point:

python src/main.py

or run the training notebook:

notebooks/01_training.ipynb

---

📚 Skills Demonstrated

This project demonstrates practical experience with:

- Python programming
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Supervised Machine Learning
- Classification algorithms
- Model evaluation
- Feature engineering
- Cross-validation
- Confusion matrix analysis
- Model persistence
- GUI development
- End-to-end ML workflow

---

🔮 Future Improvements

Potential future improvements include:

- Hyperparameter tuning using GridSearchCV / RandomizedSearchCV
- Advanced feature engineering
- Improved model comparison
- Explainable AI using SHAP or similar techniques
- REST API deployment
- Web-based prediction interface
- Dockerization
- Cloud deployment
- Automated ML pipeline
- Model monitoring and versioning

---

👩‍💻 Author

Aylin Behnia

Computer Engineering Student | Aspiring Machine Learning Engineer

Interested in Machine Learning, Data Analysis, Python, and SQL.

🔗 GitHub: https://github.com/aylinbehnia

---

⚠️ Disclaimer

This project is developed for educational and portfolio purposes only.

The predictions generated by this system are not intended to replace professional medical advice, diagnosis, or treatment.
