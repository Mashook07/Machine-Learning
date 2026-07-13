# Machine Learning Repository - Learning Phase

Welcome to the Machine Learning Learning Repository! This repository contains a collection of projects illustrating fundamental machine learning concepts, exploratory data analysis (EDA), preprocessing pipelines, model building, and modular python script implementations.

---

## Repository Structure

```
├── .gitignore               # Git ignore configuration
├── ford.csv                 # Ford Used Car price dataset
├── car-price.ipynb          # EDA & regression notebook for Ford cars
├── car_price_predictor.py   # CLI training and evaluation script for Ford cars
├── heart.csv                # Heart Disease dataset
├── heart.ipynb              # EDA & Preprocessing notebook for Heart Disease
├── heart_disease_classifier.py # CLI classification script for Heart Disease prediction
├── insurance.csv            # Medical Insurance premium dataset
├── ml.ipynb                 # Linear Regression notebook for Insurance charges prediction
├── insurance_predictor.py   # CLI regression script for Insurance charges prediction
├── Copy_of_TitanicModels.ipynb  # Titanic survival prediction ML notebook
├── app.py                   # Streamlit web app for model deployment
├── requirements.txt         # Project dependencies setup
└── README.md                # Repository documentation
```

---

## Project 1: Heart Disease Classification

### **Dataset Overview (`heart.csv`)**
A combination of multiple datasets containing 918 observations with 12 attributes. It predicts heart disease based on clinical features:
* **Numerical Features**: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`.
* **Categorical Features**: `Sex`, `ChestPainType`, `FastingBS`, `RestingECG`, `ExerciseAngina`, `ST_Slope`.
* **Target variable**: `HeartDisease` (1 = heart disease, 0 = normal).

### **Workflow & Analysis**
1. **Exploratory Data Analysis (EDA)**: Checking distributions and target class ratios.
2. **Imputation**: Zero values in `Cholesterol` and `RestingBP` are imputed with their non-zero means.
3. **Encoding & Scaling**: Categorical features are encoded and numerical features are standardized.

---

## Project 2: Insurance Charges Regression

### **Dataset Overview (`insurance.csv`)**
Consists of medical insurance customer profiles, aiming to predict charges based on physical, demographic, and lifestyle indicators:
* **Features**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
* **Target variable**: `charges` (yearly medical cost).

### **Workflow & Analysis**
1. **Correlation Testing**: Features are evaluated against the target using Pearson Correlation and Chi-Square contingency testing.
2. **Feature Selection**: An optimized feature subset is selected based on statistical significance.

---

## Project 3: Car Price Prediction (New)

### **Dataset Overview (`ford.csv`)**
Contains used Ford car details to predict their prices:
* **Features**: `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`.
* **Target variable**: `price`.

### **Workflow & Analysis**
1. **EDA**: Analyzing price trends relative to vehicle age, mileage, transmission type, and fuel type.
2. **Categorical Encoding**: Compares performance between One-Hot Encoding (`pd.get_dummies`) and Label Encoding (`LabelEncoder`).
3. **Model**: Trains a Linear Regression model on both encoded variations.

---

## Getting Started

### **Environment Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/Mashook07/Machine-Learning.git
   cd Machine-Learning
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Jupyter Notebooks**
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
Open any `.ipynb` notebook and execute the cells.

### **Running the Modular CLI Scripts**

This repository includes production-ready Python command-line tools for training, evaluating, and saving models.

#### 1. **Car Price Predictor**
Train a regression model using either `onehot` or `label` encoding:
```bash
# Run using onehot encoding and save the model
python car_price_predictor.py --data ford.csv --encoding onehot --save car_model.joblib

# Run using label encoding
python car_price_predictor.py --data ford.csv --encoding label
```

#### 2. **Insurance Charges Predictor**
Perform correlation tests and train an insurance charge regression model:
```bash
# Run with statistical significance tests and a subset of features
python insurance_predictor.py --data insurance.csv --stats --feature_subset --save insurance_model.joblib
```

#### 3. **Heart Disease Classifier**
Train classification models (logistic regression, random forest, or SVM) to predict heart disease risk:
```bash
# Train a Random Forest classifier and save the pipeline
python heart_disease_classifier.py --data heart.csv --model random_forest --save heart_model.joblib

# Train a Support Vector Machine classifier
python heart_disease_classifier.py --data heart.csv --model svm
```

---

## Project 4: Titanic Survival Prediction

### **Dataset Overview**
Classic Kaggle challenge — predict passenger survival aboard the Titanic based on demographic and ticket information:
* **Features**: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
* **Target variable**: `Survived` (1 = survived, 0 = did not survive).

### **Workflow & Analysis**
1. **EDA**: Visualizing survival rates by class, gender, and age groups.
2. **Feature Engineering**: Handling missing values, creating new features (title extraction, family size).
3. **Models Compared**: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
4. **Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 🚀 Web App (Streamlit)

An interactive web application is included to run predictions using trained models directly in the browser.

### **Running the App**
```bash
streamlit run app.py
```
The app lets you select a model (Heart Disease, Insurance, or Car Price) and input feature values to get real-time predictions.

---
