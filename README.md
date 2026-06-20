# Machine Learning Repository - Learning Phase

Welcome to the Machine Learning Learning Repository! This repository contains a collection of projects illustrating fundamental machine learning concepts, exploratory data analysis (EDA), preprocessing pipelines, and model building.

---

## Repository Structure

```
├── heart.csv            # Heart Disease dataset
├── heart.ipynb          # EDA & Preprocessing notebook for Heart Disease dataset
├── insurance.csv        # Medical Insurance premium dataset
├── ml.ipynb             # Linear Regression notebook for Insurance charges prediction
└── README.md            # Repository documentation
```

---

## Project 1: Heart Disease Classification (Preprocessing & EDA)

### **Dataset Overview (`heart.csv`)**
A combination of multiple datasets containing 918 observations with 12 attributes. It is a classic dataset used to predict heart disease based on clinical features:
* **Numerical Features**: `Age`, `RestingBP` (blood pressure), `Cholesterol`, `MaxHR` (max heart rate), `Oldpeak`.
* **Categorical Features**: `Sex`, `ChestPainType` (TA, ATA, NAP, ASY), `FastingBS` (fasting blood sugar), `RestingECG`, `ExerciseAngina`, `ST_Slope`.
* **Target variable**: `HeartDisease` (1 = heart disease, 0 = normal).

### **Workflow & Analysis (`heart.ipynb`)**
1. **Exploratory Data Analysis (EDA)**:
   * Checked the shape of the dataset ($918 \times 12$).
   * Verified that there are no missing (`null`) values across columns.
   * Analyzed statistical summary statistics (`describe()`).
2. **Visual Analysis**:
   * Used Seaborn and Matplotlib to analyze class distributions for categorical features, specifically visualizing `HeartDisease` distribution and the types of chest pain (`ChestPainType`).
3. **Data Preprocessing**:
   * **Categorical Encoding**: Converted categorical variables into dummy indicators using `pd.get_dummies(drop_first=True)` and casted them to integers.
   * **Feature Scaling**: Scaled the continuous numerical columns (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) using `StandardScaler` to ensure features are centered around 0 with a unit standard deviation. This prepares the data for algorithms sensitive to scale (such as SVMs, Logistic Regression, or Neural Networks).

---

## Project 2: Insurance Charges Regression (Linear Regression Model)

### **Dataset Overview (`insurance.csv`)**
Consists of medical insurance customer profiles, aiming to predict charges based on physical, demographic, and lifestyle indicators:
* **Features**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
* **Target variable**: `charges` (yearly medical cost).

### **Workflow & Analysis (`ml.ipynb`)**
1. **Correlation and Association Testing**:
   * Measured relationship between continuous variables using Pearson Correlation Coefficient (`pearsonr`).
   * Conducted statistical significance testing between categorical indicators and charges using the Chi-Square Contingency Test (`chi2_contingency`).
2. **Model Training & Evaluation**:
   * Split the dataset into training and testing subsets using `train_test_split`.
   * Standardized features to optimize gradient descent performance.
   * Trained a **Linear Regression** model (`LinearRegression`) to map features to continuous insurance costs.
   * Evaluated predictions on test data using the R-Squared Metric (`r2_score`) to analyze the percentage of variance explained by the model.

---

## Getting Started

### **Prerequisites**
Make sure you have python and standard data science libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### **Running the Notebooks**
1. Clone the repository:
   ```bash
   git clone https://github.com/Mashook07/machine-learning.git
   cd machine-learning
   ```
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Open either `heart.ipynb` or `ml.ipynb` and execute the cells.
