import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(filepath):
    """Load dataset from path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Perform pre-processing steps:
    - Handle zero values in Cholesterol and RestingBP by replacing them with non-zero means.
    - One-hot encode categorical features.
    - Scale continuous variables.
    """
    df_cleaned = df.copy()
    
    # Impute Cholesterol
    ch_mean = df_cleaned.loc[df_cleaned['Cholesterol'] != 0, 'Cholesterol'].mean()
    df_cleaned['Cholesterol'] = df_cleaned['Cholesterol'].replace(0, ch_mean).round(2)
    
    # Impute RestingBP
    bp_mean = df_cleaned.loc[df_cleaned['RestingBP'] != 0, 'RestingBP'].mean()
    df_cleaned['RestingBP'] = df_cleaned['RestingBP'].replace(0, bp_mean).round(2)
    
    # One-hot encode categorical columns
    # Pandas get_dummies will auto-detect columns of type 'object'
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    
    # Convert all boolean columns to integers
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    # Scale continuous columns
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded, scaler

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a Heart Disease Classification model.")
    parser.add_argument("--data", type=str, default="heart.csv", help="Path to heart disease dataset (default: heart.csv)")
    parser.add_argument("--model", type=str, choices=["logistic_regression", "random_forest", "svm"], default="logistic_regression", help="Classifier type to train")
    parser.add_argument("--test_size", type=float, default=0.20, help="Test split ratio (default: 0.20)")
    parser.add_argument("--save", type=str, default=None, help="Save path for trained model pipeline (e.g. heart_model.joblib)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        df = load_data(args.data)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print("Preprocessing dataset (handling zero-imputation and scaling)...")
    df_processed, scaler = preprocess_data(df)
    
    X = df_processed.drop(columns=['HeartDisease'])
    y = df_processed['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    
    print(f"Training {args.model} classifier on {X_train.shape[0]} samples...")
    if args.model == "logistic_regression":
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif args.model == "random_forest":
        clf = RandomForestClassifier(random_state=42, n_estimators=100)
    elif args.model == "svm":
        clf = SVC(random_state=42, probability=True)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy Score: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if args.save:
        save_data = {
            "model_type": args.model,
            "model": clf,
            "scaler": scaler,
            "features": X.columns.tolist()
        }
        joblib.dump(save_data, args.save)
        print(f"\nModel pipeline successfully saved to {args.save}")

if __name__ == "__main__":
    main()
