import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(filepath):
    """Load dataset from path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df, encoding="onehot"):
    """
    Preprocess dataset:
    - Splits features and target (price).
    - Encodes categorical features ('model', 'transmission', 'fuelType').
    - Scales numerical features.
    """
    X = df.drop(columns=['price'], axis=1)
    y = df['price']
    
    categorical_cols = ['model', 'transmission', 'fuelType']
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    
    scaler = StandardScaler()
    
    if encoding == "onehot":
        # One-hot encoding
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Convert boolean dummies to int
        dummy_cols = [c for c in X_encoded.columns if c not in numerical_cols]
        X_encoded[dummy_cols] = X_encoded[dummy_cols].astype(int)
        
        # Scale numerical features
        X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
        return X_encoded, y, {"scaler": scaler, "encoding": "onehot"}
        
    elif encoding == "label":
        # Label encoding
        X_encoded = X.copy()
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            encoders[col] = le
            
        # Scale all columns
        all_cols = X_encoded.columns.tolist()
        X_encoded[all_cols] = scaler.fit_transform(X_encoded[all_cols])
        return X_encoded, y, {"scaler": scaler, "encoders": encoders, "encoding": "label"}
    else:
        raise ValueError(f"Invalid encoding type: {encoding}")

def train_and_evaluate(X, y, test_size=0.33, random_state=42):
    """Split, train Linear Regression model, and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    n = X_test.shape[0]
    p = X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    
    print("\n--- Model Evaluation Results ---")
    print(f"R² Score:          {r2:.4f}")
    print(f"Adjusted R² Score: {adj_r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a Car Price Prediction model.")
    parser.add_argument("--data", type=str, default="ford.csv", help="Path to the CSV dataset (default: ford.csv)")
    parser.add_argument("--encoding", type=str, choices=["onehot", "label"], default="onehot", help="Encoding style for categorical variables")
    parser.add_argument("--test_size", type=float, default=0.33, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--save", type=str, default=None, help="File path to save the trained model and preprocessors (e.g. car_model.joblib)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        df = load_data(args.data)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print(f"Preprocessing data using '{args.encoding}' encoding...")
    X, y, preprocessors = preprocess_data(df, encoding=args.encoding)
    
    print(f"Training Linear Regression model (Test split ratio: {args.test_size})...")
    model = train_and_evaluate(X, y, test_size=args.test_size)
    
    if args.save:
        save_data = {
            "model": model,
            "preprocessors": preprocessors
        }
        joblib.dump(save_data, args.save)
        print(f"Model and preprocessors successfully saved to {args.save}")

if __name__ == "__main__":
    main()
