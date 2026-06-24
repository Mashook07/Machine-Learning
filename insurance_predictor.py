import os
import argparse
import pandas as pd
import numpy as np
import joblib
from scipy.stats import pearsonr, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(filepath):
    """Load dataset from path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Perform pre-processing steps:
    - Map 'smoker' and 'sex' variables.
    - Perform dummy encoding for 'region'.
    - Generate and encode 'bmi_category'.
    - Scale continuous variables.
    """
    df_cleaned = df.copy()
    
    # Map binary features
    df_cleaned['is_smoker'] = df_cleaned['smoker'].map({"no": 0, "yes": 1})
    df_cleaned['is_female'] = df_cleaned['sex'].map({"male": 0, "female": 1})
    df_cleaned.drop(columns=['sex', 'smoker'], inplace=True)
    
    # Region one-hot encoding
    df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
    
    # BMI categorization and encoding
    df_cleaned['bmi_category'] = pd.cut(
        df_cleaned['bmi'],
        bins=[0, 18.5, 24.9, 29.9, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
    
    # Convert all boolean columns to integers
    bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
    df_cleaned[bool_cols] = df_cleaned[bool_cols].astype(int)
    
    # Scale numeric columns
    scaler = StandardScaler()
    scale_cols = ['age', 'bmi', 'children']
    df_cleaned[scale_cols] = scaler.fit_transform(df_cleaned[scale_cols])
    
    return df_cleaned, scaler

def run_statistical_tests(df, target_col='charges'):
    """Calculate Pearson correlations and Chi-Square significance."""
    print("\n--- Pearson Correlation with Charges ---")
    features = [col for col in df.columns if col not in [target_col, 'charges_bin']]
    
    correlations = {}
    for col in features:
        # Calculate Pearson
        corr_val, _ = pearsonr(df[col], df[target_col])
        correlations[col] = corr_val
        
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation']).sort_values(by='Correlation', ascending=False)
    print(corr_df.to_string(index=False))
    
    print("\n--- Chi-Square Test of Association ---")
    df_temp = df.copy()
    df_temp['charges_bin'] = pd.qcut(df_temp[target_col], q=4, labels=False)
    
    alpha = 0.05
    chi2_results = []
    for col in features:
        contingency = pd.crosstab(df_temp[col], df_temp['charges_bin'])
        chi2_stat, p_val, _, _ = chi2_contingency(contingency)
        decision = 'Keep (p < 0.05)' if p_val < alpha else 'Drop (p >= 0.05)'
        chi2_results.append({
            'Feature': col,
            'Chi2 Stat': round(chi2_stat, 2),
            'p-value': round(p_val, 4),
            'Decision': decision
        })
    chi2_df = pd.DataFrame(chi2_results).sort_values(by='p-value')
    print(chi2_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an Insurance Charge regression model.")
    parser.add_argument("--data", type=str, default="insurance.csv", help="Path to insurance dataset (default: insurance.csv)")
    parser.add_argument("--test_size", type=float, default=0.20, help="Test split ratio (default: 0.20)")
    parser.add_argument("--stats", action="store_true", help="Run statistical significance tests before training")
    parser.add_argument("--feature_subset", action="store_true", help="Use optimized feature subset instead of all features")
    parser.add_argument("--save", type=str, default=None, help="Save path for trained model (e.g. insurance_model.joblib)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        df = load_data(args.data)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print("Preprocessing dataset...")
    df_processed, scaler = preprocess_data(df)
    
    # Statistical tests
    if args.stats:
        run_statistical_tests(df_processed)
        
    # Feature Selection
    if args.feature_subset:
        # Optimized subset from notebook
        selected_cols = ['age', 'is_female', 'bmi', 'children', 'is_smoker', 'region_southeast', 'bmi_category_Obese', 'region_northwest']
        print(f"\nUsing optimized feature subset: {selected_cols}")
        X = df_processed[selected_cols]
    else:
        # Use all processed features
        X = df_processed.drop(columns=['charges'])
        print(f"\nUsing all features: {X.columns.tolist()}")
        
    y = df_processed['charges']
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    
    print(f"Training Linear Regression model on {X_train.shape[0]} samples...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
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
    
    if args.save:
        save_data = {
            "model": model,
            "scaler": scaler,
            "features": X.columns.tolist()
        }
        joblib.dump(save_data, args.save)
        print(f"Model pipeline successfully saved to {args.save}")

if __name__ == "__main__":
    main()
