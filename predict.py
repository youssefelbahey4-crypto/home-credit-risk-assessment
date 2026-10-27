import os
import json
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import build_features
from src.preprocessing import transform_imputer

def predict_loan_default(new_data_path, models_dir="./models",
                         bureau_path=None, previous_path=None):
    """
    Loads raw input data, runs feature engineering, transforms features,
    and applies the optimal financial decision threshold (0.1773).
    """
    # 1. Load trained artifacts
    model_path = os.path.join(models_dir, "credit_risk_xgb.pkl")
    imputer_path = os.path.join(models_dir, "imputer.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    # 2. Read new input data
    print(f"--- Loading Input File: {new_data_path} ---")
    raw_df = pd.read_csv(new_data_path)

    bureau_df = pd.read_csv(bureau_path) if bureau_path else None
    previous_df = pd.read_csv(previous_path) if previous_path else None

    # 3. Apply Feature Engineering
    processed_df = build_features(raw_df, bureau_df=bureau_df, prev_df=previous_df)

    # Align features with training dataset
    X = processed_df.reindex(columns=feature_names, fill_value=0)

    # 4. Impute Missing Values
    X_imputed = transform_imputer(X, imputer_path=imputer_path)

    # 5. Predict Probabilities
    probs = model.predict_proba(X_imputed)[:, 1]

    # 6. Apply Optimal Business Decision Threshold
    metadata_path = os.path.join(models_dir, "decision_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as metadata_file:
            BUSINESS_THRESHOLD = json.load(metadata_file).get("decision_threshold", 0.1773)
    else:
        BUSINESS_THRESHOLD = 0.1773
    decisions = (probs >= BUSINESS_THRESHOLD).astype(int)

    # Output DataFrame
    results = pd.DataFrame({
        'SK_ID_CURR': raw_df['SK_ID_CURR'] if 'SK_ID_CURR' in raw_df.columns else raw_df.index,
        'Default_Probability': np.round(probs, 4),
        'Decision_Threshold': BUSINESS_THRESHOLD,
        'Loan_Status': np.where(decisions == 1, 'REJECT (High Default Risk)', 'APPROVE')
    })

    return results

if __name__ == '__main__':
    print("--- Running Standalone Inference Pipeline ---")
    # Example usage:
    # predictions = predict_loan_default('./data/application_test.csv')
    # print(predictions.head())
