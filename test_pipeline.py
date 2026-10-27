import os
import joblib
import numpy as np

def test_artifacts_exist():
    """Verify that all production model artifacts are present."""
    assert os.path.exists('models/credit_risk_xgb.pkl'), "XGBoost model file missing!"
    assert os.path.exists('models/imputer.pkl'), "Imputer artifact missing!"
    assert os.path.exists('models/feature_names.pkl'), "Feature names list missing!"
    print("✓ Artifacts Existence Test Passed.")

def test_inference_shape_and_bounds():
    """Test model probability outputs on dummy data."""
    model = joblib.load('models/credit_risk_xgb.pkl')
    feature_names = joblib.load('models/feature_names.pkl')

    # Create dummy single-row sample
    dummy_input = np.zeros((1, len(feature_names)))
    probs = model.predict_proba(dummy_input)

    assert probs.shape == (1, 2), "Inference output shape mismatch!"
    assert 0.0 <= probs[0][1] <= 1.0, "Probability value out of standard [0, 1] range!"
    print("✓ Inference Pipeline Test Passed.")

if __name__ == '__main__':
    print("--- Executing Pipeline Unit Tests ---")
    test_artifacts_exist()
    test_inference_shape_and_bounds()
    print("--- All Unit Tests Passed Successfully! ---")