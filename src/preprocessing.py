"""Fit and reuse the training-only imputer and scaler used by main01."""
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def _save(transformer, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, path)


def fit_transform_imputer(X_train, strategy="median", save_path="models/imputer.pkl"):
    """Fit only on X_train, save the fitted imputer, and return array plus imputer."""
    imputer = SimpleImputer(strategy=strategy)
    transformed = imputer.fit_transform(X_train)
    _save(imputer, save_path)
    return transformed, imputer


def transform_imputer(X, imputer_path="models/imputer.pkl"):
    """Transform data using a saved fitted imputer."""
    path = Path(imputer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fitted imputer not found: {path}")
    return joblib.load(path).transform(X)


def fit_transform_minmax(X_train, X_val=None, save_path="models/scaler.pkl"):
    """Fit MinMaxScaler on training data only; optionally transform validation data."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(X_train)
    _save(scaler, save_path)
    if X_val is None:
        return train_scaled, scaler
    return train_scaled, scaler.transform(X_val), scaler
