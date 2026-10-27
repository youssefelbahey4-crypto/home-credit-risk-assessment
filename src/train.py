"""Train XGBoost models using the setup in notebooks/main01.ipynb."""
from pathlib import Path
import joblib
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def _save_model(model, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def train_xgboost(X_train, y_train, params=None, save_path="models/credit_risk_xgb_model.pkl"):
    """Fit the notebook's initial 200-tree XGBoost benchmark and save it."""
    if params is None:
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return _save_model(model, save_path)


def tune_xgboost(X_train, y_train, save_path="models/credit_risk_xgb_model.pkl"):
    """Run main01's 10-draw, three-fold ROC-AUC randomized search and save best model."""
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        estimator=XGBClassifier(eval_metric="auc", random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    _save_model(search.best_estimator_, save_path)
    return search.best_estimator_, search
