"""Feature transformations that mirror notebooks/main01.ipynb."""
from __future__ import annotations

import numpy as np
import pandas as pd

# main01 applies sklearn LabelEncoder separately to these fixed Home Credit fields.
# These mappings reproduce its alphabetical class ordering and remain stable at inference.
_NOTEBOOK_LABEL_MAPS = {
    "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
    "FLAG_OWN_CAR": {"N": 0, "Y": 1},
    "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
    "EMERGENCYSTATE_MODE": {"No": 0, "Yes": 1, "nan": 2},
}


def prepare_application(application_df: pd.DataFrame, missing_threshold: float = 0.65) -> pd.DataFrame:
    """Apply main01's application cleaning, anomaly handling, and label encoding."""
    df = application_df.copy()
    high_missing = df.isna().mean()
    df = df.drop(columns=high_missing[high_missing > missing_threshold].index)

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"].eq(365243)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    for column, mapping in _NOTEBOOK_LABEL_MAPS.items():
        if column in df.columns:
            raw = df[column].astype(str)
            df[column] = raw.map(mapping)
            # Match LabelEncoder's string handling for missing values while making
            # previously unseen categories safely imputable at inference time.
    return df


def aggregate_bureau(bureau_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Create the same applicant-level bureau aggregates as notebook cell 27."""
    if bureau_df is None:
        return None
    required = {
        "SK_ID_CURR", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    }
    missing = required.difference(bureau_df.columns)
    if missing:
        raise ValueError(f"bureau data is missing required columns: {sorted(missing)}")
    agg = bureau_df.groupby("SK_ID_CURR").agg({
        "DAYS_CREDIT": ["min", "max", "sum"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "sum"],
    }).reset_index()
    agg.columns = ["_".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
                   for col in agg.columns]
    return agg.rename(columns={"SK_ID_CURR_": "SK_ID_CURR"})


def aggregate_previous_applications(previous_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Create the same applicant-level prior-application aggregates as cell 32."""
    if previous_df is None:
        return None
    required = {"SK_ID_CURR", "AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION"}
    missing = required.difference(previous_df.columns)
    if missing:
        raise ValueError(f"previous-application data is missing required columns: {sorted(missing)}")
    agg = previous_df.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "AMT_ANNUITY": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
    })
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]
    return agg.reset_index()


def build_features(
    main_df: pd.DataFrame,
    bureau_df: pd.DataFrame | None = None,
    prev_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build notebook-compatible model features; preserve TARGET and SK_ID_CURR."""
    df = prepare_application(main_df)
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("application data must include SK_ID_CURR")

    bureau_agg = aggregate_bureau(bureau_df)
    if bureau_agg is not None:
        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left", validate="one_to_one")

    previous_agg = aggregate_previous_applications(prev_df)
    if previous_agg is not None:
        df = df.merge(previous_agg, on="SK_ID_CURR", how="left", validate="one_to_one")

    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return pd.get_dummies(df, columns=categorical)
