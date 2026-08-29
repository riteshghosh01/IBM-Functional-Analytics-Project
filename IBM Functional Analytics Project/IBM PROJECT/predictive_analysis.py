from typing import Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def detect_ml_task(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> str:
    """Choose regression, classification, or unsupported based on target dtype."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    target = df[target_col]
    if pd.api.types.is_numeric_dtype(target):
        return "regression"
    if target.nunique() <= 20:
        return "classification"
    return "unsupported"


def build_and_evaluate_model(df: pd.DataFrame, target_col: str, feature_cols: List[str]):
    """Train a model using preprocessing and return common evaluation metrics."""
    if not feature_cols or target_col not in df.columns:
        raise ValueError("Please select valid feature columns and a target column.")

    model_df = df[feature_cols + [target_col]].dropna(subset=feature_cols + [target_col]).copy()
    if model_df.empty:
        raise ValueError("Not enough valid rows for model training.")

    X = model_df[feature_cols]
    y = model_df[target_col]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    task = detect_ml_task(model_df, target_col, feature_cols)
    if task == "unsupported":
        raise ValueError("The selected target is not suitable for automatic regression or classification.")

    if task == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    if task == "regression":
        metrics = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "mse": mean_squared_error(y_test, preds),
        }
    else:
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
        }

    result_df = pd.DataFrame({"actual": y_test.reset_index(drop=True), "predicted": preds})
    return {"task": task, "metrics": metrics, "predictions": result_df, "model": pipeline}
