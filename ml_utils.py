import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalize_target(series: pd.Series) -> pd.Series:
    """Normalize NSP labels to integers 1, 2 and 3."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any() or not set(values.astype(int).unique()).issubset({1, 2, 3}):
            raise ValueError("NSP must contain only 1, 2 and 3.")
        return values.astype(int)

    reverse = {
        "normal": 1,
        "suspect": 2,
        "suspected": 2,
        "pathologic": 3,
        "pathological": 3,
    }
    mapped = series.astype(str).str.strip().str.lower().map(reverse)
    if mapped.isna().any():
        raise ValueError("NSP must contain 1/2/3 or Normal/Suspect/Pathologic labels.")
    return mapped.astype(int)


def safe_auc(y_true, probabilities, average="weighted"):
    try:
        return roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average=average,
        )
    except ValueError:
        return np.nan


def evaluate_model(model, x_eval, y_eval):
    predictions = model.predict(x_eval)
    probabilities = model.predict_proba(x_eval)

    metrics = {
        "Accuracy": accuracy_score(y_eval, predictions),
        "AUC": safe_auc(y_eval, probabilities, average="weighted"),
        "Precision": precision_score(
            y_eval, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_eval, predictions, average="weighted", zero_division=0
        ),
        "F1": f1_score(y_eval, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_eval, predictions),
        "Macro Precision": precision_score(
            y_eval, predictions, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_eval, predictions, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(y_eval, predictions, average="macro", zero_division=0),
        "Macro AUC": safe_auc(y_eval, probabilities, average="macro"),
    }
    return metrics, predictions, probabilities
