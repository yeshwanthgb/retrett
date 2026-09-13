from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_scores(model, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities.max(axis=1)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return np.asarray(scores)
    return model.predict(x_test)


def evaluate_classifier(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    y_pred = model.predict(x_test)
    average = "binary" if y_test.nunique() == 2 else "weighted"

    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    try:
        scores = predict_scores(model, x_test)
        if y_test.nunique() == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, scores)
        else:
            probabilities = model.predict_proba(x_test)
            metrics["roc_auc"] = roc_auc_score(
                y_test, probabilities, multi_class="ovr", average="weighted"
            )
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def metrics_to_frame(metrics_by_model: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False)
