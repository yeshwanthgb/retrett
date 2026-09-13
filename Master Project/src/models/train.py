from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.explainability.shap_explainer import create_shap_summary
from src.models.evaluate import evaluate_classifier, metrics_to_frame
from src.models.model_factory import build_model_pipelines, hyperparameter_grids
from src.preprocessing.data_loader import load_maintenance_dataset
from src.preprocessing.preprocessor import split_features_target
from src.utils.io import ensure_dir, save_json, save_model
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_failure_distribution,
    plot_metric_comparison,
)


@dataclass
class TrainingResult:
    metrics_table: pd.DataFrame
    best_model_name: str
    artifact_dir: Path


def train_and_evaluate(
    data_path: Path,
    target: str,
    reports_dir: Path,
    tune: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    data = load_maintenance_dataset(data_path)
    x, y = split_features_target(data, target=target)
    label_mapping: dict[str, int] | None = None
    if y.dtype == "object" or target == "failure_type":
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(y.astype(str))
        label_mapping = {
            str(class_name): int(label_id)
            for label_id, class_name in enumerate(label_encoder.classes_)
        }
        y = pd.Series(encoded, index=y.index, name=target)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    reports_dir = ensure_dir(reports_dir)
    artifact_dir = ensure_dir(reports_dir / "artifacts")
    model_dir = ensure_dir(reports_dir / "models")
    plot_dir = ensure_dir(reports_dir / "figures")

    pipelines = build_model_pipelines(random_state=random_state)
    grids = hyperparameter_grids()
    metrics_by_model: dict[str, dict[str, object]] = {}
    fitted_models = {}

    for model_name, pipeline in pipelines.items():
        estimator = pipeline
        if tune and model_name in grids:
            estimator = GridSearchCV(
                pipeline,
                grids[model_name],
                scoring="f1_weighted",
                cv=3,
                n_jobs=-1,
            )
        estimator.fit(x_train, y_train)
        fitted = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
        fitted_models[model_name] = fitted
        metrics_by_model[model_name] = evaluate_classifier(fitted, x_test, y_test)

    metrics_table = metrics_to_frame(metrics_by_model)
    best_model_name = str(metrics_table.iloc[0]["model"])
    best_model = fitted_models[best_model_name]

    metrics_table.to_csv(artifact_dir / "model_comparison.csv", index=False)
    save_json(metrics_by_model, artifact_dir / "metrics.json")
    save_model(best_model, model_dir / "best_model.joblib")
    save_json(
        {
            "best_model_name": best_model_name,
            "target": target,
            "features": list(x.columns),
            "data_path": str(data_path),
            "label_mapping": label_mapping,
        },
        model_dir / "metadata.json",
    )

    plot_failure_distribution(data, plot_dir / "failure_distribution.png")
    plot_correlation_heatmap(data, plot_dir / "correlation_heatmap.png")
    plot_metric_comparison(metrics_table, plot_dir / "model_comparison.png")
    plot_confusion_matrix(
        metrics_by_model[best_model_name]["confusion_matrix"],
        plot_dir / "best_confusion_matrix.png",
    )
    create_shap_summary(best_model, x_train, x_test, plot_dir / "shap_summary.png")

    return TrainingResult(metrics_table, best_model_name, artifact_dir)
