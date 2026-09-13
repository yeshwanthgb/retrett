from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_shap_summary(model, x_train: pd.DataFrame, x_sample: pd.DataFrame, output_path: Path) -> None:
    """Create a SHAP summary plot when SHAP supports the fitted estimator."""
    try:
        import shap

        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed_train = preprocessor.transform(x_train)
        transformed_sample = preprocessor.transform(x_sample)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.Explainer(estimator, transformed_train)
        shap_values = explainer(transformed_sample)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(
            shap_values,
            transformed_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.with_suffix(".txt").write_text(
            f"SHAP summary could not be generated: {exc}", encoding="utf-8"
        )


def explain_single_prediction(model, row: pd.DataFrame) -> dict[str, float]:
    """Return local SHAP feature attributions for one row."""
    import shap

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(row)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.Explainer(estimator, transformed)
    shap_values = explainer(transformed)
    values = shap_values.values[0]
    if values.ndim > 1:
        values = values[:, -1]
    return dict(sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True))
