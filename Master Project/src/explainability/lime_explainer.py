from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_lime_explanation(
    model,
    x_train: pd.DataFrame,
    row: pd.DataFrame,
    output_path: Path,
    class_names: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Generate and save a local LIME explanation as HTML."""
    from lime.lime_tabular import LimeTabularExplainer

    class_names = class_names or ["normal", "failure"]
    explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=list(x_train.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    explanation = explainer.explain_instance(
        row.iloc[0].to_numpy(),
        model.predict_proba,
        num_features=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explanation.save_to_file(str(output_path))
    return explanation.as_list()
