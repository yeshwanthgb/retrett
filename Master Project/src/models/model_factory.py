from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.preprocessor import build_preprocessor

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None


def build_model_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "ann": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            early_stopping=True,
            random_state=random_state,
            max_iter=500,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def hyperparameter_grids() -> dict[str, dict[str, list[object]]]:
    return {
        "logistic_regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"],
        },
        "decision_tree": {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "random_forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "svm": {
            "model__C": [0.5, 1.0, 3.0],
            "model__gamma": ["scale", "auto"],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
        },
        "xgboost": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.1],
        },
        "ann": {
            "model__hidden_layer_sizes": [(32,), (64, 32)],
            "model__alpha": [0.0001, 0.001],
        },
    }
