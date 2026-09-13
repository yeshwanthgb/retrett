from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CANONICAL_FEATURES, ENGINEERED_FEATURES


NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    *ENGINEERED_FEATURES,
]
CATEGORICAL_FEATURES = ["machine_type"]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create physically meaningful features used by maintenance engineers."""
    df = data.copy()
    df["temperature_delta"] = df["process_temperature"] - df["air_temperature"]
    df["mechanical_power"] = (
        df["torque"] * df["rotational_speed"] * 2 * np.pi / 60.0
    )
    df["wear_torque_ratio"] = df["tool_wear"] / (df["torque"].abs() + 1e-6)
    df["thermal_stress_index"] = df["temperature_delta"] * df["torque"]
    return df


def split_features_target(
    data: pd.DataFrame, target: str = "failure_binary"
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is not available")

    features = add_engineered_features(data)
    selected = [*CANONICAL_FEATURES, *ENGINEERED_FEATURES]
    return features[selected], features[target]


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
