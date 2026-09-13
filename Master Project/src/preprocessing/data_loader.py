from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "Type": "machine_type",
    "Machine type": "machine_type",
    "machine_type": "machine_type",
    "Air temperature [K]": "air_temperature",
    "air_temperature": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "process_temperature": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "rotational_speed": "rotational_speed",
    "Torque [Nm]": "torque",
    "torque": "torque",
    "Tool wear [min]": "tool_wear",
    "tool_wear": "tool_wear",
    "Machine failure": "failure_binary",
    "machine_failure": "failure_binary",
    "failure_binary": "failure_binary",
    "Failure Type": "failure_type",
    "failure_type": "failure_type",
}

FAILURE_SUBTYPE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_LABELS = {
    "TWF": "tool_wear_failure",
    "HDF": "heat_dissipation_failure",
    "PWF": "power_failure",
    "OSF": "overstrain_failure",
    "RNF": "random_failure",
}


def load_maintenance_dataset(path: Path) -> pd.DataFrame:
    """Load an AI4I/reference-style predictive-maintenance CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = normalize_maintenance_columns(pd.read_csv(path))
    validate_required_features(data)

    if "failure_binary" not in data.columns:
        subtype_cols = [col for col in FAILURE_SUBTYPE_COLUMNS if col in data.columns]
        if subtype_cols:
            data["failure_binary"] = data[subtype_cols].max(axis=1).astype(int)
        else:
            data["failure_binary"] = 0

    if "failure_type" not in data.columns:
        data["failure_type"] = _infer_failure_type(data)

    return data


def normalize_maintenance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize reference-repository, AI4I, and project-friendly column names."""
    data = data.rename(columns={col: COLUMN_ALIASES.get(col, col) for col in data.columns})
    return data.drop(columns=[col for col in ["UDI", "Product ID"] if col in data.columns])


def validate_required_features(data: pd.DataFrame) -> None:
    missing = {
        "machine_type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    } - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {sorted(missing)}")


def _infer_failure_type(data: pd.DataFrame) -> pd.Series:
    subtype_cols = [col for col in FAILURE_SUBTYPE_COLUMNS if col in data.columns]
    if not subtype_cols:
        return data["failure_binary"].map({0: "no_failure", 1: "failure"})

    def label_row(row: pd.Series) -> str:
        for col in subtype_cols:
            if int(row.get(col, 0)) == 1:
                return FAILURE_LABELS[col]
        return "no_failure"

    return data.apply(label_row, axis=1)
