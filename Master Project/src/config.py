from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "reports" / "models" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "reports" / "models" / "metadata.json"

RANDOM_STATE = 42

CANONICAL_FEATURES = [
    "machine_type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

ENGINEERED_FEATURES = [
    "temperature_delta",
    "mechanical_power",
    "wear_torque_ratio",
    "thermal_stress_index",
]
