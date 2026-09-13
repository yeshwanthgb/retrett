from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


FAILURE_TYPES = [
    "no_failure",
    "tool_wear_failure",
    "heat_dissipation_failure",
    "power_failure",
    "overstrain_failure",
    "random_failure",
]


def generate_synthetic_maintenance_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate an AI4I-style industrial predictive-maintenance dataset."""
    rng = np.random.default_rng(seed)

    machine_type = rng.choice(["L", "M", "H"], size=rows, p=[0.55, 0.30, 0.15])
    air_temperature = rng.normal(298.2, 2.0, rows)
    process_temperature = air_temperature + rng.normal(10.0, 1.2, rows)
    rotational_speed = rng.normal(1500, 165, rows).clip(1050, 3000)
    torque = rng.normal(40, 8.5, rows).clip(3, 80)
    tool_wear = rng.integers(0, 255, rows)

    # Inject realistic abnormal operating regimes.
    high_load_idx = rng.choice(rows, size=max(rows // 12, 1), replace=False)
    torque[high_load_idx] += rng.normal(20, 5, len(high_load_idx))
    rotational_speed[high_load_idx] -= rng.normal(220, 60, len(high_load_idx))

    thermal_idx = rng.choice(rows, size=max(rows // 16, 1), replace=False)
    process_temperature[thermal_idx] += rng.normal(4.2, 1.1, len(thermal_idx))

    power_idx = rng.choice(rows, size=max(rows // 20, 1), replace=False)
    rotational_speed[power_idx] += rng.normal(900, 160, len(power_idx))
    torque[power_idx] -= rng.normal(30, 6, len(power_idx))

    rotational_speed = rotational_speed.clip(900, 3200)
    torque = torque.clip(2, 85)

    temp_delta = process_temperature - air_temperature
    mechanical_power = torque * rotational_speed * 2 * np.pi / 60.0

    tool_wear_failure = tool_wear > 215
    heat_dissipation_failure = temp_delta > 13.2
    power_failure = (mechanical_power < 3800) | (mechanical_power > 9200)
    overstrain_failure = (torque > 58) & (tool_wear > 175)
    random_failure = rng.random(rows) < 0.006

    failure_binary = (
        tool_wear_failure
        | heat_dissipation_failure
        | power_failure
        | overstrain_failure
        | random_failure
    ).astype(int)

    failure_type = np.full(rows, "no_failure", dtype=object)
    failure_type[tool_wear_failure] = "tool_wear_failure"
    failure_type[heat_dissipation_failure] = "heat_dissipation_failure"
    failure_type[power_failure] = "power_failure"
    failure_type[overstrain_failure] = "overstrain_failure"
    failure_type[random_failure] = "random_failure"

    data = pd.DataFrame(
        {
            "machine_id": [f"MCH-{idx + 1:06d}" for idx in range(rows)],
            "machine_type": machine_type,
            "air_temperature": air_temperature.round(2),
            "process_temperature": process_temperature.round(2),
            "rotational_speed": rotational_speed.round(0).astype(int),
            "torque": torque.round(2),
            "tool_wear": tool_wear,
            "failure_binary": failure_binary,
            "failure_type": failure_type,
            "TWF": tool_wear_failure.astype(int),
            "HDF": heat_dissipation_failure.astype(int),
            "PWF": power_failure.astype(int),
            "OSF": overstrain_failure.astype(int),
            "RNF": random_failure.astype(int),
        }
    )
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a large synthetic maintenance CSV")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthetic_maintenance_large.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = generate_synthetic_maintenance_dataset(rows=args.rows, seed=args.seed)
    output_path = ensure_dir(args.output.parent) / args.output.name
    data.to_csv(output_path, index=False)
    failure_rate = data["failure_binary"].mean()
    print(f"Generated {len(data):,} rows")
    print(f"Failure rows: {int(data['failure_binary'].sum()):,} ({failure_rate:.1%})")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
