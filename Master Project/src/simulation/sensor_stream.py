from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sensor_batch(rows: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    machine_type = rng.choice(["L", "M", "H"], size=rows, p=[0.5, 0.3, 0.2])
    air_temperature = rng.normal(298, 2.2, rows)
    process_temperature = air_temperature + rng.normal(10, 1.5, rows)
    rotational_speed = rng.normal(1500, 170, rows).clip(1150, 2900)
    torque = rng.normal(40, 9, rows).clip(3, 80)
    tool_wear = rng.integers(0, 260, rows)

    failure_binary = (
        (torque > 58)
        | (tool_wear > 210)
        | ((process_temperature - air_temperature) > 13.5)
    ).astype(int)

    return pd.DataFrame(
        {
            "machine_type": machine_type,
            "air_temperature": air_temperature.round(2),
            "process_temperature": process_temperature.round(2),
            "rotational_speed": rotational_speed.round(0).astype(int),
            "torque": torque.round(2),
            "tool_wear": tool_wear,
            "failure_binary": failure_binary,
        }
    )
