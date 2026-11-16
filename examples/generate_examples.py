"""Regenerate the synthetic example datasets (fixed seed, so output is reproducible).

Run from the project root:  python examples/generate_examples.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(7)


def shuffled(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1, random_state=7).reset_index(drop=True).round(2)


def transactions() -> pd.DataFrame:
    n, k = 960, 40
    normal = pd.DataFrame({
        "amount_usd": rng.lognormal(3.6, 0.6, n),
        "seconds_since_last_txn": rng.lognormal(9.5, 0.7, n),
    })
    fraud = pd.DataFrame({
        "amount_usd": rng.uniform(900, 3000, k),
        "seconds_since_last_txn": rng.uniform(5, 120, k),
    })
    return shuffled(pd.concat([normal, fraud]))


def network_traffic() -> pd.DataFrame:
    n, k = 1140, 60
    normal = pd.DataFrame({
        "bytes_sent_kb": rng.normal(120, 25, n).clip(5),
        "packets_per_sec": rng.normal(80, 15, n).clip(1),
        "duration_sec": rng.normal(30, 8, n).clip(1),
    })
    attack = pd.DataFrame({
        "bytes_sent_kb": rng.normal(600, 120, k),
        "packets_per_sec": rng.normal(400, 90, k),
        "duration_sec": rng.normal(3, 1.5, k).clip(0.2),
    })
    return shuffled(pd.concat([normal, attack]))


def sensor_quality() -> pd.DataFrame:
    n, k = 570, 30
    normal = pd.DataFrame({
        "temperature_c": rng.normal(72, 2.0, n),
        "vibration_mm_s": rng.normal(2.5, 0.4, n),
        "pressure_kpa": rng.normal(101, 3.0, n),
    })
    faulty = pd.DataFrame({
        "temperature_c": rng.normal(85, 4, k),
        "vibration_mm_s": rng.normal(5.0, 0.8, k),
        "pressure_kpa": rng.normal(92, 4, k),
    })
    return shuffled(pd.concat([normal, faulty]))


if __name__ == "__main__":
    for name, build in [("transactions", transactions), ("network_traffic", network_traffic),
                        ("sensor_quality", sensor_quality)]:
        df = build()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows")
