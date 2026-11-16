"""Isolation Forest logic, kept free of Flask so it can be tested and reused."""
from dataclasses import dataclass
from typing import IO, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MAX_ROWS = 50_000
MIN_ROWS = 10
RANDOM_STATE = 42  # fixed so repeated runs with the same settings give identical results


class DetectionError(ValueError):
    """A problem with the user's data or parameters (reported to the client as HTTP 400)."""


@dataclass(frozen=True)
class Params:
    contamination: float = 0.05
    n_estimators: int = 100

    def __post_init__(self):
        if not 0.001 <= self.contamination <= 0.5:
            raise DetectionError("contamination must be between 0.001 and 0.5.")
        if not 10 <= self.n_estimators <= 500:
            raise DetectionError("n_estimators must be between 10 and 500.")


def load_numeric_csv(stream: IO) -> tuple[pd.DataFrame, list[str], int]:
    """Parse a CSV and keep numeric columns.

    Returns (clean numeric frame, ignored non-numeric column names, rows dropped for
    missing/infinite values).
    """
    try:
        raw = pd.read_csv(stream)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DetectionError(f"Could not read the file as CSV: {exc}") from exc

    numeric = raw.select_dtypes(include="number")
    ignored = [c for c in raw.columns if c not in numeric.columns]
    if numeric.shape[1] < 2:
        raise DetectionError("The CSV needs at least two numeric columns.")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    clean = numeric.dropna().reset_index(drop=True)
    dropped = len(numeric) - len(clean)

    if len(clean) < MIN_ROWS:
        raise DetectionError(f"Need at least {MIN_ROWS} complete rows; found {len(clean)}.")
    if len(clean) > MAX_ROWS:
        raise DetectionError(f"Too many rows ({len(clean)}); the limit is {MAX_ROWS}.")
    return clean, [str(c) for c in ignored], dropped


def detect_anomalies(df: pd.DataFrame, params: Params,
                     features: Optional[Sequence[str]] = None) -> dict:
    """Fit an Isolation Forest on `features` (default: all columns) and score every row.

    Anomaly score = -score_samples(): higher means more anomalous. A row is labelled an
    anomaly (1) when its score exceeds the threshold implied by `contamination`.
    """
    features = list(features) if features else list(df.columns)
    unknown = [f for f in features if f not in df.columns]
    if unknown:
        raise DetectionError(f"Unknown feature column(s): {', '.join(unknown)}")

    model = IsolationForest(n_estimators=params.n_estimators,
                            contamination=params.contamination,
                            random_state=RANDOM_STATE, n_jobs=-1)
    labels = model.fit_predict(df[features])          # -1 = anomaly, 1 = normal
    scores = -model.score_samples(df[features])

    return {
        "columns": [str(c) for c in df.columns],
        "features": [str(f) for f in features],
        "data": {str(c): df[c].tolist() for c in df.columns},
        "scores": np.round(scores, 5).tolist(),
        "labels": (labels == -1).astype(int).tolist(),
        "threshold": round(float(-model.offset_), 5),
        "n_rows": len(df),
        "n_anomalies": int((labels == -1).sum()),
    }
