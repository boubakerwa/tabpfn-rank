"""History-aware interaction feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recpfn.data.schemas import ITEM_ID_COL, LABEL_COL, TIMESTAMP_COL


def summarize_history(history: pd.DataFrame) -> dict[str, float]:
    """Summarize a user's history before a query."""

    if history.empty:
        return {
            "hist_interactions": 0.0,
            "hist_positive": 0.0,
            "hist_positive_rate": 0.0,
            "hist_distinct_items": 0.0,
        }

    positive_count = float(history[LABEL_COL].sum())
    return {
        "hist_interactions": float(len(history)),
        "hist_positive": positive_count,
        "hist_positive_rate": positive_count / float(len(history)),
        "hist_distinct_items": float(history[ITEM_ID_COL].nunique()),
    }


def recency_in_days(history: pd.DataFrame, query_timestamp: pd.Timestamp) -> float:
    """Days since the user's most recent prior interaction."""

    if history.empty:
        return 9999.0
    delta = query_timestamp - history[TIMESTAMP_COL].max()
    return float(delta.total_seconds() / 86400.0)


def numeric_average(history: pd.DataFrame, column: str) -> float:
    """Average a numeric history column when available."""

    if history.empty or column not in history.columns:
        return 0.0
    values = pd.to_numeric(history[column], errors="coerce")
    if values.notna().sum() == 0:
        return 0.0
    return float(values.mean())


def safe_affinity_count(series: pd.Series, value: object) -> float:
    """Count how often a given value appears in a history-aligned series."""

    if series.empty or value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float((series.astype(str) == str(value)).sum())
