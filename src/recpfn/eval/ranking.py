"""Prediction post-processing helpers."""

from __future__ import annotations

import pandas as pd


def attach_metadata(
    predictions: pd.DataFrame,
    dataset: str,
    split_type: str,
    protocol: str,
    model: str,
    mode: str,
) -> pd.DataFrame:
    """Attach experiment metadata columns to a prediction frame."""

    frame = predictions.copy()
    frame["dataset"] = dataset
    frame["split_type"] = split_type
    frame["protocol"] = protocol
    frame["model"] = model
    frame["mode"] = mode
    return frame
