"""Helpers for indexing user histories and slicing them per query."""

from __future__ import annotations

from typing import Any

import pandas as pd

from recpfn.data.schemas import INTERACTION_ID_COL, TIMESTAMP_COL, USER_ID_COL


def build_user_history_index(interactions: pd.DataFrame) -> tuple[dict[Any, pd.DataFrame], pd.DataFrame]:
    """Index interactions by user for repeated pre-query history lookups."""

    if interactions.empty:
        empty = interactions.copy()
        return {}, empty

    ordered = interactions.sort_values([USER_ID_COL, TIMESTAMP_COL, INTERACTION_ID_COL]).copy()
    grouped = {
        user_id: group.reset_index(drop=True)
        for user_id, group in ordered.groupby(USER_ID_COL, sort=False)
    }
    empty = ordered.iloc[0:0].copy()
    return grouped, empty


def history_before_query(
    user_rows: pd.DataFrame,
    query_timestamp: pd.Timestamp,
    query_interaction_id: int,
) -> pd.DataFrame:
    """Return only the rows that happened before the query event."""

    if user_rows.empty:
        return user_rows

    earlier_ts = user_rows[TIMESTAMP_COL] < query_timestamp
    same_ts_earlier_id = (user_rows[TIMESTAMP_COL] == query_timestamp) & (
        user_rows[INTERACTION_ID_COL] < query_interaction_id
    )
    return user_rows[earlier_ts | same_ts_earlier_id]
