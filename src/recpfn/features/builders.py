"""Feature builder for train and test candidate rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recpfn.data.schemas import (
    ITEM_ID_COL,
    LABEL_COL,
    QUERY_INTERACTION_ID_COL,
    QUERY_TIMESTAMP_COL,
    USER_ID_COL,
)
from recpfn.features.cold_start import item_training_popularity
from recpfn.features.interactions import numeric_average, recency_in_days, safe_affinity_count, summarize_history
from recpfn.types import DatasetBundle, SplitBundle


def build_features(dataset: DatasetBundle, candidates: pd.DataFrame, split: SplitBundle) -> pd.DataFrame:
    """Construct compact tabular features for each candidate row."""

    users = dataset.users.drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL)
    items = dataset.items.drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL)
    train_interactions = split.train_interactions.copy()
    train_interactions = train_interactions.merge(
        items.add_prefix("item__"),
        left_on=ITEM_ID_COL,
        right_index=True,
        how="left",
    )

    popularity = item_training_popularity(split.train_interactions)
    item_positive_rate = (
        split.train_interactions.groupby(ITEM_ID_COL)[LABEL_COL].mean().rename("item_positive_rate")
    )

    genre_cols = [column for column in items.columns if str(column).startswith("genre_")]
    rows = []
    for _, query_group in candidates.groupby("query_id", sort=False):
        query_row = query_group.iloc[0]
        user_id = query_row[USER_ID_COL]
        query_timestamp = query_row[QUERY_TIMESTAMP_COL]
        query_interaction_id = query_row[QUERY_INTERACTION_ID_COL]

        user_history = _history_before_query(
            train_interactions,
            user_id=user_id,
            query_timestamp=query_timestamp,
            query_interaction_id=query_interaction_id,
        )
        user_meta = users.loc[user_id] if user_id in users.index else pd.Series(dtype=object)
        history_summary = summarize_history(user_history)
        history_categories = user_history.get(f"item__{dataset.context_col}", pd.Series(dtype=object))
        history_brands = user_history.get("item__brand", pd.Series(dtype=object))
        user_avg_history_price = numeric_average(user_history, "item__price")
        days_since_last = recency_in_days(user_history, query_timestamp)
        base_features = {
            **history_summary,
            "days_since_last_interaction": days_since_last,
            "user_age": _coerce_numeric(user_meta.get("age"), default=0.0),
            "user_gender": str(user_meta.get("gender", "unknown")),
            "user_occupation": str(user_meta.get("occupation", "unknown")),
            "user_avg_history_price": user_avg_history_price,
        }

        for candidate in query_group.itertuples(index=False):
            item_id = candidate.item_id
            item_meta = items.loc[item_id] if item_id in items.index else pd.Series(dtype=object)

            row = candidate._asdict()
            row.update(base_features)
            row["item_primary_category"] = str(item_meta.get(dataset.context_col, "unknown"))
            row["item_brand"] = str(item_meta.get("brand", "unknown"))
            row["item_price"] = _coerce_numeric(item_meta.get("price"), default=0.0)
            row["item_release_year"] = _coerce_numeric(item_meta.get("release_year"), default=0.0)
            row["item_genre_count"] = _coerce_numeric(item_meta.get("genre_count"), default=0.0)
            row["item_category_depth"] = _coerce_numeric(item_meta.get("category_depth"), default=0.0)
            row["item_popularity"] = float(popularity.get(item_id, 0.0))
            row["item_positive_rate"] = float(item_positive_rate.get(item_id, 0.0))
            row["item_is_cold"] = float(item_id in split.cold_item_ids)
            row["price_missing"] = float(pd.isna(item_meta.get("price")))
            row["category_affinity"] = safe_affinity_count(history_categories, item_meta.get(dataset.context_col))
            row["brand_affinity"] = safe_affinity_count(history_brands, item_meta.get("brand"))
            row["same_item_history"] = float((user_history[ITEM_ID_COL] == item_id).sum()) if not user_history.empty else 0.0
            row["price_distance_to_user_avg"] = abs(row["item_price"] - row["user_avg_history_price"])
            row["history_positive_same_category"] = float(
                (
                    (history_categories.astype(str) == str(item_meta.get(dataset.context_col, "unknown")))
                    & (user_history[LABEL_COL] == 1)
                ).sum()
            ) if not user_history.empty and dataset.context_col in item_meta.index else 0.0

            for genre_col in genre_cols:
                row[f"feat_{genre_col}"] = _coerce_numeric(item_meta.get(genre_col), default=0.0)
            rows.append(row)

    return pd.DataFrame(rows).fillna(
        {
            "user_gender": "unknown",
            "user_occupation": "unknown",
            "item_primary_category": "unknown",
            "item_brand": "unknown",
        }
    )


def _history_before_query(
    interactions: pd.DataFrame,
    user_id: object,
    query_timestamp: pd.Timestamp,
    query_interaction_id: int,
) -> pd.DataFrame:
    user_rows = interactions[interactions[USER_ID_COL] == user_id]
    earlier_ts = user_rows["timestamp"] < query_timestamp
    same_ts_earlier_id = (user_rows["timestamp"] == query_timestamp) & (
        user_rows["interaction_id"] < query_interaction_id
    )
    return user_rows[earlier_ts | same_ts_earlier_id]


def _coerce_numeric(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric):
        return default
    return numeric
