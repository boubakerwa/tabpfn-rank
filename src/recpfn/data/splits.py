"""Deterministic split builders for warm and item-cold evaluation."""

from __future__ import annotations

import math

import pandas as pd

from recpfn.data.schemas import (
    INTERACTION_ID_COL,
    ITEM_ID_COL,
    LABEL_COL,
    QUERY_ID_COL,
    QUERY_INTERACTION_ID_COL,
    QUERY_TIMESTAMP_COL,
    TIMESTAMP_COL,
    USER_ID_COL,
)
from recpfn.types import DatasetBundle, SplitBundle
from recpfn.utils import deterministic_random


def build_splits(dataset: DatasetBundle, split_type: str, seed: int = 0) -> SplitBundle:
    """Build one of the supported evaluation splits."""

    split_type = split_type.lower()
    if split_type == "warm":
        return _build_warm_split(dataset)
    if split_type in {"item_cold", "cold_items"}:
        return _build_item_cold_split(dataset, seed=seed)
    raise ValueError(f"Unsupported split type '{split_type}'.")


def _prepare_interactions(dataset: DatasetBundle) -> pd.DataFrame:
    df = dataset.interactions.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values([USER_ID_COL, TIMESTAMP_COL, INTERACTION_ID_COL]).reset_index(drop=True)
    return df


def _build_query_frame(source: pd.DataFrame, prefix: str) -> pd.DataFrame:
    queries = source.copy()
    queries[QUERY_ID_COL] = [f"{prefix}-{idx}" for idx in queries[INTERACTION_ID_COL]]
    queries[QUERY_TIMESTAMP_COL] = queries[TIMESTAMP_COL]
    queries[QUERY_INTERACTION_ID_COL] = queries[INTERACTION_ID_COL]
    return queries[
        [
            QUERY_ID_COL,
            USER_ID_COL,
            ITEM_ID_COL,
            QUERY_TIMESTAMP_COL,
            QUERY_INTERACTION_ID_COL,
            LABEL_COL,
        ]
    ]


def _build_warm_split(dataset: DatasetBundle) -> SplitBundle:
    df = _prepare_interactions(dataset)
    df["user_position"] = df.groupby(USER_ID_COL).cumcount()

    last_positive_positions = (
        df[df[LABEL_COL] == 1]
        .groupby(USER_ID_COL)["user_position"]
        .max()
        .rename("test_position")
    )
    df = df.merge(last_positive_positions, how="left", on=USER_ID_COL)
    eligible = df["test_position"].notna() & (df["test_position"] > 0)
    test_rows = df[eligible & (df["user_position"] == df["test_position"]) & (df[LABEL_COL] == 1)].copy()

    train_interactions = df[eligible & (df["user_position"] < df["test_position"])].copy()
    train_interactions = train_interactions.drop(columns=["test_position"])

    train_rows = train_interactions[train_interactions[LABEL_COL] == 1].copy()
    earliest_history_ts = train_interactions.groupby(USER_ID_COL)[TIMESTAMP_COL].min().rename("first_train_ts")
    train_rows = train_rows.merge(earliest_history_ts, on=USER_ID_COL, how="left")
    train_rows = train_rows[train_rows[TIMESTAMP_COL] > train_rows["first_train_ts"]].drop(columns=["first_train_ts"])

    test_queries = _build_query_frame(test_rows, "warm-test")
    test_queries["split"] = "test"
    train_queries = _build_query_frame(train_rows, "warm-train")
    train_queries["split"] = "train"

    return SplitBundle(
        name=f"{dataset.name}_warm",
        split_type="warm",
        train_interactions=train_interactions,
        train_queries=train_queries,
        test_queries=test_queries,
        metadata={"n_train_queries": len(train_queries), "n_test_queries": len(test_queries)},
    )


def _build_item_cold_split(dataset: DatasetBundle, seed: int = 0, cold_fraction: float = 0.15) -> SplitBundle:
    df = _prepare_interactions(dataset)
    positive_counts = df[df[LABEL_COL] == 1].groupby(ITEM_ID_COL).size().sort_values(ascending=False)
    positive_items = positive_counts.index.tolist()
    cold_count = max(1, math.ceil(len(positive_items) * cold_fraction))
    rng = deterministic_random(dataset.name, "item_cold", seed)
    cold_item_ids = set(rng.sample(positive_items, k=min(cold_count, len(positive_items))))

    train_interactions = df[~df[ITEM_ID_COL].isin(cold_item_ids)].copy()
    test_pool = df[(df[ITEM_ID_COL].isin(cold_item_ids)) & (df[LABEL_COL] == 1)].copy()

    merged = test_pool.merge(
        train_interactions.groupby(USER_ID_COL)[TIMESTAMP_COL].min().rename("first_train_ts"),
        on=USER_ID_COL,
        how="left",
    )
    merged = merged[merged["first_train_ts"].notna() & (merged[TIMESTAMP_COL] > merged["first_train_ts"])]
    test_rows = (
        merged.sort_values([USER_ID_COL, TIMESTAMP_COL, INTERACTION_ID_COL])
        .groupby(USER_ID_COL)
        .tail(1)
        .copy()
    )

    train_rows = train_interactions[train_interactions[LABEL_COL] == 1].copy()
    train_rows["train_position"] = train_rows.groupby(USER_ID_COL).cumcount()
    train_rows = train_rows[train_rows["train_position"] > 0].drop(columns=["train_position"])

    train_queries = _build_query_frame(train_rows, "item-cold-train")
    train_queries["split"] = "train"
    test_queries = _build_query_frame(test_rows, "item-cold-test")
    test_queries["split"] = "test"

    return SplitBundle(
        name=f"{dataset.name}_item_cold",
        split_type="item_cold",
        train_interactions=train_interactions,
        train_queries=train_queries,
        test_queries=test_queries,
        cold_item_ids=cold_item_ids,
        metadata={
            "cold_fraction": cold_fraction,
            "n_train_queries": len(train_queries),
            "n_test_queries": len(test_queries),
        },
    )
