"""Candidate-set builders for benchmark protocols."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from recpfn.data.schemas import (
    CANDIDATE_POSITION_COL,
    ITEM_ID_COL,
    LABEL_COL,
    PROTOCOL_COL,
    QUERY_ID_COL,
    QUERY_INTERACTION_ID_COL,
    QUERY_TIMESTAMP_COL,
    SPLIT_COL,
    USER_ID_COL,
)
from recpfn.types import DatasetBundle, SplitBundle
from recpfn.utils import deterministic_random


def build_candidates(
    dataset: DatasetBundle,
    split: SplitBundle,
    protocol: str,
    k: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Build train and test candidate rows under a fixed benchmark protocol."""

    protocol = protocol.lower()
    if protocol not in {"global_popularity", "context_popularity"}:
        raise ValueError(f"Unsupported candidate protocol '{protocol}'.")

    items = dataset.items.copy()
    items = items.drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL)
    positive_train = split.train_interactions[split.train_interactions[LABEL_COL] == 1]
    global_order = (
        positive_train.groupby(ITEM_ID_COL)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    context_col = dataset.context_col if dataset.context_col in items.columns else None
    context_orders: dict[str, list] = defaultdict(list)
    if context_col:
        context_popularity = (
            positive_train.merge(items[[context_col]], left_on=ITEM_ID_COL, right_index=True, how="left")
            .groupby([context_col, ITEM_ID_COL])
            .size()
            .sort_values(ascending=False)
            .reset_index(name="count")
        )
        for context_value, group in context_popularity.groupby(context_col):
            context_orders[str(context_value)] = group[ITEM_ID_COL].tolist()

    rows = []
    queries = pd.concat([split.train_queries, split.test_queries], ignore_index=True)
    for query in queries.itertuples(index=False):
        history = _history_before_query(split.train_interactions, query.user_id, query.query_timestamp, query.query_interaction_id)
        seen_items = set(history[ITEM_ID_COL].tolist())
        target_item = query.item_id
        target_context = None
        if context_col and target_item in items.index:
            target_context = str(items.at[target_item, context_col])

        ordered_pool = list(global_order)
        if protocol == "context_popularity" and target_context:
            context_first = context_orders.get(target_context, [])
            ordered_pool = context_first + [item for item in global_order if item not in set(context_first)]

        negatives = []
        for item_id in ordered_pool:
            if item_id == target_item or item_id in seen_items:
                continue
            negatives.append(item_id)
            if len(negatives) == max(0, k - 1):
                break

        if len(negatives) < max(0, k - 1):
            fallback_rng = deterministic_random(dataset.name, split.split_type, protocol, query.query_id, seed)
            remaining = [
                item_id
                for item_id in items.index.tolist()
                if item_id not in seen_items and item_id != target_item and item_id not in negatives
            ]
            fallback_rng.shuffle(remaining)
            negatives.extend(remaining[: max(0, k - 1 - len(negatives))])

        candidate_item_ids = [target_item, *negatives[: max(0, k - 1)]]
        for position, item_id in enumerate(candidate_item_ids):
            rows.append(
                {
                    QUERY_ID_COL: query.query_id,
                    USER_ID_COL: query.user_id,
                    ITEM_ID_COL: item_id,
                    LABEL_COL: int(item_id == target_item),
                    SPLIT_COL: query.split,
                    PROTOCOL_COL: protocol,
                    QUERY_TIMESTAMP_COL: query.query_timestamp,
                    QUERY_INTERACTION_ID_COL: query.query_interaction_id,
                    CANDIDATE_POSITION_COL: position,
                }
            )

    return pd.DataFrame(rows)


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
