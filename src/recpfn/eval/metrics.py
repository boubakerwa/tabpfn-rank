"""Ranking metrics for per-query candidate sets."""

from __future__ import annotations

import math

import pandas as pd

from recpfn.data.schemas import LABEL_COL, QUERY_ID_COL


def evaluate_rankings(
    predictions_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("ndcg@10", "recall@10", "mrr", "hitrate@10"),
) -> dict[str, float]:
    """Aggregate ranking metrics across queries."""

    per_query = evaluate_rankings_by_query(predictions_df, metrics=metrics)
    aggregated = {metric: float(per_query[metric].mean()) for metric in metrics}
    aggregated["n_queries"] = float(len(per_query))
    return aggregated


def evaluate_rankings_by_query(
    predictions_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("ndcg@10", "recall@10", "mrr", "hitrate@10"),
) -> pd.DataFrame:
    """Compute ranking metrics per query."""

    rows = []
    for query_id, group in predictions_df.groupby(QUERY_ID_COL):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        labels = ordered[LABEL_COL].astype(int).tolist()
        row = {QUERY_ID_COL: query_id}
        for metric in metrics:
            row[metric] = _score_metric(labels, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _score_metric(labels: list[int], metric: str) -> float:
    if metric == "ndcg@10":
        return _ndcg_at_k(labels, 10)
    if metric == "ndcg@5":
        return _ndcg_at_k(labels, 5)
    if metric == "recall@10":
        return _recall_at_k(labels, 10)
    if metric == "recall@5":
        return _recall_at_k(labels, 5)
    if metric == "mrr":
        return _mrr(labels)
    if metric == "map":
        return _average_precision(labels)
    if metric == "hitrate@10":
        return _hit_rate_at_k(labels, 10)
    raise ValueError(f"Unsupported metric '{metric}'.")


def _dcg_at_k(labels: list[int], k: int) -> float:
    return sum((2**label - 1) / math.log2(idx + 2) for idx, label in enumerate(labels[:k]))


def _ndcg_at_k(labels: list[int], k: int) -> float:
    ideal = sorted(labels, reverse=True)
    denom = _dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return _dcg_at_k(labels, k) / denom


def _recall_at_k(labels: list[int], k: int) -> float:
    total = sum(labels)
    if total == 0:
        return 0.0
    return sum(labels[:k]) / total


def _hit_rate_at_k(labels: list[int], k: int) -> float:
    return float(any(label == 1 for label in labels[:k]))


def _mrr(labels: list[int]) -> float:
    for idx, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / idx
    return 0.0


def _average_precision(labels: list[int]) -> float:
    hits = 0
    precision_sum = 0.0
    for idx, label in enumerate(labels, start=1):
        if label == 1:
            hits += 1
            precision_sum += hits / idx
    if hits == 0:
        return 0.0
    return precision_sum / hits
