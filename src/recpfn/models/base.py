"""Base model wrappers and public model APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from recpfn.data.schemas import ITEM_ID_COL, LABEL_COL, PROTOCOL_COL, QUERY_ID_COL
from recpfn.utils import deterministic_random

RESERVED_COLUMNS = {
    QUERY_ID_COL,
    "user_id",
    ITEM_ID_COL,
    LABEL_COL,
    "split",
    PROTOCOL_COL,
    "query_timestamp",
    "query_interaction_id",
    "candidate_position",
    "score",
    "model",
    "mode",
    "left_item_id",
    "right_item_id",
    "dataset",
    "split_type",
    "status",
    "error",
}


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Infer feature columns by excluding reserved metadata columns."""

    return [column for column in frame.columns if column not in RESERVED_COLUMNS]


@dataclass
class DummyMatrixEncoder:
    """One-hot encoder implemented with pandas get_dummies."""

    columns_: list[str] | None = None

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.get_dummies(frame.copy(), dummy_na=True)
        self.columns_ = matrix.columns.tolist()
        return matrix.astype(float)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.get_dummies(frame.copy(), dummy_na=True)
        columns = self.columns_ or []
        return matrix.reindex(columns=columns, fill_value=0.0).astype(float)


class BasePointwiseRanker:
    """Base wrapper with aligned one-hot encoding."""

    def __init__(self) -> None:
        self.encoder = DummyMatrixEncoder()
        self.model = None

    def build_model(self):
        raise NotImplementedError

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str = LABEL_COL) -> "BasePointwiseRanker":
        x = self.encoder.fit_transform(frame[feature_cols])
        y = frame[target_col].astype(int).to_numpy()
        self.model = self.build_model()
        self.model.fit(x, y)
        return self

    def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")
        x = self.encoder.transform(frame[feature_cols])
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        decision = self.model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))


class PopularityRanker:
    """Simple rule-based baseline using precomputed popularity features."""

    def __init__(self, feature_name: str = "item_popularity") -> None:
        self.feature_name = feature_name

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str = LABEL_COL) -> "PopularityRanker":
        del frame, feature_cols, target_col
        return self

    def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        del feature_cols
        values = pd.to_numeric(frame[self.feature_name], errors="coerce").fillna(0.0)
        return values.to_numpy(dtype=float)


class RecentPopularityRanker(PopularityRanker):
    """Baseline that discounts stale interactions by recency."""

    def __init__(self) -> None:
        super().__init__(feature_name="item_popularity")

    def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        del feature_cols
        popularity = pd.to_numeric(frame["item_popularity"], errors="coerce").fillna(0.0)
        recency = pd.to_numeric(frame["days_since_last_interaction"], errors="coerce").fillna(9999.0)
        return (popularity / (1.0 + recency)).to_numpy(dtype=float)


class SklearnGBDTRanker(BasePointwiseRanker):
    """Fallback tree model when optional libraries are unavailable."""

    def build_model(self):
        return HistGradientBoostingClassifier(random_state=0)


class SklearnLogisticRanker(BasePointwiseRanker):
    """Lightweight linear baseline / fallback."""

    def build_model(self):
        return LogisticRegression(max_iter=1000)


def fit_pointwise(
    model_name: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = LABEL_COL,
    group_col: str = QUERY_ID_COL,
):
    """Fit a pointwise reranker by model name."""

    del group_col
    ranker = _build_ranker(model_name)
    return ranker.fit(train_df, feature_cols, target_col=target_col)


def predict_scores(model, candidate_df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    """Predict pointwise scores."""

    cols = feature_cols or infer_feature_columns(candidate_df)
    return model.predict_proba(candidate_df, cols)


def fit_pairwise(
    model_name: str,
    pair_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = LABEL_COL,
):
    """Fit a pairwise preference model."""

    ranker = _build_ranker(model_name)
    return ranker.fit(pair_df, feature_cols, target_col=target_col)


def predict_preferences(model, pair_df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    """Predict pairwise preference probabilities."""

    cols = feature_cols or infer_feature_columns(pair_df)
    return model.predict_proba(pair_df, cols)


def build_pairwise_training_rows(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    max_pairs_per_query: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Construct pairwise training rows from pointwise candidate features."""

    rows: list[dict] = []
    for query_id, group in feature_df.groupby(QUERY_ID_COL):
        positives = group[group[LABEL_COL] == 1]
        negatives = group[group[LABEL_COL] == 0]
        if positives.empty or negatives.empty:
            continue

        positive_row = positives.iloc[0]
        weights = (negatives["item_popularity"].fillna(0.0) + 1.0).to_list()
        sample_size = min(max_pairs_per_query, len(negatives))
        sampled_indices = _sample_weighted_indices(weights, sample_size, query_id, seed)
        sampled_negatives = negatives.iloc[sampled_indices]
        for _, negative_row in sampled_negatives.iterrows():
            rows.append(_pair_row(query_id, positive_row, negative_row, feature_cols, label=1))
            rows.append(_pair_row(query_id, negative_row, positive_row, feature_cols, label=0))

    return pd.DataFrame(rows)


def score_pairwise_candidates(
    model,
    candidate_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Score candidate sets by averaging pairwise win probabilities."""

    pair_rows = []
    pair_index = []
    for query_id, group in candidate_df.groupby(QUERY_ID_COL):
        group = group.reset_index(drop=True)
        for left_idx, left_row in group.iterrows():
            for right_idx, right_row in group.iterrows():
                if left_idx == right_idx:
                    continue
                pair_rows.append(_pair_row(query_id, left_row, right_row, feature_cols, label=0))
                pair_index.append((query_id, left_idx, right_idx))

    if not pair_rows:
        scored = candidate_df.copy()
        scored["score"] = 0.0
        return scored

    pair_df = pd.DataFrame(pair_rows)
    pair_feature_cols = infer_feature_columns(pair_df)
    probabilities = predict_preferences(model, pair_df, pair_feature_cols)

    score_map: dict[tuple[str, int], list[float]] = {}
    for (query_id, left_idx, _), probability in zip(pair_index, probabilities):
        score_map.setdefault((query_id, left_idx), []).append(float(probability))

    scored_rows = []
    for query_id, group in candidate_df.groupby(QUERY_ID_COL):
        group = group.reset_index(drop=True).copy()
        scores = []
        for idx in range(len(group)):
            votes = score_map.get((query_id, idx), [0.0])
            scores.append(float(np.mean(votes)))
        group["score"] = scores
        scored_rows.append(group)
    return pd.concat(scored_rows, ignore_index=True)


def _sample_weighted_indices(weights: Iterable[float], sample_size: int, query_id: str, seed: int) -> list[int]:
    rng = deterministic_random("pairwise", query_id, seed)
    weight_list = list(weights)
    positions = list(range(len(weight_list)))
    if sample_size >= len(weight_list):
        return positions

    selected: list[int] = []
    available = positions.copy()
    available_weights = weight_list.copy()
    while available and len(selected) < sample_size:
        choice = rng.choices(available, weights=available_weights, k=1)[0]
        choice_pos = available.index(choice)
        selected.append(choice)
        del available[choice_pos]
        del available_weights[choice_pos]
    return selected


def _pair_row(query_id: str, left_row: pd.Series, right_row: pd.Series, feature_cols: list[str], label: int) -> dict:
    row = {
        QUERY_ID_COL: query_id,
        LABEL_COL: label,
        "left_item_id": left_row[ITEM_ID_COL],
        "right_item_id": right_row[ITEM_ID_COL],
    }
    for feature in feature_cols:
        row[f"a__{feature}"] = left_row[feature]
        row[f"b__{feature}"] = right_row[feature]
    return row


def _build_ranker(model_name: str):
    normalized = model_name.lower()
    if normalized == "popularity":
        return PopularityRanker()
    if normalized == "recent_popularity":
        return RecentPopularityRanker()
    if normalized == "sklearn_gbdt":
        return SklearnGBDTRanker()
    if normalized == "sklearn_logreg":
        return SklearnLogisticRanker()
    if normalized == "xgboost":
        from recpfn.models.xgboost_ranker import XGBoostRanker

        return XGBoostRanker()
    if normalized == "catboost":
        from recpfn.models.catboost_ranker import CatBoostRanker

        return CatBoostRanker()
    if normalized == "tabpfn":
        from recpfn.models.tabpfn_pointwise import TabPFNPointwiseRanker

        return TabPFNPointwiseRanker()
    raise ValueError(f"Unsupported model '{model_name}'.")
