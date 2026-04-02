"""TabPFN-backed classifier wrappers."""

from __future__ import annotations

import pandas as pd

from recpfn.data.schemas import LABEL_COL
from recpfn.exceptions import OptionalDependencyNotAvailable
from recpfn.models.base import BasePointwiseRanker
from recpfn.utils import read_env_str


class TabPFNPointwiseRanker(BasePointwiseRanker):
    """Binary classifier wrapper used for pointwise and pairwise ranking."""

    def build_model(self):
        return _build_tabpfn_classifier()


class TabPFNNativePointwiseRanker(BasePointwiseRanker):
    """TabPFN wrapper that keeps mixed numeric/categorical features intact."""

    def __init__(self) -> None:
        super().__init__()
        self.categorical_cols_: list[str] = []

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str = LABEL_COL) -> "TabPFNNativePointwiseRanker":
        x, categorical_cols = _prepare_tabpfn_native_frame(frame[feature_cols])
        categorical_indices = [x.columns.get_loc(column) for column in categorical_cols]
        y = frame[target_col].astype(int).to_numpy()
        self.categorical_cols_ = categorical_cols
        self.model = _build_tabpfn_classifier(categorical_indices=categorical_indices)
        self.model.fit(x, y)
        return self

    def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]):
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")
        x, _ = _prepare_tabpfn_native_frame(frame[feature_cols], categorical_cols=self.categorical_cols_)
        return self.model.predict_proba(x)[:, 1]


def _prepare_tabpfn_native_frame(
    frame: pd.DataFrame,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    prepared = frame.copy()
    inferred_categorical_cols = categorical_cols or [
        column for column in prepared.columns if _is_categorical_series(prepared[column])
    ]
    for column in prepared.columns:
        if column in inferred_categorical_cols:
            prepared[column] = prepared[column].astype("string")
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared, inferred_categorical_cols


def _is_categorical_series(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _build_tabpfn_classifier(categorical_indices: list[int] | None = None):
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as exc:
        raise OptionalDependencyNotAvailable(
            "tabpfn is not installed. Install recpfn with the 'tabpfn' extra."
        ) from exc

    configured_version = _resolve_tabpfn_version(ModelVersion, read_env_str("RECPFN_TABPFN_VERSION", "v2"))
    return TabPFNClassifier.create_default_for_version(
        configured_version,
        device="cpu",
        n_estimators=4,
        categorical_features_indices=categorical_indices,
    )


def _resolve_tabpfn_version(model_version_enum, configured_value: str):
    normalized = configured_value.strip().lower()
    mapping = {
        "v2": model_version_enum.V2,
        "2": model_version_enum.V2,
        "v2.5": model_version_enum.V2_5,
        "2.5": model_version_enum.V2_5,
    }
    if normalized not in mapping:
        raise ValueError(
            "Unsupported RECPFN_TABPFN_VERSION value "
            f"'{configured_value}'. Expected one of: v2, 2, v2.5, 2.5."
        )
    return mapping[normalized]
