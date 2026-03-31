"""CatBoost-backed pointwise and pairwise classifier wrapper."""

from __future__ import annotations

from recpfn.exceptions import OptionalDependencyNotAvailable
from recpfn.models.base import BasePointwiseRanker


class CatBoostRanker(BasePointwiseRanker):
    """Binary classifier wrapper used for pointwise and pairwise ranking."""

    def build_model(self):
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise OptionalDependencyNotAvailable(
                "catboost is not installed. Install recpfn with the 'models' extra."
            ) from exc

        return CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_seed=0,
        )
