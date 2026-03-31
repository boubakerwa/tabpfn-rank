"""XGBoost-backed pointwise and pairwise classifier wrapper."""

from __future__ import annotations

from recpfn.exceptions import OptionalDependencyNotAvailable
from recpfn.models.base import BasePointwiseRanker


class XGBoostRanker(BasePointwiseRanker):
    """Binary classifier wrapper used for pointwise and pairwise ranking."""

    def build_model(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise OptionalDependencyNotAvailable(
                "xgboost is not installed. Install recpfn with the 'models' extra."
            ) from exc

        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=0,
            n_jobs=1,
        )
