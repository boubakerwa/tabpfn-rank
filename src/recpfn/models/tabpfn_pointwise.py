"""TabPFN-backed classifier wrapper."""

from __future__ import annotations

from recpfn.exceptions import OptionalDependencyNotAvailable
from recpfn.models.base import BasePointwiseRanker


class TabPFNPointwiseRanker(BasePointwiseRanker):
    """Binary classifier wrapper used for pointwise and pairwise ranking."""

    def build_model(self):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:
            raise OptionalDependencyNotAvailable(
                "tabpfn is not installed. Install recpfn with the 'tabpfn' extra."
            ) from exc

        # Prefer the openly downloadable v2 checkpoint so local benchmark runs work
        # without requiring gated-model authentication.
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device="cpu",
            n_estimators=4,
        )
