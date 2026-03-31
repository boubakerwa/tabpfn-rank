"""TabPFN-backed classifier wrapper."""

from __future__ import annotations

from recpfn.exceptions import OptionalDependencyNotAvailable
from recpfn.models.base import BasePointwiseRanker


class TabPFNPointwiseRanker(BasePointwiseRanker):
    """Binary classifier wrapper used for pointwise and pairwise ranking."""

    def build_model(self):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise OptionalDependencyNotAvailable(
                "tabpfn is not installed. Install recpfn with the 'tabpfn' extra."
            ) from exc

        return TabPFNClassifier(device="cpu")
