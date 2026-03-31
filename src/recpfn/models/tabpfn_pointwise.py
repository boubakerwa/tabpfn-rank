"""TabPFN-backed classifier wrapper."""

from __future__ import annotations

from recpfn.exceptions import OptionalDependencyNotAvailable
from recpfn.models.base import BasePointwiseRanker
from recpfn.utils import read_env_str


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

        configured_version = _resolve_tabpfn_version(ModelVersion, read_env_str("RECPFN_TABPFN_VERSION", "v2"))
        return TabPFNClassifier.create_default_for_version(
            configured_version,
            device="cpu",
            n_estimators=4,
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
