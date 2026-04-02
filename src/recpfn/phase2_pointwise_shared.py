"""Shared constants and helpers for the Phase 2 pointwise workflow."""

from __future__ import annotations

from pathlib import Path

PRIMARY_DATASET = "movielens_100k"
SECONDARY_DATASET = "amazon_baby_products"
PRIMARY_SPLITS = ["warm", "item_cold"]
PRIMARY_PROTOCOLS = ["global_popularity", "context_popularity"]
PRIMARY_POINTWISE_MODELS = ["xgboost", "catboost", "tabpfn", "tabpfn_native"]
TREE_MODELS = {"xgboost", "catboost"}
TABPFN_MODELS = {"tabpfn", "tabpfn_native"}
PRIMARY_TRAIN_FRACTIONS = [0.1, 0.2, 0.5, 1.0]
PRIMARY_SEEDS = [0, 1, 2]
PRIMARY_K = 20
PRIMARY_MAX_TRAIN_QUERIES = 100
PRIMARY_MAX_TEST_QUERIES = 100
AMAZON_MAX_TRAIN_QUERIES = 50
AMAZON_MAX_TEST_QUERIES = 50
K_SENSITIVITY_VALUES = [20, 50, 100]
FEATURE_ABLATION_SETS = [
    "full",
    "no_user_metadata",
    "no_item_metadata",
    "no_interaction_history",
    "metadata_only",
]
KEY_SLICE_DEFINITIONS = [
    (PRIMARY_DATASET, "warm", "global_popularity"),
    (PRIMARY_DATASET, "warm", "context_popularity"),
    (PRIMARY_DATASET, "item_cold", "global_popularity"),
    (PRIMARY_DATASET, "item_cold", "context_popularity"),
]
BOOTSTRAP_SLICES = [
    (PRIMARY_DATASET, "warm", "global_popularity"),
    (PRIMARY_DATASET, "item_cold", "global_popularity"),
    (PRIMARY_DATASET, "item_cold", "context_popularity"),
]
BOOTSTRAP_TRAIN_FRACTIONS = [0.1, 0.2, 0.5, 1.0]
BOOTSTRAP_COMPARISONS = [
    "tabpfn_native_minus_tabpfn",
    "best_tabpfn_minus_best_tree",
]
PHASE2_GROUPS = {
    "primary",
    "k_sensitivity",
    "amazon_sanity",
    "feature_group_ablation",
}
CANONICAL_TABPFN_VERSION = "v2.5"


def fraction_dir_name(train_fraction: float) -> str:
    return f"train_{int(train_fraction * 100):03d}"


def seed_dir_name(seed: int) -> str:
    return f"seed_{int(seed):03d}"


def k_dir_name(k: int) -> str:
    return f"k_{int(k):03d}"


def feature_set_dir_name(feature_set: str) -> str:
    return f"feature_set_{feature_set}"


def infer_phase2_group(path: Path, raw_root: Path) -> str | None:
    """Infer the top-level Phase 2 group from a raw artifact path."""

    relative = path.relative_to(raw_root)
    if not relative.parts:
        return None
    group = relative.parts[0]
    return group if group in PHASE2_GROUPS else None
