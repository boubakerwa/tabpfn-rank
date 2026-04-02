"""Raw benchmark runner for the Phase 2 pointwise validation matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from recpfn.benchmark_runner import (
    DEFAULT_UNIT_TIMEOUT_SECONDS,
    ProgressTracker,
    concat_results as _concat_results,
    run_unit_matrix as _run_unit_matrix,
    temporary_env as _temporary_env,
)
from recpfn.phase2_pointwise_shared import (
    AMAZON_MAX_TEST_QUERIES,
    AMAZON_MAX_TRAIN_QUERIES,
    CANONICAL_TABPFN_VERSION,
    FEATURE_ABLATION_SETS,
    K_SENSITIVITY_VALUES,
    PRIMARY_DATASET,
    PRIMARY_K,
    PRIMARY_MAX_TEST_QUERIES,
    PRIMARY_MAX_TRAIN_QUERIES,
    PRIMARY_POINTWISE_MODELS,
    PRIMARY_PROTOCOLS,
    PRIMARY_SEEDS,
    PRIMARY_SPLITS,
    PRIMARY_TRAIN_FRACTIONS,
    SECONDARY_DATASET,
    TREE_MODELS,
    feature_set_dir_name,
    fraction_dir_name,
    k_dir_name,
    seed_dir_name,
)
from recpfn.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the raw Phase 2 pointwise validation units.")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--run-output-dir", default="paper/results_phase2_pointwise_runs")
    parser.add_argument("--primary-dataset", default=PRIMARY_DATASET)
    parser.add_argument("--secondary-dataset", default=SECONDARY_DATASET)
    parser.add_argument("--seeds", nargs="+", type=int, default=PRIMARY_SEEDS)
    parser.add_argument("--train-fractions", nargs="+", type=float, default=PRIMARY_TRAIN_FRACTIONS)
    parser.add_argument("--k-values", nargs="+", type=int, default=K_SENSITIVITY_VALUES)
    parser.add_argument("--max-train-queries", type=int, default=PRIMARY_MAX_TRAIN_QUERIES)
    parser.add_argument("--max-test-queries", type=int, default=PRIMARY_MAX_TEST_QUERIES)
    parser.add_argument("--amazon-max-train-queries", type=int, default=AMAZON_MAX_TRAIN_QUERIES)
    parser.add_argument("--amazon-max-test-queries", type=int, default=AMAZON_MAX_TEST_QUERIES)
    parser.add_argument("--k", type=int, default=PRIMARY_K)
    parser.add_argument("--unit-timeout-seconds", type=int, default=DEFAULT_UNIT_TIMEOUT_SECONDS)
    parser.add_argument("--skip-k-sensitivity", action="store_true")
    parser.add_argument("--skip-amazon-sanity", action="store_true")
    parser.add_argument("--skip-feature-ablation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_phase2_pointwise_raw(
        cache_dir=args.cache_dir,
        run_output_dir=args.run_output_dir,
        primary_dataset=args.primary_dataset,
        secondary_dataset=args.secondary_dataset,
        seeds=args.seeds,
        train_fractions=args.train_fractions,
        k_values=args.k_values,
        k=args.k,
        max_train_queries=args.max_train_queries,
        max_test_queries=args.max_test_queries,
        amazon_max_train_queries=args.amazon_max_train_queries,
        amazon_max_test_queries=args.amazon_max_test_queries,
        unit_timeout_seconds=args.unit_timeout_seconds,
        include_k_sensitivity=not args.skip_k_sensitivity,
        include_amazon_sanity=not args.skip_amazon_sanity,
        include_feature_ablation=not args.skip_feature_ablation,
    )
    print(results.to_string(index=False))
    print(f"\nRaw output root: {Path(args.run_output_dir).resolve()}")


def run_phase2_pointwise_raw(
    cache_dir: str | Path = "data",
    run_output_dir: str | Path = "paper/results_phase2_pointwise_runs",
    primary_dataset: str = PRIMARY_DATASET,
    secondary_dataset: str | None = SECONDARY_DATASET,
    seeds: list[int] | None = None,
    train_fractions: list[float] | None = None,
    k_values: list[int] | None = None,
    k: int = PRIMARY_K,
    max_train_queries: int = PRIMARY_MAX_TRAIN_QUERIES,
    max_test_queries: int = PRIMARY_MAX_TEST_QUERIES,
    amazon_max_train_queries: int = AMAZON_MAX_TRAIN_QUERIES,
    amazon_max_test_queries: int = AMAZON_MAX_TEST_QUERIES,
    unit_timeout_seconds: int = DEFAULT_UNIT_TIMEOUT_SECONDS,
    include_k_sensitivity: bool = True,
    include_amazon_sanity: bool = True,
    include_feature_ablation: bool = True,
) -> pd.DataFrame:
    """Run the raw Phase 2 pointwise benchmark units."""

    seeds = list(seeds or PRIMARY_SEEDS)
    train_fractions = list(train_fractions or PRIMARY_TRAIN_FRACTIONS)
    k_values = list(k_values or K_SENSITIVITY_VALUES)
    raw_root = ensure_dir(run_output_dir)

    total_units = _primary_unit_count(seeds, train_fractions, split_count=len(PRIMARY_SPLITS), protocol_count=len(PRIMARY_PROTOCOLS))
    if include_k_sensitivity:
        total_units += _k_sensitivity_unit_count(seeds, k_values, split_count=len(PRIMARY_SPLITS), protocol_count=len(PRIMARY_PROTOCOLS))
    if include_amazon_sanity and secondary_dataset:
        total_units += _amazon_unit_count(seeds, split_count=len(PRIMARY_SPLITS), protocol_count=len(PRIMARY_PROTOCOLS))
    if include_feature_ablation:
        total_units += _feature_ablation_unit_count(protocol_count=2)
    tracker = ProgressTracker(total_units=total_units)

    print(
        f"Phase 2 pointwise raw sweep: {total_units} benchmark units "
        f"({len(seeds)} seeds, {len(train_fractions)} train fractions).",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    with _temporary_env("RECPFN_TABPFN_VERSION", CANONICAL_TABPFN_VERSION):
        with _temporary_env("TABPFN_ALLOW_CPU_LARGE_DATASET", "1"):
            primary = _run_primary_matrix(
                cache_dir=cache_dir,
                output_dir=raw_root / "primary",
                dataset_name=primary_dataset,
                seeds=seeds,
                train_fractions=train_fractions,
                k=k,
                max_train_queries=max_train_queries,
                max_test_queries=max_test_queries,
                tracker=tracker,
                timeout_seconds=unit_timeout_seconds,
            )
            frames.append(primary)

            best_tree_by_slice = _select_best_tree_models(primary)
            selection_path = raw_root / "best_tree_selection.json"
            selection_path.write_text(json.dumps(best_tree_by_slice, indent=2, sort_keys=True), encoding="utf-8")

            if include_k_sensitivity:
                frames.append(
                    _run_k_sensitivity(
                        cache_dir=cache_dir,
                        output_dir=raw_root / "k_sensitivity",
                        dataset_name=primary_dataset,
                        seeds=seeds,
                        k_values=k_values,
                        best_tree_by_slice=best_tree_by_slice,
                        max_train_queries=max_train_queries,
                        max_test_queries=max_test_queries,
                        tracker=tracker,
                        timeout_seconds=unit_timeout_seconds,
                    )
                )

            if include_amazon_sanity and secondary_dataset:
                frames.append(
                    _run_amazon_sanity(
                        cache_dir=cache_dir,
                        output_dir=raw_root / "amazon_sanity",
                        dataset_name=secondary_dataset,
                        seeds=seeds,
                        k=k,
                        max_train_queries=amazon_max_train_queries,
                        max_test_queries=amazon_max_test_queries,
                        tracker=tracker,
                        timeout_seconds=unit_timeout_seconds,
                    )
                )

            if include_feature_ablation:
                frames.append(
                    _run_feature_group_ablation(
                        cache_dir=cache_dir,
                        output_dir=raw_root / "feature_group_ablation",
                        dataset_name=primary_dataset,
                        k=k,
                        max_train_queries=max_train_queries,
                        max_test_queries=max_test_queries,
                        tracker=tracker,
                        timeout_seconds=unit_timeout_seconds,
                    )
                )

    return _concat_results(frames)


def _run_primary_matrix(
    cache_dir: str | Path,
    output_dir: Path,
    dataset_name: str,
    seeds: list[int],
    train_fractions: list[float],
    k: int,
    max_train_queries: int,
    max_test_queries: int,
    tracker: ProgressTracker,
    timeout_seconds: int,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        for train_fraction in train_fractions:
            capped_train_queries = max(1, math.ceil(max_train_queries * train_fraction))
            seed_output_dir = output_dir / fraction_dir_name(train_fraction) / seed_dir_name(seed)
            for split_type in PRIMARY_SPLITS:
                results = _run_unit_matrix(
                    dataset_name=dataset_name,
                    split_type=split_type,
                    protocols=PRIMARY_PROTOCOLS,
                    models=PRIMARY_POINTWISE_MODELS,
                    mode="pointwise",
                    cache_dir=cache_dir,
                    output_dir=seed_output_dir,
                    seed=seed,
                    k=k,
                    max_train_queries=capped_train_queries,
                    max_test_queries=max_test_queries,
                    tracker=tracker,
                    timeout_seconds=timeout_seconds,
                    train_fraction=float(train_fraction),
                )
                frames.append(_annotate_phase2_group(results, "primary"))
    return _concat_results(frames)


def _run_k_sensitivity(
    cache_dir: str | Path,
    output_dir: Path,
    dataset_name: str,
    seeds: list[int],
    k_values: list[int],
    best_tree_by_slice: dict[str, str],
    max_train_queries: int,
    max_test_queries: int,
    tracker: ProgressTracker,
    timeout_seconds: int,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        for candidate_k in k_values:
            seed_output_dir = output_dir / k_dir_name(candidate_k) / seed_dir_name(seed)
            for split_type in PRIMARY_SPLITS:
                for protocol in PRIMARY_PROTOCOLS:
                    selected_tree = best_tree_by_slice.get(f"{split_type}::{protocol}", "xgboost")
                    models = [selected_tree, "tabpfn", "tabpfn_native"]
                    results = _run_unit_matrix(
                        dataset_name=dataset_name,
                        split_type=split_type,
                        protocols=[protocol],
                        models=models,
                        mode="pointwise",
                        cache_dir=cache_dir,
                        output_dir=seed_output_dir,
                        seed=seed,
                        k=int(candidate_k),
                        max_train_queries=max_train_queries,
                        max_test_queries=max_test_queries,
                        tracker=tracker,
                        timeout_seconds=timeout_seconds,
                        train_fraction=1.0,
                    )
                    frames.append(_annotate_phase2_group(results, "k_sensitivity"))
    return _concat_results(frames)


def _run_amazon_sanity(
    cache_dir: str | Path,
    output_dir: Path,
    dataset_name: str,
    seeds: list[int],
    k: int,
    max_train_queries: int,
    max_test_queries: int,
    tracker: ProgressTracker,
    timeout_seconds: int,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        seed_output_dir = output_dir / seed_dir_name(seed)
        for split_type in PRIMARY_SPLITS:
            results = _run_unit_matrix(
                dataset_name=dataset_name,
                split_type=split_type,
                protocols=PRIMARY_PROTOCOLS,
                models=PRIMARY_POINTWISE_MODELS,
                mode="pointwise",
                cache_dir=cache_dir,
                output_dir=seed_output_dir,
                seed=seed,
                k=k,
                max_train_queries=max_train_queries,
                max_test_queries=max_test_queries,
                tracker=tracker,
                timeout_seconds=timeout_seconds,
                train_fraction=1.0,
            )
            frames.append(_annotate_phase2_group(results, "amazon_sanity"))
    return _concat_results(frames)


def _run_feature_group_ablation(
    cache_dir: str | Path,
    output_dir: Path,
    dataset_name: str,
    k: int,
    max_train_queries: int,
    max_test_queries: int,
    tracker: ProgressTracker,
    timeout_seconds: int,
) -> pd.DataFrame:
    frames = []
    for feature_set in FEATURE_ABLATION_SETS:
        feature_output_dir = output_dir / feature_set_dir_name(feature_set) / seed_dir_name(0)
        results = _run_unit_matrix(
            dataset_name=dataset_name,
            split_type="item_cold",
            protocols=PRIMARY_PROTOCOLS,
            models=["tabpfn", "tabpfn_native"],
            mode="pointwise",
            cache_dir=cache_dir,
            output_dir=feature_output_dir,
            seed=0,
            k=k,
            max_train_queries=max_train_queries,
            max_test_queries=max_test_queries,
            tracker=tracker,
            timeout_seconds=timeout_seconds,
            train_fraction=1.0,
            feature_set=feature_set,
        )
        frames.append(_annotate_phase2_group(results, "feature_group_ablation"))
    return _concat_results(frames)


def _select_best_tree_models(primary_results: pd.DataFrame) -> dict[str, str]:
    subset = primary_results[
        (primary_results["phase2_group"] == "primary")
        & (primary_results["train_fraction"] == 1.0)
        & (primary_results["model"].isin(TREE_MODELS))
        & (primary_results["status"] == "ok")
    ].copy()
    mapping: dict[str, str] = {}
    if subset.empty:
        return mapping

    aggregated = (
        subset.groupby(["split_type", "protocol", "model"], as_index=False)
        .agg(
            ndcg_mean=("ndcg@10", "mean"),
            runtime_median=("runtime_seconds", "median"),
        )
        .sort_values(["split_type", "protocol", "ndcg_mean", "runtime_median", "model"], ascending=[True, True, False, True, True])
    )
    for (split_type, protocol), group in aggregated.groupby(["split_type", "protocol"], sort=False):
        mapping[f"{split_type}::{protocol}"] = str(group.iloc[0]["model"])
    return mapping


def _annotate_phase2_group(frame: pd.DataFrame, phase2_group: str) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["phase2_group"] = phase2_group
    return annotated


def _primary_unit_count(seeds: list[int], train_fractions: list[float], split_count: int, protocol_count: int) -> int:
    return len(seeds) * len(train_fractions) * split_count * protocol_count * len(PRIMARY_POINTWISE_MODELS)


def _k_sensitivity_unit_count(seeds: list[int], k_values: list[int], split_count: int, protocol_count: int) -> int:
    return len(seeds) * len(k_values) * split_count * protocol_count * 3


def _amazon_unit_count(seeds: list[int], split_count: int, protocol_count: int) -> int:
    return len(seeds) * split_count * protocol_count * len(PRIMARY_POINTWISE_MODELS)


def _feature_ablation_unit_count(protocol_count: int) -> int:
    return len(FEATURE_ABLATION_SETS) * protocol_count * 2


if __name__ == "__main__":
    main()
