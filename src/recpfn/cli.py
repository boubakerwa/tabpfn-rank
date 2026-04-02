"""Command-line entry point for benchmark runs."""

from __future__ import annotations

import argparse

from recpfn.features.groups import FEATURE_SET_CHOICES
from recpfn.rerank.pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RecPFN benchmark experiments.")
    parser.add_argument("--dataset", default="movielens_100k")
    parser.add_argument("--split", default="warm", choices=["warm", "item_cold"])
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["global_popularity", "context_popularity"],
        choices=["global_popularity", "context_popularity"],
    )
    parser.add_argument(
        "--pointwise-models",
        nargs="+",
        default=["popularity", "recent_popularity", "xgboost", "catboost", "tabpfn"],
    )
    parser.add_argument(
        "--pairwise-models",
        nargs="+",
        default=[],
    )
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="paper/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-test-queries", type=int, default=None)
    parser.add_argument("--feature-set", default="full", choices=list(FEATURE_SET_CHOICES))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results, artifacts = run_experiment(
        dataset_name=args.dataset,
        split_type=args.split,
        protocols=args.protocols,
        pointwise_models=args.pointwise_models,
        pairwise_models=args.pairwise_models,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        k=args.k,
        max_train_queries=args.max_train_queries,
        max_test_queries=args.max_test_queries,
        feature_set=args.feature_set,
    )
    if results.empty:
        print("No results produced.")
        return
    print(results.to_string(index=False))
    if artifacts.benchmark_table_path:
        print(f"\nBenchmark table: {artifacts.benchmark_table_path}")
