"""Run a single benchmark unit in a fresh Python process."""

from __future__ import annotations

import argparse

from recpfn.rerank.pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one benchmark unit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["warm", "item_cold"])
    parser.add_argument("--protocol", required=True, choices=["global_popularity", "context_popularity"])
    parser.add_argument("--mode", required=True, choices=["pointwise", "pairwise"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-test-queries", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pointwise_models = [args.model] if args.mode == "pointwise" else []
    pairwise_models = [args.model] if args.mode == "pairwise" else []
    results, artifacts = run_experiment(
        dataset_name=args.dataset,
        split_type=args.split,
        protocols=[args.protocol],
        pointwise_models=pointwise_models,
        pairwise_models=pairwise_models,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        k=args.k,
        max_train_queries=args.max_train_queries,
        max_test_queries=args.max_test_queries,
    )
    if results.empty:
        print("No results produced.")
        return
    print(results.to_string(index=False))
    if artifacts.summary_csv_path:
        print(f"\nSummary CSV: {artifacts.summary_csv_path}")


if __name__ == "__main__":
    main()
