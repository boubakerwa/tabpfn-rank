"""Focused MovieLens low-data ladder for the final Phase 1 decision step."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from recpfn.eval.reports import save_benchmark_table, save_summary_csv
from recpfn.phase1_decision import (
    CANONICAL_K,
    CANONICAL_MAX_TEST_QUERIES,
    CANONICAL_MAX_TRAIN_QUERIES,
    CANONICAL_SPLITS,
    CANONICAL_TABPFN_VERSION,
    ProgressTracker,
    _concat_results,
    _infer_train_fraction,
    _temporary_env,
    _tie_break_unit_count,
    evaluate_decision_outcome,
    load_existing_phase1_results,
    run_movie_lens_tie_break,
    snapshot_status,
    summarize_protocol_overlap,
    write_decision_memo,
    write_next_steps_plan,
)
from recpfn.types import Phase1DecisionArtifacts
from recpfn.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the focused MovieLens low-data Phase 1 decision step.")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--baseline-run-output-dir", default="paper/results_phase1_decision_runs_final")
    parser.add_argument("--low-data-run-output-dir", default="paper/results_phase1_low_data_runs")
    parser.add_argument("--output-dir", default="paper/phase1_low_data")
    parser.add_argument("--plots-output-dir", default="paper/figures/phase1_low_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-queries", type=int, default=CANONICAL_MAX_TRAIN_QUERIES)
    parser.add_argument("--max-test-queries", type=int, default=CANONICAL_MAX_TEST_QUERIES)
    parser.add_argument("--k", type=int, default=CANONICAL_K)
    parser.add_argument(
        "--reuse-existing-low-data",
        action="store_true",
        help="Do not rerun the low-data ladder. Merge existing low-data results from --low-data-run-output-dir.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results, artifacts, decision = run_phase1_low_data_step(
        cache_dir=args.cache_dir,
        baseline_run_output_dir=args.baseline_run_output_dir,
        low_data_run_output_dir=args.low_data_run_output_dir,
        output_dir=args.output_dir,
        plots_output_dir=args.plots_output_dir,
        seed=args.seed,
        max_train_queries=args.max_train_queries,
        max_test_queries=args.max_test_queries,
        k=args.k,
        reuse_existing_low_data=args.reuse_existing_low_data,
    )
    print(results.to_string(index=False))
    print(f"\nDecision outcome: {decision['outcome']}")
    if artifacts.decision_memo_path:
        print(f"Decision memo: {artifacts.decision_memo_path}")
    if artifacts.benchmark_table_path:
        print(f"Benchmark table: {artifacts.benchmark_table_path}")
    if artifacts.tie_break_results_path:
        print(f"Low-data results: {artifacts.tie_break_results_path}")
    if artifacts.plots_output_dir:
        print(f"Plots: {artifacts.plots_output_dir}")


def run_phase1_low_data_step(
    cache_dir: str | Path = "data",
    baseline_run_output_dir: str | Path = "paper/results_phase1_decision_runs_final",
    low_data_run_output_dir: str | Path = "paper/results_phase1_low_data_runs",
    output_dir: str | Path = "paper/phase1_low_data",
    plots_output_dir: str | Path = "paper/figures/phase1_low_data",
    seed: int = 0,
    max_train_queries: int = CANONICAL_MAX_TRAIN_QUERIES,
    max_test_queries: int = CANONICAL_MAX_TEST_QUERIES,
    k: int = CANONICAL_K,
    reuse_existing_low_data: bool = False,
) -> tuple[pd.DataFrame, Phase1DecisionArtifacts, dict[str, object]]:
    """Freeze the canonical snapshot, run the MovieLens low-data ladder, and emit final Phase 1 artifacts."""

    decision_root = ensure_dir(output_dir)
    low_data_run_root = ensure_dir(low_data_run_output_dir)
    plots_root = ensure_dir(plots_output_dir)
    artifacts = Phase1DecisionArtifacts(
        output_dir=decision_root,
        run_output_dir=low_data_run_root,
        plots_output_dir=plots_root,
    )

    baseline_results = load_existing_phase1_results(baseline_run_output_dir)
    if "phase" in baseline_results.columns:
        baseline_results = baseline_results[baseline_results["phase"] == "canonical"].copy()
    baseline_snapshot = snapshot_status(baseline_results)
    artifacts.baseline_results_path = save_summary_csv(baseline_results, decision_root / "baseline_results.csv")

    overlap_notes = [
        summarize_protocol_overlap(
            dataset_name=dataset_name,
            split_type=split_type,
            cache_dir=cache_dir,
            seed=seed,
            k=k,
            max_train_queries=max_train_queries,
            max_test_queries=max_test_queries,
        )
        for dataset_name in ("movielens_100k", "amazon_baby_products")
        for split_type in CANONICAL_SPLITS
    ]

    if reuse_existing_low_data:
        low_data_summary_path = Path(output_dir) / "low_data_results.csv"
        if low_data_summary_path.exists():
            low_data_results = pd.read_csv(low_data_summary_path)
        else:
            low_data_results = load_existing_phase1_results(low_data_run_root)
            if not low_data_results.empty:
                low_data_results = low_data_results.copy()
                low_data_results["phase"] = "tie_break"
                if "train_fraction" not in low_data_results.columns:
                    low_data_results["train_fraction"] = pd.NA
                low_data_results["train_fraction"] = low_data_results["train_fraction"].fillna(
                    low_data_results["max_train_queries"].map(_infer_train_fraction)
                )
        if low_data_results.empty:
            raise FileNotFoundError(f"No low-data results.csv files found under {low_data_run_root}.")
    else:
        tracker = ProgressTracker(total_units=_tie_break_unit_count())
        print(
            f"Phase 1 low-data sweep: {tracker.total_units} benchmark units "
            "(MovieLens only, warm + item_cold, global_popularity).",
            flush=True,
        )
        with _temporary_env("RECPFN_TABPFN_VERSION", CANONICAL_TABPFN_VERSION):
            with _temporary_env("TABPFN_ALLOW_CPU_LARGE_DATASET", "1"):
                low_data_results = run_movie_lens_tie_break(
                    cache_dir=cache_dir,
                    output_dir=low_data_run_root,
                    seed=seed,
                    k=k,
                    canonical_max_train_queries=max_train_queries,
                    max_test_queries=max_test_queries,
                    tracker=tracker,
                )

    artifacts.tie_break_results_path = save_summary_csv(low_data_results, decision_root / "low_data_results.csv")
    merged = _concat_results([baseline_results, low_data_results])
    artifacts.merged_results_path = save_summary_csv(merged, decision_root / "merged_results.csv")
    artifacts.benchmark_table_path = save_benchmark_table(
        merged,
        decision_root / "benchmark.md",
        columns=[
            "phase",
            "dataset",
            "split_type",
            "protocol",
            "mode",
            "model",
            "train_fraction",
            "ndcg@10",
            "recall@10",
            "mrr",
            "hitrate@10",
            "runtime_seconds",
            "n_queries",
        ],
        sort_by=["phase", "dataset", "split_type", "protocol", "mode", "train_fraction", "model"],
    )

    decision = evaluate_decision_outcome(merged, overlap_notes)
    artifacts.decision_memo_path = write_decision_memo(
        merged,
        decision,
        overlap_notes,
        decision_root / "decision.md",
        snapshot=baseline_snapshot,
    )
    artifacts.next_steps_plan_path = write_next_steps_plan(
        decision,
        decision_root / "next_steps.md",
        snapshot=baseline_snapshot,
        low_data_ran=True,
    )
    _generate_plots_from_summary(artifacts.merged_results_path, plots_root)
    return merged, artifacts, decision


def _generate_plots_from_summary(summary_csv_path: Path, plots_output_dir: Path) -> None:
    """Regenerate the figure set from one merged summary CSV."""

    command = [
        sys.executable,
        "experiments/plot_phase1_results.py",
        "--summary-csv",
        str(summary_csv_path),
        "--output-dir",
        str(plots_output_dir),
        "--title-suffix",
        "(canonical + low-data)",
    ]
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
