"""Canonical phase-one benchmark sweep and decision reporting."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from recpfn.benchmark_runner import (
    DEFAULT_UNIT_TIMEOUT_SECONDS,
    ProgressTracker,
    concat_results as _concat_results,
    run_unit_matrix as _run_unit_matrix,
    run_unit_subprocess as _run_unit_subprocess,
    temporary_env as _temporary_env,
)
from recpfn.data.loaders import load_dataset
from recpfn.data.splits import build_splits
from recpfn.eval.reports import save_benchmark_table, save_summary_csv
from recpfn.rerank.candidate_sets import build_candidates
from recpfn.rerank.pipeline import limit_split_queries
from recpfn.types import Phase1DecisionArtifacts
from recpfn.utils import ensure_dir, read_env_int

CANONICAL_DATASETS = ["movielens_100k", "amazon_baby_products"]
CANONICAL_SPLITS = ["warm", "item_cold"]
CANONICAL_PROTOCOLS = ["global_popularity", "context_popularity"]
CANONICAL_POINTWISE_MODELS = ["popularity", "recent_popularity", "tabpfn", "xgboost", "catboost"]
CANONICAL_PAIRWISE_MODELS = ["tabpfn", "xgboost", "catboost"]
CANONICAL_MAX_TRAIN_QUERIES = 100
CANONICAL_MAX_TEST_QUERIES = 100
CANONICAL_K = 20
CANONICAL_TABPFN_VERSION = "v2.5"
PRIMARY_DATASET = "movielens_100k"
SECONDARY_DATASET = "amazon_baby_products"
TREE_MODELS = {"xgboost", "catboost"}
BASELINE_MODELS = {"popularity", "recent_popularity", *TREE_MODELS}
LEARNED_MODELS = {"xgboost", "catboost", "tabpfn"}
TIE_BREAK_FRACTIONS = [0.1, 0.2, 0.5, 1.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical Phase 1 decision sweep.")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="paper/phase1_decision")
    parser.add_argument("--run-output-dir", default="paper/results_phase1_decision_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-queries", type=int, default=CANONICAL_MAX_TRAIN_QUERIES)
    parser.add_argument("--max-test-queries", type=int, default=CANONICAL_MAX_TEST_QUERIES)
    parser.add_argument("--k", type=int, default=CANONICAL_K)
    parser.add_argument("--unit-timeout-seconds", type=int, default=DEFAULT_UNIT_TIMEOUT_SECONDS)
    parser.add_argument("--skip-tie-break", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Do not rerun benchmarks. Merge and summarize existing results.csv files under --run-output-dir.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.reuse_existing:
        results, artifacts, decision = summarize_existing_phase1_runs(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            run_output_dir=args.run_output_dir,
            seed=args.seed,
            max_train_queries=args.max_train_queries,
            max_test_queries=args.max_test_queries,
            k=args.k,
            unit_timeout_seconds=args.unit_timeout_seconds,
        )
    else:
        results, artifacts, decision = run_phase1_decision(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            run_output_dir=args.run_output_dir,
            seed=args.seed,
            max_train_queries=args.max_train_queries,
            max_test_queries=args.max_test_queries,
            k=args.k,
            allow_tie_break=not args.skip_tie_break,
            unit_timeout_seconds=args.unit_timeout_seconds,
        )
    print(results.to_string(index=False))
    print(f"\nDecision outcome: {decision['outcome']}")
    if artifacts.decision_memo_path:
        print(f"Decision memo: {artifacts.decision_memo_path}")
    if artifacts.benchmark_table_path:
        print(f"Benchmark table: {artifacts.benchmark_table_path}")
    if artifacts.next_steps_plan_path:
        print(f"Next steps: {artifacts.next_steps_plan_path}")


def run_phase1_decision(
    cache_dir: str | Path = "data",
    output_dir: str | Path = "paper/phase1_decision",
    run_output_dir: str | Path = "paper/results_phase1_decision_runs",
    seed: int = 0,
    max_train_queries: int = CANONICAL_MAX_TRAIN_QUERIES,
    max_test_queries: int = CANONICAL_MAX_TEST_QUERIES,
    k: int = CANONICAL_K,
    allow_tie_break: bool = True,
    unit_timeout_seconds: int = DEFAULT_UNIT_TIMEOUT_SECONDS,
) -> tuple[pd.DataFrame, Phase1DecisionArtifacts, dict[str, object]]:
    """Run the canonical benchmark matrix and emit decision artifacts."""

    decision_root = ensure_dir(output_dir)
    raw_run_root = ensure_dir(run_output_dir)
    artifacts = Phase1DecisionArtifacts(output_dir=decision_root, run_output_dir=raw_run_root)
    overlap_notes = []
    result_frames = []
    tracker = ProgressTracker(total_units=_canonical_unit_count())

    print(
        f"Phase 1 canonical sweep: {tracker.total_units} benchmark units "
        f"({len(CANONICAL_DATASETS)} datasets, {len(CANONICAL_SPLITS)} splits).",
        flush=True,
    )

    with _temporary_env("RECPFN_TABPFN_VERSION", CANONICAL_TABPFN_VERSION):
        with _temporary_env("TABPFN_ALLOW_CPU_LARGE_DATASET", "1"):
            for dataset_name in CANONICAL_DATASETS:
                for split_type in CANONICAL_SPLITS:
                    overlap_notes.append(
                        summarize_protocol_overlap(
                            dataset_name=dataset_name,
                            split_type=split_type,
                            cache_dir=cache_dir,
                            seed=seed,
                            k=k,
                            max_train_queries=max_train_queries,
                            max_test_queries=max_test_queries,
                        )
                    )
                    pointwise_results = _run_unit_matrix(
                        dataset_name=dataset_name,
                        split_type=split_type,
                        protocols=CANONICAL_PROTOCOLS,
                        models=CANONICAL_POINTWISE_MODELS,
                        mode="pointwise",
                        cache_dir=cache_dir,
                        output_dir=Path(raw_run_root) / "canonical_pointwise",
                        seed=seed,
                        k=k,
                        max_train_queries=max_train_queries,
                        max_test_queries=max_test_queries,
                        tracker=tracker,
                        timeout_seconds=unit_timeout_seconds,
                    )
                    pairwise_results = _run_unit_matrix(
                        dataset_name=dataset_name,
                        split_type=split_type,
                        protocols=CANONICAL_PROTOCOLS,
                        models=CANONICAL_PAIRWISE_MODELS,
                        mode="pairwise",
                        cache_dir=cache_dir,
                        output_dir=Path(raw_run_root) / "canonical_pairwise",
                        seed=seed,
                        k=k,
                        max_train_queries=max_train_queries,
                        max_test_queries=max_test_queries,
                        tracker=tracker,
                        timeout_seconds=unit_timeout_seconds,
                    )
                    results = _concat_results([pointwise_results, pairwise_results]).copy()
                    results["phase"] = "canonical"
                    results["train_fraction"] = pd.NA
                    result_frames.append(results)

            merged = _concat_results(result_frames)
            decision = evaluate_decision_outcome(merged, overlap_notes)

            if allow_tie_break and decision["outcome"] == "run one tie-break sweep":
                tie_break_units = _tie_break_unit_count()
                tracker.extend(tie_break_units)
                print(f"Tie-break triggered: adding {tie_break_units} benchmark units.", flush=True)
                tie_break_results = run_movie_lens_tie_break(
                    cache_dir=cache_dir,
                    output_dir=Path(raw_run_root) / "tie_break",
                    seed=seed,
                    k=k,
                    canonical_max_train_queries=max_train_queries,
                    max_test_queries=max_test_queries,
                    tracker=tracker,
                    timeout_seconds=unit_timeout_seconds,
                )
                artifacts.tie_break_results_path = save_summary_csv(
                    tie_break_results,
                    decision_root / "tie_break_results.csv",
                )
                merged = _concat_results([merged, tie_break_results])
                decision = evaluate_decision_outcome(merged, overlap_notes)

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
    artifacts.decision_memo_path = write_decision_memo(
        merged,
        decision,
        overlap_notes,
        decision_root / "decision.md",
    )
    artifacts.next_steps_plan_path = write_next_steps_plan(
        decision,
        decision_root / "next_steps.md",
        low_data_ran=bool((merged.get("phase") == "tie_break").any()) if not merged.empty and "phase" in merged.columns else False,
    )
    return merged, artifacts, decision


def summarize_existing_phase1_runs(
    cache_dir: str | Path = "data",
    output_dir: str | Path = "paper/phase1_decision",
    run_output_dir: str | Path = "paper/results_phase1_decision_runs",
    seed: int = 0,
    max_train_queries: int = CANONICAL_MAX_TRAIN_QUERIES,
    max_test_queries: int = CANONICAL_MAX_TEST_QUERIES,
    k: int = CANONICAL_K,
    unit_timeout_seconds: int = DEFAULT_UNIT_TIMEOUT_SECONDS,
) -> tuple[pd.DataFrame, Phase1DecisionArtifacts, dict[str, object]]:
    """Merge already-completed Phase 1 units and emit decision artifacts without rerunning benchmarks."""

    del unit_timeout_seconds
    decision_root = ensure_dir(output_dir)
    raw_run_root = ensure_dir(run_output_dir)
    artifacts = Phase1DecisionArtifacts(output_dir=decision_root, run_output_dir=raw_run_root)
    results = load_existing_phase1_results(raw_run_root)
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
        for dataset_name in CANONICAL_DATASETS
        for split_type in CANONICAL_SPLITS
    ]
    decision = evaluate_decision_outcome(results, overlap_notes)
    snapshot = snapshot_status(results)
    artifacts.merged_results_path = save_summary_csv(results, decision_root / "merged_results.csv")
    artifacts.benchmark_table_path = save_benchmark_table(
        results,
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
    artifacts.decision_memo_path = write_decision_memo(
        results,
        decision,
        overlap_notes,
        decision_root / "decision.md",
        snapshot=snapshot,
    )
    artifacts.next_steps_plan_path = write_next_steps_plan(
        decision,
        decision_root / "next_steps.md",
        snapshot=snapshot,
        low_data_ran=bool((results.get("phase") == "tie_break").any()) if not results.empty and "phase" in results.columns else False,
    )
    return results, artifacts, decision


def run_movie_lens_tie_break(
    cache_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    k: int,
    canonical_max_train_queries: int,
    max_test_queries: int,
    tracker: ProgressTracker,
    timeout_seconds: int = DEFAULT_UNIT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Run the low-data tie-break sweep on MovieLens only."""

    frames = []
    for split_type in CANONICAL_SPLITS:
        for train_fraction in TIE_BREAK_FRACTIONS:
            max_train_queries = max(1, math.ceil(canonical_max_train_queries * train_fraction))
            fraction_dir = Path(output_dir) / f"train_{int(train_fraction * 100):03d}"
            pointwise_results = _run_unit_matrix(
                dataset_name=PRIMARY_DATASET,
                split_type=split_type,
                protocols=["global_popularity"],
                models=["xgboost", "catboost", "tabpfn"],
                mode="pointwise",
                cache_dir=cache_dir,
                output_dir=fraction_dir / "pointwise",
                seed=seed,
                k=k,
                max_train_queries=max_train_queries,
                max_test_queries=max_test_queries,
                tracker=tracker,
                timeout_seconds=timeout_seconds,
                train_fraction=float(train_fraction),
            )
            pairwise_results = _run_unit_matrix(
                dataset_name=PRIMARY_DATASET,
                split_type=split_type,
                protocols=["global_popularity"],
                models=["tabpfn", "xgboost", "catboost"],
                mode="pairwise",
                cache_dir=cache_dir,
                output_dir=fraction_dir / "pairwise",
                seed=seed,
                k=k,
                max_train_queries=max_train_queries,
                max_test_queries=max_test_queries,
                tracker=tracker,
                timeout_seconds=timeout_seconds,
                train_fraction=float(train_fraction),
            )
            results = _concat_results([pointwise_results, pairwise_results]).copy()
            results["phase"] = "tie_break"
            results["train_fraction"] = float(train_fraction)
            frames.append(results)
    return _concat_results(frames)


def summarize_protocol_overlap(
    dataset_name: str,
    split_type: str,
    cache_dir: str | Path,
    seed: int,
    k: int,
    max_train_queries: int | None,
    max_test_queries: int | None,
) -> dict[str, object]:
    """Measure how distinct the two candidate protocols are on capped test queries."""

    dataset = load_dataset(dataset_name, cache_dir=cache_dir, seed=seed)
    split = limit_split_queries(
        build_splits(dataset, split_type=split_type, seed=seed),
        max_train_queries=max_train_queries,
        max_test_queries=max_test_queries,
    )
    global_candidates = build_candidates(dataset, split, protocol="global_popularity", k=k, seed=seed)
    context_candidates = build_candidates(dataset, split, protocol="context_popularity", k=k, seed=seed)

    global_sets = _query_item_sets(global_candidates[global_candidates["split"] == "test"])
    context_sets = _query_item_sets(context_candidates[context_candidates["split"] == "test"])
    common_queries = sorted(set(global_sets) & set(context_sets))
    if not common_queries:
        return {
            "dataset": dataset_name,
            "split_type": split_type,
            "n_queries": 0,
            "identical_rate": 0.0,
            "mean_jaccard": 0.0,
            "collapsed": False,
        }

    identical = 0
    jaccards = []
    for query_id in common_queries:
        global_items = global_sets[query_id]
        context_items = context_sets[query_id]
        if global_items == context_items:
            identical += 1
        union = global_items | context_items
        jaccards.append(len(global_items & context_items) / max(1, len(union)))

    identical_rate = identical / len(common_queries)
    mean_jaccard = sum(jaccards) / len(jaccards)
    return {
        "dataset": dataset_name,
        "split_type": split_type,
        "n_queries": len(common_queries),
        "identical_rate": identical_rate,
        "mean_jaccard": mean_jaccard,
        "collapsed": identical_rate >= 0.9 or mean_jaccard >= 0.95,
    }


def evaluate_decision_outcome(
    results: pd.DataFrame,
    overlap_notes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Apply the decision rubric to merged benchmark results."""

    ok = results[results["status"] == "ok"].copy()
    tie_break_ran = bool((ok.get("phase") == "tie_break").any()) if not ok.empty and "phase" in ok.columns else False
    overlap_notes = overlap_notes or []

    pairwise_gains = _tabpfn_pairwise_gains(ok, PRIMARY_DATASET)
    meaningful_pairwise_rows = [
        row
        for row in pairwise_gains
        if row["ndcg_gain"] >= 0.02 or row["recall_gain"] >= 0.02
    ]
    competitive_rows = _tabpfn_competitive_rows(ok)
    amazon_saturated = _amazon_is_saturated(ok)
    movie_close = _movie_lens_is_close(ok)
    protocol_collapsed = [note for note in overlap_notes if note["collapsed"]]
    runtime_penalty = _tabpfn_runtime_penalty(ok)

    reasons = []
    if meaningful_pairwise_rows:
        best_gain = max(meaningful_pairwise_rows, key=lambda row: row["ndcg_gain"])
        reasons.append(
            "Pairwise TabPFN cleared the positive threshold "
            f"on {best_gain['split_type']} / {best_gain['protocol']} "
            f"with +{best_gain['ndcg_gain']:.4f} NDCG@10."
        )
    if competitive_rows:
        best_competitive = max(competitive_rows, key=lambda row: row["tabpfn_ndcg"])
        reasons.append(
            "A TabPFN variant was competitive with the tree baselines "
            f"on {best_competitive['split_type']} / {best_competitive['protocol']}."
        )
    if amazon_saturated:
        reasons.append("Amazon was saturated under the current oracle-positive benchmark and is non-decisive.")
    if protocol_collapsed:
        reasons.append("At least one context-popularity protocol collapsed close to global popularity.")
    if runtime_penalty is not None and runtime_penalty > 1.5:
        reasons.append(f"TabPFN median runtime was {runtime_penalty:.2f}x the best tree baseline on matched slices.")

    if meaningful_pairwise_rows or competitive_rows:
        outcome = "drill further"
    elif not tie_break_ran and amazon_saturated and movie_close:
        outcome = "run one tie-break sweep"
        reasons.append("MovieLens is close enough that a low-data tie-break is still justified.")
    else:
        outcome = "stop at benchmark"
        reasons.append("Tree baselines dominated the informative settings without a compensating TabPFN win.")

    return {
        "outcome": outcome,
        "pairwise_gains": pairwise_gains,
        "competitive_rows": competitive_rows,
        "amazon_saturated": amazon_saturated,
        "movie_lens_close": movie_close,
        "protocol_overlap": overlap_notes,
        "runtime_penalty_ratio": runtime_penalty,
        "reasons": reasons,
    }


def write_decision_memo(
    results: pd.DataFrame,
    decision: dict[str, object],
    overlap_notes: list[dict[str, object]],
    path: str | Path,
    snapshot: dict[str, object] | None = None,
) -> Path:
    """Write a short markdown memo explaining the decision outcome."""

    path = Path(path)
    ensure_dir(path.parent)
    ok = results[results["status"] == "ok"].copy()
    canonical = ok[ok["phase"] == "canonical"] if "phase" in ok.columns else ok
    tie_break = ok[ok["phase"] == "tie_break"] if "phase" in ok.columns else pd.DataFrame()
    movielens_best = _best_rows(canonical, PRIMARY_DATASET)
    amazon_best = _best_rows(canonical, SECONDARY_DATASET)
    low_data_best = _best_rows(tie_break, PRIMARY_DATASET) if not tie_break.empty else {"best_tree": None, "best_tabpfn": None}
    amazon_review_cap = read_env_int("RECPFN_AMAZON_MAX_REVIEWS")
    amazon_meta_cap = read_env_int("RECPFN_AMAZON_MAX_META")
    provisional = amazon_review_cap is not None or amazon_meta_cap is not None

    canonical_k = _single_run_value(canonical, "k", CANONICAL_K)
    canonical_train_cap = _single_run_value(canonical, "max_train_queries", CANONICAL_MAX_TRAIN_QUERIES)
    canonical_test_cap = _single_run_value(canonical, "max_test_queries", CANONICAL_MAX_TEST_QUERIES)
    canonical_tabpfn_version = _single_run_value(canonical, "tabpfn_version", CANONICAL_TABPFN_VERSION)

    lines = [
        "# Phase 1 Decision Memo",
        "",
        f"Outcome: **{decision['outcome']}**",
        "",
    ]
    if snapshot is not None:
        lines.extend(
            [
                "## Snapshot Status",
                "",
                f"- Completed units: `{snapshot['completed_units']}/{snapshot['expected_units']}`",
                f"- Missing units: `{snapshot['missing_units_count']}`",
            ]
        )
        if snapshot["missing_units"]:
            lines.append("- Missing labels:")
            lines.extend([f"  - `{label}`" for label in snapshot["missing_units"][:8]])
            if snapshot["missing_units_count"] > 8:
                lines.append(f"  - ... and `{snapshot['missing_units_count'] - 8}` more")
        lines.append("")
    lines.extend(
        [
        "## Canonical Contract",
        "",
        f"- TabPFN version: `{canonical_tabpfn_version}`",
        f"- Candidate size: `K={canonical_k}`",
        f"- Canonical query caps: `train={canonical_train_cap}`, `test={canonical_test_cap}`",
        "- CPU large-dataset override: `TABPFN_ALLOW_CPU_LARGE_DATASET=1` during the canonical sweep",
        f"- Datasets: `{PRIMARY_DATASET}`, `{SECONDARY_DATASET}`",
        f"- Splits: `{', '.join(CANONICAL_SPLITS)}`",
        f"- Protocols: `{', '.join(CANONICAL_PROTOCOLS)}`",
        "",
        "## Decision Reasons",
        "",
    ])
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend(
        [
            "",
            "## Best Canonical Rows",
            "",
            f"- MovieLens best tree row: {_format_best_row(movielens_best.get('best_tree'))}",
            f"- MovieLens best TabPFN row: {_format_best_row(movielens_best.get('best_tabpfn'))}",
            f"- Amazon best tree row: {_format_best_row(amazon_best.get('best_tree'))}",
            f"- Amazon best TabPFN row: {_format_best_row(amazon_best.get('best_tabpfn'))}",
        ]
    )
    if not tie_break.empty:
        lines.extend(
            [
                "",
                "## Low-Data Ladder Highlights",
                "",
                f"- MovieLens low-data best tree row: {_format_best_row(low_data_best.get('best_tree'))}",
                f"- MovieLens low-data best TabPFN row: {_format_best_row(low_data_best.get('best_tabpfn'))}",
            ]
        )
        lines.extend(["", "## Low-Data Scorecard", ""])
        lines.extend(_format_low_data_scorecard(tie_break))
    lines.extend(
        [
            "",
            "## Pairwise TabPFN Checks",
            "",
        ]
    )
    pairwise_gains = decision.get("pairwise_gains", [])
    if pairwise_gains:
        for row in pairwise_gains:
            lines.append(
                "- "
                f"{row['split_type']} / {row['protocol']} / {row['phase']}: "
                f"NDCG gain {row['ndcg_gain']:.4f}, Recall gain {row['recall_gain']:.4f}"
            )
    else:
        lines.append("- No matched pointwise/pairwise TabPFN comparison rows were available.")

    lines.extend(["", "## Protocol Distinctness", ""])
    for note in overlap_notes:
        label = "collapsed" if note["collapsed"] else "distinct"
        lines.append(
            "- "
            f"{note['dataset']} / {note['split_type']}: {label}, "
            f"identical_rate={note['identical_rate']:.3f}, mean_jaccard={note['mean_jaccard']:.3f}"
        )

    lines.extend(["", "## Caveats", ""])
    if provisional:
        lines.append(
            "- Amazon used a capped local slice, so this decision is provisional rather than fully final."
        )
    else:
        lines.append("- Amazon used the uncapped local ingest path.")
    if decision["amazon_saturated"]:
        lines.append("- Amazon was too easy under the current protocol and was treated as secondary evidence only.")
    lines.append("- Existing exploratory result folders were excluded from this memo and from the merged benchmark table.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_next_steps_plan(
    decision: dict[str, object],
    path: str | Path,
    snapshot: dict[str, object] | None = None,
    low_data_ran: bool = False,
) -> Path:
    """Write the recommended next steps that follow from the current decision."""

    path = Path(path)
    ensure_dir(path.parent)
    outcome = decision["outcome"]
    lines = [
        "# Phase 1 Next Steps",
        "",
        f"Current outcome: **{outcome}**",
        "",
    ]
    if snapshot is not None:
        lines.extend(
            [
                f"Snapshot coverage: `{snapshot['completed_units']}/{snapshot['expected_units']}` canonical units completed.",
                "",
            ]
        )

    if outcome == "drill further" and low_data_ran:
        lines.extend(
            [
                "## Recommended Plan",
                "",
                "1. Freeze Phase 1 and treat pointwise TabPFN as the primary positive story.",
                "2. Keep pairwise TabPFN as an ablation unless a later focused experiment changes the picture.",
                "3. Start MVP 3 around small-data and item-cold pointwise reranking, not a general pairwise method claim.",
                "4. Turn the current evidence into one tight writeup, one benchmark table, and one concise figure set.",
            ]
        )
    elif outcome == "drill further":
        lines.extend(
            [
                "## Recommended Plan",
                "",
                "1. Freeze the current canonical evidence set and use it as the interim baseline.",
                "2. Run the focused MovieLens low-data ladder at `10%`, `20%`, `50%`, and `100%` train scale.",
                "3. Compare `xgboost`, `catboost`, and `tabpfn` in both pointwise and pairwise modes on `warm` and `item_cold`.",
                "4. Reframe MVP 3 around pointwise TabPFN for small-data and cold-start reranking if the low-data signal holds.",
                "5. Keep pairwise TabPFN as an ablation unless it produces a clear win in the low-data ladder.",
            ]
        )
    elif outcome == "run one tie-break sweep":
        lines.extend(
            [
                "## Recommended Plan",
                "",
                "1. Run the MovieLens-only low-data tie-break sweep on `global_popularity`.",
                "2. Use `warm` and `item_cold` splits with train fractions `10%`, `20%`, `50%`, and `100%`.",
                "3. Limit the model matrix to `xgboost`, `catboost`, and `tabpfn` in pointwise and pairwise modes.",
                "4. Promote the project only if TabPFN is clearly competitive or wins in the low-data regime.",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommended Plan",
                "",
                "1. Stop deeper method work and keep the repo focused on the benchmark artifact.",
                "2. Clean up the benchmark narrative and plot set for a small OSS and blog-level release.",
                "3. Do not expand pairwise or method work unless new evidence appears from a separate dataset.",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_existing_phase1_results(run_output_dir: str | Path) -> pd.DataFrame:
    """Load existing per-unit results.csv files and annotate them for summary-only workflows."""

    run_root = Path(run_output_dir)
    paths = sorted(run_root.glob("**/results.csv"))
    if not paths:
        raise FileNotFoundError(f"No results.csv files found under {run_root}.")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        fallback_phase = "tie_break" if "tie_break" in path.parts else "canonical"
        phase = None
        if "phase" not in frame.columns:
            frame["phase"] = fallback_phase
            phase = fallback_phase
        else:
            frame["phase"] = frame["phase"].fillna(fallback_phase)
            non_null_phase = frame["phase"].dropna()
            phase = str(non_null_phase.iloc[0]) if not non_null_phase.empty else fallback_phase
        if "train_fraction" not in frame.columns:
            frame["train_fraction"] = pd.NA
        if phase == "tie_break":
            frame["train_fraction"] = frame["train_fraction"].fillna(frame["max_train_queries"].map(_infer_train_fraction))
        frames.append(frame)

    return _concat_results(frames)


def snapshot_status(results: pd.DataFrame) -> dict[str, object]:
    """Report how many canonical units have completed and which are still missing."""

    canonical = results.copy()
    if "phase" in canonical.columns:
        canonical = canonical[canonical["phase"] == "canonical"].copy()
    observed = {
        _unit_label(row["dataset"], row["split_type"], row["protocol"], row["mode"], row["model"])
        for _, row in canonical.iterrows()
    }
    expected = _expected_canonical_labels()
    missing = sorted(expected - observed)
    return {
        "completed_units": len(observed),
        "expected_units": len(expected),
        "missing_units_count": len(missing),
        "missing_units": missing,
    }


def _query_item_sets(frame: pd.DataFrame) -> dict[str, set[object]]:
    return {
        str(query_id): set(group["item_id"].tolist())
        for query_id, group in frame.groupby("query_id")
    }


def _tabpfn_pairwise_gains(results: pd.DataFrame, dataset_name: str) -> list[dict[str, object]]:
    subset = results[(results["dataset"] == dataset_name) & (results["model"] == "tabpfn")]
    if subset.empty:
        return []
    merge_cols = ["dataset", "split_type", "protocol", "phase"]
    if "train_fraction" in subset.columns:
        merge_cols.append("train_fraction")
    pointwise = subset[subset["mode"] == "pointwise"][merge_cols + ["ndcg@10", "recall@10"]]
    pairwise = subset[subset["mode"] == "pairwise"][merge_cols + ["ndcg@10", "recall@10"]]
    if pointwise.empty or pairwise.empty:
        return []
    merged = pointwise.merge(pairwise, on=merge_cols, suffixes=("_pointwise", "_pairwise"))
    rows = []
    for _, merged_row in merged.iterrows():
        rows.append(
            {
                "dataset": merged_row["dataset"],
                "split_type": merged_row["split_type"],
                "protocol": merged_row["protocol"],
                "phase": merged_row["phase"],
                "train_fraction": merged_row.get("train_fraction"),
                "ndcg_gain": merged_row["ndcg@10_pairwise"] - merged_row["ndcg@10_pointwise"],
                "recall_gain": merged_row["recall@10_pairwise"] - merged_row["recall@10_pointwise"],
            }
        )
    return rows


def _tabpfn_competitive_rows(results: pd.DataFrame) -> list[dict[str, object]]:
    phase_series = results["phase"] if "phase" in results.columns else pd.Series(index=results.index, dtype=object)
    subset = results[
        (results["dataset"] == PRIMARY_DATASET)
        & (
            (results["split_type"] == "item_cold")
            | (phase_series == "tie_break")
        )
    ]
    if subset.empty:
        return []

    compare_cols = ["split_type", "protocol", "phase"]
    if "train_fraction" in subset.columns:
        compare_cols.append("train_fraction")
    rows = []
    for keys, group in subset.groupby(compare_cols, dropna=False):
        group = group[group["status"] == "ok"]
        tree_rows = group[group["model"].isin(TREE_MODELS)]
        tabpfn_rows = group[group["model"] == "tabpfn"]
        if tree_rows.empty or tabpfn_rows.empty:
            continue
        best_tree = tree_rows.sort_values("ndcg@10", ascending=False).iloc[0]
        best_tabpfn = tabpfn_rows.sort_values("ndcg@10", ascending=False).iloc[0]
        if best_tabpfn["ndcg@10"] >= (best_tree["ndcg@10"] * 0.99):
            record = {
                "split_type": best_tabpfn["split_type"],
                "protocol": best_tabpfn["protocol"],
                "phase": best_tabpfn.get("phase", "canonical"),
                "train_fraction": best_tabpfn.get("train_fraction"),
                "tabpfn_mode": best_tabpfn["mode"],
                "tabpfn_ndcg": float(best_tabpfn["ndcg@10"]),
                "best_tree_model": best_tree["model"],
                "best_tree_mode": best_tree["mode"],
                "best_tree_ndcg": float(best_tree["ndcg@10"]),
            }
            rows.append(record)
    return rows


def _movie_lens_is_close(results: pd.DataFrame) -> bool:
    phase_series = results["phase"] if "phase" in results.columns else pd.Series(index=results.index, dtype=object)
    subset = results[(results["dataset"] == PRIMARY_DATASET) & (phase_series == "canonical")]
    if subset.empty:
        return False

    for (split_type, protocol), group in subset.groupby(["split_type", "protocol"]):
        tree_rows = group[group["model"].isin(TREE_MODELS)]
        tabpfn_rows = group[group["model"] == "tabpfn"]
        if tree_rows.empty or tabpfn_rows.empty:
            continue
        best_tree = tree_rows["ndcg@10"].max()
        best_tabpfn = tabpfn_rows["ndcg@10"].max()
        if abs(best_tree - best_tabpfn) <= 0.03:
            return True
    return False


def _amazon_is_saturated(results: pd.DataFrame) -> bool:
    phase_series = results["phase"] if "phase" in results.columns else pd.Series(index=results.index, dtype=object)
    subset = results[
        (results["dataset"] == SECONDARY_DATASET)
        & (phase_series == "canonical")
        & (results["model"].isin(LEARNED_MODELS))
    ]
    if subset.empty:
        return False
    return float((subset["ndcg@10"] >= 0.98).mean()) >= 0.8


def _tabpfn_runtime_penalty(results: pd.DataFrame) -> float | None:
    phase_series = results["phase"] if "phase" in results.columns else pd.Series(index=results.index, dtype=object)
    subset = results[(results["status"] == "ok") & (phase_series == "canonical")]
    compare_cols = ["dataset", "split_type", "protocol"]
    ratios = []
    for _, group in subset.groupby(compare_cols):
        tabpfn_rows = group[group["model"] == "tabpfn"]
        tree_rows = group[group["model"].isin(TREE_MODELS)]
        if tabpfn_rows.empty or tree_rows.empty:
            continue
        best_tree_runtime = tree_rows["runtime_seconds"].min()
        best_tabpfn_runtime = tabpfn_rows["runtime_seconds"].min()
        if best_tree_runtime > 0:
            ratios.append(best_tabpfn_runtime / best_tree_runtime)
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def _best_rows(results: pd.DataFrame, dataset_name: str) -> dict[str, pd.Series | None]:
    dataset_rows = results[(results["dataset"] == dataset_name) & (results["status"] == "ok")]
    tree_rows = dataset_rows[dataset_rows["model"].isin(TREE_MODELS)]
    tabpfn_rows = dataset_rows[dataset_rows["model"] == "tabpfn"]
    return {
        "best_tree": tree_rows.sort_values("ndcg@10", ascending=False).iloc[0] if not tree_rows.empty else None,
        "best_tabpfn": tabpfn_rows.sort_values("ndcg@10", ascending=False).iloc[0] if not tabpfn_rows.empty else None,
    }


def _format_best_row(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    train_fraction = row.get("train_fraction")
    train_fraction_text = f", train_fraction={train_fraction:.1f}" if pd.notna(train_fraction) else ""
    return (
        f"{row['mode']} {row['model']} on {row['split_type']} / {row['protocol']}"
        f"{train_fraction_text} with NDCG@10={row['ndcg@10']:.4f}, "
        f"Recall@10={row['recall@10']:.4f}, runtime={row['runtime_seconds']:.2f}s"
    )


def _single_run_value(frame: pd.DataFrame, column: str, default: object) -> object:
    if column not in frame.columns:
        return default
    values = frame[column].dropna().drop_duplicates().tolist()
    if not values:
        return default
    return values[0]


def _format_low_data_scorecard(tie_break: pd.DataFrame) -> list[str]:
    rows = []
    subset = tie_break[
        (tie_break["dataset"] == PRIMARY_DATASET)
        & (tie_break["protocol"] == "global_popularity")
        & (tie_break["status"] == "ok")
    ].copy()
    if subset.empty:
        return ["- No low-data rows were available."]

    for split_type in CANONICAL_SPLITS:
        split_rows = subset[subset["split_type"] == split_type]
        if split_rows.empty:
            continue
        for train_fraction in TIE_BREAK_FRACTIONS:
            group = split_rows[split_rows["train_fraction"] == train_fraction]
            if group.empty:
                continue
            pointwise = group[(group["model"] == "tabpfn") & (group["mode"] == "pointwise")]
            pairwise = group[(group["model"] == "tabpfn") & (group["mode"] == "pairwise")]
            tree_rows = group[group["model"].isin(TREE_MODELS)]
            if pointwise.empty or pairwise.empty or tree_rows.empty:
                continue
            pointwise_row = pointwise.iloc[0]
            pairwise_row = pairwise.iloc[0]
            tree_row = tree_rows.sort_values(["ndcg@10", "recall@10", "mrr"], ascending=False).iloc[0]
            best_row = group.sort_values(["ndcg@10", "recall@10", "mrr"], ascending=False).iloc[0]
            rows.append(
                "- "
                f"{split_type} / {int(train_fraction * 100)}%: "
                f"best={best_row['model']} {best_row['mode']} ({best_row['ndcg@10']:.4f}), "
                f"TabPFN pointwise={pointwise_row['ndcg@10']:.4f}, "
                f"TabPFN pairwise={pairwise_row['ndcg@10']:.4f}, "
                f"best tree={tree_row['model']} {tree_row['mode']} ({tree_row['ndcg@10']:.4f})"
            )
    return rows or ["- No low-data rows were available."]


def _unit_label(dataset: str, split_type: str, protocol: str, mode: str, model: str) -> str:
    return f"{dataset} {split_type} {protocol} {mode} {model}"


def _expected_canonical_labels() -> set[str]:
    labels = set()
    for dataset_name in CANONICAL_DATASETS:
        for split_type in CANONICAL_SPLITS:
            for protocol in CANONICAL_PROTOCOLS:
                for model in CANONICAL_POINTWISE_MODELS:
                    labels.add(_unit_label(dataset_name, split_type, protocol, "pointwise", model))
                for model in CANONICAL_PAIRWISE_MODELS:
                    labels.add(_unit_label(dataset_name, split_type, protocol, "pairwise", model))
    return labels


def _infer_train_fraction(max_train_queries: object) -> float | pd.NA:
    if pd.isna(max_train_queries):
        return pd.NA
    try:
        value = int(max_train_queries)
    except (TypeError, ValueError):
        return pd.NA
    for fraction in TIE_BREAK_FRACTIONS:
        if max(1, math.ceil(CANONICAL_MAX_TRAIN_QUERIES * fraction)) == value:
            return fraction
    return pd.NA


def _canonical_unit_count() -> int:
    return len(CANONICAL_DATASETS) * len(CANONICAL_SPLITS) * (
        len(CANONICAL_PROTOCOLS) * len(CANONICAL_POINTWISE_MODELS)
        + len(CANONICAL_PROTOCOLS) * len(CANONICAL_PAIRWISE_MODELS)
    )


def _tie_break_unit_count() -> int:
    return len(CANONICAL_SPLITS) * len(TIE_BREAK_FRACTIONS) * (3 + 3)


if __name__ == "__main__":
    main()
