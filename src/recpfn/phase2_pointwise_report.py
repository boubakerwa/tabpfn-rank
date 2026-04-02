"""Reporting, uncertainty estimation, and decision memo for Phase 2 pointwise validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recpfn.data.schemas import QUERY_ID_COL
from recpfn.eval.metrics import evaluate_rankings_by_query
from recpfn.eval.reports import save_benchmark_table, save_summary_csv
from recpfn.phase2_pointwise_shared import (
    BOOTSTRAP_SLICES,
    BOOTSTRAP_TRAIN_FRACTIONS,
    CANONICAL_TABPFN_VERSION,
    K_SENSITIVITY_VALUES,
    KEY_SLICE_DEFINITIONS,
    PRIMARY_DATASET,
    PRIMARY_K,
    PRIMARY_PROTOCOLS,
    PRIMARY_SPLITS,
    PRIMARY_TRAIN_FRACTIONS,
    SECONDARY_DATASET,
    TABPFN_MODELS,
    TREE_MODELS,
    infer_phase2_group,
)
from recpfn.types import Phase2PointwiseArtifacts
from recpfn.utils import ensure_dir, stable_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize and score existing Phase 2 raw runs.")
    parser.add_argument("--run-output-dir", default="paper/results_phase2_pointwise_runs")
    parser.add_argument("--output-dir", default="paper/phase2_pointwise")
    parser.add_argument("--plots-output-dir", default="paper/figures/phase2_pointwise")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_summary, artifacts, decision = run_phase2_pointwise_report(
        run_output_dir=args.run_output_dir,
        output_dir=args.output_dir,
        plots_output_dir=args.plots_output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(raw_summary.to_string(index=False))
    print(f"\nNative adapter outcome: {decision['native_adapter_outcome']}")
    if artifacts.decision_memo_path:
        print(f"Decision memo: {artifacts.decision_memo_path}")
    if artifacts.benchmark_table_path:
        print(f"Benchmark table: {artifacts.benchmark_table_path}")


def run_phase2_pointwise_report(
    run_output_dir: str | Path = "paper/results_phase2_pointwise_runs",
    output_dir: str | Path = "paper/phase2_pointwise",
    plots_output_dir: str | Path = "paper/figures/phase2_pointwise",
    bootstrap_replicates: int = 2000,
) -> tuple[pd.DataFrame, Phase2PointwiseArtifacts, dict[str, object]]:
    """Build all Phase 2 summary artifacts from existing raw runs."""

    raw_root = Path(run_output_dir)
    summary_root = ensure_dir(output_dir)
    plots_root = ensure_dir(plots_output_dir)
    artifacts = Phase2PointwiseArtifacts(output_dir=summary_root, run_output_dir=raw_root, plots_output_dir=plots_root)

    raw_summary = load_phase2_raw_results(raw_root)
    artifacts.raw_summary_path = save_summary_csv(raw_summary, summary_root / "raw_summary.csv")

    aggregated = aggregate_phase2_results(raw_summary)
    artifacts.aggregated_results_path = save_summary_csv(aggregated, summary_root / "aggregated_results.csv")

    per_query = load_phase2_per_query_metrics(raw_root)
    bootstrap = compute_bootstrap_delta_summary(aggregated, per_query, replicates=bootstrap_replicates)
    artifacts.bootstrap_delta_summary_path = save_summary_csv(bootstrap, summary_root / "bootstrap_delta_summary.csv")

    k_sensitivity = aggregated[aggregated["phase2_group"] == "k_sensitivity"].copy()
    artifacts.k_sensitivity_results_path = save_summary_csv(k_sensitivity, summary_root / "k_sensitivity_results.csv")

    amazon_sanity = aggregated[aggregated["phase2_group"] == "amazon_sanity"].copy()
    artifacts.amazon_sanity_results_path = save_summary_csv(amazon_sanity, summary_root / "amazon_sanity_results.csv")

    feature_group_ablation = aggregated[aggregated["phase2_group"] == "feature_group_ablation"].copy()
    artifacts.feature_group_ablation_path = save_summary_csv(
        feature_group_ablation,
        summary_root / "feature_group_ablation.csv",
    )

    decision = evaluate_phase2_outcome(aggregated, bootstrap, k_sensitivity, amazon_sanity)
    artifacts.benchmark_table_path = save_benchmark_table(
        aggregated,
        summary_root / "benchmark.md",
        columns=[
            "phase2_group",
            "dataset",
            "split_type",
            "protocol",
            "model",
            "feature_set",
            "k",
            "train_fraction",
            "seed_count",
            "ndcg@10_mean",
            "ndcg@10_std",
            "recall@10_mean",
            "mrr_mean",
            "hitrate@10_mean",
            "runtime_seconds_median",
        ],
        sort_by=[
            "phase2_group",
            "dataset",
            "split_type",
            "protocol",
            "feature_set",
            "k",
            "train_fraction",
            "model",
        ],
    )
    artifacts.decision_memo_path = write_phase2_decision_memo(
        aggregated,
        bootstrap,
        k_sensitivity,
        amazon_sanity,
        feature_group_ablation,
        decision,
        summary_root / "decision.md",
    )
    generate_phase2_plots(aggregated, bootstrap, plots_root)
    return raw_summary, artifacts, decision


def load_phase2_raw_results(raw_root: str | Path) -> pd.DataFrame:
    """Load all Phase 2 results.csv files and annotate them with group metadata."""

    raw_root = Path(raw_root)
    paths = sorted(raw_root.glob("**/results.csv"))
    if not paths:
        raise FileNotFoundError(f"No results.csv files found under {raw_root}.")

    frames = []
    for path in paths:
        phase2_group = infer_phase2_group(path, raw_root)
        if phase2_group is None:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["phase2_group"] = phase2_group
        if "feature_set" not in frame.columns:
            frame["feature_set"] = "full"
        else:
            frame["feature_set"] = frame["feature_set"].fillna("full")
        frame["source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_phase2_results(raw_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw result rows across seeds."""

    ok = raw_summary[raw_summary["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "phase2_group",
        "dataset",
        "split_type",
        "protocol",
        "mode",
        "model",
        "k",
        "max_train_queries",
        "max_test_queries",
        "train_fraction",
        "feature_set",
        "tabpfn_version",
    ]
    group_cols = [column for column in group_cols if column in ok.columns]

    aggregated = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            ndcg_10_mean=("ndcg@10", "mean"),
            ndcg_10_std=("ndcg@10", "std"),
            recall_10_mean=("recall@10", "mean"),
            recall_10_std=("recall@10", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
            hitrate_10_mean=("hitrate@10", "mean"),
            hitrate_10_std=("hitrate@10", "std"),
            runtime_seconds_median=("runtime_seconds", "median"),
            n_queries_median=("n_queries", "median"),
        )
        .reset_index()
    )
    return aggregated.rename(
        columns={
            "ndcg_10_mean": "ndcg@10_mean",
            "ndcg_10_std": "ndcg@10_std",
            "recall_10_mean": "recall@10_mean",
            "recall_10_std": "recall@10_std",
            "hitrate_10_mean": "hitrate@10_mean",
            "hitrate_10_std": "hitrate@10_std",
        }
    )


def load_phase2_per_query_metrics(raw_root: str | Path) -> pd.DataFrame:
    """Compute per-query metrics from saved prediction files."""

    raw_root = Path(raw_root)
    prediction_paths = sorted(raw_root.glob("**/*_predictions.csv"))
    frames = []
    for path in prediction_paths:
        phase2_group = infer_phase2_group(path, raw_root)
        if phase2_group is None:
            continue
        predictions = pd.read_csv(path)
        if predictions.empty:
            continue
        metrics = evaluate_rankings_by_query(predictions, metrics=("ndcg@10",))
        metadata = predictions.iloc[0].to_dict()
        for column in [
            "dataset",
            "split_type",
            "protocol",
            "model",
            "mode",
            "seed",
            "k",
            "train_fraction",
            "feature_set",
            "max_train_queries",
            "max_test_queries",
            "tabpfn_version",
        ]:
            metrics[column] = metadata.get(column)
        metrics["phase2_group"] = phase2_group
        metrics["prediction_path"] = str(path)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_bootstrap_delta_summary(
    aggregated: pd.DataFrame,
    per_query: pd.DataFrame,
    replicates: int = 2000,
) -> pd.DataFrame:
    """Compute paired query-level bootstrap deltas for the key Phase 2 comparisons."""

    rows = []
    key_primary = aggregated[
        (aggregated["phase2_group"] == "primary")
        & (aggregated["k"] == PRIMARY_K)
    ].copy()
    if key_primary.empty or per_query.empty:
        return pd.DataFrame()

    best_models = _best_models_by_slice_and_fraction(key_primary)
    for dataset_name, split_type, protocol in BOOTSTRAP_SLICES:
        for train_fraction in BOOTSTRAP_TRAIN_FRACTIONS:
            native_model = "tabpfn_native"
            ohe_model = "tabpfn"
            for seed in sorted(per_query["seed"].dropna().unique()):
                native_vs_ohe = _bootstrap_for_model_pair(
                    per_query=per_query,
                    dataset_name=dataset_name,
                    split_type=split_type,
                    protocol=protocol,
                    seed=int(seed),
                    train_fraction=float(train_fraction),
                    left_model=native_model,
                    right_model=ohe_model,
                    comparison="tabpfn_native_minus_tabpfn",
                    replicates=replicates,
                )
                if native_vs_ohe:
                    rows.append(native_vs_ohe)

                best_tab_model = best_models.get((dataset_name, split_type, protocol, float(train_fraction)), {}).get(
                    "best_tabpfn_model"
                )
                best_tree_model = best_models.get((dataset_name, split_type, protocol, float(train_fraction)), {}).get(
                    "best_tree_model"
                )
                if best_tab_model and best_tree_model:
                    best_tab_vs_tree = _bootstrap_for_model_pair(
                        per_query=per_query,
                        dataset_name=dataset_name,
                        split_type=split_type,
                        protocol=protocol,
                        seed=int(seed),
                        train_fraction=float(train_fraction),
                        left_model=best_tab_model,
                        right_model=best_tree_model,
                        comparison="best_tabpfn_minus_best_tree",
                        replicates=replicates,
                    )
                    if best_tab_vs_tree:
                        rows.append(best_tab_vs_tree)
    return pd.DataFrame(rows)


def _bootstrap_for_model_pair(
    per_query: pd.DataFrame,
    dataset_name: str,
    split_type: str,
    protocol: str,
    seed: int,
    train_fraction: float,
    left_model: str,
    right_model: str,
    comparison: str,
    replicates: int,
) -> dict[str, object] | None:
    left = per_query[
        (per_query["phase2_group"] == "primary")
        & (per_query["dataset"] == dataset_name)
        & (per_query["split_type"] == split_type)
        & (per_query["protocol"] == protocol)
        & (per_query["seed"] == seed)
        & (per_query["k"] == PRIMARY_K)
        & (per_query["train_fraction"] == train_fraction)
        & (per_query["model"] == left_model)
    ][[QUERY_ID_COL, "ndcg@10"]].rename(columns={"ndcg@10": "left"})
    right = per_query[
        (per_query["phase2_group"] == "primary")
        & (per_query["dataset"] == dataset_name)
        & (per_query["split_type"] == split_type)
        & (per_query["protocol"] == protocol)
        & (per_query["seed"] == seed)
        & (per_query["k"] == PRIMARY_K)
        & (per_query["train_fraction"] == train_fraction)
        & (per_query["model"] == right_model)
    ][[QUERY_ID_COL, "ndcg@10"]].rename(columns={"ndcg@10": "right"})
    merged = left.merge(right, on=QUERY_ID_COL, how="inner")
    if merged.empty:
        return None

    deltas = (merged["left"] - merged["right"]).to_numpy(dtype=float)
    rng = np.random.default_rng(stable_seed(dataset_name, split_type, protocol, seed, comparison, train_fraction))
    bootstrap_means = rng.choice(deltas, size=(replicates, len(deltas)), replace=True).mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "dataset": dataset_name,
        "split_type": split_type,
        "protocol": protocol,
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "comparison": comparison,
        "left_model": left_model,
        "right_model": right_model,
        "metric": "ndcg@10",
        "n_queries": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_excludes_zero_positive": bool(lower > 0.0),
    }


def evaluate_phase2_outcome(
    aggregated: pd.DataFrame,
    bootstrap: pd.DataFrame,
    k_sensitivity: pd.DataFrame,
    amazon_sanity: pd.DataFrame,
) -> dict[str, object]:
    """Apply the explicit Phase 2 decision rules."""

    key_primary = aggregated[
        (aggregated["phase2_group"] == "primary")
        & (aggregated["train_fraction"] == 1.0)
        & (aggregated["k"] == PRIMARY_K)
        & (aggregated["dataset"] == PRIMARY_DATASET)
    ].copy()
    key_models = _best_models_by_slice(key_primary)
    native_slice_rows = _native_vs_ohe_key_slice_rows(key_primary)
    positive_native_slices = sum(1 for row in native_slice_rows if row["delta_ndcg"] > 0)

    item_cold_bootstrap = bootstrap[
        (bootstrap["comparison"] == "tabpfn_native_minus_tabpfn")
        & (bootstrap["split_type"] == "item_cold")
        & (bootstrap["train_fraction"] == 1.0)
    ].copy()
    positive_bootstrap_support = False
    bootstrap_support_rows = []
    for protocol in PRIMARY_PROTOCOLS:
        subset = item_cold_bootstrap[item_cold_bootstrap["protocol"] == protocol]
        positive_seed_count = int(subset["ci_excludes_zero_positive"].sum()) if not subset.empty else 0
        bootstrap_support_rows.append({"protocol": protocol, "positive_seed_count": positive_seed_count})
        if positive_seed_count >= 2:
            positive_bootstrap_support = True

    runtime_ratios = [row["runtime_ratio"] for row in native_slice_rows if pd.notna(row["runtime_ratio"])]
    median_runtime_ratio = float(np.median(runtime_ratios)) if runtime_ratios else float("nan")
    runtime_ok = bool(runtime_ratios) and median_runtime_ratio <= 1.1

    if positive_native_slices >= 3 and positive_bootstrap_support and runtime_ok:
        native_outcome = "promote primary adapter"
    elif positive_native_slices >= 2 or positive_bootstrap_support:
        native_outcome = "keep as targeted cold-start variant"
    else:
        native_outcome = "do not promote"

    tabpfn_vs_tree_rows = _best_tabpfn_vs_tree_rows(key_primary, key_models)
    competitive_slices = sum(1 for row in tabpfn_vs_tree_rows if row["tabpfn_within_one_percent"])
    pointwise_story_holds = competitive_slices >= 2

    k50_rows = _evaluate_k50_retention(k_sensitivity, key_models)
    amazon_notes = _summarize_amazon(amazon_sanity)

    reasons = [
        f"tabpfn_native had positive mean NDCG@10 deltas on {positive_native_slices}/4 key MovieLens slices.",
        f"Item-cold bootstrap support was positive on at least one slice for {'yes' if positive_bootstrap_support else 'no'} >=2/3 seeds.",
        f"Median native-vs-ohe runtime ratio across key slices was {median_runtime_ratio:.3f}."
        if runtime_ratios
        else "No runtime ratio could be computed for the native-vs-ohe comparison.",
        f"Best TabPFN variant was within 1% of the best tree on {competitive_slices}/4 key slices.",
    ]
    for row in k50_rows:
        reasons.append(
            f"K=50 retention on {row['split_type']} / {row['protocol']} with {row['model']} was "
            f"{row['retention_ratio']:.4f}."
        )

    return {
        "native_adapter_outcome": native_outcome,
        "pointwise_story_holds": pointwise_story_holds,
        "median_runtime_ratio_native_vs_ohe": median_runtime_ratio,
        "positive_native_slice_count": positive_native_slices,
        "native_slice_rows": native_slice_rows,
        "bootstrap_support_rows": bootstrap_support_rows,
        "tabpfn_vs_tree_rows": tabpfn_vs_tree_rows,
        "k50_rows": k50_rows,
        "amazon_notes": amazon_notes,
        "reasons": reasons,
    }


def write_phase2_decision_memo(
    aggregated: pd.DataFrame,
    bootstrap: pd.DataFrame,
    k_sensitivity: pd.DataFrame,
    amazon_sanity: pd.DataFrame,
    feature_group_ablation: pd.DataFrame,
    decision: dict[str, object],
    path: str | Path,
) -> Path:
    """Write the Phase 2 decision memo."""

    path = Path(path)
    ensure_dir(path.parent)
    lines = [
        "# Phase 2 Pointwise Decision Memo",
        "",
        f"TabPFN version: `{CANONICAL_TABPFN_VERSION}`",
        f"Native adapter outcome: **{decision['native_adapter_outcome']}**",
        f"Pointwise TabPFN story holds: **{'yes' if decision['pointwise_story_holds'] else 'no'}**",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in decision["reasons"]])

    lines.extend(["", "## Native vs One-Hot Key Slices", ""])
    for row in decision["native_slice_rows"]:
        lines.append(
            "- "
            f"{row['split_type']} / {row['protocol']}: "
            f"delta={row['delta_ndcg']:.4f}, runtime_ratio={row['runtime_ratio']:.4f}"
        )

    lines.extend(["", "## Bootstrap Support", ""])
    if bootstrap.empty:
        lines.append("- No bootstrap rows were available.")
    else:
        for _, row in bootstrap.iterrows():
            lines.append(
                "- "
                f"{row['split_type']} / {row['protocol']} / {int(float(row['train_fraction']) * 100)}% / "
                f"{row['comparison']} / seed {int(row['seed'])}: "
                f"mean_delta={row['mean_delta']:.4f}, CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            )

    lines.extend(["", "## K Sensitivity", ""])
    lines.append("- Tree comparisons in this section use the tree model selected from the primary 100% train sweep for each split/protocol slice.")
    if k_sensitivity.empty:
        lines.append("- No K sensitivity rows were available.")
    else:
        for row in decision["k50_rows"]:
            lines.append(
                "- "
                f"{row['split_type']} / {row['protocol']} / {row['model']}: "
                f"K50 retention={row['retention_ratio']:.4f}"
            )

    lines.extend(["", "## Amazon Sanity", ""])
    lines.append("- Amazon is secondary evidence here and was run with capped query counts for directional sanity checking.")
    for note in decision["amazon_notes"]:
        lines.append(f"- {note}")

    lines.extend(["", "## Feature Group Ablation", ""])
    if feature_group_ablation.empty:
        lines.append("- No feature-group ablation rows were available.")
    else:
        ablation_subset = feature_group_ablation[
            (feature_group_ablation["dataset"] == PRIMARY_DATASET)
            & (feature_group_ablation["split_type"] == "item_cold")
        ].sort_values(["protocol", "model", "feature_set"])
        for _, row in ablation_subset.iterrows():
            lines.append(
                "- "
                f"{row['protocol']} / {row['model']} / {row['feature_set']}: "
                f"NDCG@10={row['ndcg@10_mean']:.4f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_phase2_plots(aggregated: pd.DataFrame, bootstrap: pd.DataFrame, output_dir: Path) -> None:
    """Generate the required Phase 2 comparison plots."""

    ensure_dir(output_dir)
    plot_adapter_delta_by_train_fraction(aggregated, output_dir / "adapter_delta_by_train_fraction.png")
    plot_runtime_by_train_fraction(aggregated, output_dir / "runtime_by_train_fraction.png")
    plot_metric_by_k(aggregated, output_dir / "metric_by_k.png")
    plot_native_minus_ohe_by_slice(aggregated, output_dir / "native_minus_one_hot_by_slice.png")
    plot_best_tabpfn_vs_tree(aggregated, output_dir / "best_tabpfn_vs_best_tree.png")


def plot_adapter_delta_by_train_fraction(aggregated: pd.DataFrame, path: Path) -> None:
    subset = _primary_subset(aggregated)
    delta_rows = []
    for (split_type, protocol, train_fraction), group in subset.groupby(["split_type", "protocol", "train_fraction"], dropna=False):
        native = group[group["model"] == "tabpfn_native"]
        ohe = group[group["model"] == "tabpfn"]
        if native.empty or ohe.empty:
            continue
        delta_rows.append(
            {
                "split_type": split_type,
                "protocol": protocol,
                "train_fraction": train_fraction,
                "delta": float(native.iloc[0]["ndcg@10_mean"] - ohe.iloc[0]["ndcg@10_mean"]),
            }
        )
    delta = pd.DataFrame(delta_rows)
    if delta.empty:
        _save_empty_plot(path, "No primary adapter delta data available.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split_type in zip(axes, PRIMARY_SPLITS):
        split_rows = delta[delta["split_type"] == split_type]
        for protocol, group in split_rows.groupby("protocol"):
            ordered = group.sort_values("train_fraction")
            ax.plot(ordered["train_fraction"], ordered["delta"], marker="o", linewidth=2, label=protocol)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(split_type.replace("_", " ").title())
        ax.set_xlabel("Train fraction")
        ax.set_xticks(PRIMARY_TRAIN_FRACTIONS)
        ax.set_xticklabels([f"{int(value * 100)}%" for value in PRIMARY_TRAIN_FRACTIONS])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("tabpfn_native - tabpfn (NDCG@10)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_by_train_fraction(aggregated: pd.DataFrame, path: Path) -> None:
    subset = _primary_subset(aggregated)
    tab_subset = subset[subset["model"].isin(TABPFN_MODELS)]
    if tab_subset.empty:
        _save_empty_plot(path, "No TabPFN runtime data available.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split_type in zip(axes, PRIMARY_SPLITS):
        split_rows = tab_subset[tab_subset["split_type"] == split_type]
        for (protocol, model), group in split_rows.groupby(["protocol", "model"]):
            ordered = group.sort_values("train_fraction")
            ax.plot(
                ordered["train_fraction"],
                ordered["runtime_seconds_median"],
                marker="o",
                linewidth=2,
                label=f"{protocol} / {model}",
            )
        ax.set_title(split_type.replace("_", " ").title())
        ax.set_xlabel("Train fraction")
        ax.set_xticks(PRIMARY_TRAIN_FRACTIONS)
        ax.set_xticklabels([f"{int(value * 100)}%" for value in PRIMARY_TRAIN_FRACTIONS])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Median runtime (seconds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_k(aggregated: pd.DataFrame, path: Path) -> None:
    subset = aggregated[aggregated["phase2_group"] == "k_sensitivity"].copy()
    if subset.empty:
        _save_empty_plot(path, "No K sensitivity data available.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, (split_type, protocol) in zip(axes.ravel(), [(s, p) for s in PRIMARY_SPLITS for p in PRIMARY_PROTOCOLS]):
        group = subset[(subset["split_type"] == split_type) & (subset["protocol"] == protocol)]
        for model, model_rows in group.groupby("model"):
            ordered = model_rows.sort_values("k")
            ax.plot(ordered["k"], ordered["ndcg@10_mean"], marker="o", linewidth=2, label=model)
        ax.set_title(f"{split_type} / {protocol}")
        ax.set_xlabel("Candidate size K")
        ax.set_xticks(K_SENSITIVITY_VALUES)
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel("Mean NDCG@10")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_native_minus_ohe_by_slice(aggregated: pd.DataFrame, path: Path) -> None:
    subset = _primary_subset(aggregated)
    rows = []
    for split_type, protocol in [(s, p) for s in PRIMARY_SPLITS for p in PRIMARY_PROTOCOLS]:
        group = subset[(subset["split_type"] == split_type) & (subset["protocol"] == protocol) & (subset["train_fraction"] == 1.0)]
        native = group[group["model"] == "tabpfn_native"]
        ohe = group[group["model"] == "tabpfn"]
        if native.empty or ohe.empty:
            continue
        rows.append(
            {
                "setting": f"{split_type}\n{protocol}",
                "delta": float(native.iloc[0]["ndcg@10_mean"] - ohe.iloc[0]["ndcg@10_mean"]),
            }
        )
    delta = pd.DataFrame(rows)
    if delta.empty:
        _save_empty_plot(path, "No 100% train adapter comparison data available.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#059669" if value >= 0 else "#dc2626" for value in delta["delta"]]
    ax.bar(delta["setting"], delta["delta"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("tabpfn_native - tabpfn (NDCG@10)")
    ax.set_title("Native vs One-Hot on 100% Train Slices")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_best_tabpfn_vs_tree(aggregated: pd.DataFrame, path: Path) -> None:
    subset = _primary_subset(aggregated)
    best_models = _best_models_by_slice(subset[subset["train_fraction"] == 1.0])
    rows = []
    for dataset_name, split_type, protocol in KEY_SLICE_DEFINITIONS:
        group = subset[
            (subset["dataset"] == dataset_name)
            & (subset["split_type"] == split_type)
            & (subset["protocol"] == protocol)
            & (subset["train_fraction"] == 1.0)
        ]
        if group.empty:
            continue
        best_tree_model = best_models.get((dataset_name, split_type, protocol), {}).get("best_tree_model")
        best_tab_model = best_models.get((dataset_name, split_type, protocol), {}).get("best_tabpfn_model")
        if not best_tree_model or not best_tab_model:
            continue
        tree_score = float(group[group["model"] == best_tree_model].iloc[0]["ndcg@10_mean"])
        tab_score = float(group[group["model"] == best_tab_model].iloc[0]["ndcg@10_mean"])
        rows.append(
            {
                "setting": f"{split_type}\n{protocol}",
                "best_tree": tree_score,
                "best_tabpfn": tab_score,
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        _save_empty_plot(path, "No best-model comparison data available.")
        return
    positions = np.arange(len(comparison))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions - width / 2, comparison["best_tree"], width=width, label="best tree", color="#2563eb")
    ax.bar(positions + width / 2, comparison["best_tabpfn"], width=width, label="best TabPFN", color="#7c3aed")
    ax.set_xticks(positions)
    ax.set_xticklabels(comparison["setting"].tolist())
    ax.set_ylabel("Mean NDCG@10")
    ax.set_title("Best TabPFN Variant vs Best Tree on Key Slices")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _primary_subset(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated[
        (aggregated["phase2_group"] == "primary")
        & (aggregated["dataset"] == PRIMARY_DATASET)
        & (aggregated["feature_set"] == "full")
        & (aggregated["k"] == PRIMARY_K)
        & (aggregated["mode"] == "pointwise")
    ].copy()


def _best_models_by_slice(aggregated: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, str]]:
    mapping: dict[tuple[str, str, str], dict[str, str]] = {}
    for (dataset_name, split_type, protocol), group in aggregated.groupby(["dataset", "split_type", "protocol"], dropna=False):
        tree_rows = group[group["model"].isin(TREE_MODELS)].sort_values(
            ["ndcg@10_mean", "runtime_seconds_median", "model"],
            ascending=[False, True, True],
        )
        tab_rows = group[group["model"].isin(TABPFN_MODELS)].sort_values(
            ["ndcg@10_mean", "runtime_seconds_median", "model"],
            ascending=[False, True, True],
        )
        mapping[(dataset_name, split_type, protocol)] = {
            "best_tree_model": str(tree_rows.iloc[0]["model"]) if not tree_rows.empty else "",
            "best_tabpfn_model": str(tab_rows.iloc[0]["model"]) if not tab_rows.empty else "",
        }
    return mapping


def _best_models_by_slice_and_fraction(aggregated: pd.DataFrame) -> dict[tuple[str, str, str, float], dict[str, str]]:
    mapping: dict[tuple[str, str, str, float], dict[str, str]] = {}
    for (dataset_name, split_type, protocol, train_fraction), group in aggregated.groupby(
        ["dataset", "split_type", "protocol", "train_fraction"],
        dropna=False,
    ):
        tree_rows = group[group["model"].isin(TREE_MODELS)].sort_values(
            ["ndcg@10_mean", "runtime_seconds_median", "model"],
            ascending=[False, True, True],
        )
        tab_rows = group[group["model"].isin(TABPFN_MODELS)].sort_values(
            ["ndcg@10_mean", "runtime_seconds_median", "model"],
            ascending=[False, True, True],
        )
        mapping[(dataset_name, split_type, protocol, float(train_fraction))] = {
            "best_tree_model": str(tree_rows.iloc[0]["model"]) if not tree_rows.empty else "",
            "best_tabpfn_model": str(tab_rows.iloc[0]["model"]) if not tab_rows.empty else "",
        }
    return mapping


def _native_vs_ohe_key_slice_rows(aggregated: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for dataset_name, split_type, protocol in KEY_SLICE_DEFINITIONS:
        group = aggregated[
            (aggregated["dataset"] == dataset_name)
            & (aggregated["split_type"] == split_type)
            & (aggregated["protocol"] == protocol)
            & (aggregated["phase2_group"] == "primary")
            & (aggregated["train_fraction"] == 1.0)
            & (aggregated["k"] == PRIMARY_K)
            & (aggregated["feature_set"] == "full")
        ]
        native = group[group["model"] == "tabpfn_native"]
        ohe = group[group["model"] == "tabpfn"]
        if native.empty or ohe.empty:
            continue
        native_row = native.iloc[0]
        ohe_row = ohe.iloc[0]
        rows.append(
            {
                "dataset": dataset_name,
                "split_type": split_type,
                "protocol": protocol,
                "delta_ndcg": float(native_row["ndcg@10_mean"] - ohe_row["ndcg@10_mean"]),
                "runtime_ratio": float(native_row["runtime_seconds_median"] / ohe_row["runtime_seconds_median"])
                if float(ohe_row["runtime_seconds_median"]) > 0
                else float("nan"),
            }
        )
    return rows


def _best_tabpfn_vs_tree_rows(
    aggregated: pd.DataFrame,
    best_models: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    rows = []
    for dataset_name, split_type, protocol in KEY_SLICE_DEFINITIONS:
        group = aggregated[
            (aggregated["dataset"] == dataset_name)
            & (aggregated["split_type"] == split_type)
            & (aggregated["protocol"] == protocol)
        ]
        if group.empty:
            continue
        best_tree_model = best_models.get((dataset_name, split_type, protocol), {}).get("best_tree_model")
        best_tab_model = best_models.get((dataset_name, split_type, protocol), {}).get("best_tabpfn_model")
        if not best_tree_model or not best_tab_model:
            continue
        tree_row = group[group["model"] == best_tree_model].iloc[0]
        tab_row = group[group["model"] == best_tab_model].iloc[0]
        rows.append(
            {
                "split_type": split_type,
                "protocol": protocol,
                "best_tree_model": best_tree_model,
                "best_tabpfn_model": best_tab_model,
                "best_tree_ndcg": float(tree_row["ndcg@10_mean"]),
                "best_tabpfn_ndcg": float(tab_row["ndcg@10_mean"]),
                "tabpfn_within_one_percent": float(tab_row["ndcg@10_mean"]) >= float(tree_row["ndcg@10_mean"]) * 0.99,
            }
        )
    return rows


def _evaluate_k50_retention(
    k_sensitivity: pd.DataFrame,
    best_models: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    rows = []
    if k_sensitivity.empty:
        return rows
    for protocol in PRIMARY_PROTOCOLS:
        split_type = "item_cold"
        model = best_models.get((PRIMARY_DATASET, split_type, protocol), {}).get("best_tabpfn_model")
        if not model:
            continue
        subset = k_sensitivity[
            (k_sensitivity["dataset"] == PRIMARY_DATASET)
            & (k_sensitivity["split_type"] == split_type)
            & (k_sensitivity["protocol"] == protocol)
            & (k_sensitivity["model"] == model)
        ]
        k20 = subset[subset["k"] == 20]
        k50 = subset[subset["k"] == 50]
        if k20.empty or k50.empty:
            continue
        base = float(k20.iloc[0]["ndcg@10_mean"])
        retained = float(k50.iloc[0]["ndcg@10_mean"])
        rows.append(
            {
                "split_type": split_type,
                "protocol": protocol,
                "model": model,
                "retention_ratio": retained / base if base > 0 else float("nan"),
            }
        )
    return rows


def _summarize_amazon(amazon_sanity: pd.DataFrame) -> list[str]:
    if amazon_sanity.empty:
        return ["Amazon sanity runs were not available."]

    notes = []
    for split_type in PRIMARY_SPLITS:
        split_rows = amazon_sanity[amazon_sanity["split_type"] == split_type]
        if split_rows.empty:
            continue
        global_rows = split_rows[split_rows["protocol"] == "global_popularity"]
        if not global_rows.empty:
            saturated = bool((global_rows["ndcg@10_mean"] >= 0.98).all())
            notes.append(
                f"Amazon {split_type} / global_popularity was {'saturated' if saturated else 'not saturated'}."
            )
        context_rows = split_rows[split_rows["protocol"] == "context_popularity"]
        if not context_rows.empty:
            best = context_rows.sort_values(
                ["ndcg@10_mean", "runtime_seconds_median", "model"],
                ascending=[False, True, True],
            ).iloc[0]
            notes.append(
                f"Amazon {split_type} / context_popularity best model: {best['model']} "
                f"(NDCG@10={best['ndcg@10_mean']:.4f})."
            )
    return notes


def _save_empty_plot(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
