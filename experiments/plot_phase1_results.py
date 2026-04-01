"""Generate summary plots from Phase 1 benchmark result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLORS = {
    "tabpfn": "#0f766e",
    "xgboost": "#b45309",
    "catboost": "#2563eb",
    "popularity": "#6b7280",
    "recent_popularity": "#9ca3af",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase 1 benchmark plots from results.csv files.")
    parser.add_argument(
        "--results-root",
        default="paper/results_phase1_decision_runs_final",
        help="Root directory containing per-unit results.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures/phase1_final",
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--title-suffix",
        default="",
        help="Optional suffix appended to plot titles, e.g. '(64/64 units)'.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = load_results(args.results_root)
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit("No successful result rows were found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok["setting"] = ok["dataset"] + " | " + ok["split_type"] + " | " + ok["protocol"]

    suffix = f" {args.title_suffix.strip()}".rstrip() if args.title_suffix.strip() else ""

    paths = [
        plot_quality_vs_runtime(ok, output_dir, suffix),
        plot_best_by_setting(ok, output_dir, suffix),
        plot_tabpfn_vs_best_tree_gap(ok, output_dir, suffix),
        plot_pairwise_vs_pointwise_tabpfn(ok, output_dir, suffix),
    ]

    print("Wrote plots:")
    for path in paths:
        print(path)


def load_results(results_root: str | Path) -> pd.DataFrame:
    root = Path(results_root)
    paths = sorted(root.glob("**/results.csv"))
    if not paths:
        raise FileNotFoundError(f"No results.csv files found under {root}.")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_quality_vs_runtime(results: pd.DataFrame, output_dir: Path, suffix: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in results.groupby("model"):
        ax.scatter(
            group["runtime_seconds"],
            group["ndcg@10"],
            s=70,
            alpha=0.8,
            label=model,
            color=MODEL_COLORS.get(model, "#111827"),
        )

    for _, row in results.iterrows():
        if row["model"] == "tabpfn" and row["ndcg@10"] > 0.9:
            ax.annotate(
                f"{row['dataset'].split('_')[0]}\n{row['split_type']}\n{row['protocol'].split('_')[0]}",
                (row["runtime_seconds"], row["ndcg@10"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_ylabel("NDCG@10")
    ax.set_title(f"Phase 1 Results: Quality vs Runtime{suffix}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncols=3)
    fig.tight_layout()

    path = output_dir / "quality_vs_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_by_setting(results: pd.DataFrame, output_dir: Path, suffix: str) -> Path:
    best = (
        results.sort_values(["ndcg@10", "recall@10", "mrr"], ascending=False)
        .groupby("setting", as_index=False)
        .first()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = range(len(best))
    colors = [MODEL_COLORS.get(model, "#111827") for model in best["model"]]
    ax.barh(list(positions), best["ndcg@10"], color=colors)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(
        [f"{setting} | {model} ({mode})" for setting, model, mode in zip(best["setting"], best["model"], best["mode"])],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Best NDCG@10")
    ax.set_title(f"Best Completed Model per Setting{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / "best_by_setting.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_tabpfn_vs_best_tree_gap(results: pd.DataFrame, output_dir: Path, suffix: str) -> Path:
    rows = []
    for setting, group in results.groupby("setting"):
        tab = group[group["model"] == "tabpfn"]
        tree = group[group["model"].isin(["xgboost", "catboost"])]
        if tab.empty or tree.empty:
            continue

        best_tab = tab.sort_values(["ndcg@10", "recall@10", "mrr"], ascending=False).iloc[0]
        best_tree = tree.sort_values(["ndcg@10", "recall@10", "mrr"], ascending=False).iloc[0]
        rows.append(
            {
                "setting": setting,
                "gap": float(best_tab["ndcg@10"] - best_tree["ndcg@10"]),
            }
        )

    gap_df = pd.DataFrame(rows).sort_values("gap", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#dc2626" if gap < 0 else "#059669" for gap in gap_df["gap"]]
    ax.barh(gap_df["setting"], gap_df["gap"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Best TabPFN NDCG@10 - Best Tree NDCG@10")
    ax.set_title(f"Where TabPFN Helps{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / "tabpfn_vs_best_tree_gap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pairwise_vs_pointwise_tabpfn(results: pd.DataFrame, output_dir: Path, suffix: str) -> Path:
    rows = []
    for setting, group in results[results["model"] == "tabpfn"].groupby("setting"):
        pointwise = group[group["mode"] == "pointwise"]
        pairwise = group[group["mode"] == "pairwise"]
        if pointwise.empty or pairwise.empty:
            continue

        pointwise_row = pointwise.iloc[0]
        pairwise_row = pairwise.iloc[0]
        rows.append(
            {
                "setting": setting,
                "ndcg_gain": float(pairwise_row["ndcg@10"] - pointwise_row["ndcg@10"]),
            }
        )

    pair_df = pd.DataFrame(rows).sort_values("ndcg_gain", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#dc2626" if gain < 0 else "#059669" for gain in pair_df["ndcg_gain"]]
    ax.barh(pair_df["setting"], pair_df["ndcg_gain"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Pairwise TabPFN - Pointwise TabPFN (NDCG@10)")
    ax.set_title(f"Pairwise TabPFN Gains{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / "pairwise_vs_pointwise_tabpfn.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
