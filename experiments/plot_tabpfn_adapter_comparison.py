"""Generate focused comparison plots for `tabpfn` vs `tabpfn_native`."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLORS = {
    "tabpfn": "#0f766e",
    "tabpfn_native": "#7c3aed",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot tabpfn vs tabpfn_native comparison summaries.")
    parser.add_argument(
        "--summary-csv",
        required=True,
        help="Input summary CSV containing tabpfn and tabpfn_native rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures/tabpfn_adapter_comparison",
        help="Directory where plots and the optional HTML index will be written.",
    )
    parser.add_argument(
        "--metric",
        default="ndcg@10",
        help="Primary metric column to plot.",
    )
    parser.add_argument(
        "--title-suffix",
        default="",
        help="Optional suffix appended to plot titles.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip generating the local HTML index.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_summary_csv(args.summary_csv)
    comparison = prepare_comparison_frame(summary)
    if comparison.empty:
        raise SystemExit("No rows for tabpfn/tabpfn_native were found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" {args.title_suffix.strip()}".rstrip() if args.title_suffix.strip() else ""

    paths = generate_comparison_plots(comparison, output_dir=output_dir, metric=args.metric, suffix=suffix)
    if not args.no_html:
        paths.append(write_html_index(comparison, paths, output_dir, suffix=suffix, metric=args.metric))

    print("Wrote comparison artifacts:")
    for path in paths:
        print(path)


def load_summary_csv(summary_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(summary_csv)


def prepare_comparison_frame(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    if "status" in frame.columns:
        frame = frame[frame["status"] == "ok"].copy()
    frame = frame[frame["model"].isin(["tabpfn", "tabpfn_native"])].copy()
    if frame.empty:
        return frame

    frame["model"] = pd.Categorical(frame["model"], categories=["tabpfn", "tabpfn_native"], ordered=True)
    for column in ("train_fraction", "ndcg@10", "recall@10", "mrr", "runtime_seconds"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values([column for column in ["split_type", "protocol", "train_fraction", "model"] if column in frame.columns])


def generate_comparison_plots(
    comparison: pd.DataFrame,
    output_dir: Path,
    metric: str = "ndcg@10",
    suffix: str = "",
) -> list[Path]:
    has_train_fraction = "train_fraction" in comparison.columns and comparison["train_fraction"].notna().any()
    if has_train_fraction:
        return [
            plot_metric_by_train_fraction(comparison, output_dir, metric=metric, suffix=suffix),
            plot_runtime_by_train_fraction(comparison, output_dir, suffix=suffix),
            plot_metric_gain_by_train_fraction(comparison, output_dir, metric=metric, suffix=suffix),
        ]
    return [
        plot_metric_by_setting(comparison, output_dir, metric=metric, suffix=suffix),
        plot_runtime_by_setting(comparison, output_dir, suffix=suffix),
        plot_metric_gain_by_setting(comparison, output_dir, metric=metric, suffix=suffix),
    ]


def plot_metric_by_train_fraction(comparison: pd.DataFrame, output_dir: Path, metric: str, suffix: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split_type in zip(axes, sorted(comparison["split_type"].dropna().unique())):
        subset = comparison[comparison["split_type"] == split_type]
        for model, group in subset.groupby("model"):
            group = group.sort_values("train_fraction")
            ax.plot(
                group["train_fraction"],
                group[metric],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS.get(str(model), "#111827"),
                label=str(model),
            )
        ax.set_title(split_type.replace("_", " ").title())
        ax.set_xlabel("Train fraction")
        ax.set_xticks(sorted(subset["train_fraction"].dropna().unique()))
        ax.set_xticklabels([f"{int(value * 100)}%" for value in sorted(subset["train_fraction"].dropna().unique())])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"TabPFN Adapter Comparison: {metric.upper()} vs Train Fraction{suffix}", y=1.07)
    fig.tight_layout()

    path = output_dir / f"{metric.replace('@', '_')}_by_train_fraction.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_runtime_by_train_fraction(comparison: pd.DataFrame, output_dir: Path, suffix: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split_type in zip(axes, sorted(comparison["split_type"].dropna().unique())):
        subset = comparison[comparison["split_type"] == split_type]
        for model, group in subset.groupby("model"):
            group = group.sort_values("train_fraction")
            ax.plot(
                group["train_fraction"],
                group["runtime_seconds"],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS.get(str(model), "#111827"),
                label=str(model),
            )
        ax.set_title(split_type.replace("_", " ").title())
        ax.set_xlabel("Train fraction")
        ax.set_xticks(sorted(subset["train_fraction"].dropna().unique()))
        ax.set_xticklabels([f"{int(value * 100)}%" for value in sorted(subset["train_fraction"].dropna().unique())])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Runtime (seconds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"TabPFN Adapter Comparison: Runtime vs Train Fraction{suffix}", y=1.07)
    fig.tight_layout()

    path = output_dir / "runtime_by_train_fraction.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_gain_by_train_fraction(comparison: pd.DataFrame, output_dir: Path, metric: str, suffix: str) -> Path:
    gain = _gain_table(comparison, value_col=metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#059669" if value >= 0 else "#dc2626" for value in gain["gain"]]
    positions = range(len(gain))
    ax.barh(list(positions), gain["gain"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(
        [
            f"{split_type} | {int(train_fraction * 100)}%"
            for split_type, train_fraction in zip(gain["split_type"], gain["train_fraction"])
        ]
    )
    ax.set_xlabel(f"Native - OHE {metric.upper()}")
    ax.set_title(f"TabPFN Native Gain Over One-Hot{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / f"native_minus_ohe_{metric.replace('@', '_')}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_by_setting(comparison: pd.DataFrame, output_dir: Path, metric: str, suffix: str) -> Path:
    settings = _setting_table(comparison, value_col=metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = range(len(settings))
    colors = [MODEL_COLORS.get(model, "#111827") for model in settings["best_model"]]
    ax.barh(list(positions), settings[metric], color=colors)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(settings["setting"].tolist())
    ax.set_xlabel(metric.upper())
    ax.set_title(f"TabPFN Adapter Comparison by Setting{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / f"{metric.replace('@', '_')}_by_setting.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_runtime_by_setting(comparison: pd.DataFrame, output_dir: Path, suffix: str) -> Path:
    settings = _setting_table(comparison, value_col="runtime_seconds")
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = range(len(settings))
    colors = [MODEL_COLORS.get(model, "#111827") for model in settings["best_model"]]
    ax.barh(list(positions), settings["runtime_seconds"], color=colors)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(settings["setting"].tolist())
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title(f"TabPFN Adapter Runtime by Setting{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / "runtime_by_setting.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_gain_by_setting(comparison: pd.DataFrame, output_dir: Path, metric: str, suffix: str) -> Path:
    gain = _gain_table(comparison, value_col=metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#059669" if value >= 0 else "#dc2626" for value in gain["gain"]]
    positions = range(len(gain))
    ax.barh(list(positions), gain["gain"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(gain["setting"].tolist())
    ax.set_xlabel(f"Native - OHE {metric.upper()}")
    ax.set_title(f"TabPFN Native Gain Over One-Hot by Setting{suffix}")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    path = output_dir / f"native_minus_ohe_{metric.replace('@', '_')}_by_setting.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_html_index(
    comparison: pd.DataFrame,
    plot_paths: list[Path],
    output_dir: Path,
    suffix: str,
    metric: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = _summary_lines(comparison, metric=metric)
    image_tags = "\n".join(
        f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}</h2><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></section>'
        for path in plot_paths
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TabPFN Adapter Comparison{html.escape(suffix)}</title>
  <style>
    :root {{
      color-scheme: light;
      --fg: #0f172a;
      --muted: #475569;
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.5;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .intro {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(15, 23, 42, 0.06);
    }}
    .grid {{
      display: grid;
      gap: 20px;
    }}
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 8px 32px rgba(15, 23, 42, 0.06);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    ul {{ margin: 0; padding-left: 20px; color: var(--muted); }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <div class="intro">
      <h1>TabPFN Adapter Comparison{html.escape(suffix)}</h1>
      <p class="muted">Focused comparison between <code>tabpfn</code> and <code>tabpfn_native</code> from <code>{html.escape(metric)}</code> summary rows.</p>
      <ul>
        {''.join(f'<li>{html.escape(line)}</li>' for line in summary_lines)}
      </ul>
    </div>
    <div class="grid">
      {image_tags}
    </div>
  </main>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def _setting_table(comparison: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for setting, group in comparison.groupby(_setting_columns(comparison)):
        if isinstance(setting, tuple):
            split_type, protocol = setting[:2]
            train_fraction = setting[2] if len(setting) > 2 else None
        else:
            split_type, protocol, train_fraction = setting, "", None
        pivot = group.pivot_table(index=[], columns="model", values=value_col, aggfunc="mean")
        if not {"tabpfn", "tabpfn_native"}.issubset(set(pivot.columns)):
            continue
        pivot_row = pivot.iloc[0]
        best_model = "tabpfn_native" if pivot_row["tabpfn_native"] >= pivot_row["tabpfn"] else "tabpfn"
        best_value = float(max(pivot_row["tabpfn"], pivot_row["tabpfn_native"]))
        label = f"{split_type} | {protocol}" if train_fraction is None else f"{split_type} | {protocol} | {train_fraction:.1f}"
        rows.append(
            {
                "setting": label,
                value_col: best_value,
                "best_model": best_model,
            }
        )
    return pd.DataFrame(rows)


def _gain_table(comparison: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    grouped = comparison.groupby(_setting_columns(comparison))
    for setting, group in grouped:
        pivot = group.pivot_table(index=[], columns="model", values=value_col, aggfunc="mean")
        if not {"tabpfn", "tabpfn_native"}.issubset(set(pivot.columns)):
            continue
        pivot_row = pivot.iloc[0]
        split_type = setting[0] if isinstance(setting, tuple) else setting
        protocol = setting[1] if isinstance(setting, tuple) and len(setting) > 1 else ""
        train_fraction = setting[2] if isinstance(setting, tuple) and len(setting) > 2 else None
        rows.append(
            {
                "split_type": split_type,
                "protocol": protocol,
                "train_fraction": train_fraction,
                "setting": f"{split_type} | {protocol}" if train_fraction is None else f"{split_type} | {protocol} | {train_fraction:.1f}",
                "gain": float(pivot_row["tabpfn_native"] - pivot_row["tabpfn"]),
            }
        )
    return pd.DataFrame(rows).sort_values("gain", ascending=True)


def _setting_columns(comparison: pd.DataFrame) -> list[str]:
    columns = ["split_type", "protocol"]
    if "train_fraction" in comparison.columns and comparison["train_fraction"].notna().any():
        columns.append("train_fraction")
    return columns


def _summary_lines(comparison: pd.DataFrame, metric: str) -> list[str]:
    comparison = comparison.copy()
    comparison[metric] = pd.to_numeric(comparison[metric], errors="coerce")
    if "train_fraction" in comparison.columns and comparison["train_fraction"].notna().any():
        gain = _gain_table(comparison, value_col=metric)
        if gain.empty:
            return ["No paired rows were available for summary statistics."]
        best = gain.sort_values("gain", ascending=False).iloc[0]
        worst = gain.iloc[0]
        mean_gain = gain["gain"].mean()
        return [
            f"Mean native-vs-ohe {metric}: {mean_gain:+.4f}",
            f"Best setting: {best['setting']} ({best['gain']:+.4f})",
            f"Worst setting: {worst['setting']} ({worst['gain']:+.4f})",
        ]
    setting = _setting_table(comparison, value_col=metric)
    if setting.empty:
        return ["No paired rows were available for summary statistics."]
    return [f"Prepared {len(setting)} comparison settings from the summary CSV."]


if __name__ == "__main__":
    main()
