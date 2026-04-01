"""Experiment reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recpfn.utils import ensure_dir


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    predictions.to_csv(path, index=False)
    return path


def save_metrics(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return path


def save_benchmark_table(
    results: pd.DataFrame,
    path: str | Path,
    columns: list[str] | None = None,
    sort_by: list[str] | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    selected_columns = columns or [
        "dataset",
        "split_type",
        "protocol",
        "mode",
        "model",
        "ndcg@10",
        "recall@10",
        "mrr",
        "hitrate@10",
        "runtime_seconds",
    ]
    selected_columns = [column for column in selected_columns if column in results.columns]
    selected_sort = sort_by or ["dataset", "split_type", "protocol", "mode", "model"]
    selected_sort = [column for column in selected_sort if column in results.columns]
    table = results[results["status"] == "ok"][selected_columns]
    if selected_sort:
        table = table.sort_values(selected_sort)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(_to_markdown(table))
        handle.write("\n")
    return path


def save_summary_csv(results: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    results.to_csv(path, index=False)
    return path


def _to_markdown(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    rows = [[_format_cell(row[column]) for column in headers] for _, row in table.iterrows()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(values: list[str]) -> str:
        padded = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
