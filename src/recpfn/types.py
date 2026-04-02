"""Shared dataclasses and type aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    """Normalized dataset tables plus metadata needed by the pipeline."""

    name: str
    users: pd.DataFrame
    items: pd.DataFrame
    interactions: pd.DataFrame
    user_feature_columns: list[str]
    item_feature_columns: list[str]
    timestamp_col: str = "timestamp"
    label_col: str = "label"
    event_col: str = "event"
    context_col: str = "primary_category"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitBundle:
    """Train/test decomposition for one evaluation regime."""

    name: str
    split_type: str
    train_interactions: pd.DataFrame
    train_queries: pd.DataFrame
    test_queries: pd.DataFrame
    cold_item_ids: set[Any] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    """Paths emitted by the experiment runner."""

    output_dir: Path
    predictions_paths: list[Path] = field(default_factory=list)
    metrics_paths: list[Path] = field(default_factory=list)
    benchmark_table_path: Path | None = None
    summary_csv_path: Path | None = None


@dataclass
class Phase1DecisionArtifacts:
    """Paths emitted by the phase-one decision sweep."""

    output_dir: Path
    run_output_dir: Path
    baseline_results_path: Path | None = None
    merged_results_path: Path | None = None
    benchmark_table_path: Path | None = None
    decision_memo_path: Path | None = None
    tie_break_results_path: Path | None = None
    next_steps_plan_path: Path | None = None
    plots_output_dir: Path | None = None


@dataclass
class Phase2PointwiseArtifacts:
    """Paths emitted by the phase-two pointwise validation workflow."""

    output_dir: Path
    run_output_dir: Path
    plots_output_dir: Path
    raw_summary_path: Path | None = None
    aggregated_results_path: Path | None = None
    bootstrap_delta_summary_path: Path | None = None
    k_sensitivity_results_path: Path | None = None
    amazon_sanity_results_path: Path | None = None
    feature_group_ablation_path: Path | None = None
    benchmark_table_path: Path | None = None
    decision_memo_path: Path | None = None
