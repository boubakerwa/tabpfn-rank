"""End-to-end experiment runner."""

from __future__ import annotations

from dataclasses import replace
import time
from pathlib import Path

import pandas as pd

from recpfn.data.loaders import load_dataset
from recpfn.data.splits import build_splits
from recpfn.eval.metrics import evaluate_rankings
from recpfn.eval.ranking import attach_metadata
from recpfn.eval.reports import save_benchmark_table, save_metrics, save_predictions, save_summary_csv
from recpfn.features.builders import build_features
from recpfn.features.groups import select_feature_columns
from recpfn.models.base import (
    build_pairwise_training_rows,
    fit_pairwise,
    fit_pointwise,
    infer_feature_columns,
    predict_scores,
    score_pairwise_candidates,
)
from recpfn.rerank.candidate_sets import build_candidates
from recpfn.types import RunArtifacts, SplitBundle
from recpfn.utils import ensure_dir, read_env_str


def run_experiment(
    dataset_name: str,
    split_type: str,
    protocols: list[str],
    pointwise_models: list[str],
    pairwise_models: list[str],
    cache_dir: str | Path = "data",
    output_dir: str | Path = "paper/results",
    seed: int = 0,
    k: int = 20,
    max_train_queries: int | None = None,
    max_test_queries: int | None = None,
    train_fraction: float | None = None,
    feature_set: str = "full",
) -> tuple[pd.DataFrame, RunArtifacts]:
    """Run one experiment sweep and persist outputs."""

    dataset = load_dataset(dataset_name, cache_dir=cache_dir, seed=seed)
    split = build_splits(dataset, split_type=split_type, seed=seed)
    split = limit_split_queries(
        split,
        max_train_queries=max_train_queries,
        max_test_queries=max_test_queries,
    )
    artifact_dir = ensure_dir(Path(output_dir) / dataset.name / split.split_type)
    result_rows = []
    artifacts = RunArtifacts(output_dir=artifact_dir)
    run_metadata = _build_run_metadata(
        seed=seed,
        k=k,
        max_train_queries=max_train_queries,
        max_test_queries=max_test_queries,
        train_fraction=train_fraction,
        feature_set=feature_set,
    )

    for protocol in protocols:
        candidates = build_candidates(dataset, split, protocol=protocol, k=k, seed=seed)
        candidates = limit_candidate_queries(
            candidates,
            max_train_queries=max_train_queries,
            max_test_queries=max_test_queries,
        )

        features = build_features(dataset, candidates, split)
        train_df = features[features["split"] == "train"].copy()
        test_df = features[features["split"] == "test"].copy()
        feature_cols = select_feature_columns(infer_feature_columns(features), feature_set=feature_set)

        for model_name in pointwise_models:
            start = time.perf_counter()
            try:
                model = fit_pointwise(model_name, train_df, feature_cols)
                scored = test_df.copy()
                scored["score"] = predict_scores(model, scored, feature_cols)
            except Exception as exc:
                result_rows.append(
                    _failure_row(
                        dataset.name,
                        split.split_type,
                        protocol,
                        model_name,
                        "pointwise",
                        exc,
                        run_metadata=run_metadata,
                    )
                )
                continue
            runtime = time.perf_counter() - start
            predictions = attach_metadata(scored, dataset.name, split.split_type, protocol, model_name, "pointwise")
            predictions = _attach_run_metadata(predictions, run_metadata)
            metrics = evaluate_rankings(predictions)
            metrics["runtime_seconds"] = runtime
            metrics.update(
                {
                    "dataset": dataset.name,
                    "split_type": split.split_type,
                    "protocol": protocol,
                    "model": model_name,
                    "mode": "pointwise",
                    "status": "ok",
                    **run_metadata,
                }
            )
            result_rows.append(metrics)
            artifacts.predictions_paths.append(
                save_predictions(
                    predictions,
                    artifact_dir / f"{protocol}_pointwise_{model_name}_predictions.csv",
                )
            )
            artifacts.metrics_paths.append(
                save_metrics(metrics, artifact_dir / f"{protocol}_pointwise_{model_name}_metrics.json")
            )

        if pairwise_models:
            pair_train_df = build_pairwise_training_rows(train_df, feature_cols, max_pairs_per_query=10, seed=seed)
            pair_feature_cols = infer_feature_columns(pair_train_df) if not pair_train_df.empty else []
            for model_name in pairwise_models:
                if pair_train_df.empty:
                    result_rows.append(
                        _failure_row(
                            dataset.name,
                            split.split_type,
                            protocol,
                            model_name,
                            "pairwise",
                            RuntimeError("No pairwise training rows were generated."),
                            run_metadata=run_metadata,
                        )
                    )
                    continue
                start = time.perf_counter()
                try:
                    model = fit_pairwise(model_name, pair_train_df, pair_feature_cols)
                    scored = score_pairwise_candidates(model, test_df, feature_cols)
                except Exception as exc:
                    result_rows.append(
                        _failure_row(
                            dataset.name,
                            split.split_type,
                            protocol,
                            model_name,
                            "pairwise",
                            exc,
                            run_metadata=run_metadata,
                        )
                    )
                    continue
                runtime = time.perf_counter() - start
                predictions = attach_metadata(scored, dataset.name, split.split_type, protocol, model_name, "pairwise")
                predictions = _attach_run_metadata(predictions, run_metadata)
                metrics = evaluate_rankings(predictions)
                metrics["runtime_seconds"] = runtime
                metrics.update(
                    {
                        "dataset": dataset.name,
                        "split_type": split.split_type,
                        "protocol": protocol,
                        "model": model_name,
                        "mode": "pairwise",
                        "status": "ok",
                        **run_metadata,
                    }
                )
                result_rows.append(metrics)
                artifacts.predictions_paths.append(
                    save_predictions(
                        predictions,
                        artifact_dir / f"{protocol}_pairwise_{model_name}_predictions.csv",
                    )
                )
                artifacts.metrics_paths.append(
                    save_metrics(metrics, artifact_dir / f"{protocol}_pairwise_{model_name}_metrics.json")
                )

    results = pd.DataFrame(result_rows)
    if not results.empty:
        artifacts.summary_csv_path = save_summary_csv(results, artifact_dir / "results.csv")
        artifacts.benchmark_table_path = save_benchmark_table(results, artifact_dir / "benchmark.md")
    return results, artifacts


def limit_candidate_queries(
    candidates: pd.DataFrame,
    max_train_queries: int | None = None,
    max_test_queries: int | None = None,
) -> pd.DataFrame:
    """Limit train/test candidate rows to the first N queries per split."""

    limited = candidates.copy()
    if max_train_queries is not None:
        train_query_ids = limited[limited["split"] == "train"]["query_id"].drop_duplicates().head(max_train_queries)
        limited = limited[(limited["split"] != "train") | (limited["query_id"].isin(train_query_ids))]
    if max_test_queries is not None:
        test_query_ids = limited[limited["split"] == "test"]["query_id"].drop_duplicates().head(max_test_queries)
        limited = limited[(limited["split"] != "test") | (limited["query_id"].isin(test_query_ids))]
    return limited


def limit_split_queries(
    split: SplitBundle,
    max_train_queries: int | None = None,
    max_test_queries: int | None = None,
) -> SplitBundle:
    """Limit split query frames before candidate generation."""

    limited_train_queries = split.train_queries.copy()
    limited_test_queries = split.test_queries.copy()
    if max_train_queries is not None:
        limited_train_queries = limited_train_queries.head(max_train_queries).copy()
    if max_test_queries is not None:
        limited_test_queries = limited_test_queries.head(max_test_queries).copy()
    return replace(
        split,
        train_queries=limited_train_queries,
        test_queries=limited_test_queries,
        metadata={
            **split.metadata,
            "n_train_queries": len(limited_train_queries),
            "n_test_queries": len(limited_test_queries),
        },
    )


def _build_run_metadata(
    seed: int,
    k: int,
    max_train_queries: int | None,
    max_test_queries: int | None,
    train_fraction: float | None = None,
    feature_set: str = "full",
) -> dict[str, object]:
    return {
        "tabpfn_version": read_env_str("RECPFN_TABPFN_VERSION", "v2"),
        "k": int(k),
        "max_train_queries": max_train_queries,
        "max_test_queries": max_test_queries,
        "seed": int(seed),
        "train_fraction": train_fraction,
        "feature_set": feature_set,
    }


def _attach_run_metadata(frame: pd.DataFrame, run_metadata: dict[str, object]) -> pd.DataFrame:
    enriched = frame.copy()
    for key, value in run_metadata.items():
        enriched[key] = value
    return enriched


def _failure_row(
    dataset: str,
    split_type: str,
    protocol: str,
    model_name: str,
    mode: str,
    exc: Exception,
    run_metadata: dict[str, object] | None = None,
) -> dict:
    row = {
        "dataset": dataset,
        "split_type": split_type,
        "protocol": protocol,
        "model": model_name,
        "mode": mode,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "runtime_seconds": 0.0,
        "n_queries": 0.0,
        "ndcg@10": float("nan"),
        "recall@10": float("nan"),
        "mrr": float("nan"),
        "hitrate@10": float("nan"),
    }
    if run_metadata:
        row.update(run_metadata)
    return row
