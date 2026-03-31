"""End-to-end experiment runner."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from recpfn.data.loaders import load_dataset
from recpfn.data.splits import build_splits
from recpfn.eval.metrics import evaluate_rankings
from recpfn.eval.ranking import attach_metadata
from recpfn.eval.reports import save_benchmark_table, save_metrics, save_predictions, save_summary_csv
from recpfn.features.builders import build_features
from recpfn.models.base import (
    build_pairwise_training_rows,
    fit_pairwise,
    fit_pointwise,
    infer_feature_columns,
    predict_scores,
    score_pairwise_candidates,
)
from recpfn.rerank.candidate_sets import build_candidates
from recpfn.types import RunArtifacts
from recpfn.utils import ensure_dir


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
) -> tuple[pd.DataFrame, RunArtifacts]:
    """Run one experiment sweep and persist outputs."""

    dataset = load_dataset(dataset_name, cache_dir=cache_dir, seed=seed)
    split = build_splits(dataset, split_type=split_type, seed=seed)
    artifact_dir = ensure_dir(Path(output_dir) / dataset.name / split.split_type)
    result_rows = []
    artifacts = RunArtifacts(output_dir=artifact_dir)

    for protocol in protocols:
        candidates = build_candidates(dataset, split, protocol=protocol, k=k, seed=seed)
        if max_train_queries is not None:
            train_query_ids = candidates[candidates["split"] == "train"]["query_id"].drop_duplicates().head(max_train_queries)
            candidates = candidates[
                (candidates["split"] != "train") | (candidates["query_id"].isin(train_query_ids))
            ]
        if max_test_queries is not None:
            test_query_ids = candidates[candidates["split"] == "test"]["query_id"].drop_duplicates().head(max_test_queries)
            candidates = candidates[
                (candidates["split"] != "test") | (candidates["query_id"].isin(test_query_ids))
            ]

        features = build_features(dataset, candidates, split)
        train_df = features[features["split"] == "train"].copy()
        test_df = features[features["split"] == "test"].copy()
        feature_cols = infer_feature_columns(features)

        for model_name in pointwise_models:
            start = time.perf_counter()
            try:
                model = fit_pointwise(model_name, train_df, feature_cols)
                scored = test_df.copy()
                scored["score"] = predict_scores(model, scored, feature_cols)
            except Exception as exc:
                result_rows.append(
                    _failure_row(dataset.name, split.split_type, protocol, model_name, "pointwise", exc)
                )
                continue
            runtime = time.perf_counter() - start
            predictions = attach_metadata(scored, dataset.name, split.split_type, protocol, model_name, "pointwise")
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
                        )
                    )
                    continue
                start = time.perf_counter()
                try:
                    model = fit_pairwise(model_name, pair_train_df, pair_feature_cols)
                    scored = score_pairwise_candidates(model, test_df, feature_cols)
                except Exception as exc:
                    result_rows.append(
                        _failure_row(dataset.name, split.split_type, protocol, model_name, "pairwise", exc)
                    )
                    continue
                runtime = time.perf_counter() - start
                predictions = attach_metadata(scored, dataset.name, split.split_type, protocol, model_name, "pairwise")
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


def _failure_row(dataset: str, split_type: str, protocol: str, model_name: str, mode: str, exc: Exception) -> dict:
    return {
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
