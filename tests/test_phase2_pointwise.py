import subprocess

import pandas as pd
import pytest

from recpfn.data.loaders import load_dataset
from recpfn.data.schemas import INTERACTION_ID_COL, ITEM_ID_COL, LABEL_COL, QUERY_ID_COL, TIMESTAMP_COL, USER_ID_COL
from recpfn.data.splits import build_splits
from recpfn.eval.reports import save_predictions, save_summary_csv
from recpfn.features.builders import build_features
from recpfn.features.cold_start import item_training_popularity
from recpfn.features.groups import select_feature_columns
from recpfn.features.interactions import numeric_average, recency_in_days, safe_affinity_count, summarize_history
from recpfn.models.tabpfn_pointwise import TabPFNNativePointwiseRanker
from recpfn.phase1_decision import ProgressTracker, _run_unit_subprocess
from recpfn.phase2_pointwise_report import (
    compute_bootstrap_delta_summary,
    evaluate_phase2_outcome,
    generate_phase2_plots,
    run_phase2_pointwise_report,
)
from recpfn.phase2_pointwise_run import _run_k_sensitivity, _select_best_tree_models, run_phase2_pointwise_raw


def test_feature_builder_history_index_matches_legacy_scan():
    dataset = load_dataset("synthetic")
    split = build_splits(dataset, "warm", seed=0)
    candidates = pd.concat(
        [
            split.train_queries.assign(split="train"),
            split.test_queries.assign(split="test"),
        ],
        ignore_index=True,
    )
    candidate_rows = candidates[[QUERY_ID_COL, USER_ID_COL, ITEM_ID_COL, "split", "query_timestamp", "query_interaction_id"]]
    current = build_features(dataset, candidate_rows, split).sort_values([QUERY_ID_COL, ITEM_ID_COL]).reset_index(drop=True)
    legacy = _legacy_build_features(dataset, candidate_rows, split).sort_values([QUERY_ID_COL, ITEM_ID_COL]).reset_index(drop=True)

    pd.testing.assert_frame_equal(current, legacy)


def test_run_unit_subprocess_returns_timeout_failure_row(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=1)

    monkeypatch.setattr("recpfn.benchmark_runner.subprocess.run", fake_run)
    tracker = ProgressTracker(total_units=1)

    result = _run_unit_subprocess(
        dataset_name="synthetic",
        split_type="warm",
        protocol="global_popularity",
        mode="pointwise",
        model="popularity",
        cache_dir="data",
        output_dir=tmp_path,
        seed=0,
        k=4,
        max_train_queries=2,
        max_test_queries=1,
        tracker=tracker,
        timeout_seconds=1,
    )

    assert result.iloc[0]["status"] == "error"
    assert "TimeoutExpired" in result.iloc[0]["error"]


def test_tabpfn_native_requires_matching_feature_columns():
    ranker = TabPFNNativePointwiseRanker()
    ranker.model = object()
    ranker.feature_columns_ = ["user_gender", "item_price"]
    ranker.categorical_cols_ = ["user_gender"]

    with pytest.raises(ValueError, match="match the training feature columns exactly"):
        ranker.predict_proba(
            pd.DataFrame({"user_gender": ["F"], "item_price": [10.0], "item_brand": ["acme"]}),
            ["user_gender", "item_brand"],
        )


def test_select_feature_columns_supports_named_feature_sets():
    feature_cols = [
        "user_age",
        "user_gender",
        "user_occupation",
        "item_primary_category",
        "item_brand",
        "item_price",
        "feat_genre_action",
        "hist_interactions",
        "category_affinity",
        "price_distance_to_user_avg",
    ]

    assert select_feature_columns(feature_cols, "full") == feature_cols
    assert select_feature_columns(feature_cols, "no_user_metadata") == [
        "item_primary_category",
        "item_brand",
        "item_price",
        "feat_genre_action",
        "hist_interactions",
        "category_affinity",
        "price_distance_to_user_avg",
    ]
    assert select_feature_columns(feature_cols, "metadata_only") == [
        "user_age",
        "user_gender",
        "user_occupation",
        "item_primary_category",
        "item_brand",
        "item_price",
        "feat_genre_action",
    ]


def test_phase2_raw_runner_smoke_writes_tree_selection(tmp_path, monkeypatch):
    def fake_unit_matrix(**kwargs):
        models = kwargs["models"]
        protocols = kwargs["protocols"]
        rows = []
        for protocol in protocols:
            for model in models:
                rows.append(
                    {
                        "dataset": kwargs["dataset_name"],
                        "split_type": kwargs["split_type"],
                        "protocol": protocol,
                        "mode": kwargs["mode"],
                        "model": model,
                        "status": "ok",
                        "ndcg@10": 0.80 if model == "xgboost" else 0.78,
                        "recall@10": 0.90,
                        "mrr": 0.75,
                        "hitrate@10": 0.90,
                        "runtime_seconds": 2.0,
                        "n_queries": float(kwargs.get("max_test_queries") or 1),
                        "seed": kwargs["seed"],
                        "k": kwargs["k"],
                        "max_train_queries": kwargs.get("max_train_queries"),
                        "max_test_queries": kwargs.get("max_test_queries"),
                        "train_fraction": kwargs.get("train_fraction"),
                        "feature_set": kwargs.get("feature_set", "full"),
                        "tabpfn_version": "v2.5",
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr("recpfn.phase2_pointwise_run._run_unit_matrix", fake_unit_matrix)
    monkeypatch.setattr("recpfn.phase2_pointwise_run.PRIMARY_PROTOCOLS", ["global_popularity"])
    monkeypatch.setattr("recpfn.phase2_pointwise_run.PRIMARY_SPLITS", ["warm"])
    monkeypatch.setattr("recpfn.phase2_pointwise_run.PRIMARY_POINTWISE_MODELS", ["xgboost", "tabpfn"])
    monkeypatch.setattr("recpfn.phase2_pointwise_run.FEATURE_ABLATION_SETS", ["full"])

    results = run_phase2_pointwise_raw(
        cache_dir="data",
        run_output_dir=tmp_path,
        primary_dataset="synthetic",
        secondary_dataset=None,
        seeds=[0],
        train_fractions=[1.0],
        k_values=[20],
        include_k_sensitivity=False,
        include_amazon_sanity=False,
        include_feature_ablation=False,
    )

    assert not results.empty
    assert (tmp_path / "best_tree_selection.json").exists()
    assert set(results["phase2_group"]) == {"primary"}


def test_select_best_tree_models_uses_best_full_data_tree_per_slice():
    primary = pd.DataFrame(
        [
            _result_row(model="xgboost", seed=0, split_type="warm", protocol="global_popularity", ndcg=0.81),
            _result_row(model="catboost", seed=0, split_type="warm", protocol="global_popularity", ndcg=0.84),
            _result_row(model="xgboost", seed=1, split_type="warm", protocol="global_popularity", ndcg=0.82),
            _result_row(model="catboost", seed=1, split_type="warm", protocol="global_popularity", ndcg=0.85),
            _result_row(model="xgboost", seed=0, split_type="item_cold", protocol="context_popularity", ndcg=0.88),
            _result_row(model="catboost", seed=0, split_type="item_cold", protocol="context_popularity", ndcg=0.86),
        ]
    )

    mapping = _select_best_tree_models(primary.assign(phase2_group="primary", train_fraction=1.0, status="ok"))

    assert mapping["warm::global_popularity"] == "catboost"
    assert mapping["item_cold::context_popularity"] == "xgboost"


def test_phase2_report_smoke_from_existing_raw_outputs(tmp_path):
    raw_root = tmp_path / "runs"
    unit_dir = raw_root / "primary" / "unit_a"
    unit_dir.mkdir(parents=True)

    results = pd.DataFrame(
        [
            _result_row(model="xgboost", seed=0, split_type="warm", protocol="global_popularity", ndcg=0.81),
            _result_row(model="tabpfn", seed=0, split_type="warm", protocol="global_popularity", ndcg=0.80),
            _result_row(model="tabpfn_native", seed=0, split_type="warm", protocol="global_popularity", ndcg=0.82),
        ]
    )
    save_summary_csv(results, unit_dir / "results.csv")
    for row in results.to_dict("records"):
        save_predictions(_prediction_frame(row), unit_dir / f"{row['protocol']}_pointwise_{row['model']}_predictions.csv")

    raw_summary, artifacts, decision = run_phase2_pointwise_report(
        run_output_dir=raw_root,
        output_dir=tmp_path / "summary",
        plots_output_dir=tmp_path / "plots",
        bootstrap_replicates=50,
    )

    assert not raw_summary.empty
    assert artifacts.raw_summary_path is not None and artifacts.raw_summary_path.exists()
    assert artifacts.aggregated_results_path is not None and artifacts.aggregated_results_path.exists()
    assert artifacts.decision_memo_path is not None and artifacts.decision_memo_path.exists()
    assert decision["native_adapter_outcome"] in {
        "promote primary adapter",
        "keep as targeted cold-start variant",
        "do not promote",
    }


def test_bootstrap_delta_summary_returns_paired_query_ci_rows():
    aggregated = pd.DataFrame(
        [
            _aggregated_row("warm", "global_popularity", "xgboost", 0.70, train_fraction=0.1),
            _aggregated_row("warm", "global_popularity", "tabpfn", 0.73, train_fraction=0.1),
            _aggregated_row("warm", "global_popularity", "tabpfn_native", 0.76, train_fraction=0.1),
            _aggregated_row("warm", "global_popularity", "xgboost", 0.79),
            _aggregated_row("warm", "global_popularity", "tabpfn", 0.80),
            _aggregated_row("warm", "global_popularity", "tabpfn_native", 0.82),
            _aggregated_row("item_cold", "global_popularity", "xgboost", 0.81),
            _aggregated_row("item_cold", "global_popularity", "tabpfn", 0.83),
            _aggregated_row("item_cold", "global_popularity", "tabpfn_native", 0.86),
            _aggregated_row("item_cold", "context_popularity", "xgboost", 0.78),
            _aggregated_row("item_cold", "context_popularity", "tabpfn", 0.84),
            _aggregated_row("item_cold", "context_popularity", "tabpfn_native", 0.87),
        ]
    )
    per_query = pd.concat(
        [
            _per_query_rows("warm", "global_popularity", "xgboost", [0.6, 0.7, 0.8], train_fraction=0.1),
            _per_query_rows("warm", "global_popularity", "tabpfn", [0.65, 0.74, 0.81], train_fraction=0.1),
            _per_query_rows("warm", "global_popularity", "tabpfn_native", [0.70, 0.78, 0.83], train_fraction=0.1),
            _per_query_rows("warm", "global_popularity", "xgboost", [0.7, 0.8, 0.9]),
            _per_query_rows("warm", "global_popularity", "tabpfn", [0.75, 0.82, 0.88]),
            _per_query_rows("warm", "global_popularity", "tabpfn_native", [0.80, 0.86, 0.91]),
            _per_query_rows("item_cold", "global_popularity", "xgboost", [0.77, 0.79, 0.82]),
            _per_query_rows("item_cold", "global_popularity", "tabpfn", [0.80, 0.83, 0.85]),
            _per_query_rows("item_cold", "global_popularity", "tabpfn_native", [0.84, 0.87, 0.88]),
            _per_query_rows("item_cold", "context_popularity", "xgboost", [0.73, 0.76, 0.80]),
            _per_query_rows("item_cold", "context_popularity", "tabpfn", [0.82, 0.84, 0.86]),
            _per_query_rows("item_cold", "context_popularity", "tabpfn_native", [0.85, 0.88, 0.90]),
        ],
        ignore_index=True,
    )

    bootstrap = compute_bootstrap_delta_summary(aggregated, per_query, replicates=100)

    assert not bootstrap.empty
    assert {"comparison", "ci_lower", "ci_upper", "mean_delta", "train_fraction"}.issubset(bootstrap.columns)
    assert (bootstrap["n_queries"] == 3).all()
    assert {0.1, 1.0}.issubset(set(bootstrap["train_fraction"].tolist()))


def test_k_sensitivity_preserves_fixed_caps_across_k(monkeypatch, tmp_path):
    calls = []

    def fake_unit_matrix(**kwargs):
        calls.append((kwargs["k"], kwargs["max_train_queries"], kwargs["max_test_queries"]))
        return pd.DataFrame(
            [
                {
                    "dataset": kwargs["dataset_name"],
                    "split_type": kwargs["split_type"],
                    "protocol": kwargs["protocols"][0],
                    "mode": kwargs["mode"],
                    "model": kwargs["models"][0],
                    "status": "ok",
                    "ndcg@10": 0.8,
                    "recall@10": 0.9,
                    "mrr": 0.8,
                    "hitrate@10": 0.9,
                    "runtime_seconds": 1.0,
                    "n_queries": 10.0,
                    "seed": kwargs["seed"],
                    "k": kwargs["k"],
                    "max_train_queries": kwargs["max_train_queries"],
                    "max_test_queries": kwargs["max_test_queries"],
                    "train_fraction": kwargs["train_fraction"],
                    "feature_set": kwargs.get("feature_set", "full"),
                    "tabpfn_version": "v2.5",
                }
            ]
        )

    monkeypatch.setattr("recpfn.phase2_pointwise_run._run_unit_matrix", fake_unit_matrix)
    monkeypatch.setattr("recpfn.phase2_pointwise_run.PRIMARY_SPLITS", ["warm"])
    monkeypatch.setattr("recpfn.phase2_pointwise_run.PRIMARY_PROTOCOLS", ["global_popularity"])

    tracker = ProgressTracker(total_units=3)
    results = _run_k_sensitivity(
        cache_dir="data",
        output_dir=tmp_path,
        dataset_name="synthetic",
        seeds=[0],
        k_values=[20, 50, 100],
        best_tree_by_slice={"warm::global_popularity": "xgboost"},
        max_train_queries=100,
        max_test_queries=100,
        tracker=tracker,
        timeout_seconds=10,
    )

    assert not results.empty
    assert calls == [(20, 100, 100), (50, 100, 100), (100, 100, 100)]


def test_phase2_plot_generation_smoke(tmp_path):
    aggregated = pd.DataFrame(
        [
            _aggregated_row("warm", "global_popularity", "xgboost", 0.81, train_fraction=0.1),
            _aggregated_row("warm", "global_popularity", "tabpfn", 0.80, train_fraction=0.1),
            _aggregated_row("warm", "global_popularity", "tabpfn_native", 0.82, train_fraction=0.1),
            _aggregated_row("warm", "global_popularity", "xgboost", 0.83, train_fraction=1.0),
            _aggregated_row("warm", "global_popularity", "tabpfn", 0.84, train_fraction=1.0),
            _aggregated_row("warm", "global_popularity", "tabpfn_native", 0.85, train_fraction=1.0),
            _aggregated_row("warm", "context_popularity", "xgboost", 0.75, train_fraction=1.0),
            _aggregated_row("warm", "context_popularity", "tabpfn", 0.76, train_fraction=1.0),
            _aggregated_row("warm", "context_popularity", "tabpfn_native", 0.74, train_fraction=1.0),
            _aggregated_row("item_cold", "global_popularity", "xgboost", 0.79, train_fraction=1.0),
            _aggregated_row("item_cold", "global_popularity", "tabpfn", 0.84, train_fraction=1.0),
            _aggregated_row("item_cold", "global_popularity", "tabpfn_native", 0.86, train_fraction=1.0),
            _aggregated_row("item_cold", "context_popularity", "xgboost", 0.78, train_fraction=1.0),
            _aggregated_row("item_cold", "context_popularity", "tabpfn", 0.85, train_fraction=1.0),
            _aggregated_row("item_cold", "context_popularity", "tabpfn_native", 0.87, train_fraction=1.0),
            _aggregated_row("warm", "global_popularity", "xgboost", 0.82, phase2_group="k_sensitivity", k=20),
            _aggregated_row("warm", "global_popularity", "tabpfn", 0.83, phase2_group="k_sensitivity", k=20),
            _aggregated_row("warm", "global_popularity", "tabpfn_native", 0.84, phase2_group="k_sensitivity", k=20),
            _aggregated_row("warm", "global_popularity", "xgboost", 0.80, phase2_group="k_sensitivity", k=50),
            _aggregated_row("warm", "global_popularity", "tabpfn", 0.82, phase2_group="k_sensitivity", k=50),
            _aggregated_row("warm", "global_popularity", "tabpfn_native", 0.83, phase2_group="k_sensitivity", k=50),
        ]
    )

    generate_phase2_plots(aggregated, pd.DataFrame(), tmp_path)

    for name in [
        "adapter_delta_by_train_fraction.png",
        "runtime_by_train_fraction.png",
        "metric_by_k.png",
        "native_minus_one_hot_by_slice.png",
        "best_tabpfn_vs_best_tree.png",
    ]:
        assert (tmp_path / name).exists()


def test_evaluate_phase2_outcome_promotes_native_adapter():
    aggregated = _phase2_key_aggregated(native_positive_slices=3, runtime_native=4.0, runtime_ohe=5.0)
    bootstrap = _phase2_bootstrap(positive_seed_count=2, train_fraction=1.0)

    decision = evaluate_phase2_outcome(aggregated, bootstrap, pd.DataFrame(), pd.DataFrame())

    assert decision["native_adapter_outcome"] == "promote primary adapter"


def test_evaluate_phase2_outcome_keeps_targeted_when_runtime_ratio_is_bad():
    aggregated = _phase2_key_aggregated(native_positive_slices=3, runtime_native=6.0, runtime_ohe=5.0)
    bootstrap = _phase2_bootstrap(positive_seed_count=2, train_fraction=1.0)

    decision = evaluate_phase2_outcome(aggregated, bootstrap, pd.DataFrame(), pd.DataFrame())

    assert decision["native_adapter_outcome"] == "keep as targeted cold-start variant"
    assert decision["median_runtime_ratio_native_vs_ohe"] > 1.1


def test_evaluate_phase2_outcome_does_not_promote_without_signal():
    aggregated = _phase2_key_aggregated(native_positive_slices=1, runtime_native=4.0, runtime_ohe=5.0)
    bootstrap = _phase2_bootstrap(positive_seed_count=0, train_fraction=1.0)

    decision = evaluate_phase2_outcome(aggregated, bootstrap, pd.DataFrame(), pd.DataFrame())

    assert decision["native_adapter_outcome"] == "do not promote"


def _legacy_build_features(dataset, candidates, split):
    users = dataset.users.drop_duplicates(subset=[USER_ID_COL]).set_index(USER_ID_COL)
    items = dataset.items.drop_duplicates(subset=[ITEM_ID_COL]).set_index(ITEM_ID_COL)
    train_interactions = split.train_interactions.copy().merge(
        items.add_prefix("item__"),
        left_on=ITEM_ID_COL,
        right_index=True,
        how="left",
    )
    popularity = item_training_popularity(split.train_interactions)
    item_positive_rate = split.train_interactions.groupby(ITEM_ID_COL)[LABEL_COL].mean().rename("item_positive_rate")
    genre_cols = [column for column in items.columns if str(column).startswith("genre_")]
    rows = []
    for _, query_group in candidates.groupby(QUERY_ID_COL, sort=False):
        query_row = query_group.iloc[0]
        user_id = query_row[USER_ID_COL]
        query_timestamp = query_row["query_timestamp"]
        query_interaction_id = query_row["query_interaction_id"]
        user_history = train_interactions[
            (train_interactions[USER_ID_COL] == user_id)
            & (
                (train_interactions[TIMESTAMP_COL] < query_timestamp)
                | (
                    (train_interactions[TIMESTAMP_COL] == query_timestamp)
                    & (train_interactions[INTERACTION_ID_COL] < query_interaction_id)
                )
            )
        ].copy()
        user_meta = users.loc[user_id] if user_id in users.index else pd.Series(dtype=object)
        history_summary = summarize_history(user_history)
        history_categories = user_history.get(f"item__{dataset.context_col}", pd.Series(dtype=object))
        history_brands = user_history.get("item__brand", pd.Series(dtype=object))
        user_avg_history_price = numeric_average(user_history, "item__price")
        days_since_last = recency_in_days(user_history, query_timestamp)
        base_features = {
            **history_summary,
            "days_since_last_interaction": days_since_last,
            "user_age": _coerce_numeric(user_meta.get("age"), default=0.0),
            "user_gender": str(user_meta.get("gender", "unknown")),
            "user_occupation": str(user_meta.get("occupation", "unknown")),
            "user_avg_history_price": user_avg_history_price,
        }
        for candidate in query_group.itertuples(index=False):
            item_id = candidate.item_id
            item_meta = items.loc[item_id] if item_id in items.index else pd.Series(dtype=object)
            row = candidate._asdict()
            row.update(base_features)
            row["item_primary_category"] = str(item_meta.get(dataset.context_col, "unknown"))
            row["item_brand"] = str(item_meta.get("brand", "unknown"))
            row["item_price"] = _coerce_numeric(item_meta.get("price"), default=0.0)
            row["item_release_year"] = _coerce_numeric(item_meta.get("release_year"), default=0.0)
            row["item_genre_count"] = _coerce_numeric(item_meta.get("genre_count"), default=0.0)
            row["item_category_depth"] = _coerce_numeric(item_meta.get("category_depth"), default=0.0)
            row["item_popularity"] = float(popularity.get(item_id, 0.0))
            row["item_positive_rate"] = float(item_positive_rate.get(item_id, 0.0))
            row["item_is_cold"] = float(item_id in split.cold_item_ids)
            row["price_missing"] = float(pd.isna(item_meta.get("price")))
            row["category_affinity"] = safe_affinity_count(history_categories, item_meta.get(dataset.context_col))
            row["brand_affinity"] = safe_affinity_count(history_brands, item_meta.get("brand"))
            row["same_item_history"] = float((user_history[ITEM_ID_COL] == item_id).sum()) if not user_history.empty else 0.0
            row["price_distance_to_user_avg"] = abs(row["item_price"] - row["user_avg_history_price"])
            row["history_positive_same_category"] = (
                float(
                    (
                        (history_categories.astype(str) == str(item_meta.get(dataset.context_col, "unknown")))
                        & (user_history[LABEL_COL] == 1)
                    ).sum()
                )
                if not user_history.empty and dataset.context_col in item_meta.index
                else 0.0
            )
            for genre_col in genre_cols:
                row[f"feat_{genre_col}"] = _coerce_numeric(item_meta.get(genre_col), default=0.0)
            rows.append(row)
    return pd.DataFrame(rows).fillna(
        {
            "user_gender": "unknown",
            "user_occupation": "unknown",
            "item_primary_category": "unknown",
            "item_brand": "unknown",
        }
    )


def _coerce_numeric(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _result_row(
    *,
    model: str,
    seed: int,
    split_type: str,
    protocol: str,
    ndcg: float,
    dataset: str = "movielens_100k",
    phase2_group: str = "primary",
    train_fraction: float = 1.0,
    k: int = 20,
    feature_set: str = "full",
) -> dict[str, object]:
    return {
        "phase2_group": phase2_group,
        "dataset": dataset,
        "split_type": split_type,
        "protocol": protocol,
        "mode": "pointwise",
        "model": model,
        "status": "ok",
        "seed": seed,
        "train_fraction": train_fraction,
        "k": k,
        "max_train_queries": 100,
        "max_test_queries": 100,
        "feature_set": feature_set,
        "tabpfn_version": "v2.5",
        "ndcg@10": ndcg,
        "recall@10": 0.90,
        "mrr": 0.80,
        "hitrate@10": 0.90,
        "runtime_seconds": 5.0,
        "n_queries": 2.0,
    }


def _prediction_frame(result_row: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            QUERY_ID_COL: "q1",
            "item_id": "i1",
            LABEL_COL: 1,
            "score": 0.9,
            "dataset": result_row["dataset"],
            "split_type": result_row["split_type"],
            "protocol": result_row["protocol"],
            "model": result_row["model"],
            "mode": result_row["mode"],
            "seed": result_row["seed"],
            "k": result_row["k"],
            "train_fraction": result_row["train_fraction"],
            "feature_set": result_row["feature_set"],
            "max_train_queries": result_row["max_train_queries"],
            "max_test_queries": result_row["max_test_queries"],
            "tabpfn_version": result_row["tabpfn_version"],
        },
        {
            QUERY_ID_COL: "q1",
            "item_id": "i2",
            LABEL_COL: 0,
            "score": 0.2,
            "dataset": result_row["dataset"],
            "split_type": result_row["split_type"],
            "protocol": result_row["protocol"],
            "model": result_row["model"],
            "mode": result_row["mode"],
            "seed": result_row["seed"],
            "k": result_row["k"],
            "train_fraction": result_row["train_fraction"],
            "feature_set": result_row["feature_set"],
            "max_train_queries": result_row["max_train_queries"],
            "max_test_queries": result_row["max_test_queries"],
            "tabpfn_version": result_row["tabpfn_version"],
        },
    ]
    return pd.DataFrame(rows)


def _aggregated_row(
    split_type: str,
    protocol: str,
    model: str,
    ndcg: float,
    *,
    dataset: str = "movielens_100k",
    phase2_group: str = "primary",
    train_fraction: float = 1.0,
    k: int = 20,
    runtime_seconds_median: float | None = None,
) -> dict[str, object]:
    return {
        "phase2_group": phase2_group,
        "dataset": dataset,
        "split_type": split_type,
        "protocol": protocol,
        "mode": "pointwise",
        "model": model,
        "k": k,
        "max_train_queries": 100,
        "max_test_queries": 100,
        "train_fraction": train_fraction,
        "feature_set": "full",
        "tabpfn_version": "v2.5",
        "seed_count": 3,
        "ndcg@10_mean": ndcg,
        "ndcg@10_std": 0.01,
        "recall@10_mean": 0.9,
        "recall@10_std": 0.01,
        "mrr_mean": 0.8,
        "mrr_std": 0.01,
        "hitrate@10_mean": 0.9,
        "hitrate@10_std": 0.01,
        "runtime_seconds_median": runtime_seconds_median
        if runtime_seconds_median is not None
        else (5.0 if model == "tabpfn" else 4.0),
        "n_queries_median": 100.0,
    }


def _per_query_rows(
    split_type: str,
    protocol: str,
    model: str,
    ndcgs: list[float],
    train_fraction: float = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            QUERY_ID_COL: [f"q{i}" for i in range(len(ndcgs))],
            "ndcg@10": ndcgs,
            "dataset": "movielens_100k",
            "split_type": split_type,
            "protocol": protocol,
            "model": model,
            "mode": "pointwise",
            "seed": 0,
            "k": 20,
            "train_fraction": train_fraction,
            "feature_set": "full",
            "max_train_queries": 100,
            "max_test_queries": 100,
            "tabpfn_version": "v2.5",
            "phase2_group": "primary",
        }
    )


def _phase2_key_aggregated(native_positive_slices: int, runtime_native: float, runtime_ohe: float) -> pd.DataFrame:
    rows = []
    ordered_slices = [
        ("warm", "global_popularity"),
        ("warm", "context_popularity"),
        ("item_cold", "global_popularity"),
        ("item_cold", "context_popularity"),
    ]
    for index, (split_type, protocol) in enumerate(ordered_slices):
        native_beats = index < native_positive_slices
        native_ndcg = 0.86 if native_beats else 0.80
        ohe_ndcg = 0.84 if native_beats else 0.82
        tree_ndcg = 0.85 if protocol == "global_popularity" else 0.83
        rows.extend(
            [
                _aggregated_row(split_type, protocol, "xgboost", tree_ndcg, runtime_seconds_median=1.0),
                _aggregated_row(split_type, protocol, "tabpfn", ohe_ndcg, runtime_seconds_median=runtime_ohe),
                _aggregated_row(
                    split_type,
                    protocol,
                    "tabpfn_native",
                    native_ndcg,
                    runtime_seconds_median=runtime_native,
                ),
            ]
        )
    return pd.DataFrame(rows)


def _phase2_bootstrap(positive_seed_count: int, train_fraction: float) -> pd.DataFrame:
    rows = []
    for protocol in ["global_popularity", "context_popularity"]:
        for seed in [0, 1, 2]:
            positive = protocol == "context_popularity" and seed < positive_seed_count
            rows.append(
                {
                    "dataset": "movielens_100k",
                    "split_type": "item_cold",
                    "protocol": protocol,
                    "seed": seed,
                    "train_fraction": train_fraction,
                    "comparison": "tabpfn_native_minus_tabpfn",
                    "left_model": "tabpfn_native",
                    "right_model": "tabpfn",
                    "metric": "ndcg@10",
                    "n_queries": 100,
                    "mean_delta": 0.02 if positive else -0.01,
                    "ci_lower": 0.005 if positive else -0.02,
                    "ci_upper": 0.03 if positive else 0.005,
                    "ci_excludes_zero_positive": positive,
                }
            )
    return pd.DataFrame(rows)
