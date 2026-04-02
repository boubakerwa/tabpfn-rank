import pandas as pd

from recpfn.phase1_decision import evaluate_decision_outcome, load_existing_phase1_results, snapshot_status


def test_decision_outcome_drills_further_on_meaningful_tabpfn_pairwise_gain():
    results = pd.DataFrame(
        [
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "tabpfn", 0.81, 0.90, 0.80, 8.0),
            _row("movielens_100k", "warm", "global_popularity", "pairwise", "tabpfn", 0.84, 0.91, 0.82, 10.0),
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "xgboost", 0.86, 0.92, 0.84, 1.0),
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "catboost", 0.85, 0.91, 0.83, 1.2),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "xgboost", 0.99, 0.99, 0.98, 1.0),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "tabpfn", 0.99, 0.99, 0.98, 4.0),
        ]
    )

    decision = evaluate_decision_outcome(results, overlap_notes=[])

    assert decision["outcome"] == "drill further"


def test_decision_outcome_requests_tie_break_when_amazon_is_saturated_and_movielens_is_close():
    results = pd.DataFrame(
        [
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "tabpfn", 0.84, 0.90, 0.83, 8.0),
            _row("movielens_100k", "warm", "global_popularity", "pairwise", "tabpfn", 0.85, 0.90, 0.84, 10.0),
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "xgboost", 0.86, 0.91, 0.84, 1.0),
            _row("movielens_100k", "item_cold", "global_popularity", "pointwise", "tabpfn", 0.82, 0.89, 0.80, 8.5),
            _row("movielens_100k", "item_cold", "global_popularity", "pointwise", "xgboost", 0.84, 0.90, 0.82, 1.1),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "xgboost", 0.99, 0.99, 0.98, 1.0),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "catboost", 0.99, 0.99, 0.98, 1.1),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "tabpfn", 0.99, 0.99, 0.98, 4.5),
            _row("amazon_baby_products", "item_cold", "global_popularity", "pointwise", "xgboost", 0.99, 0.99, 0.98, 1.0),
            _row("amazon_baby_products", "item_cold", "global_popularity", "pointwise", "tabpfn", 0.99, 0.99, 0.98, 4.3),
        ]
    )

    decision = evaluate_decision_outcome(results, overlap_notes=[])

    assert decision["outcome"] == "run one tie-break sweep"


def test_decision_outcome_stops_after_uninformative_tie_break():
    results = pd.DataFrame(
        [
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "tabpfn", 0.79, 0.88, 0.78, 9.0, phase="tie_break", train_fraction=0.1),
            _row("movielens_100k", "warm", "global_popularity", "pairwise", "tabpfn", 0.80, 0.88, 0.79, 10.5, phase="tie_break", train_fraction=0.1),
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "xgboost", 0.85, 0.91, 0.83, 1.0, phase="tie_break", train_fraction=0.1),
            _row("movielens_100k", "item_cold", "global_popularity", "pointwise", "tabpfn", 0.77, 0.86, 0.76, 9.1, phase="tie_break", train_fraction=0.5),
            _row("movielens_100k", "item_cold", "global_popularity", "pointwise", "catboost", 0.84, 0.90, 0.82, 1.2, phase="tie_break", train_fraction=0.5),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "xgboost", 0.99, 0.99, 0.98, 1.0),
            _row("amazon_baby_products", "warm", "global_popularity", "pointwise", "tabpfn", 0.99, 0.99, 0.98, 4.2),
        ]
    )

    decision = evaluate_decision_outcome(results, overlap_notes=[])

    assert decision["outcome"] == "stop at benchmark"


def test_load_existing_phase1_results_marks_phase_and_snapshot(tmp_path):
    result_path = (
        tmp_path
        / "canonical_pointwise"
        / "movielens_100k__warm__global_popularity__pointwise__tabpfn"
        / "movielens_100k"
        / "warm"
        / "results.csv"
    )
    result_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _row("movielens_100k", "warm", "global_popularity", "pointwise", "tabpfn", 0.83, 0.90, 0.81, 20.0)
        ]
    ).drop(columns=["phase", "train_fraction"]).to_csv(result_path, index=False)

    results = load_existing_phase1_results(tmp_path)
    snapshot = snapshot_status(results)

    assert results.iloc[0]["phase"] == "canonical"
    assert snapshot["completed_units"] == 1
    assert snapshot["expected_units"] == 64
    assert snapshot["missing_units_count"] == 63


def test_load_existing_phase1_results_prefers_phase_column_over_path(tmp_path):
    result_path = (
        tmp_path
        / "canonical_pointwise"
        / "movielens_100k__warm__global_popularity__pointwise__tabpfn"
        / "movielens_100k"
        / "warm"
        / "results.csv"
    )
    result_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _row(
                "movielens_100k",
                "warm",
                "global_popularity",
                "pointwise",
                "tabpfn",
                0.83,
                0.90,
                0.81,
                20.0,
                phase="tie_break",
                train_fraction=0.5,
            )
        ]
    ).to_csv(result_path, index=False)

    results = load_existing_phase1_results(tmp_path)

    assert results.iloc[0]["phase"] == "tie_break"
    assert results.iloc[0]["train_fraction"] == 0.5


def _row(
    dataset: str,
    split_type: str,
    protocol: str,
    mode: str,
    model: str,
    ndcg: float,
    recall: float,
    mrr: float,
    runtime: float,
    phase: str = "canonical",
    train_fraction: float | None = None,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split_type": split_type,
        "protocol": protocol,
        "mode": mode,
        "model": model,
        "status": "ok",
        "ndcg@10": ndcg,
        "recall@10": recall,
        "mrr": mrr,
        "hitrate@10": recall,
        "runtime_seconds": runtime,
        "phase": phase,
        "train_fraction": train_fraction,
        "k": 20,
        "max_train_queries": 100,
        "max_test_queries": 100,
        "seed": 0,
        "tabpfn_version": "v2.5",
        "n_queries": 100.0,
    }
