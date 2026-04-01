from pathlib import Path

import pandas as pd

from recpfn.phase1_low_data import run_phase1_low_data_step


def test_run_phase1_low_data_step_merges_baseline_and_low_data(monkeypatch, tmp_path):
    baseline_root = (
        tmp_path
        / "baseline_runs"
        / "canonical_pointwise"
        / "movielens_100k__warm__global_popularity__pointwise__tabpfn"
        / "movielens_100k"
        / "warm"
    )
    baseline_root.mkdir(parents=True)
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
            )
        ]
    ).drop(columns=["phase", "train_fraction"]).to_csv(baseline_root / "results.csv", index=False)

    monkeypatch.setattr(
        "recpfn.phase1_low_data.summarize_protocol_overlap",
        lambda **_: {
            "dataset": "movielens_100k",
            "split_type": "warm",
            "n_queries": 10,
            "identical_rate": 0.2,
            "mean_jaccard": 0.4,
            "collapsed": False,
        },
    )
    monkeypatch.setattr(
        "recpfn.phase1_low_data.run_movie_lens_tie_break",
        lambda **_: pd.DataFrame(
            [
                _row(
                    "movielens_100k",
                    "warm",
                    "global_popularity",
                    "pointwise",
                    "tabpfn",
                    0.905,
                    0.92,
                    0.90,
                    18.0,
                    phase="tie_break",
                    train_fraction=0.1,
                ),
                _row(
                    "movielens_100k",
                    "warm",
                    "global_popularity",
                    "pointwise",
                    "xgboost",
                    0.91,
                    0.93,
                    0.90,
                    1.0,
                    phase="tie_break",
                    train_fraction=0.1,
                ),
            ]
        ),
    )
    monkeypatch.setattr("recpfn.phase1_low_data._generate_plots_from_summary", lambda *args, **kwargs: None)

    merged, artifacts, decision = run_phase1_low_data_step(
        cache_dir=tmp_path,
        baseline_run_output_dir=tmp_path / "baseline_runs",
        low_data_run_output_dir=tmp_path / "low_data_runs",
        output_dir=tmp_path / "phase1_low_data",
        plots_output_dir=tmp_path / "plots",
    )

    assert len(merged) == 3
    assert decision["outcome"] == "drill further"
    assert artifacts.baseline_results_path and artifacts.baseline_results_path.exists()
    assert artifacts.tie_break_results_path and artifacts.tie_break_results_path.exists()
    assert artifacts.merged_results_path and artifacts.merged_results_path.exists()
    assert artifacts.decision_memo_path and artifacts.decision_memo_path.exists()
    assert artifacts.next_steps_plan_path and artifacts.next_steps_plan_path.exists()


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
