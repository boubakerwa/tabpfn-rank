from recpfn.rerank.pipeline import run_experiment


def test_pipeline_smoke_runs_on_synthetic_dataset(tmp_path):
    results, artifacts = run_experiment(
        dataset_name="synthetic",
        split_type="warm",
        protocols=["global_popularity"],
        pointwise_models=["popularity", "sklearn_logreg"],
        pairwise_models=["sklearn_logreg"],
        output_dir=tmp_path,
        k=4,
        seed=0,
    )

    assert not results.empty
    assert {"pointwise", "pairwise"}.issubset(set(results["mode"]))
    assert artifacts.benchmark_table_path is not None
