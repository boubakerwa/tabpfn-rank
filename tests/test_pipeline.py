from recpfn.rerank.pipeline import run_experiment
from recpfn.models.tabpfn_pointwise import _resolve_tabpfn_version


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


def test_tabpfn_pipeline_smoke_runs_when_dependency_is_available(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("tabpfn")

    results, _ = run_experiment(
        dataset_name="synthetic",
        split_type="warm",
        protocols=["global_popularity"],
        pointwise_models=["tabpfn"],
        pairwise_models=[],
        output_dir=tmp_path,
        k=4,
        seed=0,
    )

    assert set(results["status"]) == {"ok"}


def test_tabpfn_version_parser_supports_v2_and_v25():
    class DummyVersion:
        V2 = "v2"
        V2_5 = "v2.5"

    assert _resolve_tabpfn_version(DummyVersion, "v2") == "v2"
    assert _resolve_tabpfn_version(DummyVersion, "2") == "v2"
    assert _resolve_tabpfn_version(DummyVersion, "v2.5") == "v2.5"


def test_pipeline_records_run_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("RECPFN_TABPFN_VERSION", "v2.5")

    results, _ = run_experiment(
        dataset_name="synthetic",
        split_type="warm",
        protocols=["global_popularity"],
        pointwise_models=["popularity"],
        pairwise_models=[],
        output_dir=tmp_path,
        k=4,
        seed=7,
        max_train_queries=2,
        max_test_queries=1,
    )

    assert {"tabpfn_version", "k", "max_train_queries", "max_test_queries", "seed"}.issubset(results.columns)
    row = results.iloc[0]
    assert row["tabpfn_version"] == "v2.5"
    assert row["k"] == 4
    assert row["max_train_queries"] == 2
    assert row["max_test_queries"] == 1
    assert row["seed"] == 7
