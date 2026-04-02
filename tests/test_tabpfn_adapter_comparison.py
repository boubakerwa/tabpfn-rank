from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path("/Users/wassimboubaker/TabPFN/experiments/plot_tabpfn_adapter_comparison.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("plot_tabpfn_adapter_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_adapter_comparison_plotting_runs_on_movie_lens_low_data(tmp_path):
    module = _load_script_module()
    summary_csv = Path("/Users/wassimboubaker/TabPFN/paper/results_tabpfn_native_low_data/movielens_pointwise_low_data_summary.csv")
    summary = module.load_summary_csv(summary_csv)
    comparison = module.prepare_comparison_frame(summary)

    paths = module.generate_comparison_plots(comparison, output_dir=tmp_path, metric="ndcg@10", suffix=" (test)")
    html_path = module.write_html_index(comparison, paths, tmp_path, suffix=" (test)", metric="ndcg@10")

    assert len(paths) == 3
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
    assert html_path.exists()
    assert html_path.name == "index.html"


def test_adapter_comparison_frame_keeps_only_known_models():
    module = _load_script_module()
    frame = pd.DataFrame(
        {
            "split_type": ["warm", "warm", "warm"],
            "protocol": ["global_popularity", "global_popularity", "global_popularity"],
            "model": ["tabpfn", "tabpfn_native", "xgboost"],
            "status": ["ok", "ok", "ok"],
            "ndcg@10": [0.5, 0.6, 0.7],
            "runtime_seconds": [1.0, 1.1, 1.2],
        }
    )

    comparison = module.prepare_comparison_frame(frame)

    assert list(comparison["model"].astype(str)) == ["tabpfn", "tabpfn_native"]
