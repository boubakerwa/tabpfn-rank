"""Shared subprocess runner helpers for benchmark sweeps."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from recpfn.rerank.pipeline import _failure_row
from recpfn.utils import ensure_dir, read_env_str

DEFAULT_UNIT_TIMEOUT_SECONDS = 1800


@dataclass
class ProgressTracker:
    """Simple terminal progress reporter for long benchmark sweeps."""

    total_units: int
    started_units: int = 0
    finished_units: int = 0

    def announce_start(self, label: str) -> int:
        self.started_units += 1
        unit_no = self.started_units
        print(f"[{unit_no}/{self.total_units}] Starting {label}", flush=True)
        return unit_no

    def announce_finish(self, unit_no: int, label: str, elapsed_seconds: float, status: str) -> None:
        self.finished_units += 1
        print(
            f"[{unit_no}/{self.total_units}] {status} {label} in {elapsed_seconds:.1f}s",
            flush=True,
        )

    def extend(self, additional_units: int) -> None:
        self.total_units += additional_units


def concat_results(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty result frames."""

    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def run_unit_matrix(
    dataset_name: str,
    split_type: str,
    protocols: list[str],
    models: list[str],
    mode: str,
    cache_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    k: int,
    max_train_queries: int | None,
    max_test_queries: int | None,
    tracker: ProgressTracker,
    timeout_seconds: int = DEFAULT_UNIT_TIMEOUT_SECONDS,
    train_fraction: float | None = None,
    feature_set: str = "full",
) -> pd.DataFrame:
    """Run one matrix of benchmark units through isolated subprocesses."""

    frames = []
    for protocol in protocols:
        for model in models:
            frames.append(
                run_unit_subprocess(
                    dataset_name=dataset_name,
                    split_type=split_type,
                    protocol=protocol,
                    mode=mode,
                    model=model,
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    seed=seed,
                    k=k,
                    max_train_queries=max_train_queries,
                    max_test_queries=max_test_queries,
                    tracker=tracker,
                    timeout_seconds=timeout_seconds,
                    train_fraction=train_fraction,
                    feature_set=feature_set,
                )
            )
    return concat_results(frames)


def run_unit_subprocess(
    dataset_name: str,
    split_type: str,
    protocol: str,
    mode: str,
    model: str,
    cache_dir: str | Path,
    output_dir: str | Path,
    seed: int,
    k: int,
    max_train_queries: int | None,
    max_test_queries: int | None,
    tracker: ProgressTracker,
    timeout_seconds: int = DEFAULT_UNIT_TIMEOUT_SECONDS,
    train_fraction: float | None = None,
    feature_set: str = "full",
) -> pd.DataFrame:
    """Run one benchmark unit in a fresh subprocess and return its result row(s)."""

    unit_name = f"{dataset_name}__{split_type}__{protocol}__{mode}__{model}".replace(".", "_")
    unit_output_dir = ensure_dir(Path(output_dir) / unit_name)
    log_path = unit_output_dir / "run.log"
    label = f"{dataset_name} {split_type} {protocol} {mode} {model}"
    unit_no = tracker.announce_start(label)
    started_at = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "recpfn.unit_runner",
        "--dataset",
        dataset_name,
        "--split",
        split_type,
        "--protocol",
        protocol,
        "--mode",
        mode,
        "--model",
        model,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(unit_output_dir),
        "--seed",
        str(seed),
        "--k",
        str(k),
    ]
    if max_train_queries is not None:
        command.extend(["--max-train-queries", str(max_train_queries)])
    if max_test_queries is not None:
        command.extend(["--max-test-queries", str(max_test_queries)])
    if train_fraction is not None:
        command.extend(["--train-fraction", str(train_fraction)])
    if feature_set != "full":
        command.extend(["--feature-set", feature_set])

    try:
        with log_path.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command,
                check=True,
                timeout=timeout_seconds,
                cwd=Path(__file__).resolve().parents[2],
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        result_path = unit_output_dir / dataset_name / split_type / "results.csv"
        frame = pd.read_csv(result_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        tracker.announce_finish(
            unit_no,
            f"{label} (see {log_path})",
            time.perf_counter() - started_at,
            status="FAILED",
        )
        return pd.DataFrame(
            [
                _failure_row(
                    dataset_name,
                    split_type,
                    protocol,
                    model,
                    mode,
                    exc if isinstance(exc, Exception) else RuntimeError(str(exc)),
                    run_metadata={
                        "tabpfn_version": read_env_str("RECPFN_TABPFN_VERSION", "v2"),
                        "k": int(k),
                        "max_train_queries": max_train_queries,
                        "max_test_queries": max_test_queries,
                        "seed": int(seed),
                        "train_fraction": train_fraction,
                        "feature_set": feature_set,
                    },
                )
            ]
        )

    tracker.announce_finish(unit_no, label, time.perf_counter() - started_at, status="Done")
    return frame


@contextmanager
def temporary_env(name: str, value: str) -> Iterator[None]:
    """Temporarily override one environment variable."""

    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
