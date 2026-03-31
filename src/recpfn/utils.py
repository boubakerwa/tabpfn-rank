"""Utility helpers shared across the project."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stable_seed(*parts: object) -> int:
    """Create a deterministic integer seed from arbitrary values."""

    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return int(digest[:8], 16)


def deterministic_random(*parts: object) -> random.Random:
    """Return a Random instance with a stable seed."""

    return random.Random(stable_seed(*parts))


def maybe_download(url: str, destination: str | Path) -> Path:
    """Download a file only if it is not already present."""

    destination_path = Path(destination)
    if destination_path.exists():
        return destination_path

    ensure_dir(destination_path.parent)
    with urllib.request.urlopen(url) as response, destination_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination_path


def unzip_file(archive_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a zip file if the destination folder is empty."""

    archive = Path(archive_path)
    target = ensure_dir(output_dir)
    if any(target.iterdir()):
        return target

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    return target


def iter_jsonl_gz(path: str | Path) -> Iterable[dict]:
    """Yield objects from a gzipped JSONL file."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def maybe_download_jsonl_gz_prefix(url: str, destination: str | Path, max_lines: int) -> Path:
    """Download and gzip only the first N JSONL records from a remote gzip file."""

    destination_path = Path(destination)
    if destination_path.exists():
        return destination_path

    ensure_dir(destination_path.parent)
    with urllib.request.urlopen(url) as response, gzip.GzipFile(fileobj=response) as gz_in, gzip.open(
        destination_path, "wt", encoding="utf-8"
    ) as gz_out:
        for index, raw_line in enumerate(gz_in):
            if index >= max_lines:
                break
            gz_out.write(raw_line.decode("utf-8"))
    return destination_path


def read_env_int(name: str, default: int | None = None) -> int | None:
    """Read an integer environment variable if available."""

    value = os.getenv(name)
    if value is None:
        return default
    return int(value)
