"""Paths and configuration for the landmark-gene sufficiency benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results"
WORK = HERE / "work"
REFERENCES = HERE / "references"
CONFIG = json.loads((HERE / "config.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
