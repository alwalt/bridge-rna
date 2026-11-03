from __future__ import annotations

import hashlib
import json
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
REPO = BENCHMARK.parents[1]
SHARED = Path("/home/walt/bridge-rna")
# Compatibility name used by the shared, read-only model adapters. Model assets
# deliberately resolve in the shared checkout rather than this worktree.
REPO_ROOT = SHARED
RESULTS = BENCHMARK / "results"
WORK = BENCHMARK / "work"
CONFIG = json.loads((BENCHMARK / "config.json").read_text())
RESULTS.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
