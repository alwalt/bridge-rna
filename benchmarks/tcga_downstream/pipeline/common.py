from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]
RESULTS = HERE / "results"
WORK = HERE / "work"
CONFIG = json.loads((HERE / "config.json").read_text())

RESULTS.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
