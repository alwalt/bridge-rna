from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RESULTS, WORK = HERE / "results", HERE / "work"
CONFIG = json.loads((HERE / "config.json").read_text())

def sha256(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()

def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def apply_probe(z: np.ndarray, probe: dict) -> np.ndarray:
    scale = np.asarray(probe["scale"], np.float32)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0): raise ValueError("invalid probe scale")
    return ((np.asarray(z, np.float32) - probe["mean"]) / scale) @ probe["weight"].T + probe["bias"]

def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0) or np.any(~np.isfinite(n)): raise ValueError("zero/nonfinite row norm")
    return x / n

