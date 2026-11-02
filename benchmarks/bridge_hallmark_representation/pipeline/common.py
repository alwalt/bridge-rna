"""Shared, benchmark-local utilities. No prior pathway-head code is imported."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
RESULTS, WORK = HERE / "results", HERE / "work"
CONFIG = json.loads((HERE / "config.json").read_text())

def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (HERE / p).resolve()

def sha256(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""): h.update(block)
    return h.hexdigest()

def bh_fdr(pvalues):
    p = np.asarray(pvalues, float); shape = p.shape; flat = p.ravel()
    valid = np.isfinite(flat); out = np.full(flat.size, np.nan)
    q = flat[valid]; order = np.argsort(q); ranked = q[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked)+1))[::-1])[::-1]
    tmp = np.empty_like(adjusted); tmp[order] = np.minimum(adjusted, 1.0); out[valid] = tmp
    return out.reshape(shape)

def read_vocab() -> list[str]:
    frame = pd.read_csv(resolve(CONFIG["canonical_genes"]))
    for col in ("human_gene", "gene", "gene_symbol", "symbol", "human_symbol"):
        if col in frame: return frame[col].astype(str).str.upper().tolist()
    raise ValueError(f"Cannot identify human gene column: {list(frame)}")

def read_gmt() -> dict[str, set[str]]:
    sets = {}
    with resolve(CONFIG["hallmark_gmt"]).open() as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) >= 3: sets[fields[0]] = {x.upper() for x in fields[2:] if x}
    if len(sets) != 50: raise AssertionError(f"Expected 50 Hallmarks, found {len(sets)}")
    return dict(sorted(sets.items()))

def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")

def assert_matrix(x, rows: int | None, cols: int, name: str):
    if x.ndim != 2 or x.shape[1] != cols or (rows is not None and x.shape[0] != rows):
        raise AssertionError(f"{name}: unexpected shape {x.shape}")
    if not np.isfinite(x).all(): raise AssertionError(f"{name}: non-finite values")
