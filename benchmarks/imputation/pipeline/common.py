"""Shared paths, gene alignment, masking, and metrics for TCGA imputation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]
WORK = HERE / "work"
RESULTS = HERE / "results"
CONFIG = json.loads((HERE / "config.json").read_text())


def norm_gene(value: object) -> str:
    return str(value).strip().split(".")[0].upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mask(sample_ids: list[str], genes: list[str], ratio: float, seed: int) -> np.ndarray:
    """Create an exact-size deterministic mask for every sample.

    Hash-derived random streams make shared-vocabulary masks independent of
    model ordering while remaining identical across model adapters.
    """
    count = max(1, int(round(len(genes) * ratio)))
    mask = np.zeros((len(sample_ids), len(genes)), dtype=bool)
    for row, sample_id in enumerate(sample_ids):
        token = f"{seed}|{ratio:.8f}|{sample_id}".encode()
        row_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        chosen = np.random.default_rng(row_seed).choice(len(genes), count, replace=False)
        mask[row, chosen] = True
    return mask


def row_metrics(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson, Spearman, and MSE for equally shaped sample-by-masked arrays."""
    left = truth - truth.mean(axis=1, keepdims=True)
    right = prediction - prediction.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    pearson = np.divide((left * right).sum(axis=1), denom,
                        out=np.full(len(truth), np.nan), where=denom > 0)
    spearman = np.empty(len(truth), dtype=np.float64)
    for row in range(len(truth)):
        a, b = rankdata(truth[row]), rankdata(prediction[row])
        ac, bc = a - a.mean(), b - b.mean()
        d = np.linalg.norm(ac) * np.linalg.norm(bc)
        spearman[row] = np.dot(ac, bc) / d if d else np.nan
    mse = np.mean((truth - prediction) ** 2, axis=1)
    return pearson, spearman, mse


def tpm_log1p(counts: np.ndarray, lengths_bp: np.ndarray) -> np.ndarray:
    lengths_kb = np.asarray(lengths_bp, dtype=np.float64) / 1000.0
    rate = np.asarray(counts, dtype=np.float64) / lengths_kb[None, :]
    totals = rate.sum(axis=1, keepdims=True)
    tpm = np.divide(rate, totals, out=np.zeros_like(rate), where=totals > 0) * 1e6
    return np.log1p(tpm).astype(np.float32)
