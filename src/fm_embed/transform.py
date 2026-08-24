"""Gene-vocab alignment and normalization, matching model training assumptions."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .vocab import norm_gene

PreprocessMode = str  # one of: raw, cpm, tpm, log1p_raw, log1p_cpm, log1p_tpm


def align_to_vocab(
    matrix: pd.DataFrame,
    canonical_genes: Sequence[str],
    genes_are_columns: bool = True,
) -> np.ndarray:
    """Reindex an expression matrix to the canonical gene order.

    `matrix` must be indexed by sample id, with genes either as columns
    (genes_are_columns=True) or as the index (genes_are_columns=False, in
    which case `matrix` is gene x sample and gets transposed first).
    Missing genes are zero-filled; extra genes are dropped.
    """
    if not genes_are_columns:
        matrix = matrix.T

    gene_to_column: Dict[str, str] = {norm_gene(c): c for c in matrix.columns}
    out = np.zeros((matrix.shape[0], len(canonical_genes)), dtype=np.float32)
    for idx, gene in enumerate(canonical_genes):
        col = gene_to_column.get(gene)
        if col is not None:
            out[:, idx] = matrix[col].to_numpy(dtype=np.float32)
    return out


def apply_preprocessing(counts: np.ndarray, mode: PreprocessMode = "log1p_tpm", eps: float = 1e-12) -> np.ndarray:
    """Apply the requested normalization to a [n_samples, n_genes] matrix.

    - raw:        counts as-is
    - cpm:        counts / library_size * 1e6
    - tpm:        counts, assumed already TPM-normalized upstream (no-op)
    - log1p_raw:  log1p(counts)
    - log1p_cpm:  log1p(cpm)
    - log1p_tpm:  log1p(counts), assumed already TPM-normalized upstream (model default)
    """
    mode = mode.lower().strip()
    counts = counts.astype(np.float32)

    if mode == "raw":
        return counts
    if mode == "tpm":
        return counts
    if mode in ("cpm", "log1p_cpm"):
        lib_size = counts.sum(axis=1, keepdims=True) + eps
        cpm = counts / lib_size * 1e6
        return np.log1p(cpm) if mode == "log1p_cpm" else cpm
    if mode == "log1p_raw":
        return np.log1p(counts)
    if mode == "log1p_tpm":
        return np.log1p(counts + eps)
    raise ValueError(f"Unknown preprocessing mode: {mode}")
