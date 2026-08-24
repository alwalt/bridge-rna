"""GTEx source adapter.

The GTEx parquet in this repo is gene-by-sample (rows=genes, columns=samples)
with gene symbols in the 'Description' column, not the index or column names.
Values are raw gene read counts (not TPM), so downstream preprocessing must
use a library-size-normalized mode (cpm / log1p_cpm) rather than the
already-TPM modes used for TCGA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_GTEX_PARQUET = Path(
    "/home/walt/Attention/data/gtex/GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_reads.parquet"
)


def load_gtex_matrix(
    gtex_parquet_path: Path = DEFAULT_GTEX_PARQUET,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Return (matrix indexed by sample id with gene columns, already_tpm=False).

    Genes are read from the 'Description' column, samples from all other
    columns, then transposed so the result is sample x gene.
    """
    raw = pd.read_parquet(gtex_parquet_path)
    if "Description" not in raw.columns:
        raise RuntimeError("Expected a 'Description' column with gene symbols in the GTEx parquet.")

    all_sample_cols = [c for c in raw.columns if c != "Description"]
    sample_cols = [str(s) for s in sample_ids] if sample_ids is not None else all_sample_cols
    missing = [s for s in sample_cols if s not in all_sample_cols]
    if missing:
        raise KeyError(f"Requested GTEx sample ids not found: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    gene_syms = raw["Description"].astype(str).to_numpy()
    matrix = raw[sample_cols].T
    matrix.columns = gene_syms
    matrix.index = matrix.index.astype(str)
    matrix = matrix.astype(np.float32)
    return matrix, False
