"""TCGA source adapter.

The TCGA TPM parquet in this repo is already TPM-normalized: index is
sample/file id, columns are gene symbols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

DEFAULT_TCGA_TPM = Path("/home/walt/Attention/data/tcga/tcga_tpm_unstranded_matrix.parquet")


def load_tcga_matrix(
    tpm_parquet_path: Path = DEFAULT_TCGA_TPM,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Return (matrix indexed by sample id with gene columns, already_tpm=True)."""
    matrix = pd.read_parquet(tpm_parquet_path)
    matrix.index = matrix.index.astype(str)
    if sample_ids is not None:
        matrix = matrix.reindex([str(s) for s in sample_ids])
    return matrix, True
