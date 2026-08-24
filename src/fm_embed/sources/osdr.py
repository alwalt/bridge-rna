"""OSDR (mouse) source adapter.

OSDR counts files are gene (mouse Ensembl ID) x sample raw counts. This
adapter maps mouse genes to one-to-one human orthologs and converts raw
counts to true TPM using mouse exon lengths, matching the conversion used
for the retrieval demo in demo_osdr_top5.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from ..species import (
    DEFAULT_MOUSE_EXON_LENGTHS,
    DEFAULT_ORTHOLOGS,
    build_mouse_to_human_maps,
    counts_to_tpm,
)


def load_osdr_matrix(
    counts_csv_path: Path,
    sample_ids: Optional[Sequence[str]] = None,
    orthologs_path: Path = DEFAULT_ORTHOLOGS,
    mouse_exon_lengths_path: Path = DEFAULT_MOUSE_EXON_LENGTHS,
) -> Tuple[pd.DataFrame, bool]:
    """Return (matrix indexed by sample id with human gene columns, already_tpm=True).

    `counts_csv_path` is a per-experiment counts file: first column is the
    mouse Ensembl gene id, remaining columns are one raw-count column per
    sample.
    """
    header = pd.read_csv(counts_csv_path, nrows=0)
    if header.shape[1] < 2:
        raise RuntimeError(f"OSDR counts file has no sample columns: {counts_csv_path}")

    gene_col = header.columns[0]
    all_sample_cols = list(header.columns[1:])
    sample_cols = [str(s) for s in sample_ids] if sample_ids is not None else all_sample_cols
    missing = [s for s in sample_cols if s not in all_sample_cols]
    if missing:
        raise KeyError(f"Requested OSDR sample columns not found: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    counts = pd.read_csv(counts_csv_path, usecols=[gene_col] + sample_cols)
    counts[gene_col] = counts[gene_col].astype(str).str.strip().str.strip('"').str.strip("'").str.split(".").str[0]
    counts = counts.set_index(gene_col)

    ensembl_to_human, human_length_map = build_mouse_to_human_maps(orthologs_path, mouse_exon_lengths_path)
    counts.index = counts.index.map(lambda g: ensembl_to_human.get(g))
    counts = counts[counts.index.notna()]
    counts = counts.groupby(counts.index).sum()

    tpm = counts_to_tpm(counts, human_length_map)
    matrix = tpm.T  # sample x human gene
    matrix.index = matrix.index.astype(str)
    return matrix, True
