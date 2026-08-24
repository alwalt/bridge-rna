"""Species-aware gene-length lookups and ortholog mapping, shared by adapters
that need to convert raw counts to TPM (OSDR, generic GEO series)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_ORTHOLOGS = Path("data/ensembl/orthologs_one2one.txt")
DEFAULT_MOUSE_EXON_LENGTHS = Path("data/gencode/gencode_v49_mouse_gene_exon_lengths.csv")
DEFAULT_HUMAN_EXON_LENGTHS = Path("data/gencode/gencode_v49_gene_exon_lengths.csv")


def load_exon_length_map(path: Path) -> Dict[str, float]:
    """Load a gene_symbol -> exon_length_bp map (works for human or mouse gencode csv)."""
    lengths = pd.read_csv(path).drop_duplicates("gene_symbol")
    return lengths.set_index("gene_symbol")["exon_length"].astype(float).to_dict()


def load_mouse_to_human_symbol_map(orthologs_path: Path = DEFAULT_ORTHOLOGS) -> Dict[str, str]:
    """Mouse Ensembl gene id -> human gene symbol, one-to-one orthologs only."""
    ortho = pd.read_csv(orthologs_path, sep="\t")
    ortho = ortho[ortho["Human homology type"] == "ortholog_one2one"].copy()
    ortho["Gene stable ID"] = ortho["Gene stable ID"].astype(str).str.split(".").str[0]
    return dict(zip(ortho["Gene stable ID"], ortho["Human gene name"]))


def load_human_ensembl_to_symbol_map(orthologs_path: Path = DEFAULT_ORTHOLOGS) -> Dict[str, str]:
    """Human Ensembl gene id -> human gene symbol (derived from the same ortholog table)."""
    ortho = pd.read_csv(orthologs_path, sep="\t")
    ortho["Human gene stable ID"] = ortho["Human gene stable ID"].astype(str).str.split(".").str[0]
    ortho = ortho.dropna(subset=["Human gene stable ID", "Human gene name"]).drop_duplicates("Human gene stable ID")
    return dict(zip(ortho["Human gene stable ID"], ortho["Human gene name"]))


def load_mouse_symbol_to_human_symbol_map(orthologs_path: Path = DEFAULT_ORTHOLOGS) -> Dict[str, str]:
    """Mouse gene symbol -> human gene symbol, one-to-one orthologs only."""
    ortho = pd.read_csv(orthologs_path, sep="\t")
    ortho = ortho[ortho["Human homology type"] == "ortholog_one2one"].copy()
    ortho = ortho.dropna(subset=["Gene name", "Human gene name"]).drop_duplicates("Gene name")
    return dict(zip(ortho["Gene name"], ortho["Human gene name"]))


def build_mouse_to_human_maps(
    orthologs_path: Path = DEFAULT_ORTHOLOGS,
    mouse_exon_lengths_path: Path = DEFAULT_MOUSE_EXON_LENGTHS,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Return (mouse_ensembl_id -> human_gene_symbol, human_gene_symbol -> mouse_exon_length_bp)."""
    ortho = pd.read_csv(orthologs_path, sep="\t")
    ortho = ortho[ortho["Human homology type"] == "ortholog_one2one"].copy()
    ortho["Gene stable ID"] = ortho["Gene stable ID"].astype(str).str.split(".").str[0]
    ensembl_to_human = dict(zip(ortho["Gene stable ID"], ortho["Human gene name"]))

    mouse_lengths = load_exon_length_map(mouse_exon_lengths_path)
    human_length_map: Dict[str, float] = {}
    for _, row in ortho[["Human gene name", "Gene name"]].drop_duplicates().iterrows():
        human_gene, mouse_gene = row["Human gene name"], row["Gene name"]
        if mouse_lengths.get(mouse_gene, 0) > 0 and human_gene not in human_length_map:
            human_length_map[human_gene] = mouse_lengths[mouse_gene]

    return ensembl_to_human, human_length_map


def counts_to_tpm(counts: pd.DataFrame, length_map: Dict[str, float]) -> pd.DataFrame:
    """Convert a gene x sample raw-count matrix to TPM using a gene_symbol -> exon_length_bp map.

    Genes missing from `length_map` (or with non-positive length) are dropped.
    """
    lengths_bp = pd.Series(length_map, dtype=np.float64).reindex(counts.index)
    keep_mask = lengths_bp.notna() & (lengths_bp > 0)
    counts = counts.loc[keep_mask]
    lengths_kb = (lengths_bp.loc[keep_mask] / 1000.0).astype(np.float32)

    rate = counts.div(lengths_kb, axis=0)
    denom = rate.sum(axis=0)
    tpm = rate.div(denom.replace(0, np.nan), axis=1) * 1e6
    return tpm.fillna(0.0).astype(np.float32)


def detect_gene_id_type(gene_ids) -> str:
    """Best-effort detection of gene id style: 'mouse_ensembl', 'human_ensembl', or 'symbol'."""
    sample = [str(g) for g in list(gene_ids)[:50]]
    if any(g.startswith("ENSMUSG") for g in sample):
        return "mouse_ensembl"
    if any(g.startswith("ENSG") for g in sample):
        return "human_ensembl"
    return "symbol"
