"""Canonical gene vocabulary used by the pretrained ExpressionPerformer."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_CANONICAL_GENES_PATH = Path("data/archs4/train_orthologs/canonical_genes.csv")


def norm_gene(value: object) -> str:
    """Normalize a gene identifier: strip Ensembl version suffix, upper-case."""
    return str(value).strip().split(".")[0].upper()


def load_canonical_genes(path: Path | str = DEFAULT_CANONICAL_GENES_PATH) -> List[str]:
    """Load the canonical gene list in the exact order the model expects.

    Order matters: this list defines column position 0..N-1 fed to the model.
    """
    canon_df = pd.read_csv(path)
    return list(dict.fromkeys(canon_df["gene_symbol"].astype(str).map(norm_gene)))
