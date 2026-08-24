"""Generic GEO source adapter.

Two paths, depending on whether the GEO series is already indexed by ARCHS4:

1. `lookup_archs4_embeddings` - if the GEO accessions are already part of the
   ~940k-sample ARCHS4 archive we embedded (see prepared_data/archs4_sample_embeddings_full),
   just read their precomputed embeddings directly. This is the preferred
   path: no reprocessing, exact same feature space used at training time.

2. `load_geo_counts_matrix` - for GEO series NOT in the ARCHS4 archive
   (e.g. too recent, or non-RNA-seq), load a raw counts matrix downloaded
   from GEO and convert it to TPM using the same species-aware gene-length
   logic as the OSDR adapter. Supports human or mouse, Ensembl ids or gene
   symbols.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..species import (
    DEFAULT_HUMAN_EXON_LENGTHS,
    DEFAULT_MOUSE_EXON_LENGTHS,
    DEFAULT_ORTHOLOGS,
    build_mouse_to_human_maps,
    counts_to_tpm,
    detect_gene_id_type,
    load_exon_length_map,
    load_human_ensembl_to_symbol_map,
    load_mouse_symbol_to_human_symbol_map,
)

DEFAULT_ARCHS4_EMBEDDING_DIR = Path("prepared_data/archs4_sample_embeddings_full")


def lookup_archs4_embeddings(
    geo_accessions: Sequence[str],
    embedding_dir: Path = DEFAULT_ARCHS4_EMBEDDING_DIR,
) -> pd.DataFrame:
    """Return precomputed 512-D embeddings for GEO accessions already in the ARCHS4 archive.

    Output columns: geo_accession, emb_0..emb_{dim-1}. Accessions not found in
    the archive are silently omitted; check the returned row count against
    len(geo_accessions) to see what was missing.
    """
    manifest_path = embedding_dir / "embedding_manifest.json"
    locations_path = embedding_dir / "sample_locations.parquet"
    if not manifest_path.exists() or not locations_path.exists():
        raise FileNotFoundError(f"Missing ARCHS4 embedding outputs under {embedding_dir}")

    manifest = json.loads(manifest_path.read_text())
    dim = int(manifest["embedding_dim"])
    dtype = np.float16 if manifest.get("embedding_dtype", "float16") == "float16" else np.float32
    total = int(manifest["total_samples"])

    emb_path = embedding_dir / f"sample_embeddings.{manifest.get('embedding_dtype', 'float16')}.mmap"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding memmap not found: {emb_path}")

    locations = pd.read_parquet(locations_path)
    requested = set(str(g) for g in geo_accessions)
    hits = locations[locations["geo_accession"].astype(str).isin(requested)].copy()
    if hits.empty:
        return pd.DataFrame(columns=["geo_accession"] + [f"emb_{i}" for i in range(dim)])

    vecs = np.memmap(emb_path, dtype=dtype, mode="r", shape=(total, dim))
    rows = vecs[hits["global_index"].to_numpy()].astype(np.float32)

    out = pd.DataFrame(rows, columns=[f"emb_{i}" for i in range(dim)])
    out.insert(0, "geo_accession", hits["geo_accession"].astype(str).to_numpy())
    return out.reset_index(drop=True)


def load_geo_counts_matrix(
    counts_path: Path,
    species: str,
    sample_ids: Optional[Sequence[str]] = None,
    orthologs_path: Path = DEFAULT_ORTHOLOGS,
    human_exon_lengths_path: Path = DEFAULT_HUMAN_EXON_LENGTHS,
    mouse_exon_lengths_path: Path = DEFAULT_MOUSE_EXON_LENGTHS,
) -> Tuple[pd.DataFrame, bool]:
    """Return (matrix indexed by sample id with human gene columns, already_tpm=True).

    `counts_path` is a raw-counts matrix: first column is a gene identifier
    (Ensembl id or gene symbol), remaining columns are one raw-count column
    per sample. `species` must be "human" or "mouse" — the organism is never
    guessed, since a wrong guess silently corrupts the TPM conversion.
    """
    if species not in ("human", "mouse"):
        raise ValueError(f'species must be "human" or "mouse", got {species!r}')

    header = pd.read_csv(counts_path, nrows=0)
    if header.shape[1] < 2:
        raise RuntimeError(f"GEO counts file has no sample columns: {counts_path}")

    gene_col = header.columns[0]
    all_sample_cols = list(header.columns[1:])
    sample_cols = [str(s) for s in sample_ids] if sample_ids is not None else all_sample_cols
    missing = [s for s in sample_cols if s not in all_sample_cols]
    if missing:
        raise KeyError(f"Requested GEO sample columns not found: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    counts = pd.read_csv(counts_path, usecols=[gene_col] + sample_cols)
    counts[gene_col] = counts[gene_col].astype(str).str.strip().str.strip('"').str.strip("'").str.split(".").str[0]
    counts = counts.set_index(gene_col)

    detected = detect_gene_id_type(counts.index)
    if detected == "mouse_ensembl" and species != "mouse":
        raise ValueError("Gene ids look like mouse Ensembl ids but species='human' was specified.")
    if detected == "human_ensembl" and species != "human":
        raise ValueError("Gene ids look like human Ensembl ids but species='mouse' was specified.")
    id_type = detected if detected != "symbol" else ("mouse_symbol" if species == "mouse" else "symbol")

    if id_type == "mouse_ensembl":
        ensembl_to_human, length_map = build_mouse_to_human_maps(orthologs_path, mouse_exon_lengths_path)
        counts.index = counts.index.map(lambda g: ensembl_to_human.get(g))
        counts = counts[counts.index.notna()]
    elif id_type == "mouse_symbol":
        # Compute TPM in mouse-symbol space first (exact length match), then
        # rename to human symbols so the output vocab matches other sources.
        mouse_length_map = load_exon_length_map(mouse_exon_lengths_path)
        counts = counts.groupby(counts.index).sum()
        tpm_mouse = counts_to_tpm(counts, mouse_length_map)
        symbol_to_human = load_mouse_symbol_to_human_symbol_map(orthologs_path)
        tpm_mouse.index = tpm_mouse.index.map(lambda g: symbol_to_human.get(g))
        tpm_mouse = tpm_mouse[tpm_mouse.index.notna()]
        matrix = tpm_mouse.groupby(tpm_mouse.index).sum().T
        matrix.index = matrix.index.astype(str)
        return matrix, True
    elif id_type == "human_ensembl":
        ensembl_to_symbol = load_human_ensembl_to_symbol_map(orthologs_path)
        counts.index = counts.index.map(lambda g: ensembl_to_symbol.get(g))
        counts = counts[counts.index.notna()]
        length_map = load_exon_length_map(human_exon_lengths_path)
    else:
        length_map = load_exon_length_map(human_exon_lengths_path)

    counts.index = counts.index.astype(str).str.upper()
    counts = counts.groupby(counts.index).sum()

    tpm = counts_to_tpm(counts, length_map)
    matrix = tpm.T  # sample x human gene symbol
    matrix.index = matrix.index.astype(str)
    return matrix, True
