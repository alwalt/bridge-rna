#!/usr/bin/env python3
"""Create paired ARCHS4/recount3 log1p(TPM) matrices for identical GSMs.

Both sources are aggregated into the model's canonical human gene vocabulary,
normalized with the same frozen exon-length table, and stored in identical GSM
order. No CPM, z-scoring, or batch correction is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy import sparse
from scipy.io import mmread

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = HERE / "work/final_pairs.parquet"
DEFAULT_RECOUNT = HERE / "work/recount3_counts"
DEFAULT_OUTPUT = HERE / "work/paired_expression"
DEFAULT_H5 = REPO_ROOT / "data/archs4/human_gene_v2.5.h5"
DEFAULT_LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
DEFAULT_CANONICAL = (
    REPO_ROOT / "training/scratch/working_training_tools/"
    "train_orthologs_20K_samples/canonical_genes.csv"
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def canonical_data(canonical_path: Path, lengths_path: Path) -> tuple[list[str], np.ndarray]:
    genes = pd.read_csv(canonical_path)["gene_symbol"].astype(str).str.upper().tolist()
    if len(genes) != len(set(genes)):
        raise ValueError("Canonical gene vocabulary contains duplicates")
    lengths = pd.read_csv(lengths_path).drop_duplicates("gene_symbol")
    length_map = lengths.set_index(lengths["gene_symbol"].astype(str).str.upper())["exon_length"]
    length_vector = length_map.reindex(genes).to_numpy(dtype=np.float64)
    if np.isnan(length_vector).any() or np.any(length_vector <= 0):
        missing = np.asarray(genes)[np.isnan(length_vector) | (length_vector <= 0)][:10]
        raise ValueError(f"Canonical genes lack positive exon lengths: {missing.tolist()}")
    return genes, length_vector


def aggregation_matrix(source_genes: list[str], canonical: list[str]) -> sparse.csr_matrix:
    target = {gene: index for index, gene in enumerate(canonical)}
    rows, cols = [], []
    for source_index, raw_gene in enumerate(source_genes):
        gene = str(raw_gene).split(".")[0].upper()
        if gene in target:
            rows.append(target[gene])
            cols.append(source_index)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(canonical), len(source_genes)))


def to_log1p_tpm(counts: np.ndarray, lengths_bp: np.ndarray) -> np.ndarray:
    rates = np.maximum(counts, 0).astype(np.float64, copy=False) / (lengths_bp[:, None] / 1000.0)
    denom = rates.sum(axis=0)
    tpm = np.divide(rates * 1e6, denom[None, :], out=np.zeros_like(rates), where=denom[None, :] > 0)
    return np.log1p(tpm).T.astype(np.float32)


def recount_gene_symbols(metadata_path: Path, expected_rows: int) -> list[str]:
    meta = pd.read_csv(metadata_path, low_memory=False)
    choices = ["gene_name", "gene_symbol", "symbol", "recount3_gene_id"]
    column = next((name for name in choices if name in meta), None)
    if column is None:
        raise ValueError(f"No gene-symbol column in {metadata_path}; columns={list(meta)}")
    if len(meta) != expected_rows:
        raise ValueError("recount3 gene metadata and count matrix have different row counts")
    return meta[column].fillna("").astype(str).tolist()


def process_recount(
    recount_dir: Path, gsms: list[str], canonical: list[str], lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    matrix = mmread(recount_dir / "counts.mtx").tocsr().astype(np.float32)
    source_samples = [line.strip() for line in (recount_dir / "samples.txt").read_text().splitlines()]
    if len(source_samples) != matrix.shape[1] or len(set(source_samples)) != len(source_samples):
        raise ValueError("Invalid recount3 sample list")
    missing = sorted(set(gsms) - set(source_samples))
    if missing:
        raise ValueError(f"Final pairs absent from recount3 matrix: {missing[:10]}")
    order = [source_samples.index(gsm) for gsm in gsms]
    symbols = recount_gene_symbols(recount_dir / "gene_metadata.csv", matrix.shape[0])
    aggregate = aggregation_matrix(symbols, canonical)
    aligned = (aggregate @ matrix[:, order]).toarray()
    return to_log1p_tpm(aligned, lengths), {
        "source_gene_rows": matrix.shape[0], "mapped_source_gene_rows": aggregate.nnz,
    }


def process_archs4(
    h5_path: Path, gsms: list[str], canonical: list[str], lengths: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    result = np.empty((len(gsms), len(canonical)), dtype=np.float32)
    with h5py.File(h5_path, "r") as handle:
        source_gsms = decode(handle["meta/samples/geo_accession"][:])
        index = {gsm: i for i, gsm in enumerate(source_gsms)}
        missing = sorted(set(gsms) - set(index))
        if missing:
            raise ValueError(f"Final pairs absent from ARCHS4 H5: {missing[:10]}")
        source_genes = decode(handle["meta/genes/symbol"][:])
        aggregate = aggregation_matrix(source_genes, canonical)
        expression = handle["data/expression"]
        for start in range(0, len(gsms), batch_size):
            stop = min(start + batch_size, len(gsms))
            requested = [index[gsm] for gsm in gsms[start:stop]]
            sorted_positions = np.argsort(requested)
            sorted_indices = np.asarray(requested)[sorted_positions]
            raw_sorted = expression[:, sorted_indices]
            inverse = np.argsort(sorted_positions)
            raw = raw_sorted[:, inverse]
            aligned = aggregate @ raw
            result[start:stop] = to_log1p_tpm(np.asarray(aligned), lengths)
            print(f"ARCHS4 preprocessing: {stop:,}/{len(gsms):,}")
    return result, {
        "source_gene_rows": len(source_genes), "mapped_source_gene_rows": aggregate.nnz,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--recount3-counts", type=Path, default=DEFAULT_RECOUNT)
    parser.add_argument("--archs4-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--canonical-genes", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--exon-lengths", type=Path, default=DEFAULT_LENGTHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_parquet(args.pairs).sort_values(["cohort", "gsm"]).reset_index(drop=True)
    if pairs["gsm"].duplicated().any():
        raise ValueError("Final pair table contains duplicate GSMs")
    gsms = pairs["gsm"].astype(str).tolist()
    canonical, lengths = canonical_data(args.canonical_genes, args.exon_lengths)
    recount, recount_stats = process_recount(args.recount3_counts, gsms, canonical, lengths)
    archs4, archs4_stats = process_archs4(
        args.archs4_h5, gsms, canonical, lengths, args.batch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "recount3_log1p_tpm.npy", recount)
    np.save(args.output_dir / "archs4_log1p_tpm.npy", archs4)
    pairs.to_parquet(args.output_dir / "samples.parquet", index=False)
    pd.DataFrame({"gene_index": range(len(canonical)), "gene_symbol": canonical}).to_parquet(
        args.output_dir / "genes.parquet", index=False
    )
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(gsms), "genes": len(canonical), "normalization": "log1p_tpm",
        "cpm_used": False, "batch_correction_used": False,
        "canonical_genes": str(args.canonical_genes.resolve()),
        "canonical_genes_sha256": checksum(args.canonical_genes),
        "exon_lengths": str(args.exon_lengths.resolve()),
        "exon_lengths_sha256": checksum(args.exon_lengths),
        "archs4_h5": str(args.archs4_h5.resolve()),
        "recount3_counts": str(args.recount3_counts.resolve()),
        "archs4": archs4_stats, "recount3": recount_stats,
    }
    (args.output_dir / "preprocessing_manifest.json").write_text(json.dumps(provenance, indent=2))
    print(f"Saved paired matrices: {recount.shape} to {args.output_dir}")


if __name__ == "__main__":
    main()
