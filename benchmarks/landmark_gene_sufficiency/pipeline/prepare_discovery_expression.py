#!/usr/bin/env python3
"""Extract log1p(TPM) for the frozen human discovery cohort from ARCHS4."""

from __future__ import annotations

import json
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from common import CONFIG, REPO_ROOT, RESULTS, WORK, sha256


H5 = REPO_ROOT / "data/archs4/human_gene_v2.5.h5"
HGNC = REPO_ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
MODEL_GENES = REPO_ROOT / "benchmarks/tcga_imputation/work/ours_genes.parquet"


def norm_gene(value: object) -> str:
    return str(value).strip().split(".", 1)[0].upper()


def split_symbols(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [norm_gene(item) for item in str(value).split("|") if str(item).strip()]


def hgnc_crosswalk(source_symbols: list[str]) -> dict[str, str]:
    """Map ARCHS4 symbols uniquely to current HGNC-approved symbols."""
    hgnc = pd.read_csv(HGNC, sep="\t", low_memory=False)
    hgnc = hgnc.loc[hgnc.status.eq("Approved")]
    approved = {norm_gene(x): norm_gene(x) for x in hgnc.symbol}
    previous: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in hgnc.itertuples(index=False):
        target = norm_gene(row.symbol)
        for symbol in split_symbols(row.prev_symbol): previous[symbol].add(target)
        for symbol in split_symbols(row.alias_symbol): aliases[symbol].add(target)
    mapping = {}
    for source in map(norm_gene, source_symbols):
        candidates = ({approved[source]} if source in approved else
                      previous[source] if len(previous[source]) == 1 else
                      aliases[source] if len(aliases[source]) == 1 else set())
        if len(candidates) == 1: mapping[source] = next(iter(candidates))
    reverse: dict[str, list[str]] = defaultdict(list)
    for source, target in mapping.items(): reverse[target].append(source)
    return {source: target for source, target in mapping.items()
            if len(reverse[target]) == 1 or source == target}


def tpm_log1p(counts: np.ndarray, lengths_bp: np.ndarray) -> np.ndarray:
    rate = counts.astype(np.float64) / (lengths_bp[None, :] / 1000.0)
    totals = rate.sum(axis=1, keepdims=True)
    tpm = np.divide(rate, totals, out=np.zeros_like(rate), where=totals > 0) * 1e6
    return np.log1p(tpm).astype(np.float32)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(RESULTS / "cohort_manifest.parquet")
    sample_ids = cohort.loc[cohort.role.eq("discovery"), "sample_id"].astype(str).tolist()
    model = pd.read_parquet(MODEL_GENES)
    model_symbols = model.approved_symbol.where(model.approved_symbol.notna(), None).tolist()
    lengths = pd.read_csv(LENGTHS).set_index("gene_symbol")["exon_length"]
    model_lengths = lengths.reindex(model.gene).to_numpy(float)

    with h5py.File(H5, "r") as handle:
        accessions = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in handle["meta/samples/geo_accession"][:]]
        accession_to_col = {gsm: col for col, gsm in enumerate(accessions)}
        source_symbols = [norm_gene(x.decode() if isinstance(x, bytes) else x)
                          for x in handle["meta/genes/symbol"][:]]
        source_to_approved = hgnc_crosswalk(source_symbols)
        approved_to_row = {source_to_approved[source]: row
                           for row, source in enumerate(source_symbols)
                           if source in source_to_approved}
        rows = np.asarray([approved_to_row.get(gene, -1) if gene else -1
                           for gene in model_symbols], dtype=int)
        observed = (rows >= 0) & np.isfinite(model_lengths) & (model_lengths > 0)
        matrix = np.zeros((len(sample_ids), len(rows)), dtype=np.float32)
        expression = handle["data/expression"]
        for out_row, gsm in enumerate(sample_ids):
            if gsm not in accession_to_col: raise KeyError(f"{gsm} absent from {H5}")
            raw = expression[:, accession_to_col[gsm]]
            matrix[out_row, observed] = raw[rows[observed]]
            if (out_row + 1) % 250 == 0:
                print(f"[discovery] extracted {out_row + 1:,}/{len(sample_ids):,}", flush=True)
    safe_lengths = model_lengths.copy(); safe_lengths[~observed] = 1.0
    matrix = tpm_log1p(matrix, safe_lengths)
    matrix[:, ~observed] = 0
    matrix_path = WORK / "human_discovery_log1p_tpm.npy"
    np.save(matrix_path, matrix)
    genes = model[["gene", "approved_symbol", "native_index"]].rename(
        columns={"native_index": "model_index"}
    ).copy()
    genes["observed"] = observed
    genes.to_parquet(WORK / "human_discovery_genes.parquet", index=False)
    provenance = {
        "samples": len(sample_ids), "observed_genes": int(observed.sum()),
        "source": str(H5), "preprocessing": "exon-length TPM -> natural log1p",
        "matrix_sha256": sha256(matrix_path),
        "selection_role": "human discovery; disjoint from human/mouse evaluation",
    }
    (RESULTS / "discovery_expression_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__": main()
