#!/usr/bin/env python3
"""Prepare human and mouse information-discovery expression matrices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS, WORK = HERE / "results", HERE / "work"
TCGA_PIPELINE = ROOT / "benchmarks/tcga_imputation/pipeline"
sys.path.insert(0, str(TCGA_PIPELINE))
from prepare_archs4_holdout import H5, source_rows  # noqa: E402
from common import tpm_log1p  # noqa: E402


def prepare(species: str, cohort: pd.DataFrame, model: pd.DataFrame) -> dict:
    dataset = f"archs4_holdout_{species}"
    ids = cohort.loc[cohort.species.eq(species), "sample_id"].astype(str).tolist()
    approved = model.approved_symbol.where(model.approved_symbol.notna(), None).tolist()
    with h5py.File(H5[dataset], "r") as handle:
        accessions = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in handle["meta/samples/geo_accession"][:]]
        lookup = {gsm: col for col, gsm in enumerate(accessions)}
        rows, lengths = source_rows(dataset, handle, approved)
        # Audited fallback for symbols lost when Ensembl and HGNC snapshots use
        # different stable-ID generations (including legacy IQCD, now DRC10).
        source_symbols = [str(x.decode() if isinstance(x, bytes) else x).split(".")[0]
                          for x in handle["meta/genes/symbol"][:]]
        exact_symbol_row = {symbol.upper(): row for row, symbol in enumerate(source_symbols)}
        missing = np.flatnonzero(rows < 0)
        if species == "human":
            human_lengths = pd.read_csv(ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv").set_index("gene_symbol")["exon_length"]
            for position in missing:
                symbol = str(model.iloc[position].gene).upper()
                if symbol in exact_symbol_row and symbol in human_lengths.index:
                    rows[position] = exact_symbol_row[symbol]; lengths[position] = human_lengths.loc[symbol]
        else:
            orthologs = pd.read_csv(ROOT / "data/ensembl/orthologs_one2one.txt", sep="\t", dtype=str)
            orthologs = orthologs.loc[orthologs["Human homology type"].eq("ortholog_one2one")
                & orthologs["Human orthology confidence [0 low, 1 high]"].eq("1")]
            human_to_mouse = dict(zip(orthologs["Human gene name"].str.upper(), orthologs["Gene name"]))
            mouse_lengths = pd.read_csv(ROOT / "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv").set_index("gene_symbol")["exon_length"]
            for position in missing:
                mouse_symbol = human_to_mouse.get(str(model.iloc[position].gene).upper())
                if mouse_symbol and mouse_symbol.upper() in exact_symbol_row and mouse_symbol in mouse_lengths.index:
                    rows[position] = exact_symbol_row[mouse_symbol.upper()]; lengths[position] = mouse_lengths.loc[mouse_symbol]
        observed = rows >= 0
        matrix = np.zeros((len(ids), len(rows)), dtype=np.float32)
        expression = handle["data/expression"]
        for out_row, gsm in enumerate(ids):
            if gsm not in lookup: raise KeyError(f"{gsm} absent from {H5[dataset]}")
            raw = expression[:, lookup[gsm]]
            matrix[out_row, observed] = raw[rows[observed]]
            if (out_row + 1) % 250 == 0:
                print(f"[{species}] extracted {out_row + 1:,}/{len(ids):,}", flush=True)
    matrix = tpm_log1p(matrix, lengths)
    np.save(WORK / f"{species}_information_discovery_log1p_tpm.npy", matrix)
    genes = model[["gene", "approved_symbol", "native_index"]].rename(
        columns={"native_index": "model_index"}).copy()
    genes["observed"] = observed
    genes.to_parquet(WORK / f"{species}_information_discovery_genes.parquet", index=False)
    pd.DataFrame({"sample_id": ids, "row_index": range(len(ids)), "species": species}).to_parquet(
        WORK / f"{species}_information_discovery_samples.parquet", index=False)
    return {"species": species, "samples": len(ids), "observed_genes": int(observed.sum()),
            "source": str(H5[dataset])}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_parquet(RESULTS / "information_discovery_cohort.parquet")
    model = pd.read_parquet(ROOT / "benchmarks/tcga_imputation/work/ours_genes.parquet")
    records = [prepare(species, cohort, model) for species in ("human", "mouse")]
    (RESULTS / "information_expression_provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__": main()
