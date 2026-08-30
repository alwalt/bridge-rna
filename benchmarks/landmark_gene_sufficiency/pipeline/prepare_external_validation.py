#!/usr/bin/env python3
"""Prepare independent TCGA-human and OSDR-mouse panel-validation matrices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, REPO_ROOT, RESULTS, WORK, sha256

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.sources.osdr import load_osdr_matrix

TCGA_WORK = REPO_ROOT / "benchmarks/tcga_imputation/work"
TCGA_RESULTS = REPO_ROOT / "benchmarks/tcga_imputation/results"
OSDR_META = REPO_ROOT / "data/osdr/metadata/selected_sample_metadata.tsv"


def prepare_tcga() -> dict:
    source = TCGA_WORK / "ours_log1p_tpm.npy"
    matrix = np.load(source, mmap_mode="r")
    target = WORK / "external_tcga_human_log1p_tpm.npy"
    np.save(target, np.asarray(matrix, dtype=np.float32))
    samples = pd.read_parquet(TCGA_RESULTS / "selected_tcga_samples.parquet").copy()
    samples["row_index"] = range(len(samples)); samples["dataset"] = "tcga_human"
    samples.to_parquet(RESULTS / "external_tcga_human_samples.parquet", index=False)
    genes = pd.read_parquet(TCGA_WORK / "ours_genes.parquet").rename(columns={"native_index": "model_index"})
    genes["observed"] = genes.tcga_observed.astype(bool)
    genes.to_parquet(WORK / "external_tcga_human_genes.parquet", index=False)
    return {"dataset": "tcga_human", "samples": len(matrix), "genes": matrix.shape[1],
            "source": str(source), "sha256": sha256(target)}


def prepare_osdr() -> dict:
    meta = pd.read_csv(OSDR_META, sep="\t")
    meta = meta.loc[meta.has_single_cell_rna_sequencing.eq(0)].copy()
    meta = meta.drop_duplicates(["counts_path", "id.sample name"])
    # Keep only metadata rows with a unique one-to-one count-column match.
    # Some OSDR matrices combine biological samples (for example, 1-2), and
    # those are deliberately excluded rather than mislabeled as individuals.
    def normalized(value: object) -> str:
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())
    matched = []
    for counts_path, group in meta.groupby("counts_path", sort=True):
        header = pd.read_csv(REPO_ROOT / counts_path, nrows=0).columns[1:]
        by_norm = {}
        for column in header: by_norm.setdefault(normalized(column), []).append(str(column))
        group = group.copy()
        group["_counts_column"] = group["id.sample name"].map(
            lambda value: by_norm.get(normalized(value), [None])[0]
            if len(by_norm.get(normalized(value), [])) == 1 else None)
        matched.append(group.dropna(subset=["_counts_column"]))
    meta = pd.concat(matched, ignore_index=True)
    seed = int(CONFIG["benchmark_seed"])
    capped = pd.concat([group.sample(min(len(group), 20), random_state=seed)
                        for _, group in meta.groupby("id.accession", sort=True)], ignore_index=True)
    selected = capped.sample(min(1000, len(capped)), random_state=seed).sort_values(
        ["id.accession", "id.sample name"]).reset_index(drop=True)
    vocab = pd.read_parquet(RESULTS / "final_human_informative_gene_ranking.parquet").sort_values(
        "model_index").gene.astype(str).tolist()
    matrices, rows, observed_all = [], [], np.ones(len(vocab), dtype=bool)
    for counts_path, group in selected.groupby("counts_path", sort=True):
        ids = group["_counts_column"].astype(str).tolist()
        frame, _ = load_osdr_matrix(REPO_ROOT / counts_path, ids)
        observed_all &= np.asarray([gene in frame.columns for gene in vocab], dtype=bool)
        frame = frame.reindex(columns=vocab, fill_value=0.0)
        matrices.append(np.log1p(frame.to_numpy(dtype=np.float32)))
        rows.extend(group.assign(matrix_order=range(len(group))).to_dict("records"))
        print(f"[OSDR] prepared {len(rows):,}/{len(selected):,}", flush=True)
    matrix = np.concatenate(matrices, axis=0)
    target = WORK / "external_osdr_mouse_log1p_tpm.npy"
    np.save(target, matrix)
    sample_table = pd.DataFrame(rows)
    sample_table["sample_id"] = sample_table["id.sample name"].astype(str)
    sample_table["row_index"] = range(len(sample_table)); sample_table["dataset"] = "osdr_mouse"
    sample_table.to_parquet(RESULTS / "external_osdr_mouse_samples.parquet", index=False)
    pd.DataFrame({"model_index": range(len(vocab)), "gene": vocab,
                  "observed": observed_all}).to_parquet(
        WORK / "external_osdr_mouse_genes.parquet", index=False)
    return {"dataset": "osdr_mouse", "samples": len(matrix), "genes": matrix.shape[1],
            "studies": sample_table["id.accession"].nunique(),
            "genes_observed_in_all_selected_studies": int(observed_all.sum()), "source": str(OSDR_META),
            "sha256": sha256(target)}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    records = [prepare_tcga(), prepare_osdr()]
    (RESULTS / "external_validation_provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__": main()
