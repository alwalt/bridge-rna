#!/usr/bin/env python3
"""Prepare leakage-free gene mean/median baselines from ARCHS4 training samples."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import pandas as pd

from common import REPO_ROOT, RESULTS, WORK
from prepare_tcga import BULK_INFO, OUR_LENGTHS, build_hgnc_crosswalk, norm_gene, tpm_log1p


ARCHS4_HUMAN = REPO_ROOT / "data/archs4/human_gene_v2.5.h5"
TRAIN_SPLIT = REPO_ROOT / "data/archs4/training/sample_split/train_samples.parquet"
DEFAULT_SAMPLES = 5_000
DEFAULT_SEED = 20260828


def prepare(reference_samples: int, seed: int) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    split = pd.read_parquet(TRAIN_SPLIT, columns=["sample_id", "species"])
    human = split.loc[split.species.eq("human"), "sample_id"].drop_duplicates()
    selected = human.sample(reference_samples, random_state=seed).astype(str).tolist()
    our_table = pd.read_parquet(WORK / "ours_genes.parquet")
    bulk_table = pd.read_parquet(WORK / "bulkformer_genes.parquet")
    our_lengths = pd.read_csv(OUR_LENGTHS).set_index("gene_symbol")["exon_length"]
    bulk_lengths = pd.read_csv(BULK_INFO)["gene_length"].to_numpy(float)

    with h5py.File(ARCHS4_HUMAN, "r") as handle:
        h5_symbols = [norm_gene(x.decode() if isinstance(x, bytes) else x)
                      for x in handle["meta/genes/symbol"][:]]
        _, source_to_approved = build_hgnc_crosswalk(h5_symbols)
        approved_to_row = {source_to_approved[source]: row for row, source in enumerate(h5_symbols)
                           if source in source_to_approved}
        accessions = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in handle["meta/samples/geo_accession"][:]]
        accession_to_col = {gsm: col for col, gsm in enumerate(accessions)}
        missing = [gsm for gsm in selected if gsm not in accession_to_col]
        if missing:
            raise ValueError(f"{len(missing)} selected training GSMs absent from ARCHS4 H5")
        our_rows = np.asarray([approved_to_row.get(gene, -1)
                               if pd.notna(gene) else -1 for gene in our_table.approved_symbol])
        bulk_rows = np.asarray([approved_to_row.get(gene, -1)
                                if pd.notna(gene) else -1 for gene in bulk_table.approved_symbol])
        our_values = np.empty((reference_samples, len(our_rows)), dtype=np.float32)
        bulk_values = np.empty((reference_samples, len(bulk_rows)), dtype=np.float32)
        expression = handle["data/expression"]
        our_length_vector = our_lengths.reindex(our_table.gene).to_numpy(float)
        for out_row, gsm in enumerate(selected):
            raw = expression[:, accession_to_col[gsm]]
            ours = np.zeros(len(our_rows), dtype=np.float32)
            bulk = np.zeros(len(bulk_rows), dtype=np.float32)
            ours[our_rows >= 0] = raw[our_rows[our_rows >= 0]]
            bulk[bulk_rows >= 0] = raw[bulk_rows[bulk_rows >= 0]]
            our_values[out_row] = tpm_log1p(ours[None, :], our_length_vector)[0]
            bulk_values[out_row] = tpm_log1p(bulk[None, :], bulk_lengths)[0]
            if (out_row + 1) % 500 == 0 or out_row + 1 == reference_samples:
                print(f"[baseline] samples={out_row + 1:,}/{reference_samples:,}", flush=True)

    np.savez_compressed(
        WORK / "frozen_training_gene_baselines.npz",
        ours_mean=our_values.mean(axis=0).astype(np.float32),
        ours_median=np.median(our_values, axis=0).astype(np.float32),
        bulkformer_mean=bulk_values.mean(axis=0).astype(np.float32),
        bulkformer_median=np.median(bulk_values, axis=0).astype(np.float32),
    )
    pd.DataFrame({"sample_id": selected}).to_parquet(
        RESULTS / "baseline_training_sample_ids.parquet", index=False
    )
    provenance = {
        "status": "complete", "source": str(ARCHS4_HUMAN), "split": str(TRAIN_SPLIT),
        "species": "human", "samples": reference_samples, "selection_seed": seed,
        "statistics": ["gene_mean", "gene_median"],
        "representation": "model-specific natural log1p(TPM)",
        "evaluation_samples_used": 0,
    }
    (RESULTS / "frozen_baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    prepare(args.samples, args.seed)


if __name__ == "__main__":
    main()
