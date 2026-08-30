#!/usr/bin/env python3
"""Materialize benchmark-local copies of validated ARCHS4 evaluation matrices."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from common import REPO_ROOT, RESULTS, WORK, sha256


SOURCE_WORK = REPO_ROOT / "benchmarks/tcga_imputation/work"
SOURCE_RESULTS = REPO_ROOT / "benchmarks/tcga_imputation/results"
DATASETS = ("archs4_holdout_human", "archs4_holdout_mouse")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    cohorts = pd.read_parquet(RESULTS / "cohort_manifest.parquet")
    source_selected = pd.read_parquet(SOURCE_RESULTS / "archs4_holdout_selected_samples.parquet")
    records = []
    for dataset in DATASETS:
        species = dataset.rsplit("_", 1)[-1]
        expected = cohorts.loc[(cohorts.role == "evaluation") & (cohorts.species == species),
                               "sample_id"].astype(str).tolist()
        source_ids = source_selected.loc[source_selected.dataset.eq(dataset), "sample_id"].astype(str).tolist()
        if expected != source_ids:
            raise ValueError(f"{dataset}: frozen IDs/order differ from validated source matrix")
        for suffix in ("log1p_tpm.npy", "genes.parquet"):
            source = SOURCE_WORK / f"{dataset}_{suffix}"
            target = WORK / f"{dataset}_{suffix}"
            shutil.copy2(source, target)
        matrix = np.load(WORK / f"{dataset}_log1p_tpm.npy", mmap_mode="r")
        genes = pd.read_parquet(WORK / f"{dataset}_genes.parquet")
        records.append({"dataset": dataset, "samples": matrix.shape[0],
                        "model_positions": matrix.shape[1],
                        "observed_genes": int(genes.observed.sum()),
                        "matrix_sha256": sha256(WORK / f"{dataset}_log1p_tpm.npy")})
    manifest = {"preprocessing": "species-specific exon-length TPM -> natural log1p",
                "source_benchmark": str(SOURCE_WORK), "datasets": records}
    (RESULTS / "expression_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
