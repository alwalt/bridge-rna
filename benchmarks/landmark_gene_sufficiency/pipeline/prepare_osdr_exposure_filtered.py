#!/usr/bin/env python3
"""Create a derived OSDR cohort excluding known pretraining/seen-study samples."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import REPO_ROOT, RESULTS, WORK


def main() -> None:
    samples_path = RESULTS / "external_osdr_mouse_samples.parquet"
    matrix_path = WORK / "external_osdr_mouse_log1p_tpm.npy"
    genes_path = WORK / "external_osdr_mouse_genes.parquet"
    manifest_path = REPO_ROOT / "data/manifests/sample_manifest.parquet"
    samples = pd.read_parquet(samples_path).sort_values("row_index").reset_index(drop=True)
    matrix = np.load(matrix_path, mmap_mode="r")
    if len(samples) != len(matrix):
        raise ValueError("OSDR sample/matrix order mismatch")
    manifest = pd.read_parquet(manifest_path,
        columns=["gsm", "split", "study_exposure", "gse_candidates_str"]).drop_duplicates("gsm")
    manifest["gsm"] = manifest.gsm.astype("string")
    annotated = samples.merge(manifest, left_on="sample_id", right_on="gsm", how="left",
                              validate="many_to_one")
    is_gsm = annotated.sample_id.astype(str).str.fullmatch(r"GSM\d+")
    known_seen = annotated.split.isin(["train", "val"]) | annotated.study_exposure.eq("seen_study")
    keep = ~known_seen
    filtered = annotated.loc[keep].copy()
    filtered["exposure_audit"] = np.select(
        [filtered.study_exposure.eq("unseen_study"), is_gsm.loc[keep] & filtered.gsm.isna()],
        ["confirmed_unseen_study", "gsm_absent_from_manifest"],
        default="non_geo_unresolved")
    filtered["original_row_index"] = filtered.row_index.astype(int)
    filtered["row_index"] = np.arange(len(filtered), dtype=int)
    output_matrix = WORK / "external_osdr_mouse_filtered_log1p_tpm.npy"
    output_samples = RESULTS / "external_osdr_mouse_filtered_samples.parquet"
    output_genes = WORK / "external_osdr_mouse_filtered_genes.parquet"
    np.save(output_matrix, np.asarray(matrix[filtered.original_row_index.to_numpy(int)], dtype=np.float32))
    filtered.to_parquet(output_samples, index=False)
    pd.read_parquet(genes_path).to_parquet(output_genes, index=False)
    provenance = {"source_samples": str(samples_path), "source_matrix": str(matrix_path),
        "exposure_manifest": str(manifest_path), "source_samples_count": len(samples),
        "excluded_known_seen": int(known_seen.sum()), "retained_samples": len(filtered),
        "retained_exposure_audit": filtered.exposure_audit.value_counts().to_dict(),
        "policy": "Exclude manifest-linked train/val samples and samples from seen studies; "
                  "do not relabel unresolved samples as confirmed unseen."}
    (RESULTS / "external_osdr_mouse_filtered_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
