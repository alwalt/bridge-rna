#!/usr/bin/env python3
"""Align the released processed expression to native model vocabularies.

The release does not identify an invertible raw-count source. Its values look like
batch-corrected log expression and include negatives. We preserve them for the raw/PCA
controls. For FM inputs, negatives are clipped to zero and values are treated as log2(TPM+1),
then converted to natural-log units. This is explicit approximation, not exact TPM recovery.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, SHARED, WORK, write_json


def aligned(frame: pd.DataFrame, vocabulary: list[str], missing: float) -> tuple[np.ndarray, np.ndarray]:
    source = {str(column): i for i, column in enumerate(frame.columns)}
    indices = np.asarray([source.get(str(gene), -1) for gene in vocabulary])
    observed = indices >= 0
    output = np.full((len(frame), len(vocabulary)), missing, dtype=np.float32)
    output[:, observed] = frame.iloc[:, indices[observed]].to_numpy(dtype=np.float32)
    return output, observed


def main() -> None:
    source = pd.read_csv(CONFIG["source"]["expression"], low_memory=False)
    meta = ["depmap_id", "cell_line_display_name", "lineage_1", "lineage_2",
            "lineage_3", "lineage_6", "lineage_4"]
    expression = source.drop(columns=meta)
    raw = expression.to_numpy(dtype=np.float32)
    np.save(WORK / "raw_expression.npy", raw)
    canonical = pd.read_csv(SHARED / "data/ensembl/canonical_genes.csv")
    bridge, bridge_seen = aligned(expression, canonical.gene_symbol.astype(str).tolist(), 0.0)
    bridge = np.clip(bridge, 0, None) * np.float32(np.log(2.0))
    np.save(WORK / "bridge_log1p_tpm_approx.npy", bridge)
    bulk = pd.read_csv(SHARED / "model/BulkFormer/data/bulkformer_gene_info.csv")
    bulk_values, bulk_seen = aligned(expression, bulk.gene_symbol.astype(str).tolist(), -10.0)
    present = bulk_values != -10.0
    bulk_values[present] = np.clip(bulk_values[present], 0, None) * np.float32(np.log(2.0))
    np.save(WORK / "bulkformer_log1p_tpm_approx.npy", bulk_values)
    source[meta].to_parquet(WORK / "sample_metadata.parquet", index=False)
    write_json(RESULTS / "input_preprocessing.json", {
        "source_semantics": "released processed expression; exact transform undocumented",
        "raw_pca": "released values unchanged",
        "fm_approximation": "clip negative values to zero; multiply by ln(2), assuming log2(TPM+1)",
        "bridge_genes_observed": int(bridge_seen.sum()),
        "bridge_genes_total": len(bridge_seen),
        "bulkformer_genes_observed": int(bulk_seen.sum()),
        "bulkformer_genes_total": len(bulk_seen),
        "limitation": "Not equivalent to natural log1p(TPM) if the released matrix was batch corrected."
    })
    print(f"saved raw={raw.shape}, bridge={bridge.shape}, bulkformer={bulk_values.shape}")


if __name__ == "__main__":
    main()
