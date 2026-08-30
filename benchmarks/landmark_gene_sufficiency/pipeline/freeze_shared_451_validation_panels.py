#!/usr/bin/env python3
"""Freeze Shared-451 and 50 deterministic size-matched random panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, sha256

OUT = RESULTS / "frozen_shared_451_validation_panels.parquet"
PROVENANCE = RESULTS / "frozen_shared_451_validation_panels_provenance.json"
SHARED = RESULTS / "shared_top_451_genes.parquet"
RANKING = RESULTS / "final_human_informative_gene_ranking.parquet"
PANEL_SIZE = 451
RANDOM_REPLICATES = 50


def main() -> None:
    if OUT.exists() or PROVENANCE.exists():
        raise FileExistsError(
            f"Frozen panels already exist at {OUT}. Never silently refreeze after validation."
        )
    shared = pd.read_parquet(SHARED)
    ranking = pd.read_parquet(RANKING).sort_values("model_index")
    if len(shared) != PANEL_SIZE or shared.gene.nunique() != PANEL_SIZE:
        raise AssertionError("Expected the frozen Shared-451 gene set")
    if len(ranking) != 15165 or ranking.model_index.nunique() != 15165:
        raise AssertionError("Expected the complete 15,165-gene universe")
    gene_to_index = ranking.set_index("gene").model_index.astype(int)
    shared_indices = gene_to_index.reindex(shared.gene.astype(str))
    if shared_indices.isna().any() or shared_indices.nunique() != PANEL_SIZE:
        raise AssertionError("Shared-451 does not map uniquely to the model vocabulary")

    rows = []
    for gene, index in zip(shared.gene.astype(str), shared_indices.astype(int)):
        rows.append({"panel_id": "shared_451", "panel_type": "shared_451",
                     "selection_species": "human_mouse", "replicate": pd.NA,
                     "seed": pd.NA, "model_index": index, "gene": gene})
    universe = ranking.model_index.to_numpy(int)
    index_to_gene = ranking.set_index("model_index").gene.astype(str)
    seed = int(CONFIG["benchmark_seed"])
    for replicate in range(RANDOM_REPLICATES):
        panel_seed = seed + 20000 + replicate
        selected = np.sort(np.random.default_rng(panel_seed).choice(
            universe, PANEL_SIZE, replace=False))
        for index in selected:
            rows.append({"panel_id": f"random_451_r{replicate:02d}",
                         "panel_type": "random_451", "selection_species": "none",
                         "replicate": replicate, "seed": panel_seed,
                         "model_index": int(index),
                         "gene": str(index_to_gene.loc[index])})
    panels = pd.DataFrame(rows)
    counts = panels.groupby("panel_id").size()
    if len(counts) != RANDOM_REPLICATES + 1 or not counts.eq(PANEL_SIZE).all():
        raise AssertionError(counts)
    panels.to_parquet(OUT, index=False)
    provenance = {
        "status": "frozen_before_shared_451_external_validation",
        "panel_size": PANEL_SIZE, "random_replicates": RANDOM_REPLICATES,
        "benchmark_seed": seed, "random_seed_offset": 20000,
        "shared_input": {"path": str(SHARED), "sha256": sha256(SHARED)},
        "ranking_input": {"path": str(RANKING), "sha256": sha256(RANKING)},
        "frozen_panel_sha256": sha256(OUT),
        "policy": "Do not modify or regenerate after viewing validation results.",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(counts.groupby(counts).size().to_string())
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
