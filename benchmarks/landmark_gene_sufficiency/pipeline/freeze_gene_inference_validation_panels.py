#!/usr/bin/env python3
"""Freeze gene-inference Top-1000, controls, and all model-mapped L1000 genes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, sha256


OUT = RESULTS / "frozen_gene_inference_validation_panels.parquet"
PROVENANCE = RESULTS / "frozen_gene_inference_validation_panels_provenance.json"
PANEL_SIZE = 1000
RANDOM_REPLICATES = 50


def main() -> None:
    if OUT.exists() or PROVENANCE.exists():
        raise FileExistsError(f"Frozen artifact already exists: {OUT}")
    paths = {species: RESULTS / f"gene_inference_{species}_consensus_ranking.parquet"
             for species in ("human", "mouse")}
    rankings = {species: pd.read_parquet(path) for species, path in paths.items()}
    rows = []
    for species, ranking in rankings.items():
        selected = ranking.nsmallest(PANEL_SIZE, "gene_inference_rank")
        for row in selected.itertuples(index=False):
            rows.append({"panel_id": f"{species}_gene_inference_top1000",
                "panel_type": "gene_inference_top1000", "selection_species": species,
                "replicate": pd.NA, "seed": pd.NA, "model_index": int(row.model_index),
                "gene": str(row.gene), "selection_score": float(row.consensus_score),
                "selection_rank": int(row.gene_inference_rank)})
    merged = rankings["human"][["model_index", "gene", "consensus_score"]].merge(
        rankings["mouse"][["model_index", "consensus_score"]], on="model_index",
        suffixes=("_human", "_mouse"), validate="one_to_one")
    merged["cross_species_score"] = (merged.consensus_score_human.rank(pct=True)
                                      + merged.consensus_score_mouse.rank(pct=True)) / 2
    merged["cross_species_rank"] = merged.cross_species_score.rank(
        ascending=False, method="min").astype("Int64")
    selected = merged.nsmallest(PANEL_SIZE, "cross_species_rank")
    for row in selected.itertuples(index=False):
        rows.append({"panel_id": "conserved_gene_inference_top1000",
            "panel_type": "gene_inference_top1000", "selection_species": "human_mouse",
            "replicate": pd.NA, "seed": pd.NA, "model_index": int(row.model_index),
            "gene": str(row.gene), "selection_score": float(row.cross_species_score),
            "selection_rank": int(row.cross_species_rank)})
    l1000_path = RESULTS / "l1000_model_mapping.parquet"
    l1000 = pd.read_parquet(l1000_path)
    l1000 = l1000.loc[l1000.mapping_status.eq("mapped")].sort_values("model_index")
    if len(l1000) != 922 or l1000.model_index.nunique() != 922:
        raise ValueError(f"Expected all 922 uniquely model-mapped L1000 genes, found {len(l1000)}")
    for row in l1000.itertuples(index=False):
        rows.append({"panel_id": "l1000_all_mapped_922", "panel_type": "l1000_all_mapped",
            "selection_species": "external", "replicate": pd.NA, "seed": pd.NA,
            "model_index": int(row.model_index), "gene": str(row.model_symbol),
            "selection_score": np.nan, "selection_rank": pd.NA})
    universe = rankings["human"].model_index.to_numpy(int)
    genes = rankings["human"].set_index("model_index").gene.astype(str)
    seed = int(CONFIG["benchmark_seed"])
    for replicate in range(RANDOM_REPLICATES):
        panel_seed = seed + 30000 + replicate
        selected = np.sort(np.random.default_rng(panel_seed).choice(
            universe, PANEL_SIZE, replace=False))
        for index in selected:
            rows.append({"panel_id": f"random_1000_r{replicate:02d}",
                "panel_type": "random_1000", "selection_species": "none",
                "replicate": replicate, "seed": panel_seed, "model_index": int(index),
                "gene": str(genes.loc[index]), "selection_score": np.nan,
                "selection_rank": pd.NA})
    panels = pd.DataFrame(rows)
    counts = panels.groupby("panel_id").size()
    expected = {"human_gene_inference_top1000": 1000,
        "mouse_gene_inference_top1000": 1000,
        "conserved_gene_inference_top1000": 1000, "l1000_all_mapped_922": 922}
    expected.update({f"random_1000_r{i:02d}": 1000 for i in range(RANDOM_REPLICATES)})
    if counts.to_dict() != expected:
        raise AssertionError(counts)
    if panels.duplicated(["panel_id", "model_index"]).any():
        raise AssertionError("Duplicate gene within a frozen panel")
    panels.to_parquet(OUT, index=False)
    provenance = {"status": "frozen_before_external_validation",
        "objective": "masked-gene prediction across external biological samples",
        "top_panel_size": PANEL_SIZE, "random_replicates": RANDOM_REPLICATES,
        "benchmark_seed": seed, "random_seed_offset": 30000,
        "ranking_inputs": {species: {"path": str(path), "sha256": sha256(path)}
                           for species, path in paths.items()},
        "cross_species_selection": "mean percentile rank of human and mouse size-consensus scores",
        "l1000_input": {"path": str(l1000_path), "sha256": sha256(l1000_path),
            "official_landmarks": 978, "model_mapped": 922, "not_in_model": 56},
        "frozen_panel_sha256": sha256(OUT),
        "policy": "Do not modify or regenerate after viewing external validation results."}
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(counts.groupby(counts).size().to_string())
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
