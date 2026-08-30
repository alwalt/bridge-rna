#!/usr/bin/env python3
"""Freeze ranked and control gene panels before held-out validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS

OUT = RESULTS / "frozen_validation_panels.parquet"
PROVENANCE = RESULTS / "frozen_validation_panels_provenance.json"
N = 921
RANDOM_REPLICATES = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    # Immutability guard: never silently refreeze after validation is examined.
    if OUT.exists() or PROVENANCE.exists():
        raise FileExistsError(
            f"Frozen panels already exist at {OUT}. Delete explicitly only if invalidating all validation results."
        )
    rows = []
    ranking_paths = {
        species: RESULTS / f"final_{species}_informative_gene_ranking.parquet"
        for species in ("human", "mouse")
    }
    rankings = {species: pd.read_parquet(path) for species, path in ranking_paths.items()}
    for species, frame in rankings.items():
        eligible = frame.dropna(subset=["information_rank"]).sort_values(
            ["information_rank", "model_index"], kind="stable"
        )
        selections = {f"{species}_top": eligible.head(N), f"{species}_bottom": eligible.tail(N)}
        for panel_id, selected in selections.items():
            for row in selected.itertuples(index=False):
                rows.append({"panel_id": panel_id, "panel_type": panel_id.rsplit("_", 1)[-1],
                    "selection_species": species, "replicate": pd.NA, "seed": pd.NA,
                    "model_index": int(row.model_index), "gene": row.gene,
                    "selection_score": float(row.information_score),
                    "selection_rank": int(row.information_rank)})
    l1000 = pd.read_parquet(RESULTS / "l1000_model_mapping.parquet")
    l1000 = l1000.loc[l1000.jointly_evaluable].sort_values("model_index")
    if len(l1000) != N: raise ValueError(f"Expected {N} jointly evaluable L1000 genes, found {len(l1000)}")
    for row in l1000.itertuples(index=False):
        rows.append({"panel_id": "l1000", "panel_type": "l1000", "selection_species": "external",
            "replicate": pd.NA, "seed": pd.NA, "model_index": int(row.model_index),
            "gene": row.model_symbol, "selection_score": np.nan, "selection_rank": pd.NA})
    universe = np.arange(15165, dtype=int)
    genes = rankings["human"].set_index("model_index").gene
    seed = int(CONFIG["benchmark_seed"])
    for replicate in range(RANDOM_REPLICATES):
        panel_seed = seed + 10000 + replicate
        selected = np.sort(np.random.default_rng(panel_seed).choice(universe, N, replace=False))
        for index in selected:
            rows.append({"panel_id": f"random_921_r{replicate:02d}", "panel_type": "random",
                "selection_species": "none", "replicate": replicate, "seed": panel_seed,
                "model_index": int(index), "gene": genes.loc[index],
                "selection_score": np.nan, "selection_rank": pd.NA})
    panels = pd.DataFrame(rows)
    counts = panels.groupby("panel_id").size()
    if not counts.eq(N).all(): raise AssertionError(counts[counts.ne(N)])
    panels.to_parquet(OUT, index=False)
    provenance = {
        "status": "frozen_before_validation", "panel_size": N,
        "random_replicates": RANDOM_REPLICATES, "benchmark_seed": seed,
        "ranking_inputs": {species: {"path": str(path), "sha256": sha256(path)}
                           for species, path in ranking_paths.items()},
        "l1000_input": {"path": str(RESULTS / "l1000_model_mapping.parquet"),
                         "sha256": sha256(RESULTS / "l1000_model_mapping.parquet")},
        "frozen_panel_sha256": sha256(OUT),
        "policy": "Do not modify or regenerate after viewing held-out validation results.",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(counts.groupby(counts).size().to_string())
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__": main()
