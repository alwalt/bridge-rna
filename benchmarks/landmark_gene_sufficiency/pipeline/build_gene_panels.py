#!/usr/bin/env python3
"""Build deterministic random sufficiency panels and the mapped L1000 panel."""

from __future__ import annotations

import json
import hashlib

import numpy as np
import pandas as pd

from common import CONFIG, REPO_ROOT, RESULTS, WORK


def main() -> None:
    human = pd.read_parquet(WORK / "archs4_holdout_human_genes.parquet").observed.to_numpy(bool)
    mouse = pd.read_parquet(WORK / "archs4_holdout_mouse_genes.parquet").observed.to_numpy(bool)
    common = np.flatnonzero(human & mouse)
    vocab = pd.read_csv(REPO_ROOT / "data/ensembl/canonical_genes.csv")
    mapping = pd.read_parquet(RESULTS / "l1000_model_mapping.parquet")
    mapping["observed_human"] = mapping.model_index.map(
        lambda value: bool(human[int(value)]) if pd.notna(value) else False
    )
    mapping["observed_mouse"] = mapping.model_index.map(
        lambda value: bool(mouse[int(value)]) if pd.notna(value) else False
    )
    mapping["jointly_evaluable"] = (
        mapping.mapping_status.eq("mapped") & mapping.observed_human & mapping.observed_mouse
    )
    mapping.to_parquet(RESULTS / "l1000_model_mapping.parquet", index=False)
    mapping.to_csv(RESULTS / "l1000_model_mapping.csv", index=False)
    l1000 = np.sort(mapping.loc[mapping.jointly_evaluable, "model_index"].astype(int).unique())
    if len(l1000) == 0:
        raise ValueError("No L1000 genes are jointly evaluable")

    rng_seed = int(CONFIG["benchmark_seed"])
    panel_rows, gene_rows = [], []

    def add_panel(panel_type: str, panel_id: str, indices: np.ndarray,
                  replicate: int | None, seed: int | None) -> None:
        panel_rows.append({"panel_type": panel_type, "panel_id": panel_id,
                           "visible_gene_count": len(indices), "replicate": replicate,
                           "seed": seed, "selection_species": "none"})
        gene_rows.extend({"panel_id": panel_id, "model_index": int(index),
                          "gene_symbol": vocab.gene_symbol.iloc[index]}
                         for index in indices)

    add_panel("l1000", "l1000", l1000, None, None)
    specifications = [("random_curve", int(size)) for size in CONFIG["random_visible_counts"]]
    specifications.append(("random_l1000_matched", len(l1000)))
    for panel_type, size in specifications:
        for replicate in range(int(CONFIG["random_panel_replicates"])):
            digest = hashlib.sha256(
                f"{rng_seed}|{panel_type}|{size}|{replicate}".encode()
            ).digest()
            token = int.from_bytes(digest[:8], "little")
            indices = np.sort(np.random.default_rng(token).choice(common, size=size, replace=False))
            panel_id = f"{panel_type}__n{size:04d}__r{replicate:02d}"
            add_panel(panel_type, panel_id, indices, replicate, rng_seed + replicate)
    panels = pd.DataFrame(panel_rows)
    genes = pd.DataFrame(gene_rows)
    panels.to_parquet(RESULTS / "panel_manifest.parquet", index=False)
    panels.to_csv(RESULTS / "panel_manifest.csv", index=False)
    genes.to_parquet(RESULTS / "panel_genes.parquet", index=False)
    summary = {"model_positions": len(human), "jointly_observed_genes": len(common),
               "l1000_source_genes": len(mapping), "l1000_model_overlap": int(mapping.mapping_status.eq("mapped").sum()),
               "l1000_jointly_evaluable": len(l1000), "panels": len(panels)}
    (RESULTS / "panel_provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
