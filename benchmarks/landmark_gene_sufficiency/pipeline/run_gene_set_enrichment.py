#!/usr/bin/env python3
"""Run GO:BP and Reactome enrichment with the frozen 15,165-gene background."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from common import RESULTS

URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-prefix", default="final",
                        help="Ranking artifact prefix (for example: final or definitive)")
    args = parser.parse_args()
    prefix = args.ranking_prefix
    rankings = {species: pd.read_parquet(RESULTS / f"{prefix}_{species}_informative_gene_ranking.parquet")
                for species in ("human", "mouse")}
    n = 921
    top = {species: set(frame.nsmallest(n, "information_rank").gene.astype(str))
           for species, frame in rankings.items()}
    shared = sorted(top["human"] & top["mouse"])
    human = rankings["human"].set_index("gene")
    mouse = rankings["mouse"].set_index("gene")
    l1000 = set(pd.read_parquet(RESULTS / "l1000_model_mapping.parquet").loc[
        lambda frame: frame.jointly_evaluable, "model_symbol"].astype(str))
    shared_table = pd.DataFrame({
        "gene": shared,
        "human_rank": [int(human.loc[gene, "information_rank"]) for gene in shared],
        "mouse_rank": [int(mouse.loc[gene, "information_rank"]) for gene in shared],
        "human_score": [float(human.loc[gene, "information_score"]) for gene in shared],
        "mouse_score": [float(mouse.loc[gene, "information_score"]) for gene in shared],
        "human_expression_mean": [float(human.loc[gene, "expression_mean"]) for gene in shared],
        "mouse_expression_mean": [float(mouse.loc[gene, "expression_mean"]) for gene in shared],
        "human_detection_fraction": [float(human.loc[gene, "detection_fraction"]) for gene in shared],
        "mouse_detection_fraction": [float(mouse.loc[gene, "detection_fraction"]) for gene in shared],
        "in_l1000": [gene in l1000 for gene in shared],
    })
    shared_table["mean_rank"] = shared_table[["human_rank", "mouse_rank"]].mean(axis=1)
    shared_table["max_rank"] = shared_table[["human_rank", "mouse_rank"]].max(axis=1)
    shared_table = shared_table.sort_values(["mean_rank", "max_rank", "gene"])
    stem = "" if prefix == "final" else f"{prefix}_"
    shared_name = "shared_top_451_genes" if prefix == "final" else f"{prefix}_shared_top_genes"
    shared_table.to_parquet(RESULTS / f"{shared_name}.parquet", index=False)
    shared_table.to_csv(RESULTS / f"{shared_name}.csv", index=False)
    queries = {f"shared_top_{len(shared)}": sorted(top["human"] & top["mouse"]),
               "human_top_921": sorted(top["human"]), "mouse_top_921": sorted(top["mouse"])}
    background = rankings["human"].sort_values("model_index").gene.astype(str).tolist()
    if len(background) != 15165 or len(set(background)) != 15165:
        raise AssertionError("Enrichment background must be exactly 15,165 unique eligible genes")
    payload = {"organism": "hsapiens", "query": queries, "sources": ["GO:BP", "REAC"],
               "user_threshold": 0.05, "domain_scope": "custom",
               "background": background, "no_evidences": False}
    response = requests.post(URL, json=payload, timeout=300)
    response.raise_for_status()
    raw = response.json()
    (RESULTS / f"{stem}informative_gene_enrichment_raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    result = pd.DataFrame(raw.get("result", []))
    if not result.empty:
        keep = [column for column in ["query", "source", "native", "name", "p_value", "significant",
            "term_size", "query_size", "intersection_size", "effective_domain_size", "precision",
            "recall", "intersection"] if column in result]
        result = result[keep].sort_values(["query", "source", "p_value"])
    result.to_parquet(RESULTS / f"{stem}informative_gene_enrichment.parquet", index=False)
    provenance = {"service": "g:Profiler g:GOSt", "endpoint": URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "organism": "hsapiens",
        "sources": ["GO:BP", "REAC"], "correction_method": "g:SCS", "threshold": 0.05,
        "domain_scope": "custom", "background_genes": len(background),
        "background_unique_genes": len(set(background)), "ranking_prefix": prefix,
        "queries": {name: len(genes) for name, genes in queries.items()},
        "service_meta": raw.get("meta", {})}
    (RESULTS / f"{stem}informative_gene_enrichment_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(result.groupby(["query", "source"]).size().to_string() if not result.empty else "No enriched terms")


if __name__ == "__main__": main()
