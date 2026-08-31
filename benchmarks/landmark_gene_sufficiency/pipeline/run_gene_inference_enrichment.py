#!/usr/bin/env python3
"""GO Biological Process and Reactome enrichment for conserved gene-inference genes."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import requests

from common import RESULTS, sha256


URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
PANEL_PATH = RESULTS / "frozen_gene_inference_validation_panels.parquet"
RANKING_PATH = RESULTS / "gene_inference_human_consensus_ranking.parquet"
STEM = "gene_inference_conserved_top1000_enrichment"


def main() -> None:
    panels = pd.read_parquet(PANEL_PATH)
    query = panels.loc[panels.panel_id.eq("conserved_gene_inference_top1000"), "gene"].astype(str).tolist()
    if len(query) != 1000 or len(set(query)) != 1000:
        raise AssertionError("Expected the frozen conserved Top-1000 panel")
    ranking = pd.read_parquet(RANKING_PATH).sort_values("model_index")
    background = ranking.gene.astype(str).tolist()
    if len(background) != 15165 or len(set(background)) != 15165:
        raise AssertionError("Background must be exactly 15,165 unique model genes")
    payload = {"organism": "hsapiens", "query": {"conserved_gene_inference_top1000": query},
        "sources": ["GO:BP", "REAC"], "user_threshold": 0.05,
        "domain_scope": "custom", "background": background, "no_evidences": False}
    response = requests.post(URL, json=payload, timeout=300)
    response.raise_for_status()
    raw = response.json()
    (RESULTS / f"{STEM}_raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    result = pd.DataFrame(raw.get("result", []))
    if not result.empty:
        keep = [column for column in ["query", "source", "native", "name", "p_value",
            "significant", "term_size", "query_size", "intersection_size",
            "effective_domain_size", "precision", "recall", "intersection"] if column in result]
        result = result[keep].sort_values(["source", "p_value", "native"])
    result.to_parquet(RESULTS / f"{STEM}.parquet", index=False)
    result.to_csv(RESULTS / f"{STEM}.csv", index=False)
    result_meta = raw.get("meta", {}).get("result_metadata", {})
    summary = pd.DataFrame([{"source": source,
        "tested_terms": int(result_meta.get(source, {}).get("number_of_terms", 0)),
        "significant_terms": int((result.source == source).sum()) if "source" in result else 0,
        "query_genes": len(query), "background_genes": len(background)}
        for source in ("GO:BP", "REAC")])
    summary.to_parquet(RESULTS / f"{STEM}_summary.parquet", index=False)
    summary.to_csv(RESULTS / f"{STEM}_summary.csv", index=False)
    provenance = {"service": "g:Profiler g:GOSt", "endpoint": URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "organism": "hsapiens",
        "sources": ["GO:BP", "REAC"], "correction_method": "g:SCS", "threshold": 0.05,
        "domain_scope": "custom", "query": "conserved_gene_inference_top1000",
        "query_genes": len(query), "background_genes": len(background),
        "background_unique_genes": len(set(background)),
        "frozen_panel": {"path": str(PANEL_PATH), "sha256": sha256(PANEL_PATH)},
        "background_ranking": {"path": str(RANKING_PATH), "sha256": sha256(RANKING_PATH)},
        "service_meta": raw.get("meta", {})}
    (RESULTS / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(result.groupby("source").size().to_string() if not result.empty else "No enriched terms")
    print("\nSummary\n" + summary.to_string(index=False))
    if not result.empty:
        print("\nTop terms\n" + result.groupby("source", group_keys=False).head(10)[
            ["source", "native", "name", "p_value", "intersection_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
