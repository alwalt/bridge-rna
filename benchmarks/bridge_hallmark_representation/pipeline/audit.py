#!/usr/bin/env python3
"""Phase 0: resolve authoritative assets and freeze cohort/gene-set manifests."""
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import CONFIG, HERE, RESULTS, resolve, sha256, read_vocab, read_gmt, save_json

def file_record(path, role):
    p = resolve(path); return {"role": role, "path": str(p), "exists": p.exists(),
        "bytes": p.stat().st_size if p.exists() else None, "sha256": sha256(p) if p.exists() else None}

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    vocab, hallmarks = read_vocab(), read_gmt()
    if len(vocab) != 15165 or len(set(vocab)) != len(vocab): raise AssertionError("Canonical vocabulary invariant failed")
    gmeta = pd.read_parquet(resolve(CONFIG["gtex_metadata"]))
    gmeta = gmeta.loc[gmeta.species.eq("human")].iloc[:np.load(resolve(CONFIG["gtex_expression"]), mmap_mode="r").shape[0]].copy()
    gmeta["row_index"] = np.arange(len(gmeta)); gmeta["donor_id"] = gmeta.sample_id.str.split("-").str[:2].str.join("-")
    if gmeta.sample_id.duplicated().any(): raise AssertionError("Duplicate GTEx sample IDs")
    gmeta.to_parquet(RESULTS / "gtex_sample_manifest.parquet", index=False)
    membership=[]
    universe=set(vocab)
    for name, genes in hallmarks.items():
        overlap=sorted(genes & universe)
        membership.append({"hallmark":name,"source_genes":len(genes),"bridge_genes":len(overlap),
                           "mapping_loss":len(genes)-len(overlap),"prevalence":len(overlap)/len(vocab)})
    pd.DataFrame(membership).to_csv(RESULTS / "hallmark_gene_mapping.csv", index=False)
    model_cfg=json.loads(resolve(CONFIG["model_config"]).read_text())
    assets=[file_record(CONFIG[k],k) for k in ["checkpoint","model_config","canonical_genes","hallmark_gmt",
        "gtex_expression","gtex_embeddings","gtex_metadata","tcga_tpm","tcga_cart_metadata","gene_lengths"]]
    manifest={"benchmark":"bridge_hallmark_representation","created_by":"pipeline/audit.py",
      "python":sys.version,"platform":platform.platform(),"seed":CONFIG["seed"],"assets":assets,
      "model":{"run_id":model_cfg.get("run_id"),"architecture":model_cfg.get("architecture"),
               "normalization":model_cfg.get("normalization"),"pooling":"unweighted mean of final-layer contextual gene states",
               "frozen":True},"gene_vocabulary":{"count":len(vocab),"ordered":True},
      "hallmarks":{"collection":"MSigDB Hallmark 2020 via Enrichr/gseapy cache","count":len(hallmarks)},
      "cohorts":{"gtex_samples":len(gmeta),"tcga_samples":None},
      "reuse_decisions":{"gtex_embedding":"accepted: exact checkpoint/canonical order/natural log1p(TPM)/raw mean pooling",
        "tcga_embedding":"rejected: located caches lack a complete machine-readable preprocessing+sample-order manifest; regenerate",
        "previous_pathway_head":"prohibited and neither imported nor read as implementation input"}}
    save_json(RESULTS / "provenance_manifest.json", manifest)
    print(json.dumps({"gtex_samples":len(gmeta),"genes":len(vocab),"hallmarks":len(hallmarks)},indent=2))
if __name__ == "__main__": main()

