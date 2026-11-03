#!/usr/bin/env python3
"""Pathway enrichment for frozen Bridge/DE complementarity partitions."""

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/expanded_chembl_sensitivity"
ENDPOINT = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")


def post(payload):
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "BridgeRNA-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    partitions = pd.read_csv(OUT / "gene_partitions.csv.gz")
    recurrent = pd.read_csv(OUT / "recurrent_partition_genes.csv")
    canonical = pd.read_csv(CANONICAL).gene_symbol.astype(str).tolist()
    queries = {}
    metadata = {}
    for (condition, dataset, partition), group in partitions.groupby(["condition", "dataset", "partition"]):
        key = f"dataset|{condition}|{dataset}|{partition}"
        queries[key] = sorted(group.gene.unique())
        metadata[key] = {"scope": "dataset_partition", "condition": condition, "dataset": dataset, "partition": partition}
    for condition in sorted(recurrent.condition.unique()):
        group = recurrent[(recurrent.condition.eq(condition)) & recurrent.partition.eq("Bridge-only")]
        key = f"recurrent|{condition}|Bridge-only"
        queries[key] = sorted(group.gene.unique())
        metadata[key] = {"scope": "recurrent_partition", "condition": condition, "dataset": "recurrent_ge2", "partition": "Bridge-only"}

    responses, result_rows, meta_snapshots = {}, [], []
    # Small batches make the saved response easy to audit and retry.
    keys = list(queries)
    for start in range(0, len(keys), 8):
        batch_keys = keys[start:start+8]
        payload = {"organism": "hsapiens", "query": {k: queries[k] for k in batch_keys},
                   "sources": ["GO:BP", "REAC", "KEGG"], "user_threshold": 0.05,
                   "significance_threshold_method": "fdr", "domain_scope": "custom",
                   "background": canonical, "no_evidences": False}
        response = post(payload)
        responses[f"batch_{start//8}"] = response
        meta_snapshots.append(response.get("meta", {}))
        for row in response.get("result", []):
            query_key = row.get("query")
            if isinstance(query_key, list):
                query_key = query_key[0] if query_key else None
            # Multi-query responses use query number; map through meta when needed.
            if query_key not in metadata:
                query_index = row.get("query", 1)
                if isinstance(query_index, int) and 1 <= query_index <= len(batch_keys):
                    query_key = batch_keys[query_index-1]
            if query_key not in metadata:
                # Current API returns the query key in the `query` field; fail rather
                # than silently assigning a pathway to the wrong partition.
                raise RuntimeError(f"Unresolved g:Profiler query key: {row.get('query')!r}")
            base = metadata[query_key]
            intersection_genes = [gene for gene, evidence in zip(queries[query_key], row.get("intersections", []))
                                  if evidence]
            result_rows.append({**base, "query_id": query_key, "source": row.get("source"),
                                "term_id": row.get("native"), "term_name": row.get("name"),
                                "pvalue": row.get("p_value"), "term_size": row.get("term_size"),
                                "query_size": row.get("query_size"),
                                "intersection_size": row.get("intersection_size"),
                                "effective_domain_size": row.get("effective_domain_size"),
                                "intersection_genes": ";".join(intersection_genes)})

    (OUT / "gprofiler_raw_responses.json").write_text(json.dumps(responses) + "\n")
    results = pd.DataFrame(result_rows)
    if len(results):
        results.to_csv(OUT / "partition_pathway_enrichment.csv.gz", index=False, compression="gzip")
    summary_rows = []
    for key, genes in queries.items():
        subset = results[results.query_id.eq(key)] if len(results) else pd.DataFrame()
        hit_genes = set()
        if len(subset):
            for value in subset.intersection_genes.dropna():
                hit_genes.update(x for x in str(value).split(";") if x)
            top = subset.sort_values("pvalue").iloc[0]
        else:
            top = None
        summary_rows.append({**metadata[key], "query_id": key, "gene_count": len(genes),
                             "significant_term_count": len(subset),
                             "coherent_gene_count": len(hit_genes),
                             "coherent_gene_fraction": len(hit_genes)/max(len(genes), 1),
                             "top_source": top.source if top is not None else None,
                             "top_term_id": top.term_id if top is not None else None,
                             "top_term_name": top.term_name if top is not None else None,
                             "top_adjusted_pvalue": top.pvalue if top is not None else None})
    pd.DataFrame(summary_rows).to_csv(OUT / "partition_pathway_summary.csv", index=False)
    provenance = {"endpoint": ENDPOINT, "sources": ["GO:BP", "REAC", "KEGG"],
                  "multiple_testing": "g:Profiler FDR per query", "domain_scope": "custom",
                  "custom_background_genes": len(canonical), "meta_snapshots": meta_snapshots,
                  "raw_response_sha256": hashlib.sha256((OUT / "gprofiler_raw_responses.json").read_bytes()).hexdigest()}
    (OUT / "pathway_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
