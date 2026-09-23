#!/usr/bin/env python3
"""Frozen-background pathway analysis for radiation core/context sets."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/radiation_context_sensitivity"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
ENDPOINT = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def post(payload):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "BridgeRNA-benchmark/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    core = pd.read_csv(OUT / "radiation_core_genes.csv")
    recurrent = pd.read_csv(OUT / "recurrent_bridge_only_genes.csv")
    canonical = pd.read_csv(CANONICAL).gene_symbol.astype(str).tolist()

    queries = {}
    for method in ["Bridge", "DE"]:
        for definition in ["strict_universal", "majority_cross_study_core"]:
            key = f"{method}|{definition}"
            definitions = [definition]
            if definition == "majority_cross_study_core":
                definitions.append("strict_universal")
            queries[key] = sorted(core.loc[(core.method.eq(method)) &
                                           (core.classification.isin(definitions)), "gene"].unique())
    queries["Bridge-only|recurrent_cross_study"] = sorted(
        recurrent.loc[recurrent.dataset_count.ge(2), "gene"].unique()
    )

    # Context-restricted recurrent sets: at least two contrasts in one study and
    # absent from the other studies' top-500 lists.
    for method in ["Bridge", "DE"]:
        specific = core[(core.method.eq(method)) & (core.dataset_count.eq(1)) &
                        (core.contrast_count.ge(2))].copy()
        for dataset in ["GSE297090", "GSE297560"]:
            key = f"{method}|context_recurrent_{dataset}"
            queries[key] = sorted(specific.loc[specific.contrasts.str.contains(dataset), "gene"].unique())

    empty_queries = {key: value for key, value in queries.items() if not value}
    submitted_queries = {key: value for key, value in queries.items() if value}
    payload = {
        "organism": "hsapiens",
        "query": submitted_queries,
        "sources": ["GO:BP", "REAC", "KEGG"],
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "domain_scope": "custom",
        "background": canonical,
        "no_evidences": False,
    }
    response = post(payload)
    (OUT / "gprofiler_core_raw_response.json").write_text(json.dumps(response) + "\n")

    keys = list(submitted_queries)
    rows = []
    for result in response.get("result", []):
        query_key = result.get("query")
        if isinstance(query_key, list):
            query_key = query_key[0] if query_key else None
        if query_key not in submitted_queries and isinstance(query_key, int) and 1 <= query_key <= len(keys):
            query_key = keys[query_key - 1]
        if query_key not in submitted_queries:
            raise RuntimeError(f"Unresolved query identifier {result.get('query')!r}")
        evidence = result.get("intersections", [])
        intersection = [gene for gene, value in zip(submitted_queries[query_key], evidence) if value]
        rows.append({
            "query_id": query_key,
            "source": result.get("source"),
            "term_id": result.get("native"),
            "term_name": result.get("name"),
            "adjusted_pvalue": result.get("p_value"),
            "term_size": result.get("term_size"),
            "query_size": result.get("query_size"),
            "intersection_size": result.get("intersection_size"),
            "effective_domain_size": result.get("effective_domain_size"),
            "intersection_genes": ";".join(intersection),
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "core_pathway_enrichment.csv.gz", index=False, compression="gzip")

    summaries = []
    for key, genes in queries.items():
        subset = results[results.query_id.eq(key)].sort_values("adjusted_pvalue") if len(results) else results
        coherent = set()
        for value in subset.intersection_genes.dropna() if len(subset) else []:
            coherent.update(g for g in str(value).split(";") if g)
        top = subset.iloc[0] if len(subset) else None
        summaries.append({
            "query_id": key,
            "gene_count": len(genes),
            "significant_term_count": len(subset),
            "coherent_gene_count": len(coherent),
            "coherent_gene_fraction": len(coherent) / max(len(genes), 1),
            "top_source": top.source if top is not None else None,
            "top_term_id": top.term_id if top is not None else None,
            "top_term_name": top.term_name if top is not None else None,
            "top_adjusted_pvalue": top.adjusted_pvalue if top is not None else None,
        })
    pd.DataFrame(summaries).to_csv(OUT / "core_pathway_summary.csv", index=False)

    # Annotate every recurrent Bridge-only gene with significant pathway terms.
    bridge_only = results[results.query_id.eq("Bridge-only|recurrent_cross_study")]
    membership = {}
    for row in bridge_only.itertuples():
        for gene in str(row.intersection_genes).split(";"):
            if gene:
                membership.setdefault(gene, []).append((row.adjusted_pvalue, row.source, row.term_id, row.term_name))
    membership_rows = []
    for gene in recurrent.gene:
        terms = sorted(membership.get(gene, []))
        membership_rows.append({
            "gene": gene,
            "significant_pathway_count": len(terms),
            "top_pathways": "; ".join(f"{source}:{term_id} {name}" for _, source, term_id, name in terms[:5]),
        })
    membership_frame = pd.DataFrame(membership_rows)
    membership_frame.to_csv(OUT / "recurrent_bridge_only_pathway_membership.csv", index=False)
    recurrent.merge(membership_frame, on="gene", how="left").to_csv(
        OUT / "recurrent_bridge_only_genes_annotated.csv.gz", index=False, compression="gzip"
    )

    provenance = {
        "endpoint": ENDPOINT,
        "sources": ["GO:BP", "REAC", "KEGG"],
        "multiple_testing": "g:Profiler FDR per query",
        "domain_scope": "custom",
        "custom_background_genes": len(canonical),
        "query_sizes": {key: len(value) for key, value in queries.items()},
        "empty_queries_not_submitted": sorted(empty_queries),
        "service_meta": response.get("meta", {}),
        "raw_response_sha256": hashlib.sha256(
            (OUT / "gprofiler_core_raw_response.json").read_bytes()
        ).hexdigest(),
    }
    (OUT / "core_pathway_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
