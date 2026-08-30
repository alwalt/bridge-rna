#!/usr/bin/env python3
"""Download, verify, and map the official Broad LINCS L1000 landmark list."""

from __future__ import annotations

import gzip
import json
import urllib.request
from collections import defaultdict
from datetime import date

import pandas as pd

from common import REFERENCES, REPO_ROOT, RESULTS, sha256


URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
       "GSE92742_Broad_LINCS_gene_info.txt.gz")
RAW = REFERENCES / "GSE92742_Broad_LINCS_gene_info.txt.gz"
HGNC = REPO_ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
VOCAB = REPO_ROOT / "data/ensembl/canonical_genes.csv"


def norm(value: object) -> str:
    return str(value).strip().upper()


def main() -> None:
    REFERENCES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not RAW.is_file():
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, RAW)
    with gzip.open(RAW, "rt") as handle:
        source = pd.read_csv(handle, sep="\t", dtype=str)
    landmarks = source.loc[source.pr_is_lm.eq("1")].copy()
    if len(landmarks) != 978 or landmarks.pr_gene_id.duplicated().any():
        raise ValueError(f"Expected 978 unique landmark genes, found {len(landmarks)}")

    hgnc = pd.read_csv(HGNC, sep="\t", dtype=str)
    approved = {norm(value): norm(value) for value in hgnc.symbol}
    aliases: dict[str, set[str]] = defaultdict(set)
    entrez: dict[str, set[str]] = defaultdict(set)
    for row in hgnc.itertuples():
        target = norm(row.symbol)
        if pd.notna(row.entrez_id):
            entrez[str(row.entrez_id).strip()].add(target)
        for field in (row.prev_symbol, row.alias_symbol):
            if pd.notna(field):
                for item in str(field).split("|"):
                    aliases[norm(item)].add(target)
    unique_alias = {key: next(iter(values)) for key, values in aliases.items() if len(values) == 1}
    unique_entrez = {key: next(iter(values)) for key, values in entrez.items() if len(values) == 1}

    vocab = pd.read_csv(VOCAB)
    model_symbols = vocab.gene_symbol.map(norm).tolist()
    model_approved = [approved.get(gene, unique_alias.get(gene, gene)) for gene in model_symbols]
    approved_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, gene in enumerate(model_approved):
        approved_to_indices[gene].append(index)

    records = []
    for row in landmarks.itertuples():
        original = norm(row.pr_gene_symbol)
        gene_id = str(row.pr_gene_id).strip()
        if gene_id in unique_entrez:
            target, method = unique_entrez[gene_id], "hgnc_entrez_id"
        elif original in approved:
            target, method = approved[original], "hgnc_approved_symbol"
        elif original in unique_alias:
            target, method = unique_alias[original], "hgnc_unique_alias"
        else:
            target, method = None, "unresolved"
        indices = approved_to_indices.get(target, []) if target else []
        status = "mapped" if len(indices) == 1 else ("not_in_model" if not indices else "model_conflict")
        records.append({
            "l1000_entrez_id": gene_id, "l1000_symbol": original,
            "l1000_title": row.pr_gene_title, "hgnc_symbol": target,
            "mapping_method": method, "mapping_status": status,
            "model_symbol": model_symbols[indices[0]] if len(indices) == 1 else None,
            "model_index": indices[0] if len(indices) == 1 else pd.NA,
        })
    mapping = pd.DataFrame(records)
    if mapping.mapping_status.eq("mapped").sum() != 922:
        raise ValueError("Expected the audited 922-gene model overlap; mapping inputs changed")
    mapping.to_parquet(RESULTS / "l1000_model_mapping.parquet", index=False)
    mapping.to_csv(RESULTS / "l1000_model_mapping.csv", index=False)
    manifest = {
        "url": URL, "geo_accession": "GSE92742", "retrieved": date.today().isoformat(),
        "sha256": sha256(RAW), "selection": "pr_is_lm == 1", "source_landmarks": 978,
        "hgnc_resolved": int(mapping.hgnc_symbol.notna().sum()),
        "model_overlap": int(mapping.mapping_status.eq("mapped").sum()),
        "hgnc_file": str(HGNC), "hgnc_sha256": sha256(HGNC),
        "model_vocab": str(VOCAB), "model_vocab_sha256": sha256(VOCAB),
    }
    (REFERENCES / "l1000_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
