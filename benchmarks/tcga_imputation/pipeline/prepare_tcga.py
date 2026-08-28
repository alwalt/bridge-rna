#!/usr/bin/env python3
"""Freeze 1,000 TCGA samples and prepare both models' native log1p(TPM)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fm_embed.sources.tcga import load_tcga_h5_counts  # noqa: E402
from common import CONFIG, RESULTS, WORK, norm_gene, sha256, tpm_log1p  # noqa: E402


TCGA_H5 = REPO_ROOT / "data/tcga/tcga_matrix.h5"
OUR_GENES = REPO_ROOT / "data/ensembl/canonical_genes.csv"
OUR_LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
BULK_INFO = REPO_ROOT / "model/BulkFormer/data/bulkformer_gene_info.csv"
HGNC = REPO_ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"


def select_samples() -> tuple[list[str], pd.DataFrame]:
    with h5py.File(TCGA_H5, "r") as handle:
        ids = [value.decode() for value in handle["meta/sampleid"][:]]
        cancer = [value.decode() for value in handle["meta/cancertype"][:]]
        sample_type = [value.decode() for value in handle["meta/gdc_cases.samples.sample_type"][:]]
    metadata = pd.DataFrame({"sample_id": ids, "cancer_type": cancer, "sample_type": sample_type})
    # Restrict to primary tumors to avoid conflating tissue status with random selection.
    eligible = metadata.loc[metadata["sample_type"].eq("Primary Tumor")].copy()
    if len(eligible) < int(CONFIG["num_samples"]):
        raise ValueError(f"Only {len(eligible)} primary tumors are available")
    selected = eligible.sample(n=int(CONFIG["num_samples"]), random_state=int(CONFIG["selection_seed"]))
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    return selected["sample_id"].tolist(), selected


def split_symbols(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [norm_gene(item) for item in str(value).split("|") if str(item).strip()]


def build_hgnc_crosswalk(source_symbols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    hgnc = pd.read_csv(HGNC, sep="\t", low_memory=False)
    hgnc = hgnc.loc[hgnc["status"].eq("Approved")].copy()
    approved = {norm_gene(symbol): norm_gene(symbol) for symbol in hgnc["symbol"]}
    previous: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in hgnc.itertuples(index=False):
        target = norm_gene(row.symbol)
        for symbol in split_symbols(row.prev_symbol):
            previous[symbol].add(target)
        for symbol in split_symbols(row.alias_symbol):
            aliases[symbol].add(target)

    rows = []
    provisional: dict[str, str] = {}
    for raw in source_symbols:
        source = norm_gene(raw)
        if source in approved:
            candidates, match_type = {approved[source]}, "approved_symbol"
        elif len(previous[source]) == 1:
            candidates, match_type = previous[source], "previous_symbol"
        elif len(previous[source]) > 1:
            candidates, match_type = previous[source], "ambiguous_previous_symbol"
        elif len(aliases[source]) == 1:
            candidates, match_type = aliases[source], "alias_symbol"
        elif len(aliases[source]) > 1:
            candidates, match_type = aliases[source], "ambiguous_alias_symbol"
        else:
            candidates, match_type = set(), "unmapped"
        target = next(iter(candidates)) if len(candidates) == 1 else None
        if target is not None:
            provisional[source] = target
        rows.append({"tcga_symbol": source, "approved_symbol": target,
                     "mapping_status": match_type,
                     "candidate_symbols": "|".join(sorted(candidates))})

    # Do not silently merge multiple source columns into one current gene.
    reverse: dict[str, list[str]] = defaultdict(list)
    for source, target in provisional.items():
        reverse[target].append(source)
    conflicts = {target for target, sources in reverse.items() if len(sources) > 1}
    for row in rows:
        target = row["approved_symbol"]
        if target in conflicts:
            sources = reverse[target]
            if target in sources:
                if row["tcga_symbol"] != target:
                    row["mapping_status"] = "shadowed_by_approved_symbol"
                    provisional.pop(row["tcga_symbol"], None)
            else:
                row["mapping_status"] = "multiple_tcga_symbols_for_approved_gene"
                provisional.pop(row["tcga_symbol"], None)
    return pd.DataFrame(rows), provisional


def resolve_model_genes(genes: list[str], crosswalk: pd.DataFrame) -> list[str | None]:
    # Model vocabularies are current, but route them through the same authority
    # so comparisons are based on approved symbols rather than raw labels.
    del crosswalk  # Kept in the signature to make the shared provenance explicit.
    hgnc = pd.read_csv(
        HGNC, sep="\t", usecols=["symbol", "status", "prev_symbol", "alias_symbol"],
        low_memory=False,
    )
    hgnc = hgnc.loc[hgnc["status"].eq("Approved")]
    approved = {norm_gene(symbol): norm_gene(symbol) for symbol in hgnc["symbol"]}
    previous: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in hgnc.itertuples(index=False):
        target = norm_gene(row.symbol)
        for symbol in split_symbols(row.prev_symbol):
            previous[symbol].add(target)
        for symbol in split_symbols(row.alias_symbol):
            aliases[symbol].add(target)
    resolved = []
    for gene in genes:
        if gene in approved:
            resolved.append(approved[gene])
        elif len(previous[gene]) == 1:
            resolved.append(next(iter(previous[gene])))
        elif len(aliases[gene]) == 1:
            resolved.append(next(iter(aliases[gene])))
        else:
            resolved.append(None)
    return resolved


def align_counts(counts: pd.DataFrame, approved_genes: list[str | None],
                 source_to_approved: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    source_index = {norm_gene(column): index for index, column in enumerate(counts.columns)}
    approved_to_source = {target: source for source, target in source_to_approved.items()}
    source_indices = np.asarray([
        source_index.get(approved_to_source.get(gene, ""), -1) if gene is not None else -1
        for gene in approved_genes
    ], dtype=np.int64)
    aligned = np.zeros((len(counts), len(approved_genes)), dtype=np.float32)
    observed = source_indices >= 0
    aligned[:, observed] = counts.to_numpy()[:, source_indices[observed]]
    return aligned, observed


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample_ids, selection = select_samples()
    counts, h5_metadata = load_tcga_h5_counts(TCGA_H5, sample_ids)
    selection = selection.merge(h5_metadata.reset_index(), on="sample_id", how="left")
    selection.to_parquet(WORK / "selected_tcga_samples.parquet", index=False)
    (WORK / "selected_tcga_sample_ids.txt").write_text("\n".join(sample_ids) + "\n")
    selection.to_parquet(RESULTS / "selected_tcga_samples.parquet", index=False)
    (RESULTS / "selected_tcga_sample_ids.txt").write_text("\n".join(sample_ids) + "\n")

    crosswalk, source_to_approved = build_hgnc_crosswalk(list(counts.columns))
    crosswalk.to_parquet(WORK / "tcga_hgnc_crosswalk.parquet", index=False)
    crosswalk.to_csv(RESULTS / "tcga_hgnc_crosswalk.csv", index=False)

    our_genes = pd.read_csv(OUR_GENES)["gene_symbol"].map(norm_gene).tolist()
    our_approved = resolve_model_genes(our_genes, crosswalk)
    length_table = pd.read_csv(OUR_LENGTHS)
    length_lookup = dict(zip(length_table["gene_symbol"].map(norm_gene), length_table["exon_length"]))
    missing_lengths = [gene for gene in our_genes if gene not in length_lookup]
    if missing_lengths:
        raise ValueError(f"Our vocabulary lacks exon lengths for {len(missing_lengths)} genes")
    our_counts, our_observed = align_counts(counts, our_approved, source_to_approved)
    our_matrix = tpm_log1p(our_counts, np.asarray([length_lookup[g] for g in our_genes]))

    bulk_info = pd.read_csv(BULK_INFO)
    bulk_genes = bulk_info["gene_symbol"].map(norm_gene).tolist()
    bulk_approved = resolve_model_genes(bulk_genes, crosswalk)
    bulk_counts, bulk_observed = align_counts(counts, bulk_approved, source_to_approved)
    bulk_matrix = tpm_log1p(bulk_counts, bulk_info["gene_length"].to_numpy())
    # Published BulkFormer convention: source-unavailable genes are mask tokens.
    bulk_matrix[:, ~bulk_observed] = float(CONFIG["mask_token"])

    np.save(WORK / "ours_log1p_tpm.npy", our_matrix)
    np.save(WORK / "bulkformer_log1p_tpm.npy", bulk_matrix)
    pd.DataFrame({"gene": our_genes, "approved_symbol": our_approved,
                  "native_index": np.arange(len(our_genes)),
                  "tcga_observed": our_observed}).to_parquet(WORK / "ours_genes.parquet", index=False)
    pd.DataFrame({"gene": bulk_genes, "approved_symbol": bulk_approved,
                  "ensg_id": bulk_info["ensg_id"],
                  "native_index": np.arange(len(bulk_genes)),
                  "tcga_observed": bulk_observed}).to_parquet(WORK / "bulkformer_genes.parquet", index=False)

    # Shared evaluation requires an unambiguous one-to-one model-node mapping.
    # A few BulkFormer rows collapse to the same current HGNC symbol (usually
    # duplicated/retired Ensembl records); silently picking one would make the
    # benchmark depend on row ordering.
    our_counts_by_symbol = pd.Series([g for g in our_approved if g is not None]).value_counts()
    bulk_counts_by_symbol = pd.Series([g for g in bulk_approved if g is not None]).value_counts()
    our_index = {
        gene: index for index, gene in enumerate(our_approved)
        if gene is not None and our_observed[index] and our_counts_by_symbol[gene] == 1
    }
    bulk_index = {
        gene: index for index, gene in enumerate(bulk_approved)
        if gene is not None and bulk_observed[index] and bulk_counts_by_symbol[gene] == 1
    }
    shared = sorted(set(our_index) & set(bulk_index))
    shared_table = pd.DataFrame({
        "gene": shared,
        "ours_index": [our_index[gene] for gene in shared],
        "bulkformer_index": [bulk_index[gene] for gene in shared],
    })
    shared_table.to_csv(WORK / "shared_genes.csv", index=False)
    shared_table.to_parquet(WORK / "shared_genes.parquet", index=False)
    shared_table.to_csv(RESULTS / "shared_genes.csv", index=False)

    annotation_summary = pd.DataFrame([
        {"vocabulary": "ours_45.6m", "native_genes": len(our_genes),
         "hgnc_resolved": sum(g is not None for g in our_approved),
         "tcga_observed": int(our_observed.sum()),
         "duplicate_approved_symbols": int((our_counts_by_symbol > 1).sum())},
        {"vocabulary": "bulkformer", "native_genes": len(bulk_genes),
         "hgnc_resolved": sum(g is not None for g in bulk_approved),
         "tcga_observed": int(bulk_observed.sum()),
         "duplicate_approved_symbols": int((bulk_counts_by_symbol > 1).sum())},
        {"vocabulary": "shared_evaluable", "native_genes": len(shared),
         "hgnc_resolved": len(shared), "tcga_observed": len(shared),
         "duplicate_approved_symbols": 0},
    ])
    annotation_summary.to_csv(RESULTS / "gene_annotation_summary.csv", index=False)

    manifest = {
        "samples": len(sample_ids), "selection_seed": CONFIG["selection_seed"],
        "selection_population": "TCGA Primary Tumor", "tcga_source": str(TCGA_H5),
        "our_native_genes": len(our_genes), "our_tcga_observed_genes": int(our_observed.sum()),
        "bulkformer_native_genes": len(bulk_genes),
        "bulkformer_tcga_observed_genes": int(bulk_observed.sum()),
        "model_vocab_intersection": len(set(our_genes) & set(bulk_genes)),
        "approved_model_vocab_intersection": len(
            set(g for g in our_approved if g is not None)
            & set(g for g in bulk_approved if g is not None)
        ),
        "our_duplicate_approved_symbols": int((our_counts_by_symbol > 1).sum()),
        "bulkformer_duplicate_approved_symbols": int((bulk_counts_by_symbol > 1).sum()),
        "shared_evaluable_genes": len(shared),
        "tcga_gene_labels": len(counts.columns),
        "tcga_hgnc_mapping_counts": crosswalk["mapping_status"].value_counts().to_dict(),
        "hgnc_source": str(HGNC), "hgnc_sha256": sha256(HGNC),
        "our_preprocessing": "raw counts -> GENCODE v49 exon-length TPM -> natural log1p",
        "bulkformer_preprocessing": "raw counts -> published BulkFormer gene-length TPM -> natural log1p",
        "source_sha256": sha256(TCGA_H5),
        "bulkformer_graph_md5_expected": {
            "G_tcga.pt": "1c4d43a92514a4c5c974006d892a4ef2",
            "G_tcga_weight.pt": "eebb7f5a75d0049e00fa4d86185a2b52"
        }
    }
    (WORK / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2))
    (RESULTS / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
