#!/usr/bin/env python3
"""Audit whether HGNC reconciliation expands the model's gene universe.

This is a read-only, one-off analysis.  A candidate gene must:

1. belong to an unambiguous Ensembl mouse/human one-to-one ortholog pair;
2. resolve to one current HGNC approved human symbol;
3. occur in both local ARCHS4 H5 gene annotations;
4. have non-zero expression in TCGA (human evidence); and
5. have non-zero expression in at least one local OSDR count file (mouse evidence).

The script compares that reconciled universe with the existing 15,165-gene
vocabulary.  It does not modify training data or a model vocabulary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HGNC = ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
DEFAULT_ORTHOLOGS = ROOT / "data/ensembl/orthologs_one2one.txt"
DEFAULT_VOCAB = ROOT / "data/ensembl/canonical_genes.csv"
DEFAULT_TCGA = ROOT / "data/tcga/tcga_matrix.h5"
DEFAULT_ARCHS4_HUMAN = ROOT / "data/archs4/human_gene_v2.5.h5"
DEFAULT_ARCHS4_MOUSE = ROOT / "data/archs4/mouse_gene_v2.5.h5"
DEFAULT_OSDR = ROOT / "data/osdr/raw"
DEFAULT_HUMAN_LENGTHS = ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
DEFAULT_MOUSE_LENGTHS = ROOT / "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"
DEFAULT_OUTPUT = ROOT / "analysis/gene_universe_hgnc_audit"


def stable_id(value: object) -> str:
    """Normalize a possibly versioned Ensembl stable ID."""
    return str(value).strip().split(".", 1)[0].upper()


def symbol(value: object) -> str:
    return str(value).strip().upper()


def split_symbols(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [symbol(item) for item in str(value).split("|") if item.strip()]


def decode(values: np.ndarray) -> list[str]:
    return [item.decode() if isinstance(item, bytes) else str(item) for item in values]


def load_hgnc(path: Path) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Return unique symbol aliases, Ensembl-to-approved symbols, and protein-coding symbols."""
    frame = pd.read_csv(path, sep="\t", dtype=str)
    approved = {symbol(row.symbol): symbol(row.symbol) for row in frame.itertuples()}
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples():
        target = symbol(row.symbol)
        candidates[target].add(target)
        for field in ("prev_symbol", "alias_symbol"):
            for old in split_symbols(getattr(row, field)):
                candidates[old].add(target)
    unique_alias = {old: next(iter(targets)) for old, targets in candidates.items() if len(targets) == 1}
    ensembl_to_symbol: dict[str, str] = {}
    ensembl_candidates: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples():
        if pd.notna(row.ensembl_gene_id):
            ensembl_candidates[stable_id(row.ensembl_gene_id)].add(symbol(row.symbol))
    for gene_id, targets in ensembl_candidates.items():
        if len(targets) == 1:
            ensembl_to_symbol[gene_id] = next(iter(targets))
    protein_coding = {
        symbol(row.symbol) for row in frame.itertuples()
        if str(row.locus_group).strip().lower() == "protein-coding gene"
    }
    return unique_alias | approved, ensembl_to_symbol, protein_coding


def load_archs4_gene_ids(path: Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        return {stable_id(item) for item in decode(handle["meta/genes/ensembl_gene"][:])}


def tcga_expressed_symbols(path: Path, alias_to_approved: dict[str, str],
                           block: int = 512) -> tuple[set[str], set[str]]:
    """Find symbols with any non-zero TCGA count without loading the full matrix."""
    with h5py.File(path, "r") as handle:
        genes = [symbol(item) for item in decode(handle["meta/genes"][:])]
        matrix = handle["data/expression"]
        observed = np.zeros(len(genes), dtype=bool)
        for start in range(0, matrix.shape[0], block):
            observed |= np.any(matrix[start : start + block, :] != 0, axis=0)
    raw = {gene for gene, is_observed in zip(genes, observed) if is_observed}
    reconciled = {
        alias_to_approved[gene] for gene, is_observed in zip(genes, observed)
        if is_observed and gene in alias_to_approved
    }
    return raw, reconciled


def osdr_expressed_mouse_ids(directory: Path) -> tuple[set[str], int]:
    """Collect mouse Ensembl IDs with a non-zero count in any OSDR CSV."""
    observed: set[str] = set()
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No OSDR CSV files found under {directory}")
    ensembl_pattern = re.compile(r"^ENSMUSG\d+$")
    for path in files:
        for chunk in pd.read_csv(path, chunksize=5000):
            ids = chunk.iloc[:, 0].map(stable_id)
            numeric = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0)
            nonzero = numeric.ne(0).any(axis=1).to_numpy()
            observed.update(gene_id for gene_id, keep in zip(ids, nonzero)
                            if keep and ensembl_pattern.match(gene_id))
    return observed, len(files)


def one_to_one_orthologs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame = frame.loc[
        frame["Human homology type"].eq("ortholog_one2one")
        & frame["Human orthology confidence [0 low, 1 high]"].eq("1")
    ].copy()
    frame["mouse_ensembl"] = frame["Gene stable ID"].map(stable_id)
    frame["human_ensembl"] = frame["Human gene stable ID"].map(stable_id)
    mouse_degree = frame.groupby("mouse_ensembl")["human_ensembl"].nunique()
    human_degree = frame.groupby("human_ensembl")["mouse_ensembl"].nunique()
    frame = frame.loc[
        frame.mouse_ensembl.map(mouse_degree).eq(1)
        & frame.human_ensembl.map(human_degree).eq(1)
    ].drop_duplicates(["mouse_ensembl", "human_ensembl"])
    return frame[["mouse_ensembl", "human_ensembl", "Gene name", "Human gene name"]].rename(
        columns={"Gene name": "mouse_symbol_original", "Human gene name": "human_symbol_original"}
    )


def build_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    aliases, hgnc_by_ensembl, protein_coding = load_hgnc(args.hgnc)
    orthologs = one_to_one_orthologs(args.orthologs)
    archs4_human = load_archs4_gene_ids(args.archs4_human)
    archs4_mouse = load_archs4_gene_ids(args.archs4_mouse)
    tcga_expressed_raw, tcga_expressed = tcga_expressed_symbols(args.tcga, aliases)
    osdr_expressed, osdr_files = osdr_expressed_mouse_ids(args.osdr_dir)
    current_vocab_raw = pd.read_csv(args.current_vocab)["gene_symbol"].map(symbol)
    current_vocab = {aliases.get(item, item) for item in current_vocab_raw}
    human_lengths_raw = set(pd.read_csv(args.human_lengths)["gene_symbol"].map(symbol))
    human_lengths = {aliases.get(item, item) for item in human_lengths_raw}
    mouse_lengths = set(pd.read_csv(args.mouse_lengths)["gene_symbol"].map(symbol))

    records = []
    for row in orthologs.itertuples():
        old_human_symbol = symbol(row.human_symbol_original)
        approved_symbol = hgnc_by_ensembl.get(row.human_ensembl, aliases.get(old_human_symbol))
        record = {
            "human_ensembl": row.human_ensembl,
            "mouse_ensembl": row.mouse_ensembl,
            "human_symbol_original": old_human_symbol,
            "human_symbol_hgnc": approved_symbol,
            "mouse_symbol": row.mouse_symbol_original,
            "hgnc_resolved": approved_symbol is not None,
            "protein_coding_hgnc": approved_symbol in protein_coding if approved_symbol else False,
            "present_archs4_human": row.human_ensembl in archs4_human,
            "present_archs4_mouse": row.mouse_ensembl in archs4_mouse,
            "expressed_tcga": approved_symbol in tcga_expressed if approved_symbol else False,
            "expressed_tcga_exact_old_symbol": old_human_symbol in tcga_expressed_raw,
            "expressed_osdr": row.mouse_ensembl in osdr_expressed,
            "has_human_exon_length": approved_symbol in human_lengths if approved_symbol else False,
            "has_mouse_exon_length": symbol(row.mouse_symbol_original) in mouse_lengths,
            "in_current_vocab": approved_symbol in current_vocab if approved_symbol else False,
        }
        records.append(record)
    audit = pd.DataFrame(records)
    pair_counts = audit.loc[audit.hgnc_resolved].groupby("human_symbol_hgnc").size()
    audit["unique_pair_after_hgnc"] = audit.human_symbol_hgnc.map(pair_counts).eq(1)
    required = (
        "hgnc_resolved", "unique_pair_after_hgnc", "protein_coding_hgnc",
        "present_archs4_human", "present_archs4_mouse", "expressed_tcga", "expressed_osdr",
    )
    audit["eligible_reconciled"] = audit[list(required)].all(axis=1)
    exact_required = [key for key in required if key != "expressed_tcga"]
    audit["eligible_with_exact_tcga_symbol"] = (
        audit[exact_required].all(axis=1) & audit.expressed_tcga_exact_old_symbol
    )
    audit["training_ready"] = (
        audit.eligible_reconciled & audit.has_human_exon_length & audit.has_mouse_exon_length
    )
    audit = audit.sort_values(["eligible_reconciled", "human_symbol_hgnc"],
                              ascending=[False, True], na_position="last")
    eligible = set(audit.loc[audit.eligible_reconciled, "human_symbol_hgnc"])
    training_ready = set(audit.loc[audit.training_ready, "human_symbol_hgnc"])
    exact_symbol_eligible = set(
        audit.loc[audit.eligible_with_exact_tcga_symbol, "human_symbol_hgnc"]
    )
    added = eligible - current_vocab
    lost = current_vocab - eligible
    summary = {
        "current_vocab_genes_after_hgnc": len(current_vocab),
        "high_confidence_unique_one_to_one_pairs": len(orthologs),
        "reconciled_eligible_genes": len(eligible),
        "eligible_with_exact_tcga_symbols": len(exact_symbol_eligible),
        "genes_rescued_by_tcga_hgnc_reconciliation": len(eligible - exact_symbol_eligible),
        "genes_retained_from_current_vocab": len(eligible & current_vocab),
        "candidate_genes_added": len(added),
        "current_genes_not_eligible_under_this_audit": len(lost),
        "potential_new_vocab_size": len(eligible),
        "training_ready_genes_with_both_exon_lengths": len(training_ready),
        "training_ready_candidate_additions": len(training_ready - current_vocab),
        "training_ready_current_genes_retained": len(training_ready & current_vocab),
        "osdr_files_scanned": osdr_files,
        "filter_failure_counts": {
            key: int((~audit[key]).sum()) for key in required
        },
    }
    return audit, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hgnc", type=Path, default=DEFAULT_HGNC)
    parser.add_argument("--orthologs", type=Path, default=DEFAULT_ORTHOLOGS)
    parser.add_argument("--current-vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--tcga", type=Path, default=DEFAULT_TCGA)
    parser.add_argument("--archs4-human", type=Path, default=DEFAULT_ARCHS4_HUMAN)
    parser.add_argument("--archs4-mouse", type=Path, default=DEFAULT_ARCHS4_MOUSE)
    parser.add_argument("--osdr-dir", type=Path, default=DEFAULT_OSDR)
    parser.add_argument("--human-lengths", type=Path, default=DEFAULT_HUMAN_LENGTHS)
    parser.add_argument("--mouse-lengths", type=Path, default=DEFAULT_MOUSE_LENGTHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit, summary = build_audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output_dir / "gene_universe_audit.parquet", index=False)
    audit.to_csv(args.output_dir / "gene_universe_audit.csv", index=False)
    audit.loc[audit.eligible_reconciled & ~audit.in_current_vocab].to_csv(
        args.output_dir / "candidate_added_genes.csv", index=False
    )
    audit.loc[audit.eligible_reconciled, ["human_symbol_hgnc", "human_ensembl", "mouse_ensembl"]].to_csv(
        args.output_dir / "candidate_reconciled_vocab.csv", index=False
    )
    audit.loc[audit.training_ready, ["human_symbol_hgnc", "human_ensembl", "mouse_ensembl"]].to_csv(
        args.output_dir / "candidate_training_ready_vocab.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote audit outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
