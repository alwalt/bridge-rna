#!/usr/bin/env python3
"""Inspect the exact BulkFormer matrices and map their genes to BridgeRNA.

The published pickles were written with NumPy 2 and must be read in an
environment with NumPy >=2. This script never modifies the source files.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


if int(np.__version__.split(".")[0]) < 2:
    raise SystemExit("BulkFormer pickles require NumPy >=2; use a temporary compatible environment.")

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/gene_property_prediction"


def map_symbols(gene_ids: pd.Index) -> tuple[pd.Series, pd.Series]:
    hgnc = pd.read_csv(
        ROOT / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv",
        sep="\t",
        low_memory=False,
    )
    valid = hgnc["ensembl_gene_id"].notna() & hgnc["symbol"].notna()
    mapping = dict(
        zip(
            hgnc.loc[valid, "ensembl_gene_id"].str.replace(r"\..*$", "", regex=True),
            hgnc.loc[valid, "symbol"].str.upper(),
        )
    )
    symbols = pd.Series(gene_ids.astype(str), name="gene_id").map(mapping)
    canonical = set(pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv")["gene_symbol"].str.upper())
    return symbols, symbols.isin(canonical)


def main() -> None:
    source = BENCH / "data/source"
    processed = BENCH / "data/processed"
    results = BENCH / "results/audit"
    with (source / "gene_essentiality_expr_data.pkl").open("rb") as handle:
        expression = pickle.load(handle)
    with (source / "gene_essentiality_score.pkl").open("rb") as handle:
        scores = pickle.load(handle)
    if not isinstance(expression, pd.DataFrame) or not isinstance(scores, pd.DataFrame):
        raise TypeError("Expected both published resources to be pandas DataFrames")
    if not expression.index.equals(scores.index):
        raise AssertionError("Published expression and target cell-line order differs")
    expression_symbols, expression_mapped = map_symbols(expression.columns)
    score_symbols, score_mapped = map_symbols(scores.columns)
    pd.DataFrame(
        {"gene_id": expression.columns, "gene_symbol": expression_symbols, "maps_to_bridgerna": expression_mapped}
    ).to_csv(processed / "bulkformer_expression_gene_mapping.csv", index=False)
    pd.DataFrame(
        {
            "gene_id": scores.columns,
            "gene_symbol": score_symbols,
            "maps_to_bridgerna": score_mapped,
            "missing_values": scores.isna().sum().to_numpy(),
        }
    ).to_csv(processed / "bulkformer_target_gene_mapping.csv", index=False)
    audit = {
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "expression_shape": list(expression.shape),
        "score_shape": list(scores.shape),
        "rows_identical_order": True,
        "common_rows": len(expression.index),
        "expression_target_ensembl_overlap": len(expression.columns.intersection(scores.columns)),
        "expression_bridge_mapped": int(expression_mapped.sum()),
        "score_bridge_mapped": int(score_mapped.sum()),
        "score_missing_cells": int(scores.isna().sum().sum()),
        "score_missing_fraction": float(scores.isna().to_numpy().mean()),
        "score_genes_with_any_missing": int(scores.isna().any().sum()),
        "score_rows_with_any_missing": int(scores.isna().any(axis=1).sum()),
        "expression_min": float(expression.min().min()),
        "expression_max": float(expression.max().max()),
        "score_min": float(scores.min().min()),
        "score_max": float(scores.max().max()),
    }
    (results / "bulkformer_matrix_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
