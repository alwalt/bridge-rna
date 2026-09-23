#!/usr/bin/env python3
"""Freeze the final-release matrices, identifiers, overlaps, and CV folds.

Run this with NumPy >=2 because the released pandas pickles were serialized in
that environment. Source files are read-only; large derived arrays are written
to the configured shared work root.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parents[1]
CONFIG = json.loads((BENCH / "config.json").read_text())
SOURCE = Path(CONFIG["source_data_root"])
WORK = Path(CONFIG["shared_work_root"])
RESULTS = BENCH / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_ensembl(values: pd.Index) -> pd.Index:
    return pd.Index(values.astype(str).str.replace(r"\..*$", "", regex=True))


def bridge_mapping(gene_ids: pd.Index) -> pd.DataFrame:
    asset_root = Path(CONFIG["bridge_root"])
    hgnc_path = asset_root / "data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv"
    hgnc = pd.read_csv(hgnc_path, sep="\t", low_memory=False)
    usable = hgnc.ensembl_gene_id.notna() & hgnc.symbol.notna()
    ensembl_to_symbols: dict[str, set[str]] = {}
    for ensembl, symbol in zip(hgnc.loc[usable, "ensembl_gene_id"], hgnc.loc[usable, "symbol"]):
        ensembl_to_symbols.setdefault(str(ensembl).split(".")[0], set()).add(str(symbol).upper())
    canonical = pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv")
    symbol_to_token = dict(zip(canonical.gene_symbol.str.upper(), canonical.token_id.astype(int) - 1))
    rows = []
    for gene in clean_ensembl(gene_ids):
        symbols = sorted(ensembl_to_symbols.get(gene, set()))
        mapped = [(symbol, symbol_to_token[symbol]) for symbol in symbols if symbol in symbol_to_token]
        status = "unique" if len(mapped) == 1 else ("unmapped" if not mapped else "ambiguous")
        rows.append({
            "gene_id": gene,
            "hgnc_symbols": ";".join(symbols),
            "bridge_gene_symbol": mapped[0][0] if len(mapped) == 1 else "",
            "bridge_token_index": mapped[0][1] if len(mapped) == 1 else pd.NA,
            "bridge_mapping_status": status,
            "bridge_representable": len(mapped) == 1,
        })
    return pd.DataFrame(rows)


def distribution(values: np.ndarray, allow_nan: bool) -> dict:
    flat = values.ravel()
    finite = flat[np.isfinite(flat)] if allow_nan else flat
    quantiles = np.quantile(finite, [0, .01, .1, .25, .5, .75, .9, .99, 1])
    return {
        "finite_count": int(len(finite)),
        "missing_count": int(len(flat) - len(finite)),
        "mean": float(finite.mean()),
        "sd": float(finite.std()),
        "zero_fraction": float(np.mean(finite == 0)),
        "quantiles": dict(zip(["min", "p01", "p10", "p25", "p50", "p75", "p90", "p99", "max"], map(float, quantiles))),
    }


def save_folds(cell_lines: pd.Index) -> None:
    rows = []
    all_rows = np.arange(len(cell_lines))
    shuffled = all_rows.copy()
    np.random.RandomState(int(CONFIG["seed"])).shuffle(shuffled)
    test_chunks = np.array_split(shuffled, int(CONFIG["cv_folds"]))
    for fold, test in enumerate(test_chunks):
        train_validation = np.setdiff1d(all_rows, test, assume_unique=True)
        split_rng = np.random.RandomState(int(CONFIG["seed"]) + 1000 + fold)
        shuffled_train_validation = train_validation.copy()
        split_rng.shuffle(shuffled_train_validation)
        validation_size = int(np.ceil(len(train_validation) * float(CONFIG["validation_fraction_of_training"])))
        validation = shuffled_train_validation[:validation_size]
        train = shuffled_train_validation[validation_size:]
        assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))
        assert len(set(train) | set(validation) | set(test)) == len(cell_lines)
        for split, indices in (("train", train), ("validation", validation), ("test", test)):
            rows.extend({"fold": fold, "split": split, "row_index": int(i), "cell_line_id": cell_lines[i]} for i in indices)
    folds = pd.DataFrame(rows).sort_values(["fold", "split", "row_index"])
    folds.to_csv(RESULTS / "cv_folds.csv", index=False)


def main() -> None:
    if int(np.__version__.split(".")[0]) < 2:
        raise SystemExit("The released pickles require NumPy >=2")
    WORK.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    expr_path = SOURCE / "gene_essentiality_expr_data.pkl"
    score_path = SOURCE / "gene_essentiality_score.pkl"
    with expr_path.open("rb") as handle:
        expression = pickle.load(handle)
    with score_path.open("rb") as handle:
        dependency = pickle.load(handle)
    if not isinstance(expression, pd.DataFrame) or not isinstance(dependency, pd.DataFrame):
        raise TypeError("Expected pandas DataFrames")
    if not expression.index.equals(dependency.index):
        raise AssertionError("Expression and dependency cell-line order differs")
    for name, frame in (("expression", expression), ("dependency", dependency)):
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise AssertionError(f"{name} has duplicate identifiers")

    expression.columns = clean_ensembl(expression.columns)
    dependency.columns = clean_ensembl(dependency.columns)
    expression_values = expression.to_numpy(dtype=np.float32)
    dependency_values = dependency.to_numpy(dtype=np.float32)
    np.save(WORK / "expression_released_log2_tpm_plus_1.float32.npy", expression_values)
    np.save(WORK / "expression_fm_natural_log1p_tpm.float32.npy", expression_values * np.float32(np.log(2.0)))
    np.save(WORK / "dependency.float32.npy", dependency_values)

    pd.DataFrame({"row_index": np.arange(len(expression)), "cell_line_id": expression.index}).to_csv(
        RESULTS / "cell_lines.csv", index=False
    )
    expr_map = bridge_mapping(expression.columns)
    expr_map.insert(0, "expression_column_index", np.arange(len(expr_map)))
    target_map = bridge_mapping(dependency.columns)
    target_map.insert(0, "dependency_column_index", np.arange(len(target_map)))
    target_map["bulkformer_token_index"] = target_map.gene_id.map(
        {gene: i for i, gene in enumerate(pd.read_csv(Path(CONFIG["bulkformer_root"]) / "data/bulkformer_gene_info.csv").ensg_id.astype(str))}
    ).astype("Int64")
    target_map["bulkformer_representable"] = target_map.bulkformer_token_index.notna()
    target_map["expression_column_index"] = target_map.gene_id.map(dict(zip(expr_map.gene_id, expr_map.expression_column_index))).astype("Int64")
    target_map["present_in_expression"] = target_map.expression_column_index.notna()
    target_map["missing_dependency_count"] = np.isnan(dependency_values).sum(axis=0)
    target_map["observed_dependency_count"] = np.isfinite(dependency_values).sum(axis=0)
    target_map["bridge_bulkformer_matched"] = target_map.bridge_representable & target_map.bulkformer_representable
    target_map["all_method_common"] = target_map.bridge_bulkformer_matched & target_map.present_in_expression
    expr_map.to_csv(RESULTS / "expression_gene_overlap.csv", index=False)
    target_map.to_csv(RESULTS / "gene_overlap.csv", index=False)
    save_folds(expression.index)

    audit = {
        "expression_shape": list(expression.shape),
        "dependency_shape": list(dependency.shape),
        "cell_line_order_identical": True,
        "duplicate_cell_lines": 0,
        "duplicate_expression_genes": 0,
        "duplicate_dependency_genes": 0,
        "expression_dependency_gene_overlap": int(len(expression.columns.intersection(dependency.columns))),
        "bridge_expression_overlap": int(expr_map.bridge_representable.sum()),
        "bridge_dependency_overlap": int(target_map.bridge_representable.sum()),
        "bulkformer_dependency_overlap": int(target_map.bulkformer_representable.sum()),
        "bridge_bulkformer_matched_dependency_genes": int(target_map.bridge_bulkformer_matched.sum()),
        "all_method_common_dependency_genes": int(target_map.all_method_common.sum()),
        "expression": distribution(expression_values, allow_nan=False),
        "dependency": distribution(dependency_values, allow_nan=True),
        "dependency_genes_with_missing": int(np.isnan(dependency_values).any(axis=0).sum()),
        "cell_lines_with_missing_dependency": int(np.isnan(dependency_values).any(axis=1).sum()),
        "inputs": {
            "expression": {"path": str(expr_path), "sha256": sha256(expr_path)},
            "dependency": {"path": str(score_path), "sha256": sha256(score_path)},
        },
        "inference_transform": CONFIG["fm_inference_transform"],
        "folds": {"count": CONFIG["cv_folds"], "seed": CONFIG["seed"], "split_axis": "cell_lines"},
    }
    (RESULTS / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
