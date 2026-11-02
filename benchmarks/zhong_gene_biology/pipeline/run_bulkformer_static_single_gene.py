#!/usr/bin/env python3
"""Evaluate frozen BulkFormer static gene embeddings on Zhong GO and OMIM splits.

This adapter deliberately performs no BulkFormer forward inference and no pair
benchmark.  It reuses the exact single-gene term evaluator used for Bridge,
including training-fold-only C selection and one final holdout evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.zhong_gene_biology.pipeline.phase1_mapping_coverage import (
    build_symbol_to_entrez_maps,
    map_bridge_to_entrez,
)
from benchmarks.zhong_gene_biology.pipeline.run_single_gene_ablation import (
    C_VALUES,
    evaluate_term,
    filter_split,
    load_split_dict,
    parse_obo_names,
)


BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
REF = BENCH / "references" / "gene-embedding-benchmarks"
RESULTS = BENCH / "results" / "bulkformer_static"
LOG_PATH = RESULTS / "run.log"

BULK_ROOT = Path("/home/walt/bridge-rna/model/BulkFormer")
BULK_INFO = BULK_ROOT / "data" / "bulkformer_gene_info.csv"
GENE_INFO = BENCH / "references" / "Homo_sapiens.gene_info.gz"

MODELS = OrderedDict(
    [
        ("BulkFormer-50M", BULK_ROOT / "BulkFormer_50M.pt"),
        ("BulkFormer-147M", BULK_ROOT / "model" / "BulkFormer_147M.pt"),
    ]
)

TASKS = {
    "GO": {
        "cv": [
            REF / "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold1_dict_all.pkl",
            REF / "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold2_dict_all.pkl",
            REF / "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold3_dict_all.pkl",
        ],
        "holdout": REF / "data/data_splits/gene_level_benchmark/go_folds_splits/go_holdout_dict_all.pkl",
        "obo": REF / "data/obo/go.obo",
        "expected": 56,
    },
    "OMIM": {
        "cv": [
            REF / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold1_dict_all.pkl",
            REF / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold2_dict_all.pkl",
            REF / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold3_dict_all.pkl",
        ],
        "holdout": REF / "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_holdout_dict_all.pkl",
        "obo": REF / "data/obo/doid.obo",
        "expected": 103,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def load_static(checkpoint: Path, expected_rows: int) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint payload: {checkpoint}")
    keys = [key for key in payload if key.endswith("gene_emb_onehot_layer.weight")]
    if keys != ["module.gene_emb_onehot_layer.weight"]:
        raise AssertionError(f"Unexpected static tensor keys in {checkpoint}: {keys}")
    matrix = payload[keys[0]].detach().cpu().numpy().astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or not np.isfinite(matrix).all():
        raise AssertionError(f"Invalid static matrix from {checkpoint}: {matrix.shape}")
    return matrix


def build_mapping() -> tuple[pd.DataFrame, pd.DataFrame]:
    info = pd.read_csv(BULK_INFO, usecols=["gene_symbol", "ensg_id"])
    if len(info) != 20_010 or info["ensg_id"].duplicated().any():
        raise AssertionError("Unexpected BulkFormer vocabulary size or duplicate Ensembl IDs")
    vocab = pd.DataFrame(
        {
            "token_id": np.arange(1, len(info) + 1),
            "gene_symbol": info["gene_symbol"].astype(str).str.upper().str.strip(),
        }
    )
    official, aliases = build_symbol_to_entrez_maps(GENE_INFO)
    mapping = map_bridge_to_entrez(vocab, official, aliases).rename(
        columns={"bridge_gene_symbol": "bulkformer_gene_symbol"}
    )
    mapping.insert(2, "ensg_id", info["ensg_id"].astype(str).to_numpy())
    usable = mapping[(mapping["map_status"] == "mapped") & (~mapping["is_duplicate_entrez_mapping"])].copy()
    usable["mapped_entrez_id"] = usable["mapped_entrez_id"].astype(str)
    if usable["mapped_entrez_id"].duplicated().any():
        raise AssertionError("Duplicate Entrez IDs remain after BulkFormer filtering")
    return mapping, usable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    started = time.time()
    log(f"[startup] models={args.models}; tasks={list(TASKS)}; seed={args.seed}; C={C_VALUES}")
    log("[scope] static checkpoint tensors only; no inference, pair tasks, controls, or split generation")

    mapping, usable = build_mapping()
    mapping.to_csv(RESULTS / "bulkformer_to_entrez_mapping.tsv", sep="\t", index=False)
    status_counts = mapping["map_status"].value_counts().to_dict()
    log(f"[mapping] rows={len(mapping)} usable_unique_entrez={len(usable)} status={status_counts}")

    info = pd.read_csv(BULK_INFO, usecols=["gene_symbol"])
    allowed = set(usable["mapped_entrez_id"])
    row_index = usable["token_id"].to_numpy(np.int64) - 1
    entrez = usable["mapped_entrez_id"].to_numpy(str)
    gene_to_row = {gene: index for index, gene in enumerate(entrez)}

    summaries: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    coverage_reference: dict[str, pd.DataFrame] = {}

    for model_name in args.models:
        checkpoint = MODELS[model_name]
        matrix_native = load_static(checkpoint, len(info))
        matrix = matrix_native[row_index]
        log(f"[model] {model_name} checkpoint={checkpoint} native_shape={matrix_native.shape} usable_shape={matrix.shape}")

        for task_name, spec in TASKS.items():
            task_started = time.time()
            cv1, cv2, cv3 = [load_split_dict(path) for path in spec["cv"]]
            holdout = load_split_dict(spec["holdout"])
            terms = sorted(holdout)
            if not (set(terms) == set(cv1) == set(cv2) == set(cv3)):
                raise AssertionError(f"{task_name}: split term keys differ")
            if len(terms) != spec["expected"]:
                raise AssertionError(f"{task_name}: expected {spec['expected']} terms, found {len(terms)}")
            term_names = parse_obo_names(spec["obo"])

            rows: list[dict[str, object]] = []
            invalid: list[dict[str, object]] = []
            for position, term in enumerate(terms, start=1):
                row, reason = evaluate_term(
                    term=term,
                    term_name=term_names.get(term, ""),
                    rep_name=model_name,
                    cv1=filter_split(cv1[term], allowed),
                    cv2=filter_split(cv2[term], allowed),
                    cv3=filter_split(cv3[term], allowed),
                    holdout=filter_split(holdout[term], allowed),
                    gene_to_row=gene_to_row,
                    x=matrix,
                    seed=args.seed,
                )
                if row is None:
                    invalid.append(reason)
                else:
                    rows.append(row)
                if position % 20 == 0 or position == len(terms):
                    elapsed = time.time() - task_started
                    log(
                        f"[{model_name}/{task_name}] terms={position}/{len(terms)} "
                        f"elapsed={elapsed / 60:.1f}m rate={position / max(elapsed, 1e-9):.2f}/s"
                    )

            frame = pd.DataFrame(rows)
            if invalid or len(frame) != spec["expected"] or frame["term"].nunique() != spec["expected"]:
                pd.DataFrame(invalid).to_csv(RESULTS / f"{model_name}_{task_name}_invalid.csv", index=False)
                raise AssertionError(
                    f"{model_name}/{task_name}: expected {spec['expected']} valid terms, "
                    f"got {len(frame)}; invalid={len(invalid)}"
                )
            if frame[["AUROC", "AUPRC"]].isna().any().any():
                raise AssertionError(f"{model_name}/{task_name}: missing held-out metrics")

            output = RESULTS / f"{model_name}_{task_name}.csv"
            frame.to_csv(output, index=False)
            coverage = frame[
                ["term", "n_train_pos", "n_train_neg", "n_test_pos", "n_test_neg", "retained_total", "original_total"]
            ].sort_values("term").reset_index(drop=True)
            if task_name in coverage_reference and not coverage.equals(coverage_reference[task_name]):
                raise AssertionError(f"{task_name}: model sizes did not use identical term-level genes")
            coverage_reference[task_name] = coverage

            summaries.append(
                {
                    "Task": task_name,
                    "Model": model_name,
                    "N_terms": len(frame),
                    "Embedding_dim": matrix.shape[1],
                    "Mean_AUROC": frame["AUROC"].mean(),
                    "Median_AUROC": frame["AUROC"].median(),
                    "SD_AUROC": frame["AUROC"].std(ddof=1),
                    "Mean_AUPRC": frame["AUPRC"].mean(),
                    "Median_AUPRC": frame["AUPRC"].median(),
                    "SD_AUPRC": frame["AUPRC"].std(ddof=1),
                }
            )
            inputs.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "output": str(output),
                }
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    provenance = {
        "status": "complete",
        "scope": "BulkFormer static embeddings; Zhong GO and OMIM only",
        "excluded": ["BulkFormer forward inference", "gene-pair benchmark", "controls", "split regeneration"],
        "static_tensor": "module.gene_emb_onehot_layer.weight",
        "seed": args.seed,
        "C_values": C_VALUES,
        "svm": "RBF SVC with class_weight=balanced; C selected on three supplied training folds",
        "final_metric": "one held-out test AUROC/AUPRC per biological term",
        "vocabulary_rows": len(mapping),
        "usable_unique_entrez": len(usable),
        "mapping_status_counts": status_counts,
        "mapping_table": str((RESULTS / "bulkformer_to_entrez_mapping.tsv").resolve()),
        "gene_info": str(GENE_INFO.resolve()),
        "gene_info_sha256": sha256(GENE_INFO),
        "bulkformer_gene_info": str(BULK_INFO),
        "bulkformer_gene_info_sha256": sha256(BULK_INFO),
        "inputs": inputs,
        "elapsed_seconds": time.time() - started,
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"[done] elapsed={(time.time() - started) / 60:.1f}m")
    log("[done] summary:\n" + summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
