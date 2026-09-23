#!/usr/bin/env python3
"""Add the minimal BulkFormer-compatible TCGA classification/prognosis analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from common import REPO_ROOT, RESULTS, WORK

TCGA_H5 = REPO_ROOT / "data/tcga/tcga_matrix.h5"
CDR = REPO_ROOT / "data/tcga/Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
OUR_LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
BULK_INFO = REPO_ROOT / "model/BulkFormer/data/bulkformer_gene_info.csv"
IMPUTATION_WORK = REPO_ROOT / "benchmarks/tcga_imputation/work"
IMPUTATION_RESULTS = REPO_ROOT / "benchmarks/tcga_imputation/results"

SEED = 20260922
FOLDS = 10
PCA_COMPONENTS = 128
MODELS = ("ours_45.6m", "bulkformer_50m", "bulkformer_147m")


def say(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> pd.DataFrame:
    output = RESULTS / "bulkformer_parity_pan_cancer_manifest.csv"
    with h5py.File(TCGA_H5, "r") as handle:
        frame = pd.DataFrame({
            "h5_row": np.arange(len(handle["meta/sampleid"])),
            "sample_id": decode(handle["meta/sampleid"][:]),
            "patient_id": decode(handle["meta/gdc_cases.submitter_id"][:]),
            "project_id": decode(handle["meta/gdc_cases.project.project_id"][:]),
            "sample_type": decode(handle["meta/gdc_cases.samples.sample_type"][:]),
        })
    primary = frame.loc[frame.sample_type.eq("Primary Tumor")].copy()
    primary = primary.sort_values(["patient_id", "sample_id"]).drop_duplicates("patient_id")
    laml = frame.loc[
        frame.project_id.eq("TCGA-LAML")
        & frame.sample_type.eq("Primary Blood Derived Cancer - Peripheral Blood")
    ].copy()
    laml = laml.sort_values(["patient_id", "sample_id"]).drop_duplicates("patient_id")
    # Keep the exact existing 9,816-row cache order, then append only LAML.
    frame = pd.concat([primary, laml], ignore_index=True)
    frame["cancer_label"] = frame.project_id.str.removeprefix("TCGA-")
    cdr = pd.read_csv(CDR, sep="\t", low_memory=False).rename(columns={
        "_PATIENT": "patient_id", "OS": "event", "OS.time": "time_days"
    })
    cdr = cdr[["patient_id", "event", "time_days"]].drop_duplicates("patient_id")
    frame = frame.merge(cdr, on="patient_id", how="left", validate="one_to_one")
    frame["event"] = pd.to_numeric(frame.event, errors="coerce")
    frame["time_days"] = pd.to_numeric(frame.time_days, errors="coerce")
    frame["prognosis_usable"] = frame.event.isin([0, 1])
    frame["matrix_row"] = np.arange(len(frame))
    assert frame.patient_id.is_unique
    assert frame.cancer_label.nunique() == 33
    assert set(frame.loc[frame.cancer_label.eq("LAML"), "sample_type"]) == {
        "Primary Blood Derived Cancer - Peripheral Blood"
    }
    frame.to_csv(output, index=False)
    say(f"pan-cancer manifest: {len(frame):,} patients, "
        f"{frame.cancer_label.nunique()} classes, prognosis={frame.prognosis_usable.sum():,}")
    return frame


def mapping_assets() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, str]]:
    ours = pd.read_parquet(IMPUTATION_WORK / "ours_genes.parquet")
    bulk = pd.read_parquet(IMPUTATION_WORK / "bulkformer_genes.parquet")
    crosswalk = pd.read_csv(IMPUTATION_RESULTS / "tcga_hgnc_crosswalk.csv")
    usable = crosswalk.loc[crosswalk.mapping_status.isin(
        ["approved_symbol", "previous_symbol", "alias_symbol"])]
    approved_to_source = dict(zip(usable.approved_symbol.astype(str), usable.tcga_symbol.astype(str)))
    return ours, bulk, {}, approved_to_source


def tpm_log1p(counts: np.ndarray, lengths_bp: np.ndarray) -> np.ndarray:
    rates = counts.astype(np.float64) / (lengths_bp[None, :] / 1000.0)
    totals = rates.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(rates * 1e6, totals, out=np.zeros_like(rates), where=totals > 0)).astype(np.float32)


def prepare_inputs(manifest: pd.DataFrame) -> None:
    needed = [WORK / "bulkformer_parity_laml_ours_log1p_tpm.npy",
              WORK / "bulkformer_parity_laml_bulkformer_log1p_tpm.npy",
              WORK / "bulkformer_parity_laml_full_log1p_cpm.npy"]
    if all(path.is_file() for path in needed):
        say("reusing prepared LAML expression matrices")
        return
    laml = manifest.loc[manifest.cancer_label.eq("LAML")]
    ours, bulk, _, approved_to_source = mapping_assets()
    with h5py.File(TCGA_H5, "r") as handle:
        source_genes = decode(handle["meta/genes"][:])
        source_index = {gene.upper(): index for index, gene in enumerate(source_genes)}
        rows = laml.h5_row.to_numpy(int)
        order = np.argsort(rows)
        raw_sorted = np.asarray(handle["data/expression"][rows[order], :], dtype=np.float32)
        raw = raw_sorted[np.argsort(order)]

    def extract(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray([
            source_index.get(approved_to_source.get(str(gene), "").upper(), -1)
            if pd.notna(gene) else -1 for gene in table.approved_symbol
        ], dtype=int)
        observed = indices >= 0
        values = np.zeros((len(raw), len(table)), dtype=np.float32)
        values[:, observed] = raw[:, indices[observed]]
        return values, observed

    our_counts, _ = extract(ours)
    bulk_counts, bulk_observed = extract(bulk)
    length_table = pd.read_csv(OUR_LENGTHS)
    length_lookup = dict(zip(length_table.gene_symbol.astype(str), length_table.exon_length))
    our_lengths = np.asarray([length_lookup[gene] for gene in ours.gene.astype(str)])
    bulk_lengths = pd.read_csv(BULK_INFO).gene_length.to_numpy(float)
    our_values = tpm_log1p(our_counts, our_lengths)
    bulk_values = tpm_log1p(bulk_counts, bulk_lengths)
    bulk_values[:, ~bulk_observed] = -10.0
    totals = raw.sum(axis=1, keepdims=True, dtype=np.float64)
    full_values = np.log1p(np.divide(raw * 1e6, totals, out=np.zeros_like(raw), where=totals > 0))
    np.save(needed[0], our_values)
    np.save(needed[1], bulk_values)
    np.save(needed[2], full_values.astype(np.float32))
    say(f"prepared only missing LAML inputs: {len(laml)} patients")


def load_model(name: str, device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "benchmarks/tcga_imputation/pipeline"))
    from model_adapters import load_bulkformer, load_ours
    return load_ours(device) if name == "ours_45.6m" else load_bulkformer(name, device)


@torch.inference_mode()
def embed_laml(name: str, device: torch.device, heartbeat: int) -> None:
    output = WORK / f"bulkformer_parity_laml_{name}_embeddings.npy"
    if output.is_file():
        say(f"reusing {output.name}")
        return
    space = "ours" if name == "ours_45.6m" else "bulkformer"
    matrix = np.load(WORK / f"bulkformer_parity_laml_{space}_log1p_tpm.npy", mmap_mode="r")
    batch_size = {"ours_45.6m": 4, "bulkformer_50m": 32, "bulkformer_147m": 16}[name]
    model = load_model(name, device).eval()
    vectors = []
    started = last = time.monotonic()
    say(f"embedding only missing LAML patients: model={name} n={len(matrix)} device={device}")
    for start in range(0, len(matrix), batch_size):
        batch = torch.as_tensor(np.asarray(matrix[start:start + batch_size]).copy(),
                                dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if name == "ours_45.6m":
                embedding = model.encode(batch, normalize=False)
            else:
                embedding = model(batch, mask_prob=0.0, output_expr=False).mean(dim=1)
        vectors.append(embedding.float().cpu().numpy())
        now = time.monotonic()
        if now - last >= heartbeat or start + batch_size >= len(matrix):
            done = min(start + batch_size, len(matrix))
            say(f"heartbeat {name}: {done}/{len(matrix)} elapsed={(now-started)/60:.1f}m")
            last = now
    np.save(output, np.concatenate(vectors))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def assembled(name: str) -> np.ndarray:
    old = np.load(WORK / f"{name}_embeddings.npy", mmap_mode="r")
    new = np.load(WORK / f"bulkformer_parity_laml_{name}_embeddings.npy", mmap_mode="r")
    return np.concatenate([np.asarray(old), np.asarray(new)], axis=0)


def full_expression() -> np.ndarray:
    old = np.load(WORK / "tcga_full_log1p_cpm.npy", mmap_mode="r")
    new = np.load(WORK / "bulkformer_parity_laml_full_log1p_cpm.npy", mmap_mode="r")
    return np.concatenate([np.asarray(old), np.asarray(new)], axis=0)


def save_embedding_manifest(manifest: pd.DataFrame) -> None:
    old_n = int((~manifest.cancer_label.eq("LAML")).sum())
    new_n = int(manifest.cancer_label.eq("LAML").sum())
    rows = []
    for name in MODELS:
        matrix = assembled(name)
        assert matrix.shape[0] == len(manifest)
        rows.append({"representation": name, "patients": len(manifest),
                     "reused_patients": old_n, "newly_inferred_patients": new_n,
                     "features": matrix.shape[1],
                     "pooling": "Bridge native encode" if name == "ours_45.6m" else "mean gene-token pooling",
                     "base_cache": str(WORK / f"{name}_embeddings.npy"),
                     "incremental_cache": str(WORK / f"bulkformer_parity_laml_{name}_embeddings.npy")})
    pd.DataFrame(rows).to_csv(RESULTS / "bulkformer_parity_embedding_manifest.csv", index=False)


def fold_table(manifest: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "classification":
        eligible = np.arange(len(manifest))
        strata = manifest.cancer_label.to_numpy(str)
    else:
        eligible = np.flatnonzero(manifest.prognosis_usable.to_numpy(bool))
        strata = manifest.event.to_numpy(float)[eligible].astype(int)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(manifest), -1, dtype=int)
    for fold_id, (_, test) in enumerate(splitter.split(eligible, strata)):
        fold[eligible[test]] = fold_id
    result = manifest[["patient_id", "cancer_label"]].copy()
    result["task"] = task
    result["fold"] = fold
    result["eligible"] = fold >= 0
    assert result.loc[result.eligible, "patient_id"].is_unique
    assert set(result.loc[result.eligible, "fold"]) == set(range(FOLDS))
    return result


def pca_project(train_x: np.ndarray, test_x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(train_x), pca.transform(test_x)


def classification_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": f1_score(y, predicted, average="macro"),
        "weighted_f1": f1_score(y, predicted, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
    }


def evaluate_classification(manifest: pd.DataFrame) -> None:
    output = RESULTS / "bulkformer_parity_pan_cancer_classification_per_fold.csv"
    splits = fold_table(manifest, "classification")
    splits.to_csv(RESULTS / "bulkformer_parity_pan_cancer_classification_splits.csv", index=False)
    if output.is_file():
        say("reusing pan-cancer classification results")
        return
    y = manifest.cancer_label.to_numpy(str)
    matrices = {name: assembled(name) for name in MODELS}
    matrices["full_raw_expression"] = full_expression()
    rows = []
    predictions = []
    for fold in range(FOLDS):
        test = np.flatnonzero(splits.fold.to_numpy() == fold)
        train = np.flatnonzero(splits.fold.to_numpy() != fold)
        assert not set(manifest.patient_id.iloc[train]) & set(manifest.patient_id.iloc[test])
        for name, matrix in matrices.items():
            variants = [("random_forest", np.asarray(matrix[train]), np.asarray(matrix[test]))]
            if name != "full_raw_expression":
                train_pca, test_pca = pca_project(np.asarray(matrix[train]), np.asarray(matrix[test]), SEED + fold)
                variants = [("pca128_random_forest", train_pca, test_pca)]
            else:
                train_pca, test_pca = pca_project(np.asarray(matrix[train]), np.asarray(matrix[test]), SEED + fold)
                variants.append(("pca128_random_forest", train_pca, test_pca))
            for method, train_x, test_x in variants:
                model = RandomForestClassifier(random_state=SEED + fold, n_jobs=-1)
                model.fit(train_x, y[train])
                predicted = model.predict(test_x)
                row = {"fold": fold, "representation": name, "method": method,
                       "train_patients": len(train), "test_patients": len(test),
                       **classification_metrics(y[test], predicted)}
                rows.append(row)
                predictions.extend({"patient_id": manifest.patient_id.iloc[index], "fold": fold,
                                    "representation": name, "method": method,
                                    "observed": y[index], "predicted": value}
                                   for index, value in zip(test, predicted))
                say(f"classification fold={fold} {name}/{method} weighted_f1={row['weighted_f1']:.4f}")
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(predictions).to_csv(
        RESULTS / "bulkformer_parity_pan_cancer_classification_predictions.csv", index=False)


def evaluate_prognosis(manifest: pd.DataFrame) -> None:
    output = RESULTS / "bulkformer_parity_alive_dead_prognosis_per_fold.csv"
    splits = fold_table(manifest, "prognosis")
    splits.to_csv(RESULTS / "bulkformer_parity_alive_dead_prognosis_splits.csv", index=False)
    if output.is_file():
        say("reusing alive/dead prognosis results")
        return
    eligible = np.flatnonzero(splits.eligible.to_numpy(bool))
    y = manifest.event.fillna(0).to_numpy(np.int8)
    matrices = {name: assembled(name) for name in MODELS}
    matrices["full_raw_expression"] = full_expression()
    rows = []
    predictions = []
    for fold in range(FOLDS):
        test = np.flatnonzero(splits.fold.to_numpy() == fold)
        train = np.setdiff1d(eligible, test, assume_unique=True)
        assert not set(manifest.patient_id.iloc[train]) & set(manifest.patient_id.iloc[test])
        for name, matrix in matrices.items():
            variants = [("random_forest", np.asarray(matrix[train]), np.asarray(matrix[test]))]
            if name != "full_raw_expression":
                train_pca, test_pca = pca_project(np.asarray(matrix[train]), np.asarray(matrix[test]), SEED + fold)
                variants = [("pca128_random_forest", train_pca, test_pca)]
            else:
                train_pca, test_pca = pca_project(np.asarray(matrix[train]), np.asarray(matrix[test]), SEED + fold)
                variants.append(("pca128_random_forest", train_pca, test_pca))
            for method, train_x, test_x in variants:
                model = RandomForestClassifier(random_state=SEED + fold, n_jobs=-1)
                model.fit(train_x, y[train])
                probability = model.predict_proba(test_x)[:, 1]
                row = {"fold": fold, "representation": name, "method": method,
                       "train_patients": len(train), "test_patients": len(test),
                       "events_test": int(y[test].sum()),
                       "auroc": roc_auc_score(y[test], probability),
                       "auprc": average_precision_score(y[test], probability)}
                rows.append(row)
                predictions.extend({"patient_id": manifest.patient_id.iloc[index], "fold": fold,
                                    "representation": name, "method": method,
                                    "event": int(y[index]), "probability_dead": float(value)}
                                   for index, value in zip(test, probability))
                say(f"prognosis fold={fold} {name}/{method} AUROC={row['auroc']:.4f}")
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(predictions).to_csv(
        RESULTS / "bulkformer_parity_alive_dead_prognosis_predictions.csv", index=False)


def summarize() -> None:
    specifications = {
        "pan_cancer_classification": (
            RESULTS / "bulkformer_parity_pan_cancer_classification_per_fold.csv",
            ["macro_f1", "weighted_f1", "balanced_accuracy"]),
        "alive_dead_prognosis": (
            RESULTS / "bulkformer_parity_alive_dead_prognosis_per_fold.csv",
            ["auroc", "auprc"]),
    }
    for task, (path, metrics) in specifications.items():
        data = pd.read_csv(path)
        rows = []
        for (representation, method), frame in data.groupby(["representation", "method"]):
            row = {"task": task, "representation": representation, "method": method,
                   "folds": len(frame)}
            for metric in metrics:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_sd"] = frame[metric].std(ddof=1)
            rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS / f"bulkformer_parity_{task}_summary.csv", index=False)


def plot_results() -> None:
    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    labels = {
        "ours_45.6m": "Bridge 45.6M",
        "bulkformer_50m": "BulkFormer-50M",
        "bulkformer_147m": "BulkFormer-147M",
        "full_raw_expression": "Full raw expression",
    }
    colors = {"ours_45.6m": "#4C78A8", "bulkformer_50m": "#F58518",
              "bulkformer_147m": "#E45756", "full_raw_expression": "#72B7B2"}
    for task, metric, literature, filename in [
        ("pan_cancer_classification", "weighted_f1", 0.907,
         "bulkformer_parity_pan_cancer_classification"),
        ("alive_dead_prognosis", "auroc", 0.747,
         "bulkformer_parity_alive_dead_prognosis"),
    ]:
        data = pd.read_csv(RESULTS / f"bulkformer_parity_{task}_summary.csv")
        # Main paired bars: PCA-128+RF for FMs; direct RF is the primary raw baseline.
        data = data.loc[((data.representation != "full_raw_expression")
                         & data.method.eq("pca128_random_forest"))
                        | ((data.representation == "full_raw_expression")
                           & data.method.eq("random_forest"))].copy()
        order = ["ours_45.6m", "bulkformer_50m", "bulkformer_147m", "full_raw_expression"]
        data["order"] = data.representation.map({name: i for i, name in enumerate(order)})
        data = data.sort_values("order")
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        x = np.arange(len(data))
        ax.bar(x, data[f"{metric}_mean"], yerr=data[f"{metric}_sd"], capsize=4,
               color=[colors[value] for value in data.representation])
        ax.axhline(literature, color="black", linestyle="--", linewidth=1.4,
                   label=f"BulkFormer literature-only ({literature:.3f})")
        ax.set_xticks(x, [labels[value] for value in data.representation], rotation=15, ha="right")
        ax.set_ylabel("Weighted F1" if metric == "weighted_f1" else "AUROC")
        ax.set_title("33-class patient-level TCGA classification" if metric == "weighted_f1"
                     else "Alive/dead prediction (not time-to-event survival)")
        lower = max(0.0, float(data[f"{metric}_mean"].min()) - 0.08)
        ax.set_ylim(lower, min(1.0, float(data[f"{metric}_mean"].max()) + 0.08))
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / f"{filename}.png", dpi=220)
        fig.savefig(figures / f"{filename}.pdf")
        plt.close(fig)

    predictions = pd.read_csv(RESULTS / "bulkformer_parity_alive_dead_prognosis_predictions.csv")
    manifest = pd.read_csv(RESULTS / "bulkformer_parity_pan_cancer_manifest.csv")
    predictions = predictions.merge(
        manifest[["patient_id", "cancer_label"]], on="patient_id", how="left", validate="many_to_one")
    selected = predictions.loc[
        ((predictions.representation != "full_raw_expression")
         & predictions.method.eq("pca128_random_forest"))
        | ((predictions.representation == "full_raw_expression")
           & predictions.method.eq("random_forest"))
    ].copy()
    rows = []
    for (cancer, representation, method), frame in selected.groupby(
            ["cancer_label", "representation", "method"]):
        events = int(frame.event.sum())
        censored = int(len(frame) - events)
        rows.append({
            "cancer_label": cancer,
            "representation": representation,
            "method": method,
            "patients": len(frame),
            "events": events,
            "alive": censored,
            "auroc": roc_auc_score(frame.event, frame.probability_dead)
            if events > 0 and censored > 0 else np.nan,
            "auprc": average_precision_score(frame.event, frame.probability_dead)
            if events > 0 and censored > 0 else np.nan,
        })
    per_cancer = pd.DataFrame(rows)
    per_cancer.to_csv(RESULTS / "bulkformer_parity_alive_dead_prognosis_per_cancer.csv", index=False)

    matrix = per_cancer.pivot(index="cancer_label", columns="representation", values="auroc")
    order = ["ours_45.6m", "bulkformer_50m", "bulkformer_147m", "full_raw_expression"]
    matrix = matrix.reindex(columns=order)
    matrix = matrix.loc[matrix.mean(axis=1, skipna=True).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.4, 11.0))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlBu", vmin=0.3, vmax=0.9)
    ax.set_xticks(np.arange(len(order)), [labels[value] for value in order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_title("Per-cancer alive/dead AUROC from out-of-fold predictions")
    for row in range(len(matrix)):
        for column in range(len(order)):
            value = matrix.iloc[row, column]
            ax.text(column, row, "NA" if pd.isna(value) else f"{value:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if pd.isna(value) or 0.43 < value < 0.78 else "white")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("AUROC")
    ax.set_xlabel("Representation")
    ax.set_ylabel("TCGA cancer type")
    fig.tight_layout()
    fig.savefig(figures / "bulkformer_parity_alive_dead_prognosis_per_cancer.png", dpi=220)
    fig.savefig(figures / "bulkformer_parity_alive_dead_prognosis_per_cancer.pdf")
    plt.close(fig)


def write_provenance(manifest: pd.DataFrame) -> None:
    released = Path("/tmp/TCGA_survival.h5ad")
    provenance = {
        "created": "2026-09-22",
        "final_paper": {"doi": "10.1016/j.cels.2026.101657", "pii": "S2405471226001390"},
        "official_repository": {
            "url": "https://github.com/KangBoming/BulkFormer",
            "commit": "5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589",
        },
        "released_data": {
            "doi": "10.5281/zenodo.15744294",
            "file": "TCGA_survival.h5ad",
            "sha256": "3ae462695aafc5c8399b77ff24ff79e209d7e52f2e51e9452642548c1d5b4414",
            "audit": {"samples": 10429, "unique_patients": 9679, "classes": 33,
                      "repeated_patient_rows": 750, "normal_sample_codes_11": 709},
        },
        "local_common_cohort": {"patients": len(manifest), "classes": 33,
                                "one_sample_per_patient": True,
                                "primary_tumor_plus_laml_peripheral_blood": True},
        "evaluation": {"folds": FOLDS, "split_seed": SEED, "pca_components": PCA_COMPONENTS,
                       "pca_fit_on_training_fold_only": True,
                       "random_forest": "scikit-learn defaults; random_state fixed per fold",
                       "bulkformer_pooling": "mean gene-token pooling, matching final release notebook example"},
        "ambiguities": [
            "The final repository does not release downstream training code or fold assignments.",
            "The final feature-extraction notebook demonstrates mean pooling, while the 2025 preprint states max pooling.",
            "The final paper reports prognosis modeling but the final repository README omits prognosis metrics.",
            "The released TCGA object contains repeated patients and normal-tissue aliquots, so the literature value is not patient-level paired with the local comparison.",
        ],
        "tcga_h5": str(TCGA_H5), "tcga_h5_sha256": sha256(TCGA_H5),
        "frozen_encoders": True, "test_adaptive_method_changes": False,
    }
    (RESULTS / "bulkformer_tcga_protocol_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    manifest = build_manifest()
    prepare_inputs(manifest)
    if not args.skip_inference:
        for name in MODELS:
            embed_laml(name, device, args.heartbeat_seconds)
    missing = [name for name in MODELS if not (
        WORK / f"bulkformer_parity_laml_{name}_embeddings.npy").is_file()]
    if missing:
        raise RuntimeError(f"Missing incremental embeddings: {missing}")
    save_embedding_manifest(manifest)
    write_provenance(manifest)
    evaluate_classification(manifest)
    evaluate_prognosis(manifest)
    summarize()
    plot_results()
    say("BulkFormer-parity extension complete")


if __name__ == "__main__":
    main()
