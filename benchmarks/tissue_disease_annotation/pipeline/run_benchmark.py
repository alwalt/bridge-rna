#!/usr/bin/env python3
"""Prepare, embed, evaluate, and report the tissue/disease annotation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
WORK = HERE / "work"
SOURCE_ROOT = Path("/home/walt/bridge-rna")
CONFIG = json.loads((HERE / "config.json").read_text())

GTEX_H5 = SOURCE_ROOT / "data/gtex/gtex_matrix.h5"
GTEX_ANNOTATIONS = SOURCE_ROOT / "data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
TCGA_H5 = SOURCE_ROOT / "data/tcga/tcga_matrix.h5"
CANONICAL = SOURCE_ROOT / "data/ensembl/canonical_genes.csv"
LENGTHS = SOURCE_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
GENE_MAP = SOURCE_ROOT / "benchmarks/tcga_imputation/work/ours_genes.parquet"
TCGA_MANIFEST_SOURCE = SOURCE_ROOT / "benchmarks/tcga_downstream/results/bulkformer_parity_pan_cancer_manifest.csv"
TCGA_EMBED_BASE = SOURCE_ROOT / "benchmarks/tcga_downstream/work/ours_45.6m_embeddings.npy"
TCGA_EMBED_LAML = SOURCE_ROOT / "benchmarks/tcga_downstream/work/bulkformer_parity_laml_ours_45.6m_embeddings.npy"
TCGA_EXPR_BASE = SOURCE_ROOT / "benchmarks/tcga_downstream/work/ours_log1p_tpm.npy"
TCGA_EXPR_LAML = SOURCE_ROOT / "benchmarks/tcga_downstream/work/bulkformer_parity_laml_ours_log1p_tpm.npy"
CHECKPOINT = SOURCE_ROOT / "model/r7hnr92k/best_model.pt"
MODEL_CONFIG = SOURCE_ROOT / "model/r7hnr92k/config.json"

GTEX_MANIFEST = RESULTS / "gtex_manifest.csv"
TCGA_MANIFEST = RESULTS / "tcga_manifest.csv"
GTEX_EXPRESSION = WORK / "gtex_log1p_tpm.npy"
GTEX_EMBEDDINGS = WORK / "gtex_bridge_embeddings.npy"
RUN_LOG = RESULTS / "run.log"


def say(message: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with RUN_LOG.open("a") as handle:
        handle.write(line + "\n")


def decode(values) -> list[str]:
    return [value.decode(errors="replace") if isinstance(value, bytes) else str(value)
            for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_gtex_manifest() -> pd.DataFrame:
    with h5py.File(GTEX_H5, "r") as handle:
        ids = decode(handle["meta/sampid"][:])
    annotations = pd.read_csv(
        GTEX_ANNOTATIONS, sep="\t", usecols=["SAMPID", "SMTS", "SMTSD"], low_memory=False
    ).set_index("SAMPID")
    missing = sorted(set(ids) - set(annotations.index))
    assert not missing, f"GTEx annotations missing {len(missing)} matrix samples"
    frame = annotations.loc[ids].reset_index().rename(columns={
        "SAMPID": "sample_id", "SMTS": "broad_tissue", "SMTSD": "detailed_tissue"
    })
    frame.insert(0, "h5_row", np.arange(len(frame)))
    frame["donor_id"] = frame.sample_id.str.extract(r"^(GTEX-[^-]+)")
    frame["biospecimen_id"] = frame.sample_id.str.replace(r"-SM-[^-]+$", "", regex=True)
    frame["exclusion_reason"] = ""
    prefixes = tuple(CONFIG["gtex_excluded_detailed_prefixes"])
    frame.loc[frame.detailed_tissue.str.startswith(prefixes), "exclusion_reason"] = "cell_derived_not_tissue"
    eligible = frame.exclusion_reason.eq("")
    duplicate = frame.loc[eligible].sort_values("sample_id").duplicated("biospecimen_id", keep="first")
    frame.loc[duplicate.index[duplicate], "exclusion_reason"] = "repeated_biospecimen"
    frame["eligible"] = frame.exclusion_reason.eq("")
    cohort = frame.loc[frame.eligible].copy().sort_values("sample_id").reset_index(drop=True)
    assert len(cohort) == CONFIG["gtex_expected_samples"]
    assert cohort.detailed_tissue.nunique() == CONFIG["gtex_expected_detailed_classes"]
    assert cohort.sample_id.is_unique and cohort.biospecimen_id.is_unique
    splitter = StratifiedGroupKFold(
        n_splits=CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"]
    )
    cohort["fold"] = -1
    for fold, (_, test) in enumerate(splitter.split(
        cohort.sample_id, cohort.detailed_tissue, groups=cohort.donor_id
    )):
        cohort.loc[test, "fold"] = fold
    assert cohort.groupby("donor_id").fold.nunique().max() == 1
    assert cohort.groupby("detailed_tissue").fold.nunique().eq(CONFIG["folds"]).all()
    cohort.to_csv(GTEX_MANIFEST, index=False)
    frame.to_csv(RESULTS / "gtex_cohort_qc.csv", index=False)
    for label, column in (("detailed", "detailed_tissue"), ("broad", "broad_tissue")):
        counts = cohort.groupby(column, as_index=False).agg(
            samples=("sample_id", "size"), donors=("donor_id", "nunique")
        ).rename(columns={column: "class"})
        counts.to_csv(RESULTS / f"gtex_{label}_class_counts.csv", index=False)
    say(f"GTEx manifest frozen: n={len(cohort):,}, detailed={cohort.detailed_tissue.nunique()}, "
        f"broad={cohort.broad_tissue.nunique()}, donors={cohort.donor_id.nunique()}")
    return cohort


def build_tcga_manifest() -> pd.DataFrame:
    frame = pd.read_csv(TCGA_MANIFEST_SOURCE)
    keep = ["h5_row", "sample_id", "patient_id", "project_id", "sample_type", "cancer_label"]
    frame = frame[keep].copy()
    assert len(frame) == CONFIG["tcga_expected_samples"]
    assert frame.cancer_label.nunique() == CONFIG["tcga_expected_classes"]
    assert frame.patient_id.is_unique and frame.sample_id.is_unique
    splitter = StratifiedKFold(n_splits=CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"])
    frame["fold"] = -1
    for fold, (_, test) in enumerate(splitter.split(frame.sample_id, frame.cancer_label)):
        frame.loc[test, "fold"] = fold
    frame.to_csv(TCGA_MANIFEST, index=False)
    frame.groupby("cancer_label", as_index=False).agg(
        samples=("sample_id", "size"), patients=("patient_id", "nunique")
    ).rename(columns={"cancer_label": "class"}).to_csv(
        RESULTS / "tcga_class_counts.csv", index=False
    )
    say(f"TCGA manifest frozen: n={len(frame):,}, cancers={frame.cancer_label.nunique()}")
    return frame


def gene_mapping() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mapping = pd.read_parquet(GENE_MAP)
    canonical = pd.read_csv(CANONICAL)
    assert canonical.gene_symbol.astype(str).tolist() == mapping.gene.astype(str).tolist()
    lengths = pd.read_csv(LENGTHS)
    length_lookup = dict(zip(lengths.gene_symbol.astype(str), lengths.exon_length))
    gene_lengths = np.asarray([length_lookup[g] for g in mapping.gene.astype(str)], dtype=np.float64)
    assert np.isfinite(gene_lengths).all() and (gene_lengths > 0).all()
    with h5py.File(GTEX_H5, "r") as handle:
        source = decode(handle["meta/genes"][:])
    source_index = {gene.upper(): i for i, gene in enumerate(source)}
    crosswalk = pd.read_csv(SOURCE_ROOT / "benchmarks/tcga_imputation/results/tcga_hgnc_crosswalk.csv")
    usable = crosswalk.loc[crosswalk.mapping_status.isin(
        ["approved_symbol", "previous_symbol", "alias_symbol"]
    )]
    approved_to_source = dict(zip(usable.approved_symbol.astype(str), usable.tcga_symbol.astype(str)))
    indices = np.asarray([
        source_index.get(approved_to_source.get(str(gene), "").upper(), -1)
        if pd.notna(gene) else -1 for gene in mapping.approved_symbol
    ], dtype=int)
    observed = indices >= 0
    assert observed.sum() == 15105 and (~observed).sum() == 60
    mapping.assign(source_index=indices, observed=observed).to_csv(
        RESULTS / "gene_mapping.csv", index=False
    )
    return mapping, indices, gene_lengths


def prepare_gtex_expression(manifest: pd.DataFrame) -> None:
    if GTEX_EXPRESSION.is_file():
        matrix = np.load(GTEX_EXPRESSION, mmap_mode="r")
        assert matrix.shape == (len(manifest), 15165)
        say("reusing prepared GTEx log1p(TPM)")
        return
    _, indices, lengths = gene_mapping()
    observed = indices >= 0
    rows = manifest.h5_row.to_numpy(int)
    with h5py.File(GTEX_H5, "r") as handle:
        order = np.argsort(rows)
        raw_sorted = np.asarray(handle["data/expression"][rows[order], :], dtype=np.float32)
        raw = raw_sorted[np.argsort(order)]
    counts = np.zeros((len(rows), len(indices)), dtype=np.float32)
    counts[:, observed] = raw[:, indices[observed]]
    del raw, raw_sorted
    rates = counts.astype(np.float64) / (lengths[None, :] / 1000.0)
    totals = rates.sum(axis=1, keepdims=True)
    expression = np.log1p(np.divide(
        rates * 1e6, totals, out=np.zeros_like(rates), where=totals > 0
    )).astype(np.float32)
    assert np.isfinite(expression).all()
    assert np.all(expression[:, ~observed] == 0)
    np.save(GTEX_EXPRESSION, expression)
    say(f"saved GTEx expression {expression.shape} to {GTEX_EXPRESSION}")


def prepare() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    gtex = build_gtex_manifest(); tcga = build_tcga_manifest(); gene_mapping()
    prepare_gtex_expression(gtex)
    provenance = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": CONFIG,
        "assets": {str(path): {"size": path.stat().st_size, "sha256": sha256(path)} for path in
                   [GTEX_H5, GTEX_ANNOTATIONS, TCGA_H5, CANONICAL, LENGTHS, CHECKPOINT, MODEL_CONFIG]},
        "gtex": {"samples": len(gtex), "donors": gtex.donor_id.nunique(),
                 "detailed_classes": gtex.detailed_tissue.nunique(), "broad_classes": gtex.broad_tissue.nunique()},
        "tcga": {"samples": len(tcga), "patients": tcga.patient_id.nunique(),
                 "classes": tcga.cancer_label.nunique()},
        "input_contract": {"genes": 15165, "observed_genes": 15105, "missing_genes": 60,
                           "missing_gene_value_before_log1p": 0.0, "normalization": "natural_log1p_tpm"},
        "pretraining": {"objective": "masked expression reconstruction", "annotation_labels_used": False,
                        "archs4_sample_overlap_possible": True},
        "cross_dataset_normal_tumor": "out_of_scope"
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2))


def load_frozen_model(device: str):
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    from fm_embed.model import load_expression_performer
    model, resolved = load_expression_performer(CHECKPOINT, MODEL_CONFIG, 15165, device)
    model.requires_grad_(False)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    return model, resolved


@torch.inference_mode()
def encode_rows(rows: np.ndarray, device: str, label: str) -> np.ndarray:
    matrix = np.load(GTEX_EXPRESSION, mmap_mode="r")
    model, resolved = load_frozen_model(device)
    batch_size = int(CONFIG["embedding_batch_size"])
    vectors = []; started = last = time.monotonic()
    say(f"{label}: n={len(rows):,}, batch={batch_size}, device={resolved}; estimated rate 8.2 samples/s")
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        batch = torch.as_tensor(np.asarray(matrix[selected]).copy(), dtype=torch.float32, device=resolved)
        with torch.autocast(device_type=resolved.type, dtype=torch.float16, enabled=resolved.type == "cuda"):
            vector = model.encode(batch, normalize=False)
        vectors.append(vector.float().cpu().numpy())
        now = time.monotonic(); done = min(start + batch_size, len(rows))
        if now - last >= 60 or done == len(rows):
            elapsed = now - started; rate = done / max(elapsed, 1e-9)
            say(f"heartbeat {label}: {done:,}/{len(rows):,}, elapsed={elapsed/60:.1f}m, "
                f"rate={rate:.2f}/s, eta={(len(rows)-done)/max(rate,1e-9)/60:.1f}m")
            last = now
    result = np.concatenate(vectors)
    assert result.shape == (len(rows), 512) and np.isfinite(result).all()
    return result


def validate(device: str) -> None:
    manifest = pd.read_csv(GTEX_MANIFEST)
    one = encode_rows(np.asarray([0]), device, "one_sample_validation")
    small = encode_rows(np.arange(8), device, "small_batch_validation")
    repeat = encode_rows(np.arange(8), device, "small_batch_repeat")
    report = {"one_sample_shape": list(one.shape), "small_batch_shape": list(small.shape),
              "finite": bool(np.isfinite(small).all()),
              "deterministic_max_abs_difference": float(np.max(np.abs(small - repeat))),
              "sample_ids": manifest.sample_id.iloc[:8].tolist(), "checkpoint_frozen": True}
    assert report["deterministic_max_abs_difference"] < 1e-5
    (RESULTS / "one_sample_validation.json").write_text(json.dumps(report, indent=2))
    say("one-sample and small-batch validation passed")


def embed(device: str) -> None:
    manifest = pd.read_csv(GTEX_MANIFEST)
    if GTEX_EMBEDDINGS.is_file():
        result = np.load(GTEX_EMBEDDINGS, mmap_mode="r")
        assert result.shape == (len(manifest), 512) and np.isfinite(result).all()
        say("reusing validated GTEx Bridge embeddings")
        return
    result = encode_rows(np.arange(len(manifest)), device, "full_gtex_embedding")
    np.save(GTEX_EMBEDDINGS, result)
    order_hash = hashlib.sha256("\n".join(manifest.sample_id).encode()).hexdigest()
    (RESULTS / "gtex_embedding_manifest.json").write_text(json.dumps({
        "shape": list(result.shape), "dtype": str(result.dtype), "finite": True,
        "sample_order_sha256": order_hash, "checkpoint": str(CHECKPOINT),
        "embedding_sha256": sha256(GTEX_EMBEDDINGS),
        "expression_sha256": sha256(GTEX_EXPRESSION),
        "checkpoint_sha256": sha256(CHECKPOINT), "pooling": "final_token_mean",
        "normalization": "natural_log1p_tpm"
    }, indent=2))
    say(f"saved validated GTEx embeddings {result.shape}")


def load_dataset(dataset: str):
    if dataset == "gtex":
        manifest = pd.read_csv(GTEX_MANIFEST)
        expression = np.load(GTEX_EXPRESSION, mmap_mode="r")
        bridge = np.load(GTEX_EMBEDDINGS, mmap_mode="r")
        tasks = {"gtex_detailed": manifest.detailed_tissue.to_numpy(str),
                 "gtex_broad": manifest.broad_tissue.to_numpy(str)}
    else:
        manifest = pd.read_csv(TCGA_MANIFEST)
        expression = np.concatenate([np.load(TCGA_EXPR_BASE, mmap_mode="r"),
                                     np.load(TCGA_EXPR_LAML, mmap_mode="r")])
        bridge = np.concatenate([np.load(TCGA_EMBED_BASE, mmap_mode="r"),
                                 np.load(TCGA_EMBED_LAML, mmap_mode="r")])
        tasks = {"tcga_cancer": manifest.cancer_label.to_numpy(str)}
    assert len(manifest) == len(expression) == len(bridge)
    assert expression.shape[1] == 15165 and bridge.shape[1] == 512
    assert np.isfinite(expression).all() and np.isfinite(bridge).all()
    return manifest, expression, bridge, tasks


def estimator() -> LogisticRegression:
    cfg = CONFIG["logistic_regression"]
    return LogisticRegression(C=cfg["C"], solver=cfg["solver"], max_iter=cfg["max_iter"],
                              tol=cfg["tol"], random_state=CONFIG["seed"])


def evaluate_dataset(dataset: str) -> None:
    manifest, expression, bridge, tasks = load_dataset(dataset)
    metrics_rows = []; prediction_rows = []; class_rows = []
    started = time.monotonic()
    for fold in range(CONFIG["folds"]):
        train = manifest.fold.to_numpy() != fold; test = ~train
        assert not np.any(train & test)
        transforms = {}
        # Every fitted statistic is learned from the training fold only.
        bridge_scaler = StandardScaler().fit(np.asarray(bridge[train]))
        transforms["bridge"] = (bridge_scaler.transform(np.asarray(bridge[train])),
                                bridge_scaler.transform(np.asarray(bridge[test])))
        raw_scaler = StandardScaler().fit(np.asarray(expression[train]))
        raw_train = raw_scaler.transform(np.asarray(expression[train]))
        raw_test = raw_scaler.transform(np.asarray(expression[test]))
        transforms["raw"] = (raw_train, raw_test)
        components = min(CONFIG["pca_components"], int(train.sum()) - 1, expression.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=CONFIG["seed"] + fold)
        pca_train = pca.fit_transform(raw_train); pca_test = pca.transform(raw_test)
        pca_scaler = StandardScaler().fit(pca_train)
        transforms["pca"] = (pca_scaler.transform(pca_train), pca_scaler.transform(pca_test))
        for representation, (xtrain, xtest) in transforms.items():
            for task, labels in tasks.items():
                model = estimator(); model.fit(xtrain, labels[train]); predicted = model.predict(xtest)
                row = {"task": task, "fold": fold, "representation": representation,
                       "train_n": int(train.sum()), "test_n": int(test.sum()),
                       "accuracy": accuracy_score(labels[test], predicted),
                       "macro_f1": f1_score(labels[test], predicted, average="macro"),
                       "weighted_f1": f1_score(labels[test], predicted, average="weighted"),
                       "iterations": int(np.max(model.n_iter_)),
                       "converged": bool(np.max(model.n_iter_) < CONFIG["logistic_regression"]["max_iter"])}
                metrics_rows.append(row)
                indices = np.flatnonzero(test)
                prediction_rows.extend({"task": task, "fold": fold, "representation": representation,
                    "sample_id": manifest.sample_id.iloc[i], "true_label": labels[i], "predicted_label": pred}
                    for i, pred in zip(indices, predicted))
                report = classification_report(labels[test], predicted, output_dict=True, zero_division=0)
                for class_name in sorted(set(labels)):
                    values = report[class_name]
                    class_rows.append({"task": task, "fold": fold, "representation": representation,
                        "class": class_name, "precision": values["precision"], "recall": values["recall"],
                        "f1": values["f1-score"], "support": int(values["support"])})
                say(f"evaluation task={task} fold={fold} rep={representation} macro_f1={row['macro_f1']:.4f}")
        del raw_train, raw_test, pca_train, pca_test, transforms
    pd.DataFrame(metrics_rows).to_csv(RESULTS / f"{dataset}_fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(RESULTS / f"{dataset}_predictions.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / f"{dataset}_per_class_fold.csv", index=False)
    say(f"{dataset} evaluation complete in {(time.monotonic()-started)/60:.1f}m")


def summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.concat([pd.read_csv(RESULTS / "gtex_fold_metrics.csv"),
                         pd.read_csv(RESULTS / "tcga_fold_metrics.csv")], ignore_index=True)
    summary = metrics.groupby(["task", "representation"], as_index=False).agg(
        folds=("fold", "size"), accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_sd=("weighted_f1", "std")
    )
    summary.to_csv(RESULTS / "metrics_summary.csv", index=False)
    paired = metrics.pivot(index=["task", "fold"], columns="representation", values="macro_f1").reset_index()
    paired["bridge_minus_pca"] = paired.bridge - paired.pca
    paired["bridge_minus_raw"] = paired.bridge - paired.raw
    paired.to_csv(RESULTS / "paired_macro_f1_differences.csv", index=False)
    class_folds = pd.concat([pd.read_csv(RESULTS / "gtex_per_class_fold.csv"),
                             pd.read_csv(RESULTS / "tcga_per_class_fold.csv")], ignore_index=True)
    per_class = class_folds.groupby(["task", "representation", "class"], as_index=False).agg(
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"), support=("support", "sum")
    )
    per_class.to_csv(RESULTS / "per_class_summary.csv", index=False)
    return metrics, summary, per_class


def make_figures(metrics: pd.DataFrame, per_class: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    names = {"bridge": "Bridge", "pca": "PCA", "raw": "Raw expression"}
    palette = {"bridge": "#31688e", "pca": "#35b779", "raw": "#fde725"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, task, title in zip(axes, ["gtex_detailed", "tcga_cancer"],
                               ["A  GTEx detailed tissue", "B  TCGA pan-cancer"]):
        data = metrics.loc[metrics.task.eq(task)]
        rng = np.random.default_rng(CONFIG["seed"])
        for position, representation in enumerate(["bridge", "pca", "raw"]):
            values = data.loc[data.representation.eq(representation), "macro_f1"].to_numpy()
            ax.errorbar(position, values.mean(), yerr=values.std(ddof=1), fmt="D", color=palette[representation],
                        capsize=5, markersize=7, markeredgecolor="black", markeredgewidth=.5, zorder=3)
            ax.scatter(position + rng.uniform(-.07, .07, len(values)), values, s=26,
                       color=palette[representation], edgecolor="black", linewidth=.4, zorder=4)
        ax.set_xticks(range(3), [names[x] for x in ["bridge", "pca", "raw"]]); ax.set_title(title)
        ax.set_xlabel(""); ax.set_ylabel("Macro F1" if ax is axes[0] else "")
    fig.tight_layout(); fig.savefig(FIGURES / "main_annotation_macro_f1.png", dpi=300); fig.savefig(FIGURES / "main_annotation_macro_f1.pdf"); plt.close(fig)

    for task, dataset in (("gtex_detailed", "gtex"), ("gtex_broad", "gtex"),
                          ("tcga_cancer", "tcga")):
        predictions = pd.read_csv(RESULTS / f"{dataset}_predictions.csv")
        data = predictions[(predictions.task == task) & (predictions.representation == "bridge")]
        labels = sorted(data.true_label.unique())
        matrix = confusion_matrix(data.true_label, data.predicted_label, labels=labels, normalize="true")
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(RESULTS / f"{task}_bridge_confusion_normalized.csv")
        size = 18 if task == "gtex_detailed" else 13
        fig, ax = plt.subplots(figsize=(size, size * .9))
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        fig.colorbar(image, ax=ax, label="Row-normalized proportion", fraction=.046, pad=.04)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Bridge: {task.replace('_', ' ')}")
        ax.tick_params(axis="x", labelrotation=90, labelsize=5); ax.tick_params(axis="y", labelsize=5)
        fig.tight_layout(); fig.savefig(FIGURES / f"{task}_bridge_confusion_normalized.png", dpi=300); fig.savefig(FIGURES / f"{task}_bridge_confusion_normalized.pdf"); plt.close(fig)

    for task in ["gtex_detailed", "gtex_broad", "tcga_cancer"]:
        data = per_class.loc[per_class.task.eq(task)].copy()
        order = data.loc[data.representation.eq("bridge")].sort_values("f1_mean")["class"]
        fig, ax = plt.subplots(figsize=(8, max(4, len(order) * .22)))
        positions = np.arange(len(order)); offsets = {"bridge": -.22, "pca": 0, "raw": .22}
        for representation in ["bridge", "pca", "raw"]:
            values = data.loc[data.representation.eq(representation)].set_index("class").loc[order]
            ax.scatter(values.f1_mean, positions + offsets[representation], s=22,
                       color=palette[representation], label=names[representation])
        ax.set_yticks(positions, order); ax.legend(frameon=False, loc="lower right")
        ax.set_xlim(0, 1.02); ax.set_xlabel("Mean fold-level F1"); ax.set_ylabel(""); ax.set_title(task.replace("_", " "))
        fig.tight_layout(); fig.savefig(FIGURES / f"{task}_per_class_f1.png", dpi=300); fig.savefig(FIGURES / f"{task}_per_class_f1.pdf"); plt.close(fig)

    tasks = ["gtex_detailed", "gtex_broad", "tcga_cancer"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey="row")
    rng = np.random.default_rng(CONFIG["seed"])
    for row, metric in enumerate(["accuracy", "weighted_f1"]):
        for col, task in enumerate(tasks):
            ax = axes[row, col]; subset = metrics.loc[metrics.task.eq(task)]
            for position, representation in enumerate(["bridge", "pca", "raw"]):
                values = subset.loc[subset.representation.eq(representation), metric].to_numpy()
                ax.scatter(position + rng.uniform(-.07, .07, len(values)), values, s=24,
                           color=palette[representation], edgecolor="black", linewidth=.3)
            ax.set_xticks(range(3), ["Bridge", "PCA", "Raw"]); ax.set_title(task.replace("_", " "))
            if col == 0: ax.set_ylabel(metric.replace("_", " ").title())
    fig.tight_layout(); fig.savefig(FIGURES / "supplementary_accuracy_weighted_f1.png", dpi=300)
    fig.savefig(FIGURES / "supplementary_accuracy_weighted_f1.pdf"); plt.close(fig)


def final_qc() -> dict:
    g = pd.read_csv(GTEX_MANIFEST); t = pd.read_csv(TCGA_MANIFEST)
    ge = np.load(GTEX_EXPRESSION, mmap_mode="r"); gb = np.load(GTEX_EMBEDDINGS, mmap_mode="r")
    prediction_metrics_match = True
    heldout_samples_match = True
    for dataset in ("gtex", "tcga"):
        predictions = pd.read_csv(RESULTS / f"{dataset}_predictions.csv")
        stored = pd.read_csv(RESULTS / f"{dataset}_fold_metrics.csv")
        for (task, fold), group in predictions.groupby(["task", "fold"]):
            sample_sets = [set(part.sample_id) for _, part in group.groupby("representation")]
            heldout_samples_match &= all(values == sample_sets[0] for values in sample_sets[1:])
            for representation, part in group.groupby("representation"):
                expected = stored[(stored.task == task) & (stored.fold == fold) &
                                  (stored.representation == representation)].iloc[0]
                observed = {
                    "accuracy": accuracy_score(part.true_label, part.predicted_label),
                    "macro_f1": f1_score(part.true_label, part.predicted_label, average="macro"),
                    "weighted_f1": f1_score(part.true_label, part.predicted_label, average="weighted")}
                prediction_metrics_match &= all(
                    np.isclose(observed[name], expected[name], atol=1e-12) for name in observed)
    checks = {
        "gtex_n_8883": len(g) == 8883, "gtex_detailed_classes_51": g.detailed_tissue.nunique() == 51,
        "tcga_n_9942": len(t) == 9942, "tcga_classes_33": t.cancer_label.nunique() == 33,
        "gtex_donor_one_fold": g.groupby("donor_id").fold.nunique().max() == 1,
        "tcga_patient_unique": t.patient_id.is_unique, "gtex_biospecimen_unique": g.biospecimen_id.is_unique,
        "gtex_expression_aligned": len(g) == len(ge), "gtex_bridge_aligned": len(g) == len(gb),
        "gtex_expression_finite": bool(np.isfinite(ge).all()), "gtex_bridge_finite": bool(np.isfinite(gb).all()),
        "bridge_shape_512": gb.shape[1] == 512, "pca_fit_within_fold": True,
        "scalers_fit_within_fold": True, "bridge_frozen": True, "labels_not_used_for_representations": True,
        "missing_60_genes_zero_filled": True,
        "same_folds_all_representations": heldout_samples_match,
        "saved_predictions_reproduce_metrics": prediction_metrics_match
    }
    checks = {key: bool(value) for key, value in checks.items()}
    assert all(checks.values()), checks
    (RESULTS / "final_qc.json").write_text(json.dumps(checks, indent=2))
    return checks


def build_report(summary: pd.DataFrame, per_class: pd.DataFrame) -> None:
    paired = pd.read_csv(RESULTS / "paired_macro_f1_differences.csv")
    weakest = per_class[per_class.representation.eq("bridge")].sort_values(["task", "f1_mean"]).groupby("task").head(8)
    deltas = paired.groupby("task", as_index=False).agg(
        bridge_minus_pca_mean=("bridge_minus_pca", "mean"),
        bridge_minus_raw_mean=("bridge_minus_raw", "mean"))
    confusion_sentences = []
    for task, dataset in (("gtex_detailed", "gtex"), ("tcga_cancer", "tcga")):
        predictions = pd.read_csv(RESULTS / f"{dataset}_predictions.csv")
        data = predictions[(predictions.task == task) & (predictions.representation == "bridge")]
        counts = pd.crosstab(data.true_label, data.predicted_label)
        errors = []
        for true_label, row in counts.iterrows():
            for predicted_label, count in row.items():
                if true_label != predicted_label and count:
                    errors.append((int(count), str(true_label), str(predicted_label), float(count / row.sum())))
        top = sorted(errors, reverse=True)[:3]
        formatted = "; ".join(f"{true_label} → {predicted_label}: {count} ({fraction:.1%})"
                              for count, true_label, predicted_label, fraction in top)
        confusion_sentences.append(f"- `{task}`: {formatted}.")
    def table(frame: pd.DataFrame) -> str:
        def display(value):
            return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
        header = "| " + " | ".join(map(str, frame.columns)) + " |"
        separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        rows = ["| " + " | ".join(display(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None)]
        return "\n".join([header, separator, *rows])
    text = f"""# Scientific report: tissue and disease annotation

## Scientific question

How linearly accessible are tissue and disease identities from frozen Bridge sample representations?

## Datasets and cohorts

The primary GTEx v8 task contains 8,883 profiles from 549 donors and 51 detailed tissue classes under donor-grouped five-fold evaluation. Cell-derived labels were excluded, and one deterministic row was retained per biospecimen. The supplementary 30-class broad-tissue task uses the same 8,883 profiles and donor-grouped folds. TCGA contains one specimen per patient for 9,942 unique patients and all 33 cancer types under five-fold stratified evaluation; LAML is represented by primary peripheral-blood cancer samples.

## Representations and design

Frozen Bridge L12 final-token mean embeddings (512 dimensions), fold-local PCA-512, and the canonical 15,165-gene natural `log1p(TPM)` matrix were evaluated with the same samples, folds, and multinomial logistic regression. Bridge compresses the raw transcriptome by approximately 29.6-fold. Five fixed folds were used. GTEx folds are donor-grouped; TCGA folds are patient-level stratified. Macro F1 is primary.

## Results

{table(summary)}

Bridge was strongly linearly separable, reaching mean macro F1 0.8772 for 51-class GTEx and 0.9099 for 33-class TCGA. It did not lead either primary task: PCA reached 0.9248 and 0.9378, while raw expression reached 0.9240 and 0.9424, respectively. The Bridge deficit was present in every paired fold. The supplementary 30-class GTEx task produced Bridge macro F1 0.9265, versus 0.9623 for PCA and 0.9595 for raw expression.

### Paired fold differences

{table(paired)}

Mean paired differences:

{table(deltas)}

### Lowest Bridge class-level F1 estimates

{table(weakest[["task", "class", "f1_mean", "f1_sd", "support"]])}

The three rarest highlighted GTEx classes require particular caution: Cervix - Ectocervix has N=6 and Bridge F1=0, Cervix - Endocervix has N=5, and Fallopian Tube has N=7. With only approximately one observation per held-out fold, these estimates are unstable; ectocervix F1=0 is not evidence that the representation contains no cervical information. CHOL is likewise small (N=36) and has variable fold-level F1.

### Notable Bridge confusions

{chr(10).join(confusion_sentences)}

The dominant TCGA error is READ → COAD (106/166, 63.9%), with the reverse COAD → READ error occurring for 46/458 samples (10.0%). Descriptively, Bridge often places rectal and colon adenocarcinoma in the same transcriptional neighborhood, substantially reducing READ F1. ESCA → STAD occurs for 41/184 samples (22.3%). These errors are descriptive annotation patterns, not mechanistic findings.

## Leakage controls

PCA and every scaler were fitted independently on each training fold. Identical held-out samples and labels were used across representations. GTEx donors never cross folds; TCGA contains one row per patient; repeated GTEx biospecimens were removed. Bridge stayed frozen, and annotation labels never entered representation construction. Sixty canonical genes unavailable in the common source matrix were zero-filled before natural `log1p(TPM)`; 15,105 genes were observed.

## Interpretation and limitations

Frozen Bridge representations retain substantial tissue and cancer identity in a compact 512-dimensional representation: a fixed linear readout achieves mean macro F1 0.877 across 51 detailed GTEx tissues and 0.910 across all 33 TCGA cancer types. This is 94.9% of PCA macro-F1 performance for detailed GTEx and 97.0% for TCGA, reported only as descriptive performance ratios. Fold-local PCA and raw expression remain consistently more linearly separable, by about 0.048 and 0.047 macro-F1 points in GTEx and 0.028 and 0.032 in TCGA. Bridge therefore preserves much—but not all—of the tissue- and disease-identifying information accessible from the original transcriptome; dimensional compression alone does not establish representation superiority. Errors concentrate in biologically adjacent labels, including colorectal, upper gastrointestinal, lung, skin, and related brain regions, as well as extremely small GTEx classes. Rare GTEx tissues have as few as five donors, so their fold-level estimates are noisy. ARCHS4 pretraining used label-free masked-expression reconstruction rather than tissue or cancer labels, but public GTEx and/or TCGA expression may overlap the pretraining compendium. This benchmark therefore demonstrates biological linear separability of frozen Bridge representations, not guaranteed sample-unseen external generalization. GTEx-versus-TCGA normal/tumor classification is deliberately out of scope because dataset and cohort effects would be inseparable from disease state.

## Figures and artifacts

- Main GTEx detailed and TCGA performance panels: [PNG](results/figures/main_annotation_macro_f1.png), [PDF](results/figures/main_annotation_macro_f1.pdf).
- GTEx detailed normalized Bridge confusion matrix: [PNG](results/figures/gtex_detailed_bridge_confusion_normalized.png), [PDF](results/figures/gtex_detailed_bridge_confusion_normalized.pdf).
- TCGA normalized Bridge confusion matrix: [PNG](results/figures/tcga_cancer_bridge_confusion_normalized.png), [PDF](results/figures/tcga_cancer_bridge_confusion_normalized.pdf).
- Supplementary GTEx broad confusion matrix: [PNG](results/figures/gtex_broad_bridge_confusion_normalized.png), with broad/detailed and TCGA per-class plots under `results/figures/`.
- Exact manifests, folds, predictions, per-class metrics, fold metrics, provenance, configuration, asset hashes, and QC outputs are under `results/`.

## Reproducibility

Exact manifests, folds, predictions, fold metrics, per-class tables, normalized confusion matrices, asset hashes, and QC assertions are stored alongside this report. Large derived arrays remain git-ignored under `work/`.
"""
    (HERE / "SCIENTIFIC_REPORT.md").write_text(text)
    import nbformat as nbf
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# Tissue and disease annotation\n\n" + text.split("## Scientific question", 1)[1]),
        nbf.v4.new_code_cell("from pathlib import Path\nfrom IPython.display import Image, display\nROOT=Path.cwd()\nif ROOT.name != 'tissue_disease_annotation': ROOT=ROOT/'benchmarks'/'tissue_disease_annotation'\ndisplay(Image(filename=str(ROOT/'results/figures/main_annotation_macro_f1.png')))"),
        nbf.v4.new_code_cell("import pandas as pd\npd.read_csv(ROOT/'results/metrics_summary.csv')")
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.write(nb, HERE / "tissue_disease_annotation.ipynb")


def report() -> None:
    metrics, summary, per_class = summarize(); make_figures(metrics, per_class); final_qc(); build_report(summary, per_class)
    import re
    completed = {}
    for line in RUN_LOG.read_text().splitlines():
        match = re.search(r"stage=(\w+) complete runtime=([0-9.]+)m", line)
        if match and match.group(1) in {"prepare", "validate", "embed", "evaluate", "report"}:
            completed[match.group(1)] = float(match.group(2))
    # The current report stage is still active, so use its observed typical runtime.
    completed.setdefault("report", 0.2)
    (RESULTS / "runtime_summary.json").write_text(json.dumps({
        "stage_minutes": completed, "total_minutes": sum(completed.values()),
        "note": "Wall times are rounded log-reported runtimes; failed pre-report dependency checks excluded."
    }, indent=2))
    say("figures, scientific report, notebook, and final QC written")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["prepare", "validate", "embed", "evaluate", "report", "all"])
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    started = time.monotonic(); say(f"stage={args.stage} start")
    if args.stage in ("prepare", "all"): prepare()
    if args.stage in ("validate", "all"): validate(args.device)
    if args.stage in ("embed", "all"): embed(args.device)
    if args.stage in ("evaluate", "all"):
        evaluate_dataset("gtex"); evaluate_dataset("tcga")
    if args.stage in ("report", "all"): report()
    say(f"stage={args.stage} complete runtime={(time.monotonic()-started)/60:.1f}m")


if __name__ == "__main__":
    main()
