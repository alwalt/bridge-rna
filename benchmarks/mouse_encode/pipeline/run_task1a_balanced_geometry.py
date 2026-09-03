#!/usr/bin/env python3
"""Balanced cross-species tissue readouts for Task 1A.

This extension keeps the strict fully-unseen outputs untouched. It uses all
healthy adult unperturbed expression-ready ENCODE profiles, caches each
representation once, and repeats only balanced GTEx subsampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda maxdigits: None  # type: ignore[attr-defined]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PIPELINE = Path(__file__).resolve().parent
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from audit_expanded_cohort import DIRECT_GTEX_MATCHES
from run_task1a_retrieval import encode_frozen, load_human_tpm, load_mouse_tpm, resolve
from run_task1a_geometry import cosine, probe_readout
from src.fm_embed.model import load_expression_performer
from src.fm_embed.vocab import load_canonical_genes

HERE = Path(__file__).resolve().parents[1]
BASE_RESULTS = HERE / "results"
OUT = BASE_RESULTS / "task1a_balanced_geometry"
CACHE = OUT / "cache"
FIGURES = OUT / "figures"

FIXED_11 = [
    "adrenal", "subcutaneous adipose", "cerebellum", "heart", "liver", "lung",
    "mammary gland", "ovary", "spleen", "stomach", "testis",
]
FIXED_5 = ["heart", "liver", "lung", "spleen", "testis"]

ENCODE_LABELS = {
    "adrenal gland": "adrenal",
    "gastrocnemius": "skeletal muscle",
    "skeletal muscle tissue": "skeletal muscle",
    "heart": "heart",
    "left cerebral cortex": "cortex",
    "layer of hippocampus": "hippocampus",
    "liver": "liver",
    "lung": "lung",
    "ovary": "ovary",
    "sigmoid colon": "sigmoid colon",
    "stomach": "stomach",
    "mammary gland": "mammary gland",
    "testis": "testis",
    "pancreas": "pancreas",
    "frontal cortex": "frontal cortex",
    "subcutaneous adipose tissue": "subcutaneous adipose",
    "cerebellum": "cerebellum",
    "spleen": "spleen",
}

GTEX_LABELS = {
    "Adrenal Gland": "adrenal",
    "Muscle - Skeletal": "skeletal muscle",
    "Heart - Atrial Appendage": "heart",
    "Heart - Left Ventricle": "heart",
    "Brain - Cortex": "cortex",
    "Brain - Hippocampus": "hippocampus",
    "Liver": "liver",
    "Lung": "lung",
    "Ovary": "ovary",
    "Colon - Sigmoid": "sigmoid colon",
    "Stomach": "stomach",
    "Breast - Mammary Tissue": "mammary gland",
    "Testis": "testis",
    "Pancreas": "pancreas",
    "Brain - Frontal Cortex (BA9)": "frontal cortex",
    "Adipose - Subcutaneous": "subcutaneous adipose",
    "Brain - Cerebellum": "cerebellum",
    "Spleen": "spleen",
}

REPRESENTATIONS = ["raw_expression", "joint_pca", "bridgerna"]
READOUTS = ["centroid_cosine", "knn_cosine_k1", "linear_softmax_probe"]
DIRECTIONS = ["human_to_mouse", "mouse_to_human"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_mouse_manifest(cfg: dict) -> pd.DataFrame:
    audit = pd.read_parquet(BASE_RESULTS / "expanded_encode_cohort_audit.parquet")
    selected = audit[
        audit["healthy_non_transgenic"]
        & audit["direct_gtex_match"]
        & audit["gene_quantification_available"]
        & audit["encode_tissue"].isin(ENCODE_LABELS)
    ].copy()
    selected["sample_id"] = selected["experiment_accession"]
    selected["species"] = "mouse"
    selected["tissue"] = selected["encode_tissue"].map(ENCODE_LABELS)
    selected["source_tissue"] = selected["encode_tissue"]
    selected["study_accession"] = selected["gse"].fillna(selected["experiment_accession"])
    selected["quant_file"] = selected["gene_quantification_file"].map(
        lambda accession: str(resolve(cfg["download_directory"]) / f"{accession}.tsv")
    )
    selected["annotation"] = "ENCODE-provided RSEM effective length"
    selected["pretraining_exposure"] = selected["exposure_class"]
    missing = selected.loc[~selected.quant_file.map(lambda x: Path(x).is_file()), "quant_file"].tolist()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} eligible ENCODE quantification files: {missing[:5]}")
    if selected.sample_id.duplicated().any():
        raise ValueError("Eligible ENCODE experiment IDs are not unique")
    return selected.sort_values(["tissue", "sample_id"]).reset_index(drop=True)


def build_human_manifest(cfg: dict, tissues: set[str]) -> pd.DataFrame:
    attrs = pd.read_csv(resolve(cfg["gtex_sample_attributes"]), sep="\t", low_memory=False)
    attrs = attrs[attrs["SMTSD"].isin(GTEX_LABELS)].copy()
    attrs["tissue"] = attrs["SMTSD"].map(GTEX_LABELS)
    attrs = attrs[attrs.tissue.isin(tissues)].copy()
    attrs["subject_id"] = attrs["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    phen = pd.read_csv(resolve(cfg["gtex_subject_phenotypes"]), sep="\t")
    phen["sex"] = phen["SEX"].map({1: "male", 2: "female", "1": "male", "2": "female"}).fillna(phen["SEX"].astype(str))
    attrs = attrs.merge(phen[["SUBJID", "sex", "AGE"]], left_on="subject_id", right_on="SUBJID", how="left")
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(resolve(cfg["gtex_counts"])).schema.names)
    attrs = attrs[attrs["SAMPID"].isin(available)].copy()
    return pd.DataFrame({
        "sample_id": attrs["SAMPID"], "species": "human", "tissue": attrs["tissue"],
        "source_tissue": attrs["SMTSD"], "sex": attrs["sex"], "age": attrs["AGE"],
        "strain": pd.NA, "study_accession": "GTEx", "experiment_accession": attrs["SAMPID"],
        "biosample_accession": attrs["SAMPID"], "quant_file": str(resolve(cfg["gtex_counts"])),
        "annotation": "GENCODE v49", "pretraining_exposure": "not_applicable",
    }).sort_values(["tissue", "sample_id"]).reset_index(drop=True)


def source_availability(mouse: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for encode_tissue, gtex_tissue in DIRECT_GTEX_MATCHES.items():
        label = ENCODE_LABELS[encode_tissue]
        m = mouse[mouse.source_tissue.eq(encode_tissue)]
        h = human[human.tissue.eq(label)]
        counts = m.pretraining_exposure.value_counts()
        rows.append({
            "encode_tissue": encode_tissue, "gtex_tissue": gtex_tissue, "analysis_tissue": label,
            "eligible_expression_ready_mouse_experiments": len(m), "available_gtex_profiles": len(h),
            "exact_sample_seen": int(counts.get("exact_sample_seen", 0)),
            "same_study_seen": int(counts.get("same_study_seen", 0)),
            "fully_unseen": int(counts.get("fully_unseen", 0)),
            "unresolved": int(counts.get("unresolved", 0)),
            "included_expanded_replicated": len(m) >= 2,
            "included_fixed_11": label in FIXED_11,
            "included_fixed_5": label in FIXED_5,
        })
    return pd.DataFrame(rows).sort_values(
        ["eligible_expression_ready_mouse_experiments", "analysis_tissue"], ascending=[False, True]
    ).reset_index(drop=True)


def all_tissues_before_filtering() -> pd.DataFrame:
    """Report the expanded ENCODE pool before anatomical/readiness filtering."""
    audit = pd.read_parquet(BASE_RESULTS / "expanded_encode_cohort_audit.parquet")
    rows = []
    for tissue, group in audit.groupby("encode_tissue", dropna=False):
        healthy = group[group.healthy_non_transgenic]
        ready = healthy[healthy.gene_quantification_available]
        direct = bool(group.direct_gtex_match.any())
        rows.append({
            "encode_tissue": tissue,
            "experiments_in_expanded_pool": group.experiment_accession.nunique(),
            "healthy_adult_unperturbed_experiments": healthy.experiment_accession.nunique(),
            "healthy_expression_ready_experiments": int(ready.experiment_accession.nunique()),
            "direct_gtex_match": direct,
            "proposed_gtex_tissue": DIRECT_GTEX_MATCHES.get(tissue),
            "analysis_tissue": ENCODE_LABELS.get(tissue),
            "eligible_for_matched_benchmark": direct and tissue in ENCODE_LABELS,
        })
    return pd.DataFrame(rows).sort_values(
        ["eligible_for_matched_benchmark", "healthy_expression_ready_experiments", "encode_tissue"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def cached_or_compute_expression(
    cfg: dict, genes: list[str], human_meta: pd.DataFrame, mouse_meta: pd.DataFrame, force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hp, mp, manifest_path = CACHE / "gtex_log1p_tpm.npy", CACHE / "encode_log1p_tpm.npy", CACHE / "representation_manifest.parquet"
    if not force and hp.exists() and mp.exists() and manifest_path.exists():
        old = pd.read_parquet(manifest_path)
        expected = pd.concat([human_meta[["sample_id", "species"]], mouse_meta[["sample_id", "species"]]], ignore_index=True)
        if old[["sample_id", "species"]].equals(expected):
            log("Reusing cached expanded expression matrices")
            return np.load(hp), np.load(mp)

    # Reuse rows from the strict cache; calculate only newly added profiles.
    legacy_manifest_path = BASE_RESULTS / "expanded_task1a_sample_manifest.parquet"
    legacy_h = HERE / "work" / "expanded_gtex_log1p_tpm.npy"
    legacy_m = HERE / "work" / "expanded_encode_log1p_tpm.npy"
    human = np.zeros((len(human_meta), len(genes)), dtype=np.float32)
    mouse = np.zeros((len(mouse_meta), len(genes)), dtype=np.float32)
    reused_h = reused_m = 0
    if legacy_manifest_path.exists() and legacy_h.exists() and legacy_m.exists():
        legacy = pd.read_parquet(legacy_manifest_path)
        lhmeta = legacy[legacy.species.eq("human")].reset_index(drop=True)
        lmmeta = legacy[legacy.species.eq("mouse")].reset_index(drop=True)
        lh, lm = np.load(legacy_h), np.load(legacy_m)
        hlookup = {x: i for i, x in enumerate(lhmeta.sample_id)}
        mlookup = {x: i for i, x in enumerate(lmmeta.sample_id)}
        for i, sample in enumerate(human_meta.sample_id):
            if sample in hlookup: human[i] = lh[hlookup[sample]]; reused_h += 1
        for i, sample in enumerate(mouse_meta.sample_id):
            if sample in mlookup: mouse[i] = lm[mlookup[sample]]; reused_m += 1
    hmissing = ~human_meta.sample_id.isin(set(human_meta.sample_id.iloc[:0]) if reused_h == 0 else set())
    # Use explicit legacy membership; zero expression is a valid row and cannot identify cache misses.
    legacy_h_ids = set()
    legacy_m_ids = set()
    if legacy_manifest_path.exists():
        legacy = pd.read_parquet(legacy_manifest_path)
        legacy_h_ids = set(legacy.loc[legacy.species.eq("human"), "sample_id"])
        legacy_m_ids = set(legacy.loc[legacy.species.eq("mouse"), "sample_id"])
    hmissing = ~human_meta.sample_id.isin(legacy_h_ids)
    mmissing = ~mouse_meta.sample_id.isin(legacy_m_ids)
    log(f"Expression cache reuse: human={reused_h:,}, mouse={reused_m:,}; compute human={hmissing.sum():,}, mouse={mmissing.sum():,}")
    if hmissing.any():
        values, hmap = load_human_tpm(human_meta[hmissing].reset_index(drop=True), cfg, genes)
        human[hmissing.to_numpy()] = values
        hmap.to_parquet(OUT / "human_gene_mapping.parquet", index=False)
    if mmissing.any():
        values, mmap = load_mouse_tpm(mouse_meta[mmissing].reset_index(drop=True), cfg, genes)
        mouse[mmissing.to_numpy()] = values
        mmap.to_parquet(OUT / "mouse_gene_mapping.parquet", index=False)
    np.save(hp, human); np.save(mp, mouse)
    pd.concat([human_meta[["sample_id", "species"]], mouse_meta[["sample_id", "species"]]], ignore_index=True).to_parquet(manifest_path, index=False)
    return human, mouse


def cached_representations(
    cfg: dict, genes: list[str], human_meta: pd.DataFrame, mouse_meta: pd.DataFrame,
    human: np.ndarray, mouse: np.ndarray, args: argparse.Namespace,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    nh = len(human)
    combined = np.vstack([human, mouse])
    pca_path = CACHE / "joint_pca_scores.npy"
    if pca_path.exists() and not args.force:
        pca_scores = np.load(pca_path)
        if len(pca_scores) != len(combined):
            raise ValueError("PCA cache length mismatch; rerun with --force")
    else:
        n_components = min(int(cfg["pca_components"]), len(combined) - 1, combined.shape[1])
        log(f"Fitting one joint PCA ({n_components} components) on the expanded representation cohort")
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.base_seed)
        pca_scores = pca.fit_transform(combined).astype(np.float32)
        np.save(pca_path, pca_scores)
        np.save(CACHE / "joint_pca_components.npy", pca.components_.astype(np.float32))
        np.save(CACHE / "joint_pca_mean.npy", pca.mean_.astype(np.float32))
        pd.DataFrame({"pc": np.arange(1, n_components + 1), "variance_explained": pca.explained_variance_ratio_,
                      "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(OUT / "pca_variance.csv", index=False)

    emb_path = CACHE / "bridgerna_embeddings.npy"
    if emb_path.exists() and not args.force:
        embeddings = np.load(emb_path)
        if len(embeddings) != len(combined):
            raise ValueError("BridgeRNA cache length mismatch; rerun with --force")
        log("Reusing cached expanded BridgeRNA embeddings")
    else:
        embeddings = np.zeros((len(combined), 512), dtype=np.float32)
        missing = np.ones(len(combined), dtype=bool)
        legacy_manifest_path = BASE_RESULTS / "expanded_task1a_sample_manifest.parquet"
        legacy_emb_path = HERE / "work" / "expanded_bridgerna_embeddings.npy"
        if legacy_manifest_path.exists() and legacy_emb_path.exists():
            legacy_meta = pd.read_parquet(legacy_manifest_path)
            legacy_emb = np.load(legacy_emb_path)
            lookup = {(r.species, r.sample_id): i for i, r in legacy_meta.reset_index(drop=True).iterrows()}
            current = pd.concat([human_meta, mouse_meta], ignore_index=True)
            for i, row in current.iterrows():
                old_i = lookup.get((row.species, row.sample_id))
                if old_i is not None:
                    embeddings[i] = legacy_emb[old_i]; missing[i] = False
        log(f"BridgeRNA embedding cache: reused={(~missing).sum():,}; infer={missing.sum():,}")
        if missing.any():
            model, device = load_expression_performer(
                resolve(cfg["checkpoint"]), resolve(cfg["model_config"]), len(genes), args.device,
            )
            embeddings[missing] = encode_frozen(model, device, combined[missing], args.batch_size)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        np.save(emb_path, embeddings)
    return {
        "raw_expression": (human, mouse),
        "joint_pca": (pca_scores[:nh], pca_scores[nh:]),
        "bridgerna": (embeddings[:nh], embeddings[nh:]),
    }


def centroids(x: np.ndarray, labels: np.ndarray, tissues: list[str]) -> np.ndarray:
    return np.stack([x[labels == tissue].mean(axis=0) for tissue in tissues])


def readout_predictions(
    readout: str, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
    test_y: np.ndarray, tissues: list[str], device: torch.device,
) -> tuple[np.ndarray, float | None]:
    if readout == "centroid_cosine":
        scores = cosine(test_x, centroids(train_x, train_y, tissues))
        pred = np.asarray(tissues)[scores.argmax(axis=1)]
        true_i = np.array([tissues.index(x) for x in test_y])
        order = np.argsort(-scores, axis=1)
        ranks = np.array([np.flatnonzero(order[i] == true_i[i])[0] + 1 for i in range(len(test_y))])
        return pred, float(np.mean(1 / ranks))
    if readout == "knn_cosine_k1":
        nearest = cosine(test_x, train_x).argmax(axis=1)
        return train_y[nearest], None
    if readout == "linear_softmax_probe":
        result, pred, _ = probe_readout(train_x, train_y, test_x, test_y, tissues, False, device)
        return pred, result.get("mrr")
    raise ValueError(readout)


def evaluate_one(
    cohort: str, balance: str, seed: int, representation: str, readout: str,
    direction: str, hx: np.ndarray, hy: np.ndarray, mx: np.ndarray, my: np.ndarray,
    tissues: list[str], device: torch.device,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if direction == "human_to_mouse":
        train_x, train_y, test_x, test_y = hx, hy, mx, my
    else:
        train_x, train_y, test_x, test_y = mx, my, hx, hy
    pred, mrr = readout_predictions(readout, train_x, train_y, test_x, test_y, tissues, device)
    accuracy = float(np.mean(pred == test_y))
    macro_f1 = float(f1_score(test_y, pred, labels=tissues, average="macro", zero_division=0))
    precision, recall, tissue_f1, support = precision_recall_fscore_support(
        test_y, pred, labels=tissues, zero_division=0,
    )
    base = {"cohort": cohort, "balance": balance, "seed": seed, "representation": representation,
            "readout": readout, "direction": direction}
    result = {**base, "train_samples": len(train_y), "test_samples": len(test_y),
              "accuracy": accuracy, "macro_f1": macro_f1, "mrr": mrr}
    per = pd.DataFrame({**{k: [v] * len(tissues) for k, v in base.items()}, "tissue": tissues,
                        "samples": support, "accuracy": recall, "precision": precision, "f1": tissue_f1})
    cm = confusion_matrix(test_y, pred, labels=tissues)
    cm_long = pd.DataFrame(cm, index=tissues, columns=tissues).rename_axis("true_tissue").reset_index().melt(
        id_vars="true_tissue", var_name="predicted_tissue", value_name="count"
    )
    for key, value in reversed(list(base.items())):
        cm_long.insert(0, key, value)
    return result, per, cm_long


def balanced_indices(meta: pd.DataFrame, mouse_meta: pd.DataFrame, tissues: list[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for tissue in tissues:
        candidates = np.flatnonzero(meta.tissue.to_numpy() == tissue)
        n = int(mouse_meta.tissue.eq(tissue).sum())
        if len(candidates) < n:
            raise ValueError(f"GTEx {tissue} has {len(candidates)} profiles but needs {n}")
        selected.extend(rng.choice(candidates, n, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "balance", "representation", "readout", "direction"]
    return per_seed.groupby(keys, as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        mrr_mean=("mrr", "mean"), mrr_sd=("mrr", "std"), seeds=("seed", "nunique"),
        train_samples=("train_samples", "first"), test_samples=("test_samples", "first"),
    )


def summarize_per_tissue(per: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "balance", "representation", "readout", "direction", "tissue"]
    return per.groupby(keys, as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"), replicates=("seed", "nunique"),
        samples_per_seed=("samples", "first"),
    )


def plot_summary(summary: pd.DataFrame, cohort: str, tissues: list[str]) -> None:
    method_order = [(readout, rep) for readout in READOUTS for rep in REPRESENTATIONS]
    labels = [
        f"{readout.replace('centroid_cosine', 'Centroid cosine').replace('knn_cosine_k1', '1-NN').replace('linear_softmax_probe', 'Linear probe')} · "
        f"{rep.replace('raw_expression', 'Raw').replace('joint_pca', 'PCA').replace('bridgerna', 'BridgeRNA')}"
        for readout, rep in method_order
    ]
    data = summary[(summary.cohort == cohort) & (summary.balance == "balanced")]
    y = np.arange(len(labels), dtype=float)
    y[3:] += .45; y[6:] += .45
    height = .33
    colors = {"human_to_mouse": "#2878B5", "mouse_to_human": "#E07A1F"}
    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    for offset, direction, direction_label in [(-height / 2, "human_to_mouse", "Human→Mouse"),
                                                (height / 2, "mouse_to_human", "Mouse→Human")]:
        rows = []
        for readout, rep in method_order:
            match = data[(data.readout == readout) & (data.representation == rep) & (data.direction == direction)]
            if len(match) != 1: raise ValueError(f"Missing plot result: {cohort}/{readout}/{rep}/{direction}")
            rows.append(match.iloc[0])
        means = np.array([r.accuracy_mean for r in rows]) * 100
        errors = np.array([r.accuracy_sd for r in rows]) * 100
        bars = ax.barh(y + offset, means, height, xerr=errors, capsize=3, color=colors[direction], label=direction_label)
        ax.bar_label(bars, labels=[f"{x:.1f}%" for x in means], padding=5, fontsize=8)
    chance = 100 / len(tissues)
    ax.axvline(chance, color="#333333", ls="--", lw=1.2, label=f"Chance ({chance:.1f}%)")
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("Balanced cross-species tissue accuracy (mean ± SD, %)")
    ax.set_title(f"{cohort.replace('_', ' ').title()} ({len(tissues)} tissues)", weight="bold", loc="left")
    ax.grid(axis="x", color="#D9D9D9", lw=.7); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(FIGURES / f"balanced_methods_{cohort}.{suffix}", dpi=400 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_confusion_figures(confusions: pd.DataFrame, cohorts: dict[str, list[str]]) -> None:
    balanced = confusions[confusions.balance.eq("balanced")]
    for cohort, tissues in cohorts.items():
        for readout in READOUTS:
            fig, axes = plt.subplots(3, 2, figsize=(13, 17))
            for row, rep in enumerate(REPRESENTATIONS):
                for col, direction in enumerate(DIRECTIONS):
                    part = balanced[(balanced.cohort == cohort) & (balanced.readout == readout)
                                    & (balanced.representation == rep) & (balanced.direction == direction)]
                    matrix = part.groupby(["true_tissue", "predicted_tissue"])["count"].sum().unstack(fill_value=0).reindex(
                        index=tissues, columns=tissues, fill_value=0
                    ).to_numpy(float)
                    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
                    ax = axes[row, col]; image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                    ax.set_xticks(range(len(tissues)), tissues, rotation=45, ha="right", fontsize=7)
                    ax.set_yticks(range(len(tissues)), tissues, fontsize=7)
                    ax.set(title=f"{rep} · {direction}", xlabel="Predicted", ylabel="True")
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.55, label="Row-normalized proportion")
            fig.suptitle(f"{cohort} · {readout} · 20 balanced seeds", y=.995)
            fig.subplots_adjust(hspace=.42, wspace=.32, top=.96)
            fig.savefig(FIGURES / f"confusions_{cohort}_{readout}.png", dpi=250, bbox_inches="tight")
            plt.close(fig)


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    cfg = json.loads((HERE / "config.json").read_text())
    genes = load_canonical_genes(resolve(cfg["canonical_genes"]))
    if len(genes) != 15165: raise ValueError(f"Expected 15,165 genes, found {len(genes)}")
    mouse_meta = build_mouse_manifest(cfg)
    human_meta = build_human_manifest(cfg, set(mouse_meta.tissue))
    all_tissues_before_filtering().to_csv(OUT / "all_encode_tissues_before_filtering.csv", index=False)
    availability = source_availability(mouse_meta, human_meta)
    availability.to_csv(OUT / "all_available_tissues_before_filtering.csv", index=False)
    availability[["encode_tissue", "gtex_tissue", "analysis_tissue", "included_expanded_replicated",
                  "included_fixed_11", "included_fixed_5"]].to_csv(OUT / "tissue_mapping.csv", index=False)
    mouse_meta.to_parquet(OUT / "encode_mouse_sample_manifest.parquet", index=False)
    human_meta.to_parquet(OUT / "gtex_human_sample_manifest.parquet", index=False)
    pd.concat([human_meta, mouse_meta], ignore_index=True, sort=False).to_parquet(OUT / "sample_manifest.parquet", index=False)
    exposure = mouse_meta.groupby(["tissue", "pretraining_exposure"], as_index=False).size()
    exposure.to_csv(OUT / "mouse_pretraining_exposure_summary.csv", index=False)

    expanded = sorted(availability.loc[availability.included_expanded_replicated, "analysis_tissue"].unique())
    cohorts = {"expanded_replicated": expanded, "fixed_11": FIXED_11, "fixed_5": FIXED_5}
    for name, tissues in cohorts.items():
        absent = set(tissues) - set(mouse_meta.tissue)
        if absent: raise ValueError(f"{name} lacks mouse tissues: {sorted(absent)}")
    cohort_rows = []
    for name, tissues in cohorts.items():
        for tissue in tissues:
            cohort_rows.append({"cohort": name, "tissue": tissue,
                                "mouse_experiments": int(mouse_meta.tissue.eq(tissue).sum()),
                                "gtex_profiles_available": int(human_meta.tissue.eq(tissue).sum())})
    pd.DataFrame(cohort_rows).to_csv(OUT / "cohort_tissue_counts.csv", index=False)
    log("Expanded cohort tissues\n" + pd.DataFrame(cohort_rows).query("cohort == 'expanded_replicated'").to_string(index=False))

    human, mouse = cached_or_compute_expression(cfg, genes, human_meta, mouse_meta, args.force)
    reps = cached_representations(cfg, genes, human_meta, mouse_meta, human, mouse, args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    result_rows, tissue_rows, confusion_rows, selection_rows = [], [], [], []
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    for cohort, tissues in cohorts.items():
        hm = human_meta.tissue.isin(tissues).to_numpy(); mm = mouse_meta.tissue.isin(tissues).to_numpy()
        hmeta = human_meta[hm].reset_index(drop=True); mmeta = mouse_meta[mm].reset_index(drop=True)
        for seed in seeds:
            chosen = balanced_indices(hmeta, mmeta, tissues, seed)
            for i in chosen:
                selection_rows.append({"cohort": cohort, "seed": seed, "sample_id": hmeta.iloc[i].sample_id,
                                       "tissue": hmeta.iloc[i].tissue})
            log(f"Balanced evaluation {cohort}: seed={seed} ({seeds.index(seed)+1}/{len(seeds)})")
            for rep, (hall, mall) in reps.items():
                hx, mx = hall[hm][chosen], mall[mm]
                hy, my = hmeta.tissue.to_numpy()[chosen], mmeta.tissue.to_numpy()
                for readout in READOUTS:
                    for direction in DIRECTIONS:
                        result, per, cm = evaluate_one(cohort, "balanced", seed, rep, readout, direction,
                                                       hx, hy, mx, my, tissues, device)
                        result_rows.append(result); tissue_rows.append(per); confusion_rows.append(cm)

        log(f"Unbalanced full-GTEx reference: {cohort}")
        for rep, (hall, mall) in reps.items():
            hx, mx = hall[hm], mall[mm]
            hy, my = hmeta.tissue.to_numpy(), mmeta.tissue.to_numpy()
            for readout in READOUTS:
                for direction in DIRECTIONS:
                    result, per, cm = evaluate_one(cohort, "full_gtex_reference", args.base_seed, rep, readout,
                                                   direction, hx, hy, mx, my, tissues, device)
                    result_rows.append(result); tissue_rows.append(per); confusion_rows.append(cm)

    per_seed = pd.DataFrame(result_rows)
    per_tissue = pd.concat(tissue_rows, ignore_index=True)
    confusions = pd.concat(confusion_rows, ignore_index=True)
    summary = summarize(per_seed)
    tissue_summary = summarize_per_tissue(per_tissue)
    per_seed.to_csv(OUT / "per_seed_results.csv", index=False)
    summary.to_csv(OUT / "summary_results.csv", index=False)
    per_tissue.to_csv(OUT / "per_seed_per_tissue_results.csv", index=False)
    tissue_summary.to_csv(OUT / "per_tissue_summary.csv", index=False)
    confusions.to_parquet(OUT / "confusion_matrices_per_seed.parquet", index=False)
    pd.DataFrame(selection_rows).to_parquet(OUT / "balanced_gtex_selections.parquet", index=False)
    for cohort, tissues in cohorts.items(): plot_summary(summary, cohort, tissues)
    save_confusion_figures(confusions, cohorts)
    provenance = {
        "analysis": "Task 1A balanced representation geometry", "genes": len(genes),
        "seeds": seeds, "cohorts": cohorts, "representations": REPRESENTATIONS, "readouts": READOUTS,
        "mouse_inclusion": "all healthy adult unperturbed direct-match expression-ready experiments; exposure retained",
        "balanced_sampling": "per tissue GTEx n equals eligible mouse experiment n; without replacement",
        "pca": "fit once jointly on all cached eligible human and mouse profiles; no per-seed refitting",
        "bridgerna": {"checkpoint": cfg["checkpoint"], "frozen": True, "input": "natural log1p(TPM)"},
        "linear_probe": "source-species-only linear softmax; fixed 50 epochs; balanced loss; no feature scaling",
        "forbidden_transformations": "no alignment, fine-tuning, target supervision, batch correction, or species normalization",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    log("Balanced Task 1A complete\n" + summary[summary.balance.eq("balanced")].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Recompute cached representations")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
