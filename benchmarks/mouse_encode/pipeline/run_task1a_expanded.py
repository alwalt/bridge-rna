#!/usr/bin/env python3
"""Run corrected expanded-cohort Task 1A tissue correspondence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from run_task1a_retrieval import (
    encode_frozen, load_human_tpm, load_mouse_tpm, resolve,
)
from src.fm_embed.model import load_expression_performer
from src.fm_embed.vocab import load_canonical_genes

HERE = Path(__file__).resolve().parents[1]
RESULTS, WORK = HERE / "results", HERE / "work"
FIGURES = RESULTS / "figures" / "expanded_task1a"

TISSUES_11 = [
    "adrenal", "subcutaneous adipose", "cerebellum", "heart", "liver", "lung",
    "mammary gland", "ovary", "spleen", "stomach", "testis",
]
TISSUES_5 = ["heart", "liver", "lung", "spleen", "testis"]
ENCODE_TO_LABEL = {
    "adrenal gland": "adrenal",
    "subcutaneous adipose tissue": "subcutaneous adipose",
    "cerebellum": "cerebellum",
    "heart": "heart",
    "liver": "liver",
    "lung": "lung",
    "mammary gland": "mammary gland",
    "ovary": "ovary",
    "spleen": "spleen",
    "stomach": "stomach",
    "testis": "testis",
}
GTEX_TO_LABEL = {
    "Adrenal Gland": "adrenal",
    "Adipose - Subcutaneous": "subcutaneous adipose",
    "Brain - Cerebellum": "cerebellum",
    "Heart - Atrial Appendage": "heart",
    "Heart - Left Ventricle": "heart",
    "Liver": "liver",
    "Lung": "lung",
    "Breast - Mammary Tissue": "mammary gland",
    "Ovary": "ovary",
    "Spleen": "spleen",
    "Stomach": "stomach",
    "Testis": "testis",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_mouse_manifest(cfg: dict) -> pd.DataFrame:
    selected = pd.read_parquet(RESULTS / "proposed_all_fully_unseen_profiles.parquet")
    selected = selected[
        selected["gene_quantification_available"]
        & selected["exposure_class"].eq("fully_unseen")
        & selected["encode_tissue"].isin(ENCODE_TO_LABEL)
    ].copy()
    selected["sample_id"] = selected["experiment_accession"]
    selected["species"] = "mouse"
    selected["tissue"] = selected["encode_tissue"].map(ENCODE_TO_LABEL)
    selected["source_tissue"] = selected["encode_tissue"]
    selected["quant_file"] = selected["gene_quantification_file"].map(
        lambda x: str(resolve(cfg["download_directory"]) / f"{x}.tsv")
    )
    if len(selected) != 17 or set(selected["tissue"]) != set(TISSUES_11):
        raise ValueError(f"Expected 17 fully unseen profiles across 11 tissues; got {len(selected)} and {sorted(selected.tissue.unique())}")
    if not selected["quant_file"].map(lambda x: Path(x).is_file()).all():
        missing = selected.loc[~selected["quant_file"].map(lambda x: Path(x).is_file()), "quant_file"].tolist()
        raise FileNotFoundError(f"Missing selected ENCODE quantifications: {missing}")
    return selected.reset_index(drop=True)


def build_human_manifest(cfg: dict) -> pd.DataFrame:
    attrs = pd.read_csv(resolve(cfg["gtex_sample_attributes"]), sep="\t", low_memory=False)
    attrs = attrs[attrs["SMTSD"].isin(GTEX_TO_LABEL)].copy()
    attrs["tissue"] = attrs["SMTSD"].map(GTEX_TO_LABEL)
    attrs["subject_id"] = attrs["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    phen = pd.read_csv(resolve(cfg["gtex_subject_phenotypes"]), sep="\t")
    phen["sex"] = phen["SEX"].map({1: "male", 2: "female", "1": "male", "2": "female"}).fillna(phen["SEX"].astype(str))
    attrs = attrs.merge(phen[["SUBJID", "sex", "AGE"]], left_on="subject_id", right_on="SUBJID", how="left")
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(resolve(cfg["gtex_counts"])).schema.names)
    attrs = attrs[attrs["SAMPID"].isin(available)].copy()
    out = pd.DataFrame({
        "sample_id": attrs["SAMPID"], "species": "human", "tissue": attrs["tissue"],
        "source_tissue": attrs["SMTSD"], "sex": attrs["sex"], "age": attrs["AGE"],
        "strain": pd.NA, "study_accession": "GTEx", "experiment_accession": attrs["SAMPID"],
        "biosample_accession": attrs["SAMPID"], "quant_file": str(resolve(cfg["gtex_counts"])),
        "annotation": "GENCODE v49",
    })
    if set(out.tissue) != set(TISSUES_11):
        raise ValueError(f"GTEx is missing labels: {set(TISSUES_11) - set(out.tissue)}")
    return out.reset_index(drop=True)


def centroids(x: np.ndarray, labels: np.ndarray, tissues: list[str]) -> np.ndarray:
    return np.stack([x[labels == tissue].mean(axis=0) for tissue in tissues])


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def retrieve(
    query_x: np.ndarray, query_meta: pd.DataFrame, reference_x: np.ndarray,
    reference_meta: pd.DataFrame, tissues: list[str], representation: str,
    direction: str, cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    reference_centroids = centroids(reference_x, reference_meta.tissue.to_numpy(), tissues)
    scores = cosine(query_x, reference_centroids)
    true_index = np.array([tissues.index(x) for x in query_meta.tissue])
    order = np.argsort(-scores, axis=1)
    ranks = np.array([np.flatnonzero(order[i] == true_index[i])[0] + 1 for i in range(len(order))])
    winner = order[:, 0]
    predictions = pd.DataFrame({
        "cohort": cohort, "representation": representation, "direction": direction,
        "sample_id": query_meta.sample_id.to_numpy(), "true_tissue": query_meta.tissue.to_numpy(),
        "nearest_tissue": [tissues[i] for i in winner], "anatomical_match_rank": ranks,
        "top1_correct": winner == true_index, "reciprocal_rank": 1 / ranks,
        "nearest_cosine": scores[np.arange(len(scores)), winner],
        "anatomical_match_cosine": scores[np.arange(len(scores)), true_index],
    })
    per_tissue = predictions.groupby("true_tissue", observed=True).agg(
        samples=("sample_id", "size"), top1=("top1_correct", "mean"),
        mrr=("reciprocal_rank", "mean"), median_anatomical_rank=("anatomical_match_rank", "median"),
    ).reindex(tissues).reset_index()
    rank_positions = np.empty_like(order)
    rank_positions[np.arange(len(order))[:, None], order] = np.arange(1, len(tissues) + 1)
    full_rankings = pd.DataFrame({
        "cohort": np.repeat(cohort, len(query_meta) * len(tissues)),
        "representation": np.repeat(representation, len(query_meta) * len(tissues)),
        "direction": np.repeat(direction, len(query_meta) * len(tissues)),
        "sample_id": np.repeat(query_meta.sample_id.to_numpy(), len(tissues)),
        "true_tissue": np.repeat(query_meta.tissue.to_numpy(), len(tissues)),
        "candidate_tissue": np.tile(tissues, len(query_meta)),
        "rank": rank_positions.reshape(-1), "cosine_similarity": scores.reshape(-1),
    })
    full_rankings["is_anatomical_match"] = full_rankings["true_tissue"].eq(full_rankings["candidate_tissue"])
    return predictions, per_tissue, full_rankings, scores


def partial_eta_squared(values: np.ndarray, species: np.ndarray, tissue: np.ndarray) -> tuple[float, float]:
    """Two-factor partial eta-squared for species and tissue."""
    species_design = pd.get_dummies(species, drop_first=True, dtype=float).to_numpy()
    tissue_design = pd.get_dummies(tissue, drop_first=True, dtype=float).to_numpy()
    intercept = np.ones((len(values), 1))
    def sse(parts):
        design = np.column_stack([intercept, *parts])
        fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
        return float(np.sum((values - fitted) ** 2))
    full = sse([species_design, tissue_design])
    no_species, no_tissue = sse([tissue_design]), sse([species_design])
    return (no_species - full) / max(no_species, 1e-12), (no_tissue - full) / max(no_tissue, 1e-12)


def plot_matrix(values: np.ndarray, labels: list[str], title: str, path: Path, cmap: str, fmt: str = ".2f") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] > (np.nanmin(values) + np.nanmax(values)) / 2 else "black")
    ax.set(title=title, xlabel="Candidate/target tissue", ylabel="Query/source tissue")
    fig.colorbar(image, ax=ax, shrink=.7); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_samples(x: np.ndarray, manifest: pd.DataFrame, representation: str, path: Path, seed: int) -> None:
    xy = PCA(n_components=2, random_state=seed).fit_transform(x)
    palette = dict(zip(TISSUES_11, plt.get_cmap("tab20").colors[:len(TISSUES_11)]))
    fig, ax = plt.subplots(figsize=(11, 8))
    for species, marker in [("human", "o"), ("mouse", "^")]:
        for tissue in TISSUES_11:
            mask = (manifest.species.to_numpy() == species) & (manifest.tissue.to_numpy() == tissue)
            ax.scatter(xy[mask, 0], xy[mask, 1], c=[palette[tissue]], marker=marker,
                       s=12 if species == "human" else 65, alpha=.45 if species == "human" else .95,
                       edgecolors="none", label=f"{tissue} · {species}")
            if mask.any():
                center = xy[mask].mean(axis=0)
                ax.scatter(*center, c=[palette[tissue]], marker=marker, s=180, edgecolors="black")
    ax.set(title=f"{representation} (visualization only)", xlabel="PCA 1", ylabel="PCA 2")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_pca_diagnostics(scores: np.ndarray, pca_results: pd.DataFrame, manifest: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(pca_results.pc, pca_results.variance_explained, color="#4472c4")
    axes[0].plot(pca_results.pc, pca_results.cumulative_variance, color="#c44e52", marker="o")
    axes[0].set(xlabel="PC", ylabel="Variance fraction", title="PC1–PC20 variance")
    axes[1].plot(pca_results.pc, pca_results.species_partial_eta2, marker="o", label="Species")
    axes[1].plot(pca_results.pc, pca_results.tissue_partial_eta2, marker="o", label="Tissue")
    axes[1].set(xlabel="PC", ylabel="Partial η²", title="PC association controlling for the other factor")
    axes[1].legend(); fig.tight_layout(); fig.savefig(FIGURES / "pca_pc1_20_diagnostics.png", dpi=220); plt.close(fig)

    later = pca_results[pca_results.pc.between(5, 20)].nlargest(2, "tissue_partial_eta2").pc.astype(int).tolist()
    pairs = [(1, 2), tuple(later)]
    palette = dict(zip(TISSUES_11, plt.get_cmap("tab20").colors[:len(TISSUES_11)]))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (left, right) in zip(axes, pairs):
        for species, marker in [("human", "o"), ("mouse", "^")]:
            for tissue in TISSUES_11:
                mask = (manifest.species.to_numpy() == species) & (manifest.tissue.to_numpy() == tissue)
                ax.scatter(scores[mask, left - 1], scores[mask, right - 1], c=[palette[tissue]], marker=marker,
                           s=12 if species == "human" else 60, alpha=.45 if species == "human" else .95,
                           edgecolors="none")
        ax.set(xlabel=f"PC{left}", ylabel=f"PC{right}", title="Early PCs" if left == 1 else "Top two PC5–PC20 by tissue partial η²")
    fig.tight_layout(); fig.savefig(FIGURES / "pca_early_and_tissue_structured_pcs.png", dpi=220); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cfg = json.loads((HERE / "config.json").read_text())
    RESULTS.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    genes = load_canonical_genes(resolve(cfg["canonical_genes"]))
    if len(genes) != 15165: raise ValueError(f"Expected 15,165 genes, got {len(genes)}")
    human_meta, mouse_meta = build_human_manifest(cfg), build_mouse_manifest(cfg)
    manifest = pd.concat([human_meta, mouse_meta], ignore_index=True, sort=False)
    manifest.to_parquet(RESULTS / "expanded_task1a_sample_manifest.parquet", index=False)
    log("Cohort\n" + manifest.groupby(["species", "tissue"]).size().to_string())

    h_path, m_path = WORK / "expanded_gtex_log1p_tpm.npy", WORK / "expanded_encode_log1p_tpm.npy"
    if args.reuse_prepared and h_path.exists() and m_path.exists():
        human, mouse = np.load(h_path), np.load(m_path)
    else:
        human, hm = load_human_tpm(human_meta, cfg, genes)
        mouse, mm = load_mouse_tpm(mouse_meta, cfg, genes)
        np.save(h_path, human); np.save(m_path, mouse)
        hm.merge(mm, on=["gene_order", "gene_symbol"]).to_parquet(RESULTS / "expanded_task1a_gene_mapping.parquet", index=False)
    combined = np.vstack([human, mouse])
    n_components = min(int(cfg["pca_components"]), len(combined) - 1, len(genes))
    log(f"Fitting joint PCA with {n_components} components")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(cfg["seed"]))
    pca_scores = pca.fit_transform(combined).astype(np.float32)
    np.save(WORK / "expanded_joint_pca_scores.npy", pca_scores)
    np.save(WORK / "expanded_joint_pca_components.npy", pca.components_.astype(np.float32))
    np.save(WORK / "expanded_joint_pca_mean.npy", pca.mean_.astype(np.float32))
    pd.DataFrame({"pc": np.arange(1, n_components + 1), "variance_explained": pca.explained_variance_ratio_,
                  "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(RESULTS / "expanded_task1a_pca_variance_all.csv", index=False)

    emb_path = WORK / "expanded_bridgerna_embeddings.npy"
    if args.reuse_prepared and emb_path.exists(): embeddings = np.load(emb_path)
    else:
        model, device = load_expression_performer(resolve(cfg["checkpoint"]), resolve(cfg["model_config"]), len(genes), args.device)
        embeddings = encode_frozen(model, device, combined, args.batch_size)
        np.save(emb_path, embeddings); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    pc_rows = []
    for i in range(20):
        species_eta, tissue_eta = partial_eta_squared(pca_scores[:, i], manifest.species.to_numpy(), manifest.tissue.to_numpy())
        pc_rows.append({"pc": i + 1, "variance_explained": pca.explained_variance_ratio_[i],
                        "cumulative_variance": pca.explained_variance_ratio_[:i + 1].sum(),
                        "species_partial_eta2": species_eta, "tissue_partial_eta2": tissue_eta})
    pc_results = pd.DataFrame(pc_rows)
    pc_results.to_csv(RESULTS / "expanded_task1a_pc1_20_associations.csv", index=False)
    plot_pca_diagnostics(pca_scores, pc_results, manifest)

    nh = len(human)
    np.save(WORK / "expanded_gtex_bridgerna_embeddings.npy", embeddings[:nh])
    np.save(WORK / "expanded_encode_bridgerna_embeddings.npy", embeddings[nh:])
    representations = {"raw_expression": (human, mouse), "joint_pca": (pca_scores[:nh], pca_scores[nh:]),
                       "bridgerna": (embeddings[:nh], embeddings[nh:])}
    all_metrics, all_predictions, all_per_tissue, all_full_rankings, all_centroid_rankings = [], [], [], [], []
    for cohort, tissues in [("complete_11_tissue", TISSUES_11), ("replicated_5_tissue", TISSUES_5)]:
        hmask, mmask = human_meta.tissue.isin(tissues).to_numpy(), mouse_meta.tissue.isin(tissues).to_numpy()
        hmeta, mmeta = human_meta[hmask].reset_index(drop=True), mouse_meta[mmask].reset_index(drop=True)
        for rep, (hx_all, mx_all) in representations.items():
            hx, mx = hx_all[hmask], mx_all[mmask]
            hcent, mcent = centroids(hx, hmeta.tissue.to_numpy(), tissues), centroids(mx, mmeta.tissue.to_numpy(), tissues)
            np.save(WORK / f"expanded_centroids_{cohort}_{rep}_human.npy", hcent.astype(np.float32))
            np.save(WORK / f"expanded_centroids_{cohort}_{rep}_mouse.npy", mcent.astype(np.float32))
            similarity = cosine(hcent, mcent)
            matrix = pd.DataFrame(similarity, index=tissues, columns=tissues)
            matrix.to_csv(RESULTS / f"expanded_centroid_cosine_{cohort}_{rep}.csv")
            plot_matrix(similarity, tissues, f"{cohort} · {rep} centroid cosine", FIGURES / f"centroid_cosine_{cohort}_{rep}.png", "viridis", ".3f")
            for direction, qx, qmeta, rx, rmeta in [
                ("human_to_mouse", hx, hmeta, mx, mmeta), ("mouse_to_human", mx, mmeta, hx, hmeta),
            ]:
                pred, per_tissue, full_ranks, _ = retrieve(qx, qmeta, rx, rmeta, tissues, rep, direction, cohort)
                all_predictions.append(pred)
                all_full_rankings.append(full_ranks)
                per_tissue.insert(0, "direction", direction); per_tissue.insert(0, "representation", rep); per_tissue.insert(0, "cohort", cohort)
                all_per_tissue.append(per_tissue)
                all_metrics.append({"cohort": cohort, "representation": rep, "direction": direction,
                                    "samples": len(pred), "top1": pred.top1_correct.mean(), "mrr": pred.reciprocal_rank.mean(),
                                    "macro_tissue_top1": per_tissue.top1.mean(), "macro_tissue_mrr": per_tissue.mrr.mean()})
                cm = confusion_matrix(pred.true_tissue, pred.nearest_tissue, labels=tissues)
                pd.DataFrame(cm, index=tissues, columns=tissues).to_csv(RESULTS / f"expanded_confusion_{cohort}_{rep}_{direction}.csv")
                norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
                plot_matrix(norm, tissues, f"{cohort} · {rep} · {direction}", FIGURES / f"confusion_{cohort}_{rep}_{direction}.png", "Blues")
            for direction, sim in [("human_to_mouse", similarity), ("mouse_to_human", similarity.T)]:
                for i, tissue in enumerate(tissues):
                    order = np.argsort(-sim[i])
                    match = tissues.index(tissue)
                    all_centroid_rankings.append({"cohort": cohort, "representation": rep, "direction": direction,
                                                  "query_tissue": tissue, "nearest_cross_species_tissue": tissues[order[0]],
                                                  "anatomical_match_rank": int(np.flatnonzero(order == match)[0] + 1),
                                                  "nearest_cosine": float(sim[i, order[0]]),
                                                  "anatomical_match_cosine": float(sim[i, match])})
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(RESULTS / "expanded_task1a_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(RESULTS / "expanded_task1a_sample_rankings.parquet", index=False)
    pd.concat(all_full_rankings, ignore_index=True).to_parquet(RESULTS / "expanded_task1a_full_sample_to_centroid_rankings.parquet", index=False)
    pd.concat(all_per_tissue, ignore_index=True).to_csv(RESULTS / "expanded_task1a_per_tissue_metrics.csv", index=False)
    pd.DataFrame(all_centroid_rankings).to_csv(RESULTS / "expanded_task1a_centroid_rankings.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    rep_order = ["raw_expression", "joint_pca", "bridgerna"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for row, cohort in enumerate(["complete_11_tissue", "replicated_5_tissue"]):
        for col, direction in enumerate(["human_to_mouse", "mouse_to_human"]):
            ax = axes[row, col]
            subset = metrics_frame[(metrics_frame.cohort == cohort) & (metrics_frame.direction == direction)].set_index("representation").reindex(rep_order)
            x = np.arange(len(rep_order)); width = .2
            for offset, metric, label in [(-1.5, "top1", "Top-1"), (-.5, "mrr", "MRR"), (.5, "macro_tissue_top1", "Macro Top-1"), (1.5, "macro_tissue_mrr", "Macro MRR")]:
                ax.bar(x + offset * width, subset[metric], width, label=label)
            ax.axhline(1 / len(TISSUES_11 if cohort == "complete_11_tissue" else TISSUES_5), color="black", ls="--", lw=1, label="Chance")
            ax.set_xticks(x, ["Raw", "Joint PCA", "BridgeRNA"])
            ax.set_ylim(0, 1.05); ax.set_title(f"{cohort.replace('_', ' ')} · {direction.replace('_', ' ')}")
            if col == 0: ax.set_ylabel("Score")
    axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES / "retrieval_summary.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    for rep, (hx, mx) in representations.items():
        plot_samples(np.vstack([hx, mx]), manifest, rep, FIGURES / f"samples_and_centroids_{rep}.png", int(cfg["seed"]))
    provenance = {"cohorts": {"complete_11_tissue": TISSUES_11, "replicated_5_tissue": TISSUES_5},
                  "mouse_profiles": len(mouse_meta), "human_samples": len(human_meta), "genes": len(genes),
                  "preprocessing": "species-specific counts -> TPM -> natural log1p; no alignment/centering/batch correction",
                  "pca_fit": "joint human+mouse log1p(TPM)", "model": cfg["checkpoint"], "frozen": True,
                  "mouse_centroid_weighting": "equal weight per independent experiment"}
    (RESULTS / "expanded_task1a_provenance.json").write_text(json.dumps(provenance, indent=2))
    log("Complete\n" + metrics_frame.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reuse-prepared", action="store_true")
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
