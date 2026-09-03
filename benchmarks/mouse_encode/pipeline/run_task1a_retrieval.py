#!/usr/bin/env python3
"""Task 1A: zero-shot human↔mouse tissue-centroid retrieval.

All retrieval is performed in the original representation space. Two-dimensional
coordinates are generated only for descriptive visualization. Source expression
files are read-only; regenerable arrays are written under ``work/`` and compact
results under ``results/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fm_embed.model import load_expression_performer
from src.fm_embed.vocab import load_canonical_genes, norm_gene

HERE = Path(__file__).resolve().parents[1]
TISSUES = ["adrenal", "skeletal muscle", "heart", "cortex", "hippocampus", "liver"]
GTEX_TISSUES = {
    "Adrenal Gland": "adrenal",
    "Muscle - Skeletal": "skeletal muscle",
    "Heart - Atrial Appendage": "heart",
    "Heart - Left Ventricle": "heart",
    "Brain - Cortex": "cortex",
    "Brain - Hippocampus": "hippocampus",
    "Liver": "liver",
}
ENCODE_TISSUES = {
    "adrenal gland": "adrenal",
    "gastrocnemius": "skeletal muscle",
    "heart": "heart",
    "left cerebral cortex": "cortex",
    "layer of hippocampus": "hippocampus",
    "liver": "liver",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def joined(values: Iterable[Any]) -> str | None:
    clean = sorted({str(x).strip() for x in values if x is not None and str(x).strip() not in {"", "nan"}})
    return "; ".join(clean) if clean else None


def build_gtex_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    attrs = pd.read_csv(resolve(cfg["gtex_sample_attributes"]), sep="\t", low_memory=False)
    attrs = attrs[attrs["SMTSD"].isin(GTEX_TISSUES)].copy()
    attrs["tissue"] = attrs["SMTSD"].map(GTEX_TISSUES)
    attrs["subject_id"] = attrs["SAMPID"].str.extract(r"^(GTEX-[^-]+)")
    phen = pd.read_csv(resolve(cfg["gtex_subject_phenotypes"]), sep="\t")
    sex_map = {1: "male", 2: "female", "1": "male", "2": "female"}
    phen["sex"] = phen["SEX"].map(sex_map).fillna(phen["SEX"].astype(str))
    attrs = attrs.merge(phen[["SUBJID", "sex", "AGE"]], left_on="subject_id", right_on="SUBJID", how="left")
    available = set(pd.read_parquet(resolve(cfg["gtex_counts"]), columns=[]).columns)
    # PyArrow returns no data columns for columns=[]; inspect schema instead when needed.
    if not available:
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(resolve(cfg["gtex_counts"])).schema.names)
    attrs = attrs[attrs["SAMPID"].isin(available)].copy()
    out = pd.DataFrame({
        "sample_id": attrs["SAMPID"], "species": "human", "tissue": attrs["tissue"],
        "source_tissue": attrs["SMTSD"], "sex": attrs["sex"], "age": attrs["AGE"],
        "strain": pd.NA, "study_accession": "GTEx", "experiment_accession": attrs["SAMPID"],
        "biosample_accession": attrs["SAMPID"], "quant_file": str(resolve(cfg["gtex_counts"])),
        "annotation": "GENCODE v49", "included": True, "exclusion_reason": pd.NA,
    })
    return out.reset_index(drop=True)


def fetch_encode_experiments(accessions: list[str], cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
    else:
        cached = {}
    for i, accession in enumerate(accessions, 1):
        if accession in cached:
            continue
        log(f"ENCODE metadata {i}/{len(accessions)}: {accession}")
        response = requests.get(
            f"https://www.encodeproject.org/experiments/{accession}/?format=json",
            headers={"Accept": "application/json"}, timeout=60,
        )
        response.raise_for_status()
        cached[accession] = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=2))
    return cached


def build_encode_manifest(cfg: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    meta = pd.read_csv(resolve(cfg["metadata"]), sep="\t", low_memory=False)
    meta = meta[(meta["Output type"] == "gene quantifications") & (meta["File format"] == "tsv")].copy()
    cache = HERE / "work" / "encode_experiment_metadata.json"
    if refresh and cache.exists():
        cache.unlink()
    experiments = fetch_encode_experiments(sorted(meta["Experiment accession"].unique()), cache)
    rows = []
    for _, file_row in meta.iterrows():
        accession = file_row["Experiment accession"]
        exp = experiments[accession]
        biosamples = []
        for replicate in exp.get("replicates", []):
            bio = replicate.get("library", {}).get("biosample", {})
            if isinstance(bio, dict):
                biosamples.append(bio)
        strains = []
        donor_modifications, biosample_modifications, treatments = [], [], []
        for bio in biosamples:
            donor = bio.get("donor", {}) if isinstance(bio.get("donor", {}), dict) else {}
            strains.append(donor.get("strain_name"))
            donor_modifications.extend(donor.get("genetic_modifications", []) or [])
            biosample_modifications.extend(bio.get("genetic_modifications", []) or [])
            treatments.extend(bio.get("treatments", []) or [])
        strain = joined(strains)
        disease_strain = bool(strain and "5xfad" in strain.lower())
        modified = bool(donor_modifications or biosample_modifications or treatments or exp.get("perturbed", False))
        source_tissue = str(file_row["Biosample term name"])
        tissue = ENCODE_TISSUES.get(source_tissue)
        reasons = []
        if disease_strain:
            reasons.append("disease-model strain (5xFAD/CAST)")
        if modified:
            reasons.append("genetic modification/treatment/perturbation")
        if tissue is None:
            reasons.append("not one of six matched tissues")
        quant_path = resolve(cfg["download_directory"]) / f"{file_row['File accession']}.tsv"
        rows.append({
            "sample_id": file_row["File accession"], "species": "mouse", "tissue": tissue,
            "source_tissue": source_tissue,
            "sex": joined([b.get("sex") or b.get("model_organism_sex") for b in biosamples]),
            "age": joined([b.get("age_display") or b.get("age") for b in biosamples]),
            "strain": strain, "study_accession": accession, "experiment_accession": accession,
            "biosample_accession": joined([b.get("accession") for b in biosamples]),
            "quant_file": str(quant_path), "annotation": str(file_row["Genome annotation"]),
            "included": not reasons, "exclusion_reason": joined(reasons),
        })
    return pd.DataFrame(rows).sort_values(["tissue", "experiment_accession"], na_position="last").reset_index(drop=True)


def load_human_tpm(manifest: pd.DataFrame, cfg: dict[str, Any], genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    log(f"Reading GTEx raw counts for {len(manifest):,} selected samples")
    counts_path = resolve(cfg["gtex_counts"])
    source_meta = pd.read_parquet(counts_path, columns=["Description"])
    lengths = pd.read_csv(resolve(cfg["human_gene_lengths"])).drop_duplicates("gene_symbol")
    length_map = lengths.assign(key=lengths["gene_symbol"].map(norm_gene)).set_index("key")["exon_length"]
    keys = source_meta["Description"].map(norm_gene)
    length_bp = keys.map(length_map).to_numpy(dtype=np.float64)
    valid = np.isfinite(length_bp) & (length_bp > 0)
    out = np.zeros((len(manifest), len(genes)), dtype=np.float32)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    target_index = np.array([gene_index.get(key, -1) for key in keys], dtype=np.int32)
    observed = np.zeros(len(genes), dtype=bool)
    observed[np.unique(target_index[valid & (target_index >= 0)])] = True
    sample_ids = manifest["sample_id"].tolist()
    chunk_size = 64
    for start in range(0, len(sample_ids), chunk_size):
        ids = sample_ids[start:start + chunk_size]
        counts = pd.read_parquet(counts_path, columns=ids).to_numpy(dtype=np.float64)
        rate = counts[valid] / (length_bp[valid, None] / 1000.0)
        tpm_rows = rate / np.maximum(rate.sum(axis=0, keepdims=True), 1e-12) * 1e6
        valid_targets = target_index[valid]
        mapped = valid_targets >= 0
        # Add duplicated GENCODE symbols rather than silently selecting one row.
        for column_i in range(len(ids)):
            np.add.at(out[start + column_i], valid_targets[mapped], tpm_rows[mapped, column_i].astype(np.float32))
        log(f"GTEx TPM {min(start + chunk_size, len(sample_ids))}/{len(sample_ids)}")
    mapping = pd.DataFrame({"gene_order": np.arange(len(genes)), "gene_symbol": genes, "human_observed": observed})
    log(f"GTEx TPM complete: {observed.sum():,}/{len(genes):,} vocabulary genes observed")
    return np.log1p(out).astype(np.float32), mapping


def load_mouse_tpm(manifest: pd.DataFrame, cfg: dict[str, Any], genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    ortho = pd.read_csv(resolve(cfg["orthologs"]), sep="\t")
    ortho = ortho[ortho["Human homology type"] == "ortholog_one2one"].copy()
    ortho["mouse_id"] = ortho["Gene stable ID"].astype(str).str.split(".").str[0]
    ortho["human_symbol"] = ortho["Human gene name"].map(norm_gene)
    id_to_human = dict(zip(ortho["mouse_id"], ortho["human_symbol"]))
    gene_index = {gene: i for i, gene in enumerate(genes)}
    out = np.zeros((len(manifest), len(genes)), dtype=np.float32)
    observed = np.zeros(len(genes), dtype=bool)
    for sample_i, row in manifest.reset_index(drop=True).iterrows():
        quant = pd.read_csv(row["quant_file"], sep="\t")
        gene_ids = quant["gene_id"].astype(str).str.split(".").str[0]
        genome = gene_ids.str.startswith("ENSMUSG")
        counts = pd.to_numeric(quant["expected_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        effective = pd.to_numeric(quant["effective_length"], errors="coerce").to_numpy(dtype=np.float64)
        valid = genome.to_numpy() & np.isfinite(effective) & (effective > 0)
        rate = counts[valid] / (effective[valid] / 1000.0)
        denom = rate.sum()
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError(f"No valid genome-aligned abundance in {row['quant_file']}")
        tpm = rate / denom * 1e6
        for mouse_id, value in zip(gene_ids[valid], tpm):
            target = gene_index.get(id_to_human.get(mouse_id))
            if target is not None:
                out[sample_i, target] += np.float32(value)
                observed[target] = True
        if (sample_i + 1) % 5 == 0 or sample_i + 1 == len(manifest):
            log(f"ENCODE TPM {sample_i + 1}/{len(manifest)}")
    mapping = pd.DataFrame({"gene_order": np.arange(len(genes)), "gene_symbol": genes, "mouse_observed": observed})
    log(f"ENCODE TPM complete: {observed.sum():,}/{len(genes):,} vocabulary genes observed")
    return np.log1p(out).astype(np.float32), mapping


def cosine_scores(queries: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    c = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    return q @ c.T


@torch.inference_mode()
def encode_frozen(model: torch.nn.Module, device: torch.device, matrix: np.ndarray, batch_size: int) -> np.ndarray:
    """Standard mean-pooled frozen inference with the training-time AMP dtype."""
    chunks, started, last = [], time.monotonic(), time.monotonic()
    log(f"BridgeRNA inference: samples={len(matrix):,}, batch_size={batch_size}, device={device}")
    for start in range(0, len(matrix), batch_size):
        batch = torch.as_tensor(matrix[start:start + batch_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embedding = model.encode(batch, normalize=False)
        chunks.append(embedding.float().cpu().numpy())
        completed, now = min(start + batch_size, len(matrix)), time.monotonic()
        if now - last >= 60 or completed == len(matrix):
            elapsed = now - started
            log(f"BridgeRNA heartbeat: {completed:,}/{len(matrix):,}, elapsed={elapsed / 60:.1f}m, rate={completed / elapsed:.2f}/s")
            last = now
    return np.concatenate(chunks).astype(np.float32)


def tissue_centroids(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([x[labels == tissue].mean(axis=0) for tissue in TISSUES])


def evaluate_cross(query_x: np.ndarray, query_labels: np.ndarray, ref_x: np.ndarray, ref_labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    centroids = tissue_centroids(ref_x, ref_labels)
    scores = cosine_scores(query_x, centroids)
    true_i = np.array([TISSUES.index(x) for x in query_labels])
    order = np.argsort(-scores, axis=1)
    ranks = np.array([np.flatnonzero(order[i] == true_i[i])[0] + 1 for i in range(len(true_i))])
    pred_i = order[:, 0]
    frame = pd.DataFrame({
        "true_tissue": query_labels, "predicted_tissue": [TISSUES[i] for i in pred_i],
        "rank": ranks, "reciprocal_rank": 1.0 / ranks,
        "correct": pred_i == true_i, "top1_cosine": scores[np.arange(len(scores)), pred_i],
        "correct_tissue_cosine": scores[np.arange(len(scores)), true_i],
    })
    return frame, scores


def evaluate_within(x: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    full = tissue_centroids(x, labels)
    rows = []
    for i, tissue in enumerate(labels):
        tissue_i = TISSUES.index(tissue)
        same = np.flatnonzero(labels == tissue)
        if len(same) < 2:
            rows.append({"true_tissue": tissue, "predicted_tissue": pd.NA, "correct": pd.NA, "evaluable": False})
            continue
        centroids = full.copy()
        centroids[tissue_i] = x[same[same != i]].mean(axis=0)
        scores = cosine_scores(x[i:i + 1], centroids)[0]
        pred = int(np.argmax(scores))
        rows.append({"true_tissue": tissue, "predicted_tissue": TISSUES[pred], "correct": pred == tissue_i, "evaluable": True})
    return pd.DataFrame(rows)


def plot_embedding(x: np.ndarray, manifest: pd.DataFrame, title: str, path: Path, seed: int) -> pd.DataFrame:
    coords = PCA(n_components=2, random_state=seed).fit_transform(x)
    frame = manifest[["sample_id", "species", "tissue"]].copy()
    frame[["axis_1", "axis_2"]] = coords
    palette = dict(zip(TISSUES, plt.get_cmap("tab10").colors[:len(TISSUES)]))
    fig, ax = plt.subplots(figsize=(9, 7))
    for species, marker in [("human", "o"), ("mouse", "^")]:
        for tissue in TISSUES:
            subset = frame[(frame.species == species) & (frame.tissue == tissue)]
            ax.scatter(subset.axis_1, subset.axis_2, s=17 if species == "human" else 45,
                       marker=marker, c=[palette[tissue]], alpha=.5 if species == "human" else .9,
                       edgecolors="none", label=f"{tissue} · {species}")
            if len(subset):
                center = subset[["axis_1", "axis_2"]].mean().to_numpy()
                ax.scatter(*center, s=180, marker=marker, c=[palette[tissue]], edgecolors="black", linewidths=1.3)
    ax.set(title=title, xlabel="PCA 1 (visualization only)", ylabel="PCA 2 (visualization only)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame


def save_heatmaps(confusions: dict[str, np.ndarray], centroid_sims: dict[str, np.ndarray], figures: Path) -> None:
    def heatmap(ax, values, labels_x, labels_y, title, cmap, vmin=None, vmax=None, decimals=2):
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels_x)), labels_x, rotation=45, ha="right")
        ax.set_yticks(range(len(labels_y)), labels_y)
        threshold = (np.nanmin(values) + np.nanmax(values)) / 2
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.{decimals}f}", ha="center", va="center",
                        color="white" if values[i, j] > threshold else "black", fontsize=8)
        ax.set_title(title)
        return image

    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    for row, rep in enumerate(["raw_expression", "pca", "bridgerna"]):
        for col, direction in enumerate(["human_to_mouse", "mouse_to_human"]):
            matrix = confusions[f"{rep}:{direction}"]
            normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            heatmap(axes[row, col], normalized, TISSUES, TISSUES,
                    f"{rep.replace('_', ' ').title()} · {direction.replace('_', ' ')}", "Blues", 0, 1)
            axes[row, col].set(xlabel="Predicted centroid", ylabel="True tissue")
    fig.tight_layout(); fig.savefig(figures / "cross_species_confusion_matrices.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, rep in zip(axes, ["raw_expression", "pca", "bridgerna"]):
        heatmap(ax, centroid_sims[rep], TISSUES, TISSUES, rep.replace("_", " ").title(), "viridis", decimals=3)
        ax.set(xlabel="Mouse centroid", ylabel="Human centroid")
    fig.tight_layout(); fig.savefig(figures / "cross_species_centroid_cosine.png", dpi=220); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cfg = json.loads((HERE / "config.json").read_text())
    for key in ["results", "work"]:
        (HERE / key).mkdir(parents=True, exist_ok=True)
    figures = HERE / "results" / "figures"; figures.mkdir(parents=True, exist_ok=True)
    genes = load_canonical_genes(resolve(cfg["canonical_genes"]))
    if len(genes) != 15165:
        raise ValueError(f"Expected exact 15,165-gene vocabulary, found {len(genes)}")

    gtex_manifest = build_gtex_manifest(cfg)
    encode_audit = build_encode_manifest(cfg, args.refresh_encode_metadata)
    mouse_manifest = encode_audit[encode_audit["included"]].copy().reset_index(drop=True)
    if set(gtex_manifest.tissue) != set(TISSUES) or set(mouse_manifest.tissue) != set(TISSUES):
        raise ValueError("Both species must contain all six tissue labels")
    manifest = pd.concat([gtex_manifest, mouse_manifest], ignore_index=True)
    manifest.to_parquet(HERE / "results" / "task1a_sample_manifest.parquet", index=False)
    manifest.to_csv(HERE / "results" / "task1a_sample_manifest.csv", index=False)
    encode_audit.to_csv(HERE / "results" / "task1a_encode_cohort_audit.csv", index=False)
    log("Cohort sizes:\n" + manifest.groupby(["species", "tissue"]).size().to_string())

    human_path, mouse_path = HERE / "work" / "gtex_log1p_tpm.npy", HERE / "work" / "encode_log1p_tpm.npy"
    if args.reuse_prepared and human_path.exists() and mouse_path.exists():
        human, mouse = np.load(human_path), np.load(mouse_path)
        mapping = pd.read_parquet(HERE / "results" / "task1a_gene_mapping.parquet")
    else:
        human, hm = load_human_tpm(gtex_manifest, cfg, genes)
        mouse, mm = load_mouse_tpm(mouse_manifest, cfg, genes)
        mapping = hm.merge(mm, on=["gene_order", "gene_symbol"], how="outer")
        np.save(human_path, human); np.save(mouse_path, mouse)
        mapping.to_parquet(HERE / "results" / "task1a_gene_mapping.parquet", index=False)
    if human.shape != (len(gtex_manifest), 15165) or mouse.shape != (len(mouse_manifest), 15165):
        raise ValueError(f"Unexpected expression shapes: human={human.shape}, mouse={mouse.shape}")

    combined = np.vstack([human, mouse])
    n_pcs = min(int(cfg["pca_components"]), len(combined) - 1, combined.shape[1])
    log(f"Fitting {n_pcs}-component PCA on combined log1p(TPM)")
    pca_rep = PCA(n_components=n_pcs, svd_solver="randomized", random_state=int(cfg["seed"])).fit_transform(combined).astype(np.float32)
    human_pca, mouse_pca = pca_rep[:len(human)], pca_rep[len(human):]

    emb_path = HERE / "work" / "bridgerna_embeddings.npy"
    if args.reuse_prepared and emb_path.exists():
        embeddings = np.load(emb_path)
    else:
        log("Loading frozen BridgeRNA checkpoint")
        model, device = load_expression_performer(resolve(cfg["checkpoint"]), resolve(cfg["model_config"]), len(genes), args.device)
        embeddings = encode_frozen(model, device, combined, args.batch_size or int(cfg["batch_size"]))
        np.save(emb_path, embeddings)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    human_emb, mouse_emb = embeddings[:len(human)], embeddings[len(human):]

    reps = {
        "raw_expression": (human, mouse), "pca": (human_pca, mouse_pca),
        "bridgerna": (human_emb, mouse_emb),
    }
    h_labels, m_labels = gtex_manifest.tissue.to_numpy(), mouse_manifest.tissue.to_numpy()
    metrics, predictions, confusions, centroid_sims = [], [], {}, {}
    for rep, (hx, mx) in reps.items():
        log(f"Retrieval: {rep}")
        for direction, qx, qlabels, qmeta, rx, rlabels in [
            ("human_to_mouse", hx, h_labels, gtex_manifest, mx, m_labels),
            ("mouse_to_human", mx, m_labels, mouse_manifest, hx, h_labels),
        ]:
            pred, _ = evaluate_cross(qx, qlabels, rx, rlabels)
            pred.insert(0, "sample_id", qmeta.sample_id.to_numpy()); pred.insert(0, "representation", rep)
            pred.insert(1, "direction", direction); predictions.append(pred)
            metrics.extend([
                {"representation": rep, "direction": direction, "metric": "top1_accuracy", "value": float(pred.correct.mean()), "n": len(pred)},
                {"representation": rep, "direction": direction, "metric": "mrr", "value": float(pred.reciprocal_rank.mean()), "n": len(pred)},
            ])
            matrix = confusion_matrix(pred.true_tissue, pred.predicted_tissue, labels=TISSUES)
            confusions[f"{rep}:{direction}"] = matrix
            pd.DataFrame(matrix, index=TISSUES, columns=TISSUES).to_csv(HERE / "results" / f"confusion_{rep}_{direction}.csv")
        for species, x, labels in [("human", hx, h_labels), ("mouse", mx, m_labels)]:
            within = evaluate_within(x, labels)
            evaluable = within[within.evaluable]
            metrics.append({"representation": rep, "direction": f"within_{species}", "metric": "top1_accuracy",
                            "value": float(evaluable.correct.astype(bool).mean()), "n": len(evaluable)})
        hcent, mcent = tissue_centroids(hx, h_labels), tissue_centroids(mx, m_labels)
        centroid_sims[rep] = cosine_scores(hcent, mcent)
        pd.DataFrame(centroid_sims[rep], index=TISSUES, columns=TISSUES).to_csv(HERE / "results" / f"centroid_cosine_{rep}.csv")
        coords = plot_embedding(np.vstack([hx, mx]), manifest, rep.replace("_", " ").title(), figures / f"sample_visualization_{rep}.png", int(cfg["seed"]))
        coords.to_parquet(HERE / "results" / f"visualization_coordinates_{rep}.parquet", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(HERE / "results" / "task1a_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(HERE / "results" / "task1a_cross_species_predictions.parquet", index=False)
    save_heatmaps(confusions, centroid_sims, figures)
    provenance = {
        "task": cfg["task"], "created": pd.Timestamp.now(tz="UTC").isoformat(), "seed": int(cfg["seed"]),
        "vocabulary_genes": len(genes), "preprocessing": "species-specific gene-level counts -> TPM -> natural log1p",
        "human_annotation": "GENCODE v49 exon-union lengths", "mouse_annotation": "ENCODE GENCODE M21 RSEM effective_length",
        "normalization_exclusions": ["batch correction", "cross-species normalization", "z-scoring"],
        "checkpoint": cfg["checkpoint"], "frozen_model": True, "chance_top1": 1 / 6,
        "mouse_exclusion_rule": "exclude 5xFAD strain and any modified/treated/perturbed experiment",
        "within_mouse_note": "singleton liver has no valid leave-one-out centroid and is excluded from within-mouse accuracy",
    }
    (HERE / "results" / "task1a_provenance.json").write_text(json.dumps(provenance, indent=2))
    log("Final metrics:\n" + metrics_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Torch device for frozen-model inference")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reuse-prepared", action="store_true", help="Reuse compatible arrays already in work/")
    parser.add_argument("--refresh-encode-metadata", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
