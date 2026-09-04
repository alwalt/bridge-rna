#!/usr/bin/env python3
"""Reproduce Sanders Figure 1 and compare frozen BridgeRNA representation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.stats import gmean
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.fm_embed.encode import encode_matrix
from src.fm_embed.model import load_expression_performer
from src.fm_embed.sources.osdr import load_osdr_matrix
from src.fm_embed.transform import align_to_vocab, apply_preprocessing
from src.fm_embed.vocab import load_canonical_genes

STUDIES = [47, 48, 137, 168, 173, 242, 245]
LIBPREP = {47: "polyA", 48: "polyA", 137: "ribodepleted", 168: "ribodepleted", 173: "ribodepleted", 242: "ribodepleted", 245: "ribodepleted"}
FACILITY = {47: "UC Davis", 48: "UC Davis", 137: "UC Davis", 168: "UC Davis", 173: "UC Davis", 242: "GeneLab SPL", 245: "GeneLab SPL"}
SEQ = {47: "SE 50bp 30M", 48: "SE 50bp 30M", 137: "PE 150bp 100M", 168: "PE 150bp 100M", 173: "PE 150bp 100M", 242: "PE 149bp 60M", 245: "PE 149bp 60M"}
MISSION_COLORS = {
    "RR1_CASIS": "#F8766D",
    "RR1_NASA": "#B79F00",
    "RR3": "#00BA38",
    "RR6": "#00BFC4",
    "RR9": "#619CFF",
    "STS_135": "#F564E3",
}
OUT, WORK = HERE / "results", HERE / "work"
OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def counts_path(study: int) -> Path:
    found = sorted((ROOT / "data/osdr/raw").glob(f"GLDS-{study}_*.csv"))
    if len(found) != 1: raise RuntimeError(f"Expected one count file for OSD-{study}, found {found}")
    return found[0]


def mission(study: int, sample: str) -> str:
    if study == 47: return "RR1_CASIS"
    if study == 48: return "RR1_NASA"
    if study == 137: return "RR3"
    if study == 168: return "RR3" if "_RR3_" in sample else "RR1_NASA"
    if study == 173: return "STS_135"
    if study == 242: return "RR9"
    if study == 245: return "RR3" if "ISS-T" in sample else "RR6"
    raise ValueError(study)


def assemble_manifest() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "data/osdr/metadata/selected_sample_metadata.tsv", sep="\t", low_memory=False)
    rows = []
    for study in STUDIES:
        z = meta[(meta["id.accession"] == f"OSD-{study}") & meta["study.factor value.spaceflight"].isin(["Space Flight", "Ground Control"])].copy()
        header = set(pd.read_csv(counts_path(study), nrows=0).columns[1:])
        z = z[z["id.sample name"].isin(header)]
        for _, r in z.iterrows():
            sample = str(r["id.sample name"]); condition = "FLT" if r["study.factor value.spaceflight"] == "Space Flight" else "GC"
            preservation = r.get("study.parameter value.carcass preservation method")
            if pd.isna(preservation) or str(preservation).strip() in {"", "{Not Available}"}: preservation = r.get("study.parameter value.sample preservation method")
            rows.append({"sample_id": sample, "OSD": f"OSD-{study}", "study_number": study, "condition": condition,
                         "mission": mission(study, sample), "library_preparation": LIBPREP[study], "sequencing_facility": FACILITY[study],
                         "sequencing_parameters": SEQ[study], "preservation": preservation, "strain": r.get("study.characteristics.strain"),
                         "sex": r.get("study.characteristics.sex"), "age_at_launch": r.get("study.characteristics.age at launch"),
                         "material": r.get("study.characteristics.material type"), "counts_file": str(counts_path(study).relative_to(ROOT))})
    out = pd.DataFrame(rows).sort_values(["study_number", "condition", "sample_id"]).reset_index(drop=True)
    assert len(out) == 112 and (out.condition == "FLT").sum() == 57 and (out.condition == "GC").sum() == 55
    assert out.sample_id.is_unique
    return out


def load_intersected_counts(manifest: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    frames = []
    for study in STUDIES:
        ids = manifest.loc[manifest.study_number == study, "sample_id"].tolist()
        x = pd.read_csv(counts_path(study), usecols=lambda c: c == "Unnamed: 0" or c in ids)
        gene_col = x.columns[0]; x[gene_col] = x[gene_col].astype(str).str.split(".").str[0]
        x = x.groupby(gene_col, sort=False).sum()
        frames.append(x)
    common = sorted(set.intersection(*(set(x.index) for x in frames)))
    merged = pd.concat([x.loc[common] for x in frames], axis=1)
    merged = merged.loc[:, manifest.sample_id]
    return merged.T.to_numpy(dtype=np.float64), common


def deseq2_median_ratio(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    # DESeq2 default "ratio" estimator: genes containing any zero have undefined
    # geometric mean and are excluded from the sample-wise median ratio.
    positive_all = (counts > 0).all(axis=0)
    if not positive_all.any(): raise RuntimeError("No genes eligible for median-ratio normalization")
    log_geo = np.log(counts[:, positive_all]).mean(axis=0)
    ratios = counts[:, positive_all] / np.exp(log_geo)[None, :]
    size_factors = np.median(ratios, axis=1)
    if (size_factors <= 0).any(): raise RuntimeError("Invalid DESeq2-compatible size factor")
    return counts / size_factors[:, None], size_factors, int(positive_all.sum())


def bridgerna_inputs(manifest: pd.DataFrame, genes: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    blocks = []
    for study in STUDIES:
        ids = manifest.loc[manifest.study_number == study, "sample_id"].tolist()
        tpm, _ = load_osdr_matrix(counts_path(study), ids)
        blocks.append(tpm)
    # The per-study annotations are not perfectly identical. A model-vocabulary
    # gene absent from one source table must follow the standard alignment rule
    # (zero-filled), not become NaN through the cross-study column union.
    tpm = pd.concat(blocks).loc[manifest.sample_id].fillna(0.0)
    aligned = align_to_vocab(tpm, genes, genes_are_columns=True)
    return apply_preprocessing(aligned, "log1p_tpm"), tpm


def pca_coordinates(x: np.ndarray, prefix: str) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=min(20, x.shape[0], x.shape[1]), svd_solver="full")
    coords = pca.fit_transform(x)
    frame = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(coords.shape[1])])
    pd.DataFrame({"PC": np.arange(1, len(pca.explained_variance_ratio_)+1), "variance_explained": pca.explained_variance_ratio_,
                  "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}).to_csv(OUT / f"{prefix}_pca_variance.csv", index=False)
    return frame, pca


def direct_umap(representation: np.ndarray, prefix: str) -> pd.DataFrame:
    """Run deterministic UMAP directly in the original representation space."""
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean",
        random_state=1200132, transform_seed=1200132,
    )
    embedding = reducer.fit_transform(np.asarray(representation, dtype=np.float64))
    frame = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    frame.to_csv(OUT / f"{prefix}_umap_coordinates.csv", index=False)
    return frame


def categorical_r2(x: np.ndarray, labels: pd.Series) -> float:
    labels = labels.fillna("missing").astype(str).to_numpy(); center = x.mean(axis=0)
    total = np.square(x - center).sum()
    between = sum((labels == g).sum() * np.square(x[labels == g].mean(axis=0) - center).sum() for g in np.unique(labels))
    return float(between / total) if total else np.nan


def association_tables(representations: dict[str, np.ndarray], manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = ["OSD", "mission", "library_preparation", "condition", "sequencing_facility", "sequencing_parameters", "preservation", "strain", "sex"]
    rng = np.random.default_rng(1200132); rows, pc_rows = [], []
    for name, x in representations.items():
        coords = PCA(n_components=min(10, len(x)-1, x.shape[1]), svd_solver="full").fit_transform(x)
        for variable in variables:
            labels = manifest[variable].fillna("missing").astype(str)
            observed = categorical_r2(x, labels)
            null = np.array([categorical_r2(x, pd.Series(rng.permutation(labels.to_numpy()))) for _ in range(999)])
            silhouette = silhouette_score(coords, labels) if labels.nunique() > 1 and labels.value_counts().min() > 1 else np.nan
            rows.append({"representation": name, "variable": variable, "groups": labels.nunique(), "categorical_r2": observed,
                         "permutation_p": (1 + (null >= observed).sum()) / 1000, "silhouette_pc1_pc10": silhouette})
            for i in range(coords.shape[1]): pc_rows.append({"representation": name, "variable": variable, "PC": i+1, "eta_squared": categorical_r2(coords[:, [i]], labels)})
    return pd.DataFrame(rows), pd.DataFrame(pc_rows)


def plot_structure_metrics(assoc: pd.DataFrame, pc_assoc: pd.DataFrame) -> None:
    variables = ["OSD", "mission", "library_preparation", "sequencing_parameters", "sequencing_facility", "preservation", "strain", "sex", "condition"]
    pivot = assoc.pivot(index="variable", columns="representation", values="categorical_r2").reindex(variables)
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    y = np.arange(len(pivot)); width = .36
    for j, rep in enumerate(["expression", "BridgeRNA"]):
        ax.barh(y + (j-.5)*width, pivot[rep], height=width, label=rep, color=["#4C78A8", "#F58518"][j])
    ax.set(yticks=y, yticklabels=[v.replace("_", " ") for v in variables], xlabel="Categorical variance explained (R²)",
           title="Study, technical, and biological structure in each representation")
    ax.invert_yaxis(); ax.legend(frameon=False); ax.grid(axis="x", alpha=.2)
    fig.savefig(OUT / "categorical_structure_comparison.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "categorical_structure_comparison.pdf", bbox_inches="tight"); plt.close(fig)
    for rep in ["expression", "BridgeRNA"]:
        z = pc_assoc[(pc_assoc.representation == rep) & pc_assoc.variable.isin(variables)]
        mat = z.pivot(index="variable", columns="PC", values="eta_squared").reindex(variables)
        fig, ax = plt.subplots(figsize=(10, 6.2), layout="constrained")
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=max(.25, np.nanmax(mat)), aspect="auto")
        ax.set(xticks=np.arange(mat.shape[1]), xticklabels=[f"PC{x}" for x in mat.columns],
               yticks=np.arange(mat.shape[0]), yticklabels=[v.replace("_", " ") for v in mat.index],
               xlabel="Principal component", title=f"{rep}: categorical association across PC1–PC10")
        fig.colorbar(im, ax=ax, label="η²", pad=.02)
        fig.savefig(OUT / f"{rep.lower()}_pc_association_heatmap.png", dpi=400, bbox_inches="tight")
        fig.savefig(OUT / f"{rep.lower()}_pc_association_heatmap.pdf", bbox_inches="tight"); plt.close(fig)


def matched_plot(coords: pd.DataFrame, manifest: pd.DataFrame, variance: np.ndarray, prefix: str, title: str,
                 published_display: bool = False) -> None:
    frame = pd.concat([manifest.reset_index(drop=True), coords.reset_index(drop=True)], axis=1)
    if published_display:
        # Figure 1 displays unit-length component-score vectors (coordinates
        # near +/-0.2) rather than native score magnitudes. Normalize each PC
        # for display and choose the arbitrary PC2 sign to match the published
        # orientation. Statistical PCA coordinates/variance remain unchanged.
        frame["PC1"] = frame["PC1"] / np.linalg.norm(frame["PC1"])
        frame["PC2"] = -frame["PC2"] / np.linalg.norm(frame["PC2"])
    palettes = {"library_preparation": {"polyA": "#E76F51", "ribodepleted": "#23B5B5"}}
    for color_by, suffix in [("library_preparation", "library_preparation"), ("mission", "mission")]:
        levels = list(dict.fromkeys(frame[color_by]))
        colors = MISSION_COLORS if color_by == "mission" else palettes.get(color_by, {v: plt.cm.tab10(i % 10) for i, v in enumerate(levels)})
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for level in levels:
            for condition, marker in [("FLT", "^"), ("GC", "o")]:
                z = frame[(frame[color_by] == level) & (frame.condition == condition)]
                ax.scatter(z.PC1, z.PC2, s=52, marker=marker, color=colors[level], edgecolor="white", linewidth=.35, alpha=.9,
                           label=f"{level} | {condition}")
        ax.axhline(0,color="#BBBBBB",lw=.6); ax.axvline(0,color="#BBBBBB",lw=.6)
        ax.set(xlabel=f"PC1 ({variance[0]:.2%})", ylabel=f"PC2 ({variance[1]:.2%})", title=f"{title}\ncolored by {color_by.replace('_',' ')}; shape = FLT/GC")
        if published_display:
            ax.set_xlim(-0.22, 0.22); ax.set_ylim(-0.17, 0.23)
            ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2]); ax.set_yticks([-0.1, 0, 0.1, 0.2])
        ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",frameon=False,fontsize=8)
        fig.savefig(OUT / f"{prefix}_figure1_{suffix}.png", dpi=400, bbox_inches="tight")
        fig.savefig(OUT / f"{prefix}_figure1_{suffix}.pdf", bbox_inches="tight"); plt.close(fig)


def matched_umap_plot(coords: pd.DataFrame, manifest: pd.DataFrame, prefix: str, title: str) -> None:
    frame = pd.concat([manifest.reset_index(drop=True), coords.reset_index(drop=True)], axis=1)
    palettes = {"library_preparation": {"polyA": "#E76F51", "ribodepleted": "#23B5B5"}}
    for color_by in ["library_preparation", "mission"]:
        levels = list(dict.fromkeys(frame[color_by]))
        colors = MISSION_COLORS if color_by == "mission" else palettes.get(color_by, {v: plt.cm.tab10(i % 10) for i, v in enumerate(levels)})
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for level in levels:
            for condition, marker in [("FLT", "^"), ("GC", "o")]:
                z = frame[(frame[color_by] == level) & (frame.condition == condition)]
                ax.scatter(z.UMAP1, z.UMAP2, s=52, marker=marker, color=colors[level],
                           edgecolor="white", linewidth=.35, alpha=.9,
                           label=f"{level} | {condition}")
        ax.set(xlabel="UMAP1", ylabel="UMAP2",
               title=f"{title}\ncolored by {color_by.replace('_', ' ')}; shape = FLT/GC")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
        ax.grid(alpha=.12)
        fig.savefig(OUT / f"{prefix}_umap_{color_by}.png", dpi=400, bbox_inches="tight")
        fig.savefig(OUT / f"{prefix}_umap_{color_by}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--batch-size", type=int, default=4); args = ap.parse_args()
    manifest = assemble_manifest(); manifest.to_csv(OUT / "sample_manifest.csv", index=False)
    print(f"Cohort: {len(manifest)} samples ({(manifest.condition=='FLT').sum()} FLT, {(manifest.condition=='GC').sum()} GC)", flush=True)
    counts, intersect_genes = load_intersected_counts(manifest)
    normalized, size_factors, eligible_sf = deseq2_median_ratio(counts)
    # Figure 1's reported PC fractions (25.2%, 12.75%) are reproduced only
    # after log2(count + 1). The article states median-of-ratios normalization
    # followed by prcomp(), but does not explicitly document this log step.
    # Preserve the normalized counts separately and make the inferred plotting
    # transformation explicit in outputs/provenance.
    expression_pca_input = np.log2(normalized + 1.0)
    expr_coords, expr_pca = pca_coordinates(expression_pca_input, "expression")
    expr_coords.insert(0, "sample_id", manifest.sample_id); expr_coords.to_csv(OUT / "expression_pca_coordinates.csv", index=False)
    expr_umap = direct_umap(expression_pca_input, "expression")
    expr_umap.insert(0, "sample_id", manifest.sample_id)
    expr_umap.to_csv(OUT / "expression_umap_coordinates.csv", index=False)
    genes = load_canonical_genes(ROOT / "data/ensembl/canonical_genes.csv"); model_input, tpm = bridgerna_inputs(manifest, genes)
    model, device = load_expression_performer(ROOT / "model/r7hnr92k/best_model.pt", ROOT / "model/r7hnr92k/config.json", len(genes), args.device)
    embeddings = encode_matrix(model, device, model_input, batch_size=args.batch_size, label="task3_embedding")
    emb_coords, emb_pca = pca_coordinates(embeddings, "bridgerna")
    emb_coords.insert(0, "sample_id", manifest.sample_id); emb_coords.to_csv(OUT / "bridgerna_pca_coordinates.csv", index=False)
    emb_umap = direct_umap(embeddings, "bridgerna")
    emb_umap.insert(0, "sample_id", manifest.sample_id)
    emb_umap.to_csv(OUT / "bridgerna_umap_coordinates.csv", index=False)
    np.save(WORK / "intersected_raw_counts.npy", counts.astype(np.float32)); np.save(WORK / "deseq2_normalized_counts.npy", normalized.astype(np.float32))
    np.save(WORK / "deseq2_normalized_log2p1.npy", expression_pca_input.astype(np.float32))
    np.save(WORK / "bridgerna_log1p_tpm_inputs.npy", model_input); np.save(WORK / "bridgerna_embeddings.npy", embeddings)
    pd.Series(intersect_genes, name="ensembl_gene_id").to_csv(WORK / "intersected_gene_ids.csv", index=False)
    matched_plot(expr_coords.drop(columns="sample_id"), manifest, expr_pca.explained_variance_ratio_, "expression", "Sanders Figure 1 reproduction: log2(DESeq2-normalized counts + 1)", published_display=True)
    matched_umap_plot(expr_umap.drop(columns="sample_id"), manifest, "expression", "Direct UMAP of corrected expression")
    matched_plot(emb_coords.drop(columns="sample_id"), manifest, emb_pca.explained_variance_ratio_, "bridgerna", "Frozen BridgeRNA embedding PCA")
    matched_umap_plot(emb_umap.drop(columns="sample_id"), manifest, "bridgerna", "Direct UMAP of frozen BridgeRNA embeddings")
    assoc, pc_assoc = association_tables({"expression": expression_pca_input, "BridgeRNA": embeddings}, manifest)
    assoc.to_csv(OUT / "categorical_structure_summary.csv", index=False); pc_assoc.to_csv(OUT / "pc_categorical_associations.csv", index=False)
    plot_structure_metrics(assoc, pc_assoc)
    cohort = manifest.groupby(["OSD", "mission", "library_preparation", "condition"], as_index=False).size().rename(columns={"size":"samples"})
    cohort.to_csv(OUT / "cohort_summary.csv", index=False)
    notes = f"""# Reproduction notes\n\n- Sanders et al. report 112 liver samples: 57 FLT and 55 GC. This pipeline matches those totals exactly.\n- The seven local GeneLab unnormalized count tables were restricted to FLT and respective GC samples.\n- Counts were inner-joined on version-stripped Ensembl IDs, retaining {len(intersect_genes):,} genes.\n- The paper states DESeq2 v1.30.1 median-of-ratios normalization, followed by R `prcomp()` v4.1.0. Local R does not have DESeq2, so its default `ratio` size-factor algorithm was reproduced directly: genes containing any zero were excluded from geometric-mean estimation; each sample size factor is the median ratio to gene geometric means. {eligible_sf:,} genes contributed to size-factor estimation.\n- Before PCA, we apply `log2(DESeq2-normalized counts + 1)`, then centered, unscaled PCA matching `prcomp(center=TRUE, scale.=FALSE)`. The article does not explicitly state the log step, but it is required to reproduce Figure 1's reported variance: our PC1/PC2 are {expr_pca.explained_variance_ratio_[0]:.2%}/{expr_pca.explained_variance_ratio_[1]:.2%}, versus 25.2%/12.75% in the paper. This inferred step is disclosed rather than presented as directly documented. No batch correction is applied.\n- Mission assignments for multi-mission OSD-168 and OSD-245 use sample-name mission/return annotations consistent with Table 1: OSD-168 RR3 versus RR1 NASA; OSD-245 ISS-T as RR3 versus LAR as RR6.\n- The supplementary payload filenames are declared in the publisher XML, but direct publisher downloads returned 404 during this run. The local OSDR sample metadata plus paper Table 1 were therefore used; this limitation is explicit.\n- BridgeRNA inputs are independently generated as mouse-annotation TPM, mapped to the frozen 15,165 one-to-one vocabulary, then natural log1p(TPM). The exact same 112 samples are used, but this representation necessarily uses the model's native preprocessing rather than DESeq2-normalized full counts.\n- BridgeRNA is frozen; sample embeddings are 512-D mean-pooled contextual representations. No correction, alignment, fine-tuning, or target-label use occurs.\n"""
    (OUT / "reproduction_notes.md").write_text(notes)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "paper_doi": "10.3389/fspas.2023.1200132", "samples": len(manifest), "FLT": 57, "GC": 55,
                  "intersected_genes": len(intersect_genes), "size_factor_genes": eligible_sf, "paper_expression": "inner-joined raw counts; DESeq2 median-of-ratios; inferred log2(count+1); centered unscaled PCA",
                  "model_expression": "mouse exon-length TPM; one-to-one ortholog vocabulary; natural log1p", "canonical_genes": len(genes), "embedding_dim": embeddings.shape[1],
                  "checkpoint": "model/r7hnr92k/best_model.pt", "checkpoint_sha256": sha256(ROOT / "model/r7hnr92k/best_model.pt"), "batch_correction": False,
                  "source_files": [{"path": str(counts_path(s).relative_to(ROOT)), "sha256": sha256(counts_path(s))} for s in STUDIES]}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nPCA variance: expression", expr_pca.explained_variance_ratio_[:2], "BridgeRNA", emb_pca.explained_variance_ratio_[:2])
    print("\nCategorical structure\n", assoc.sort_values(["representation","categorical_r2"], ascending=[True,False]).to_string(index=False))


if __name__ == "__main__": main()
