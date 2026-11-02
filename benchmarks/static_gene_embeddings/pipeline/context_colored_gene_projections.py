#!/usr/bin/env python3
"""Gene-level t-SNE/UMAP colored by GTEx tissue or TCGA tumor context."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from contextual_depth_followup import VOCAB
from matched_gtex_tcga import GTEX_X, TCGA_X, HERE, LAYERS, load_model

MATCH = HERE / "results/matched_gtex_tcga"
OUT = HERE / "results/context_colored_gene_projections"
WORK = HERE / "work/context_colored_gene_projections"
SEED = 20260921


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-context", type=int, default=2)
    parser.add_argument("--genes", type=int, default=500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run.log"
    log_path.write_text("")

    def log(message: str) -> None:
        text = f"[{pd.Timestamp.utcnow().isoformat()}] {message}"
        print(text, flush=True)
        with log_path.open("a") as handle:
            handle.write(text + "\n")

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(MATCH / "cohort_manifest.csv")
    genes = pd.read_csv(VOCAB).sort_values("token_id").gene_symbol.astype(str).str.upper().to_numpy()
    rng = np.random.default_rng(SEED)
    selected_genes = np.sort(rng.choice(len(genes), args.genes, replace=False))
    pd.DataFrame({"token_id": selected_genes, "gene": genes[selected_genes]}).to_csv(
        OUT / "visualized_genes.csv", index=False
    )
    selected = manifest.groupby(["dataset", "context"], sort=False).head(args.samples_per_context)
    selected.to_csv(OUT / "sample_manifest.csv", index=False)
    arrays = {"GTEx": np.load(GTEX_X, mmap_mode="r"), "TCGA": np.load(TCGA_X, mmap_mode="r")}
    model = load_model(device)
    ids = torch.arange(len(genes), device=device)
    accumulators = {}
    counts = {}
    contexts = {dataset: selected[selected.dataset == dataset].context.unique().tolist()
                for dataset in ("GTEx", "TCGA")}
    for dataset in contexts:
        for context in contexts[dataset]:
            counts[(dataset, context)] = 0
            for layer in LAYERS:
                accumulators[(dataset, context, layer)] = np.zeros((args.genes, 512), np.float64)

    with torch.no_grad():
        for sample_number, row in enumerate(selected.itertuples(index=False), 1):
            expression = torch.as_tensor(
                np.array(arrays[row.dataset][int(row.matrix_row)], copy=True)[None], device=device
            )
            states = {"L0_static": model.gene_embedding(ids)}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model.gene_embedding(ids).unsqueeze(0) + model.ree(expression)
                for layer_index, layer_module in enumerate(model.layers, 1):
                    hidden = layer_module(hidden)
                    for layer_name, target_index in LAYERS.items():
                        if layer_index == target_index:
                            states[layer_name] = hidden[0]
            for layer_name, state in states.items():
                accumulators[(row.dataset, row.context, layer_name)] += (
                    state[selected_genes].float().cpu().numpy()
                )
            counts[(row.dataset, row.context)] += 1
            elapsed = (time.time() - started) / 60
            remaining = elapsed / sample_number * (len(selected) - sample_number)
            log(f"inference {sample_number}/{len(selected)} elapsed={elapsed:.1f}m ETA={remaining:.1f}m")

    rows = []
    residual_rows = []
    for layer in LAYERS:
        matrices, metadata = [], []
        for dataset in ("GTEx", "TCGA"):
            for context in contexts[dataset]:
                matrices.append(
                    (accumulators[(dataset, context, layer)] / counts[(dataset, context)]).astype(np.float32)
                )
                metadata.extend((dataset, context, gene) for gene in genes[selected_genes])
        matrix = np.vstack(matrices)
        pca = PCA(50, random_state=42).fit_transform(matrix)
        tsne = TSNE(2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000,
                    random_state=42).fit_transform(pca)
        projection_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                                    random_state=42).fit_transform(pca)
        frame = pd.DataFrame(metadata, columns=["dataset", "context", "gene"])
        frame["layer"] = layer
        frame[["tsne_1", "tsne_2"]] = tsne
        frame[["umap_1", "umap_2"]] = projection_umap
        rows.append(frame)
        # Remove static gene identity by subtracting each gene's mean across
        # contexts within cohort. This makes color interpretable as context shift.
        residual_blocks = []
        residual_metadata = []
        cursor = 0
        for dataset in ("GTEx", "TCGA"):
            count = len(contexts[dataset])
            block = matrix[cursor:cursor + count * args.genes].reshape(count, args.genes, 512)
            block = block - block.mean(axis=0, keepdims=True)
            residual_blocks.append(block.reshape(count * args.genes, 512))
            residual_metadata.extend((dataset, context, gene)
                                     for context in contexts[dataset]
                                     for gene in genes[selected_genes])
            cursor += count * args.genes
        residual_matrix = np.vstack(residual_blocks)
        if layer == "L0_static":
            residual_tsne = np.zeros((len(residual_matrix), 2), np.float32)
            residual_umap = np.zeros((len(residual_matrix), 2), np.float32)
        else:
            residual_pca = PCA(50, random_state=42).fit_transform(residual_matrix)
            residual_tsne = TSNE(2, perplexity=30, init="pca", learning_rate="auto",
                                 max_iter=1000, random_state=42).fit_transform(residual_pca)
            residual_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                                      random_state=42).fit_transform(residual_pca)
        residual_frame = pd.DataFrame(residual_metadata, columns=["dataset", "context", "gene"])
        residual_frame["layer"] = layer
        residual_frame[["tsne_1", "tsne_2"]] = residual_tsne
        residual_frame[["umap_1", "umap_2"]] = residual_umap
        residual_rows.append(residual_frame)
        log(f"projection complete {layer}")
    projections = pd.concat(rows, ignore_index=True)
    projections.to_csv(OUT / "context_colored_gene_projections.csv", index=False)
    residual_projections = pd.concat(residual_rows, ignore_index=True)
    residual_projections.to_csv(OUT / "context_residual_gene_projections.csv", index=False)

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), constrained_layout=True)
    palettes = {dataset: dict(zip(contexts[dataset], plt.get_cmap("tab10").colors))
                for dataset in ("GTEx", "TCGA")}
    panels = (("GTEx", "tsne_1", "tsne_2", "t-SNE"),
              ("TCGA", "tsne_1", "tsne_2", "t-SNE"),
              ("GTEx", "umap_1", "umap_2", "UMAP"),
              ("TCGA", "umap_1", "umap_2", "UMAP"))
    for layer_index, layer in enumerate(LAYERS):
        for column, (dataset, x, y, method) in enumerate(panels):
            axis = axes[layer_index, column]
            for context in contexts[dataset]:
                frame = projections[(projections.dataset == dataset) &
                                    (projections.layer == layer) &
                                    (projections.context == context)]
                axis.scatter(frame[x], frame[y], s=4, alpha=0.7,
                             color=palettes[dataset][context], label=context, rasterized=True)
            axis.set(title=f"{dataset} {layer} {method}", xticks=[], yticks=[])
            if layer_index == 0:
                axis.legend(fontsize=5, markerscale=2, loc="upper left", frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"gene_context_projections_colored_by_tissue.{extension}",
                    dpi=250, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), constrained_layout=True)
    plot_rng = np.random.default_rng(SEED)
    for layer_index, layer in enumerate(LAYERS):
        for column, (dataset, x, y, method) in enumerate(panels):
            axis = axes[layer_index, column]
            frame = residual_projections[(residual_projections.dataset == dataset) &
                                         (residual_projections.layer == layer)].copy()
            order = plot_rng.permutation(len(frame))
            frame = frame.iloc[order]
            axis.scatter(frame[x], frame[y], s=4, alpha=0.65,
                         c=[palettes[dataset][context] for context in frame.context],
                         rasterized=True)
            title = f"{dataset} {layer} residual {method}"
            if layer == "L0_static":
                title += " (zero by construction)"
            axis.set(title=title, xticks=[], yticks=[])
            if layer_index == 0:
                handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=4,
                                  color=palettes[dataset][context], label=context)
                           for context in contexts[dataset]]
                axis.legend(handles=handles, fontsize=5, loc="upper left", frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"gene_context_residuals_colored_by_tissue.{extension}",
                    dpi=250, bbox_inches="tight")
    plt.close(fig)
    (OUT / "provenance.json").write_text(json.dumps({
        "status": "complete",
        "unit": "one gene-in-context point: the selected gene token averaged across representative samples within a tissue/tumor context",
        "samples_per_context": args.samples_per_context,
        "genes_per_context": args.genes,
        "sample_selection": "first frozen samples per context from the existing balanced cohort manifest",
        "projection": "Joint GTEx+TCGA PCA50, t-SNE, and UMAP per layer with fixed parameters and seed",
        "context_residualization": "For the interpretable tissue-colored grid, subtract each selected gene's mean vector across contexts within cohort and layer before joint projection.",
        "warning": "Exploratory projection; context colors are not quantitative evidence. L0 context residuals are zero by construction.",
        "elapsed_seconds": time.time() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
