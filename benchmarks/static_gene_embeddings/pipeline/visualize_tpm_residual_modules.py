#!/usr/bin/env python3
"""Matched projections of cohort-level contextual embeddings after linear TPM residualization."""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from contextual_gene_modules import LAYERS, OUT, SEEDS, VOCAB, WORK
from contextual_module_expression_controls import residualize


def main() -> None:
    started = time.time()
    genes = (
        pd.read_csv(VOCAB).sort_values("token_id").gene_symbol.astype(str).str.upper().tolist()
    )
    rows = []

    for layer in LAYERS:
        residuals = {}
        labels = {}
        tpms = {}
        for dataset in ("GTEx", "TCGA"):
            tpms[dataset] = np.load(WORK / f"{dataset}_mean_log1p_tpm.npy")
            embeddings = np.load(WORK / f"{dataset}_{layer}_mean_tokens.float32.npy")
            residuals[dataset] = residualize(embeddings, tpms[dataset])
            labels[dataset] = (
                KMeans(10, random_state=SEEDS[0], n_init=10, algorithm="lloyd")
                .fit_predict(residuals[dataset])
                + 1
            )

        # Fit each visualization jointly across cohorts, exactly as in the original analysis.
        pair = np.vstack([residuals["GTEx"], residuals["TCGA"]])
        pca50 = PCA(50, random_state=42).fit_transform(pair)
        tsne = TSNE(
            2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=42,
        ).fit_transform(pca50)
        embedding_umap = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        ).fit_transform(pca50)

        for dataset_index, dataset in enumerate(("GTEx", "TCGA")):
            sl = slice(dataset_index * len(genes), (dataset_index + 1) * len(genes))
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "layer": layer,
                        "gene": genes,
                        "residual_cluster": labels[dataset],
                        "mean_log1p_tpm": tpms[dataset],
                        "tsne_1": tsne[sl, 0],
                        "tsne_2": tsne[sl, 1],
                        "umap_1": embedding_umap[sl, 0],
                        "umap_2": embedding_umap[sl, 1],
                    }
                )
            )
        print(f"projected {layer}", flush=True)

    projections = pd.concat(rows, ignore_index=True)
    projections.to_csv(OUT / "matched_tpm_residual_projections.csv", index=False)

    for suffix, field, cmap in (
        ("clusters", "residual_cluster", "tab10"),
        ("expression", "mean_log1p_tpm", "viridis"),
    ):
        fig, axes = plt.subplots(4, 4, figsize=(15, 14), constrained_layout=True)
        panels = (
            ("GTEx", "tsne_1", "tsne_2", "t-SNE"),
            ("TCGA", "tsne_1", "tsne_2", "t-SNE"),
            ("GTEx", "umap_1", "umap_2", "UMAP"),
            ("TCGA", "umap_1", "umap_2", "UMAP"),
        )
        for row_index, layer in enumerate(LAYERS):
            for column_index, (dataset, x, y, title) in enumerate(panels):
                subset = projections[
                    projections.dataset.eq(dataset) & projections.layer.eq(layer)
                ]
                scatter = axes[row_index, column_index].scatter(
                    subset[x], subset[y], c=subset[field], s=2, cmap=cmap, rasterized=True
                )
                axes[row_index, column_index].set(
                    title=f"{dataset} {layer} residual {title}", xticks=[], yticks=[]
                )
                if suffix == "expression":
                    fig.colorbar(scatter, ax=axes[row_index, column_index], fraction=0.035)
        for extension in ("png", "pdf"):
            fig.savefig(
                OUT / f"matched_tpm_residual_projections_by_{suffix}.{extension}",
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(fig)

    provenance = {
        "status": "complete",
        "residualization": "Within each cohort/layer, subtract the least-squares linear association of each of 512 embedding dimensions with centered cohort-mean log1p(TPM).",
        "clustering": "Independent Euclidean K-means (k=10, seed=42, n_init=10) on each 15165 x 512 residual matrix.",
        "projection": "For each layer, stack GTEx and TCGA residual vectors; joint PCA-50 followed by t-SNE and UMAP using the original matched-analysis parameters.",
        "warning": "Residualization removes only linear TPM association; t-SNE/UMAP are exploratory and cluster colors are independently assigned per cohort/layer.",
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "tpm_residual_projection_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
