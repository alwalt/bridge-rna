"""Render saved contextual residual UMAP coordinates with layers across columns."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/context_colored_gene_projections"
LAYERS = ["L0_static", "L1_early", "L6_middle", "L12_final"]
STEM = "gene_context_residuals_umap_horizontal"


def main():
    source = OUT / "context_residual_gene_projections.csv"
    frame = pd.read_csv(source)
    manifest = pd.read_csv(OUT / "sample_manifest.csv")
    assert not frame.duplicated(["dataset", "context", "gene", "layer"]).any()
    assert np.isfinite(frame[["umap_1", "umap_2"]]).all().all()
    assert (frame.loc[frame.layer.eq("L0_static"), ["umap_1", "umap_2"]] == 0).all().all()
    rng = np.random.default_rng(20260921)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 4, figsize=(17, 7))
        fig.subplots_adjust(left=0.05, right=0.83, top=0.87, bottom=0.08,
                            wspace=0.12, hspace=0.25)
        for row, dataset in enumerate(["GTEx", "TCGA"]):
            contexts = manifest.loc[manifest.dataset.eq(dataset), "context"].unique().tolist()
            palette = dict(zip(contexts, plt.get_cmap("tab10").colors))
            for col, layer in enumerate(LAYERS):
                ax = axes[row, col]
                part = frame[frame.dataset.eq(dataset) & frame.layer.eq(layer)]
                assert set(part.context) == set(contexts)
                assert part.groupby("context").gene.nunique().eq(500).all()
                part = part.iloc[rng.permutation(len(part))]
                if layer == "L0_static":
                    ax.scatter([0], [0], s=25, color="0.4")
                    ax.text(0.5, 0.35, "No contextual shift\n(all residuals = 0)",
                            ha="center", va="center", transform=ax.transAxes, color="0.4")
                else:
                    ax.scatter(part.umap_1, part.umap_2, s=4, alpha=0.65,
                               c=[palette[c] for c in part.context], rasterized=True,
                               linewidths=0)
                ax.set(xticks=[], yticks=[])
                if row == 0:
                    ax.set_title(layer.replace("_", " · "), pad=12, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(dataset, fontsize=13, fontweight="bold", labelpad=12)
                for spine in ax.spines.values():
                    spine.set_color("0.85")
            handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                              color=palette[c], label=c.replace("_", " ")) for c in contexts]
            axes[row, -1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.5),
                                 frameon=False, fontsize=9, title=f"{dataset} context")
        fig.suptitle("Gene-level contextual shifts · UMAP", fontsize=16, y=0.98)
        fig.text(0.44, 0.925, "500 fixed genes per context · two representative samples per context",
                 ha="center", fontsize=10, color="0.35")
        for extension in ["png", "pdf"]:
            fig.savefig(OUT / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps({
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_provenance": "provenance.json",
        "coordinates": "Existing UMAP coordinates reused without refitting or transformation",
        "layout": "GTEx and TCGA rows; L0, L1, L6, L12 columns",
        "draw_order_seed": 20260921,
        "limitations": "Exploratory context residual projections; two samples per context; separation is not an effect size. L0 is zero by construction."
    }, indent=2) + "\n")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
