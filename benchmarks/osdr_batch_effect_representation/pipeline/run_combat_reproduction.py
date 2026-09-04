#!/usr/bin/env python3
"""Run the four Sanders-style ComBat/ComBat-seq PCA panels."""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
WORK = HERE / "work"
RESULTS = HERE / "results"
MISSION_COLORS = {
    "RR1_CASIS": "#F8766D",
    "RR1_NASA": "#B79F00",
    "RR3": "#00BA38",
    "RR6": "#00BFC4",
    "RR9": "#619CFF",
    "STS_135": "#F564E3",
}


def export_r_inputs() -> None:
    manifest = pd.read_csv(RESULTS / "sample_manifest.csv")
    genes = pd.read_csv(WORK / "intersected_gene_ids.csv")["ensembl_gene_id"].astype(str)
    raw = np.load(WORK / "intersected_raw_counts.npy")
    logexpr = np.load(WORK / "deseq2_normalized_log2p1.npy")
    assert raw.shape == logexpr.shape == (len(manifest), len(genes))
    for name, matrix in [("combat_raw_counts.csv.gz", raw), ("combat_log2_normalized.csv.gz", logexpr)]:
        target = WORK / name
        print(f"[export] {target.name}: genes={len(genes):,}, samples={len(manifest)}", flush=True)
        pd.DataFrame(matrix.T, index=genes, columns=manifest.sample_id).to_csv(target, compression="gzip")


def plot_grid() -> None:
    manifest = pd.read_csv(RESULTS / "sample_manifest.csv")
    coords = pd.read_csv(RESULTS / "combat_pca_coordinates.csv").merge(manifest, on="sample_id", validate="many_to_one")
    variance = pd.read_csv(RESULTS / "combat_pca_variance.csv")
    layout = [("ComBat", "library_preparation"), ("ComBat-seq", "library_preparation"),
              ("ComBat", "mission"), ("ComBat-seq", "mission")]
    published_limits = {
        ("ComBat", "library_preparation"): ((-0.32, 0.15), (-0.22, 0.16)),
        ("ComBat-seq", "library_preparation"): ((-0.16, 0.32), (-0.24, 0.16)),
        ("ComBat", "mission"): ((-0.22, 0.23), (-0.16, 0.27)),
        ("ComBat-seq", "mission"): ((-0.24, 0.16), (-0.17, 0.27)),
    }

    def display_coordinate(values: pd.Series, limits: tuple[float, float]) -> pd.Series:
        """Unit-normalize a PC score vector and fit it inside published limits."""
        shown = values / np.linalg.norm(values)
        neg = abs(float(shown.min())); pos = float(shown.max())
        factors = []
        if neg > 0: factors.append(abs(limits[0]) / neg)
        if pos > 0: factors.append(limits[1] / pos)
        return shown * min(1.0, 0.96 * min(factors))

    def draw_panel(ax, label: str, method: str, batch: str, color_by: str) -> None:
        z = coords[(coords.method == method) & (coords.batch_variable == batch)].copy()
        xlim, ylim = published_limits[(method, batch)]
        z["PC1"] = display_coordinate(z["PC1"], xlim)
        z["PC2"] = display_coordinate(z["PC2"], ylim)
        levels = list(dict.fromkeys(z[color_by]))
        colors = ({"polyA": "#E76F51", "ribodepleted": "#23B5B5"} if color_by == "library_preparation"
                  else MISSION_COLORS)
        for level in levels:
            for condition, marker in [("FLT", "^"), ("GC", "o")]:
                q = z[(z[color_by] == level) & (z.condition == condition)]
                ax.scatter(q.PC1, q.PC2, marker=marker, s=48, color=colors[level],
                           edgecolor="white", linewidth=.3, alpha=.9, label=f"{level} | {condition}")
        v = variance[(variance.method == method) & (variance.batch_variable == batch)].sort_values("PC")
        ax.set(xlabel=f"PC1 ({v.variance_explained.iloc[0]:.2%})",
               ylabel=f"PC2 ({v.variance_explained.iloc[1]:.2%})",
               title=(f"{label}  {method}: {batch.replace('_', ' ')} as batch\n"
                      f"colored by {color_by.replace('_', ' ')}"))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=7, loc="best")
        ax.grid(alpha=.12)

    def save_four_panel(complementary: bool, stem: str, title: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        for label, ax, (method, batch) in zip("ABCD", axes.flat, layout):
            color_by = ("mission" if batch == "library_preparation" else "library_preparation") if complementary else batch
            draw_panel(ax, label, method, batch, color_by)
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.savefig(RESULTS / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(RESULTS / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    save_four_panel(False, "combat_four_panel_reproduction",
                    "Local reproduction of Sanders et al. Figure 2 (published coloring)")
    save_four_panel(True, "combat_four_panel_complementary_coloring",
                    "Same corrections colored by the complementary batch variable")

    # Two readable complementary-color panels for each corrected batch type.
    for batch, labels, stem, title in [
        ("library_preparation", "AB", "combat_library_batch_colored_by_mission",
         "Library preparation as batch — colored by mission"),
        ("mission", "CD", "combat_mission_batch_colored_by_library_preparation",
         "Mission as batch — colored by library preparation"),
    ]:
        subset = [(method, item_batch) for method, item_batch in layout if item_batch == batch]
        color_by = "mission" if batch == "library_preparation" else "library_preparation"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        for label, ax, (method, item_batch) in zip(labels, axes, subset):
            draw_panel(ax, label, method, item_batch, color_by)
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.savefig(RESULTS / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(RESULTS / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Direct A-D comparisons: corresponding published panel on the left and
    # the same correction coordinates with complementary coloring on the right.
    published = plt.imread(HERE / "references/Sanders_2023_Figure2_corrected.jpg")
    height, width = published.shape[:2]
    crops = {
        "A": published[:height // 2, :width // 2],
        "B": published[:height // 2, width // 2:],
        "C": published[height // 2:, :width // 2],
        "D": published[height // 2:, width // 2:],
    }
    for label, (method, batch) in zip("ABCD", layout):
        color_by = "mission" if batch == "library_preparation" else "library_preparation"
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        axes[0].imshow(crops[label])
        axes[0].set_title(f"Published panel {label}\ncolored by {batch.replace('_', ' ')}")
        axes[0].axis("off")
        draw_panel(axes[1], label, method, batch, color_by)
        axes[1].set_title(f"Switched-color panel {label}\ncolored by {color_by.replace('_', ' ')}")
        fig.suptitle(f"{method}: {batch.replace('_', ' ')} as batch", fontsize=15, fontweight="bold")
        fig.savefig(RESULTS / f"combat_published_vs_switched_panel_{label}.png", dpi=400, bbox_inches="tight")
        fig.savefig(RESULTS / f"combat_published_vs_switched_panel_{label}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    export_r_inputs()
    cmd = ["Rscript", str(HERE / "pipeline/run_combat_reproduction.R"), str(WORK), str(RESULTS), str(HERE / ".r-lib")]
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    plot_grid()
    print(f"[complete] {RESULTS / 'combat_four_panel_reproduction.png'}", flush=True)


if __name__ == "__main__":
    main()
