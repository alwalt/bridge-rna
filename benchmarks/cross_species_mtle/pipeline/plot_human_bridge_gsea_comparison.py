#!/usr/bin/env python3
"""Visualize strict-cohort human Bridge-head versus conventional GSEA.

This script uses only saved results. It creates (1) a complete 50-Hallmark
human-method agreement scatter and (2) a signed, within-column normalized
heatmap for comparing relative pathway patterns without equating NES and
Bridge-head score units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

COLS = ["Human MTLE", "Mouse acute", "Mouse intermediate", "Mouse chronic"]
PHASES = ["Acute", "Intermediate", "Chronic"]
COLORS = {
    "Both supported": "#4b7f52",
    "Bridge only": "#4d67a8",
    "GSEA only": "#d18a32",
    "Neither": "#cfcfcf",
}
CORE_LABELS = {
    "TNF-alpha Signaling via NF-kB", "Inflammatory Response",
    "Interferon Gamma Response", "Unfolded Protein Response",
    "mTORC1 Signaling",
}


def display(name: str) -> str:
    return {
        "TNF-alpha Signaling via NF-kB": "TNFα/NF-κB",
        "Inflammatory Response": "Inflammatory response",
        "Interferon Gamma Response": "Interferon-γ",
        "Unfolded Protein Response": "Unfolded protein response",
        "Oxidative Phosphorylation": "Oxidative phosphorylation",
        "Peroxisome": "Peroxisome",
        "mTORC1 Signaling": "mTORC1",
        "KRAS Signaling Dn": "KRAS DN",
    }.get(name, name)


def clean_name(value: str) -> str:
    value = value.removeprefix("HALLMARK_").replace("_", " ").title()
    for old, new in {"Nfkb": "NF-κB", "Tnfa": "TNFα", "Il6": "IL-6",
                     "Jak": "JAK", "Stat3": "STAT3", "Il2": "IL-2",
                     "Stat5": "STAT5", "Myc": "MYC", "Kras": "KRAS",
                     "Dna": "DNA", "Uv": "UV"}.items():
        value = value.replace(old, new)
    return value


def build_human_table(root: Path) -> pd.DataFrame:
    rdir = root / "results/hallmark_head"
    gsea = pd.read_csv(rdir / "conventional_hallmark_gsea.csv").query("species == 'human'")
    bridge = pd.read_csv(rdir / "strict_individual_hallmarks.csv").query("phase == 'Acute'")
    bcols = ["hallmark", "human_delta", "human_delta_ci_low", "human_delta_ci_high"]
    out = gsea[["hallmark", "nes", "fdr_q"]].merge(
        bridge[bcols], on="hallmark", validate="one_to_one")
    out["gsea_supported"] = out.fdr_q < .05
    out["bridge_supported"] = ((out.human_delta_ci_low > 0) |
                                (out.human_delta_ci_high < 0))
    out["evidence_category"] = np.select(
        [out.bridge_supported & out.gsea_supported,
         out.bridge_supported & ~out.gsea_supported,
         ~out.bridge_supported & out.gsea_supported],
        ["Both supported", "Bridge only", "GSEA only"], default="Neither")
    out["direction"] = np.where(np.sign(out.nes) == np.sign(out.human_delta),
                                "same direction", "opposite direction")
    # Scale without centering so zero remains the biological no-change point
    # and scatter quadrants exactly preserve the raw effect directions.
    out["gsea_scaled"] = out.nes / out.nes.std(ddof=0)
    out["bridge_scaled"] = out.human_delta / out.human_delta.std(ddof=0)
    out["supported_discordant"] = (
        out.direction.eq("opposite direction") &
        (out.gsea_supported | out.bridge_supported)
    )
    out["label"] = out.hallmark.isin(CORE_LABELS) | out.supported_discordant
    out["label_reason"] = np.select(
        [out.hallmark.isin(CORE_LABELS) & out.supported_discordant,
         out.supported_discordant,
         out.hallmark.isin(CORE_LABELS)],
        ["literature_relevant;supported_discordant", "supported_discordant",
         "literature_relevant"], default="")
    return out.sort_values("hallmark").reset_index(drop=True)


def spread(y: np.ndarray, lo: float, hi: float, gap: float) -> np.ndarray:
    if not len(y):
        return y
    order = np.argsort(y)
    z = np.clip(y[order].copy(), lo, hi)
    for i in range(1, len(z)):
        z[i] = max(z[i], z[i - 1] + gap)
    if z[-1] > hi:
        z -= z[-1] - hi
    out = np.empty_like(z)
    out[order] = z
    return out


def plot_scatter(table: pd.DataFrame, outbase: Path):
    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    lim = 1.13 * max(table.gsea_scaled.abs().max(), table.bridge_scaled.abs().max())
    # Opaque, explicit rectangles render uniformly across PNG/PDF/SVG viewers.
    for xy, color in [((0, 0), "#eef5ef"), ((-lim, -lim), "#eef5ef"),
                      ((-lim, 0), "#f8eeee"), ((0, -lim), "#f8eeee")]:
        ax.add_patch(Rectangle(xy, lim, lim, facecolor=color, edgecolor="none", zorder=0))
    for category, color in COLORS.items():
        q = table.loc[table.evidence_category == category]
        ax.scatter(q.gsea_scaled, q.bridge_scaled, s=62, color=color, edgecolor="white",
                   linewidth=.7, alpha=.95, zorder=3)
    ax.axhline(0, color="#666", lw=.8)
    ax.axvline(0, color="#666", lw=.8)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#888", lw=.9)
    labels = table.loc[table.label].copy()
    for choose, direction, align in [
        (labels.gsea_scaled >= 0, 1, "left"),
        (labels.gsea_scaled < 0, -1, "right"),
    ]:
        q = labels.loc[choose]
        ypos = spread(q.bridge_scaled.to_numpy(), -.88 * lim, .88 * lim, .075 * lim)
        for (_, r), yt in zip(q.iterrows(), ypos):
            xtext = np.clip(r.gsea_scaled + direction * .10 * lim,
                            -.92 * lim, .92 * lim)
            ann = ax.annotate(display(r.hallmark), (r.gsea_scaled, r.bridge_scaled),
                              xytext=(xtext, yt), ha=align, va="center", fontsize=7,
                              arrowprops=dict(arrowstyle="-", color="#777", lw=.55,
                                              shrinkA=1, shrinkB=3), zorder=4)
            ann.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])
    ax.text(.72 * lim, .91 * lim, "same direction", ha="center", fontsize=8,
            color="#436b48", fontweight="bold")
    ax.text(-.72 * lim, .91 * lim, "opposite direction", ha="center", fontsize=8,
            color="#955252", fontweight="bold")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
           xlabel="Human conventional GSEA effect (scaled by 50-pathway SD)",
           ylabel="Human Bridge-head effect (scaled by 50-pathway SD)")
    ax.set_title("Human MTLE: conventional GSEA versus frozen Bridge Hallmark head",
                 fontsize=14, fontweight="bold", pad=12)
    ax.grid(color="#e9e9e9", lw=.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=c,
                      markersize=7, label=k) for k, c in COLORS.items()]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right", fontsize=8)
    fig.text(.5, .015,
             "All 50 Hallmarks shown. Axes compare relative pathway position, not NES and head-score magnitude. "
             "Green quadrants indicate matching raw directions; red quadrants indicate opposite raw directions.",
             ha="center", fontsize=8)
    fig.subplots_adjust(bottom=.13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outbase.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def build_heatmaps(root: Path, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rdir = root / "results/hallmark_head"
    gsea = pd.read_csv(rdir / "conventional_hallmark_gsea.csv")
    bridge = pd.read_csv(rdir / "direct_hallmark_head_deltas.csv")
    human = gsea.query("species == 'human'").set_index("hallmark")
    gm = pd.DataFrame(index=order)
    gm[COLS[0]] = human.nes.reindex(order)
    for phase, col in zip(PHASES, COLS[1:]):
        gm[col] = gsea.query("species == 'mouse' and phase == @phase").set_index(
            "hallmark").nes.reindex(order)
    bm = pd.DataFrame(index=order)
    bm[COLS[0]] = bridge.query("phase == 'Acute'").set_index("hallmark").human_delta.reindex(order)
    for phase, col in zip(PHASES, COLS[1:]):
        bm[col] = bridge.query("phase == @phase").set_index("hallmark").mouse_delta.reindex(order)
    return gm, bm


def signed_column_scale(matrix: pd.DataFrame) -> pd.DataFrame:
    denom = matrix.abs().max(axis=0)
    if (denom <= 0).any():
        raise ValueError("Cannot scale a zero-effect column")
    return matrix.divide(denom, axis=1)


def draw_heatmap(ax, matrix: pd.DataFrame, title: str, show_y: bool):
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1,
                      interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(range(4), COLS, rotation=27, ha="right")
    ax.set_yticks(range(50))
    ax.set_yticklabels([clean_name(x) for x in matrix.index] if show_y else [])
    ax.tick_params(axis="y", length=0, labelsize=7.2)
    ax.tick_params(axis="x", length=0, labelsize=9)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 50, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.35, alpha=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_heatmap(gsea: pd.DataFrame, bridge: pd.DataFrame, outbase: Path):
    gs, bs = signed_column_scale(gsea), signed_column_scale(bridge)
    fig, axes = plt.subplots(1, 2, figsize=(15.7, 15.0),
                             gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.17, right=.94, bottom=.12, top=.90, wspace=.18)
    draw_heatmap(axes[0], gs, "Conventional Hallmark GSEA", True)
    image = draw_heatmap(axes[1], bs, "Frozen Bridge Hallmark functional state", False)
    cbar = fig.colorbar(image, ax=axes, fraction=.025, pad=.025)
    cbar.set_label("Signed effect relative to strongest pathway in each column", fontsize=9)
    cbar.set_ticks([-1, -.5, 0, .5, 1])
    cbar.set_ticklabels(["−100%", "−50%", "0", "+50%", "+100%"])
    fig.suptitle("Human MTLE and mouse epilepsy: relative 50-Hallmark landscape",
                 fontsize=16, fontweight="bold")
    fig.text(.5, .035,
             "Each column is divided by its own largest absolute effect. Colors therefore compare signed relative pathway patterns, "
             "not absolute effect sizes or statistical support; raw NES and Bridge-head magnitudes remain in the source table.",
             ha="center", fontsize=8.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outbase.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    return gs, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.benchmark_root.resolve()
    outdir = root / "results/hallmark_head/comparison_options"
    figdir = root / "figures/hallmark_head/comparison_options"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    table = build_human_table(root)
    table.to_csv(outdir / "human_bridge_gsea_agreement.csv", index=False)
    plot_scatter(table, figdir / "human_bridge_gsea_agreement")

    order = pd.read_csv(root / "results/hallmark_head/gsea_vs_bridge_hallmark_heatmap_values.csv").sort_values(
        "row_order").hallmark.tolist()
    gsea, bridge = build_heatmaps(root, order)
    gs, bs = plot_heatmap(gsea, bridge, figdir / "relative_gsea_vs_bridge_hallmark_heatmap")
    values = pd.DataFrame({"row_order": range(1, 51), "hallmark": order})
    for prefix, raw, scaled in (("gsea", gsea, gs), ("bridge", bridge, bs)):
        for col in COLS:
            key = col.lower().replace(" ", "_")
            values[f"{prefix}_raw_{key}"] = raw[col].to_numpy()
            values[f"{prefix}_relative_{key}"] = scaled[col].to_numpy()
    values.to_csv(outdir / "relative_gsea_vs_bridge_hallmark_heatmap_values.csv", index=False)
    provenance = {
        "analysis": "visualization only from strict saved results",
        "human_scatter": {
            "axes": "raw effect divided by SD across the same 50 pathways within each method; no centering",
            "direction_quadrants": "based on raw GSEA NES and Bridge-head delta signs",
            "gsea_support": "FDR q < 0.05",
            "bridge_support": "human biological-sample bootstrap 95% CI excludes zero",
            "labels": "five literature-relevant programs plus every opposite-direction pathway supported by either method",
        },
        "relative_heatmap": {
            "transformation": "each method/condition column divided by its maximum absolute effect",
            "preserves": ["raw sign", "within-column relative magnitude", "pathway ordering"],
            "does_not_preserve": ["absolute magnitude between columns", "absolute magnitude between methods"],
        },
        "upstream_recomputation": False,
    }
    (outdir / "human_bridge_gsea_comparison_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
