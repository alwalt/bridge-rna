#!/usr/bin/env python3
"""Frozen-output LUAD cross-space quantification and final TCGA visualization."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import pearsonr, spearmanr

from common import CONFIG, RESULTS, WORK, l2_rows, save_json

OUT = RESULTS / "luad_focused_audit"
PAN = RESULTS / "tcga_pan_cancer_positive_control"


def exact_rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), np.int32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return ranks


def summarize_values(d: pd.DataFrame, label: str) -> dict:
    return {"selection": label, "n": len(d),
            "bridge_similarity_mean": d.bridge_similarity.mean(),
            "bridge_similarity_median": d.bridge_similarity.median(),
            "hallmark_similarity_mean": d.hallmark_similarity.mean(),
            "hallmark_similarity_median": d.hallmark_similarity.median(),
            "bridge_rank_median": d.bridge_rank.median(),
            "hallmark_rank_median": d.hallmark_rank.median()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(CONFIG["tcga_manifest"]).sort_values("matrix_row").reset_index(drop=True)
    bridge = np.load(CONFIG["tcga_embeddings"], mmap_mode="r")
    hdf = pd.read_parquet(RESULTS / "tcga/hallmark_signatures.parquet")
    names = np.load(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz", allow_pickle=False)["hallmarks"].astype(str).tolist()
    hall = hdf[names].to_numpy(np.float32)
    luad_rows = np.flatnonzero(manifest.cohort.eq("LUAD").to_numpy())
    centroid_h = hall[luad_rows].mean(0)
    centroid_b = np.asarray(bridge[luad_rows], np.float32).mean(0)
    sims = l2_rows(hall[luad_rows]) @ l2_rows(centroid_h[None])[0]
    candidates = pd.DataFrame({"row": luad_rows, "similarity": sims,
                               "sample_id": manifest.iloc[luad_rows].sample_id.astype(str).to_numpy()})
    rep_row = int(candidates.sort_values(["similarity", "sample_id"], ascending=[False, True]).iloc[0].row)

    cache = np.load(WORK / "tcga33_top100_indices_scores.npz")
    centroid_hits = pd.read_csv(RESULTS / "tcga33_retrieval/centroid_top100_hits.csv")
    cancers = sorted(centroid_hits.cancer_type.unique())
    specs = [("representative_patient", rep_row, np.asarray(bridge[rep_row]), hall[rep_row]),
             ("all_patient_centroid", 9942 + cancers.index("LUAD"), centroid_b, centroid_h)]
    bref = np.load(WORK / "archs4_bridge_cosine_index.float32.npy", mmap_mode="r")
    href = np.load(WORK / "archs4_hallmark_cosine_index.float32.npy", mmap_mode="r")
    meta = pd.read_parquet(RESULTS / "archs4/sample_metadata.parquet")

    all_rows, overlap_rows, distribution_rows, summary = [], [], [], {}
    for query_type, qi, qb, qh in specs:
        bs = np.asarray(bref @ l2_rows(qb[None])[0], np.float32)
        hs = np.asarray(href @ l2_rows(qh[None])[0], np.float32)
        br, hr = exact_rank(bs), exact_rank(hs)
        bi, hi = cache["bridge_indices"][qi, :100], cache["hallmark_indices"][qi, :100]
        # Reproducibility against the saved exact top-100 cache.
        assert np.array_equal(np.argsort(br)[:100], bi)
        assert np.array_equal(np.argsort(hr)[:100], hi)
        assert np.allclose(bs[bi], cache["bridge_scores"][qi, :100], atol=2e-6)
        assert np.allclose(hs[hi], cache["hallmark_scores"][qi, :100], atol=2e-6)
        union = np.union1d(bi, hi)
        bset, hset = set(map(int, bi)), set(map(int, hi))
        for ri in union:
            m = meta.iloc[int(ri)]
            all_rows.append({"query_type": query_type, "reference_row": int(ri), "gsm": m.sample_id,
                             "gse": m.series_id, "title": m.title, "source": m.source_name_ch1,
                             "characteristics": m.characteristics_ch1,
                             "bridge_similarity": float(bs[ri]), "hallmark_similarity": float(hs[ri]),
                             "bridge_rank": int(br[ri]), "hallmark_rank": int(hr[ri]),
                             "bridge_top100": int(ri) in bset, "hallmark_top100": int(ri) in hset,
                             "bridge_top5": int(ri) in set(map(int, bi[:5])),
                             "hallmark_top5": int(ri) in set(map(int, hi[:5]))})
        qdf = pd.DataFrame(all_rows)
        qdf = qdf[qdf.query_type.eq(query_type)]
        summary[query_type] = {
            "union_candidates": len(union),
            "full_atlas_pearson_bridge_vs_hallmark_similarity": float(pearsonr(bs, hs).statistic),
            "full_atlas_spearman_bridge_vs_hallmark_similarity": float(spearmanr(bs, hs).statistic),
            "pearson_bridge_vs_hallmark_similarity": float(pearsonr(qdf.bridge_similarity, qdf.hallmark_similarity).statistic),
            "spearman_bridge_vs_hallmark_similarity": float(spearmanr(qdf.bridge_similarity, qdf.hallmark_similarity).statistic),
            "median_hallmark_rank_of_bridge_top100": float(qdf[qdf.bridge_top100].hallmark_rank.median()),
            "median_bridge_rank_of_hallmark_top100": float(qdf[qdf.hallmark_top100].bridge_rank.median()),
        }
        for k in (5, 10, 50, 100):
            a, b = set(map(int, bi[:k])), set(map(int, hi[:k]))
            overlap_rows.append({"query_type": query_type, "k": k, "intersection": len(a & b),
                                 "union": len(a | b), "jaccard": len(a & b) / len(a | b)})
        distribution_rows += [summarize_values(qdf[qdf.bridge_top100], "Bridge top 100"),
                              summarize_values(qdf[qdf.hallmark_top100], "Hallmark top 100")]
        distribution_rows[-2]["query_type"] = query_type
        distribution_rows[-1]["query_type"] = query_type

    values = pd.DataFrame(all_rows)
    values.to_csv(OUT / "luad_union_top100_cross_space_scores.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "luad_rank_overlap.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUT / "luad_selection_similarity_distributions.csv", index=False)
    save_json(OUT / "luad_cross_space_summary.json", summary)

    # Publication scatter for the centroid union of top-100 candidates.
    d = values[values.query_type.eq("all_patient_centroid")].copy()
    category = np.select([d.bridge_top5 & d.hallmark_top5, d.bridge_top5, d.hallmark_top5],
                         ["Both top 5", "Bridge top 5", "Hallmark top 5"], default="Top-100 union")
    colors = {"Top-100 union": "#a8a8a8", "Bridge top 5": "#2166ac",
              "Hallmark top 5": "#b2182b", "Both top 5": "#762a83"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in ["Top-100 union", "Bridge top 5", "Hallmark top 5", "Both top 5"]:
        mask = category == label
        ax.scatter(d.loc[mask, "bridge_similarity"], d.loc[mask, "hallmark_similarity"],
                   s=28 if label == "Top-100 union" else 85, alpha=.62 if label == "Top-100 union" else .95,
                   c=colors[label], edgecolors="white" if label != "Top-100 union" else "none", label=label, zorder=3)
    # Label only the three most explanatory samples; all top-five samples remain highlighted.
    labels = d[d.gsm.isin(["GSM3405990", "GSM5081817", "GSM5482393"])]
    offsets = {"GSM3405990": (10, -28), "GSM5081817": (-115, 18), "GSM5482393": (8, 25)}
    for row in labels.itertuples():
        text = f"{row.gsm}\n{str(row.gse).split(',')[0]}"
        if "squamous" in (str(row.title) + " " + str(row.source) + " " + str(row.characteristics)).lower():
            text += " (LUSC)"
        ax.annotate(text, (row.bridge_similarity, row.hallmark_similarity), xytext=offsets[row.gsm],
                    textcoords="offset points", fontsize=7, arrowprops={"arrowstyle": "-", "lw": .4, "color": "#555"})
    ax.set_xlabel("Bridge-512 cosine similarity\n(global transcriptomic similarity)")
    ax.set_ylabel("Hallmark-50 cosine similarity\n(functional-state similarity)")
    ax.set_title("LUAD centroid: global versus functional ARCHS4 retrieval")
    ax.grid(alpha=.18); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "luad_centroid_bridge_vs_hallmark_scatter.png", dpi=300)
    fig.savefig(OUT / "luad_centroid_bridge_vs_hallmark_scatter.pdf"); plt.close(fig)

    # Final matched TCGA geometry figure. Raw matrix is retained unchanged;
    # off-diagonal z-scoring is explicitly visualization-only.
    conv = pd.read_csv(PAN / "conventional_hallmark_cancer_similarity.csv", index_col=0)
    bh = pd.read_csv(PAN / "bridge_hallmark_cancer_similarity.csv", index_col=0)
    b512 = pd.read_csv(PAN / "bridge512_cancer_similarity.csv", index_col=0)
    profiles = pd.read_csv(PAN / "conventional_cancer_hallmark_profiles.csv", index_col=0)
    order = leaves_list(linkage(profiles.to_numpy(), method="average", metric="correlation"))
    vals = b512.to_numpy(); mask = ~np.eye(len(vals), dtype=bool); mu, sd = vals[mask].mean(), vals[mask].std()
    vis = (vals - mu) / sd; np.fill_diagonal(vis, np.nan)
    pd.DataFrame(vis, index=b512.index, columns=b512.columns).to_csv(PAN / "bridge512_offdiagonal_z_visualization_only.csv")
    fig, axes = plt.subplots(1, 4, figsize=(25, 6.5))
    mats = [(conv.to_numpy(), "Conventional Hallmark\nfunctional similarity", "RdBu_r", -1, 1),
            (bh.to_numpy(), "Bridge Hallmark-50\nfunctional similarity", "RdBu_r", -1, 1),
            (vals, "Bridge-512 raw cosine\nglobal similarity", "Reds", 0, 1),
            (vis, "Bridge-512 off-diagonal z-score\nVISUALIZATION ONLY", "RdBu_r", -3, 3)]
    names_order = np.asarray(conv.index)[order]
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, mats):
        image = ax.imshow(mat[np.ix_(order, order)], cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(names_order)), names_order, rotation=90)
        ax.set_yticks(range(len(names_order)), names_order)
        fig.colorbar(image, ax=ax, shrink=.65)
        ax.set_title(title, fontsize=10, weight="bold"); ax.tick_params(labelsize=5)
    fig.suptitle("TCGA cancer geometry across conventional, functional, and global representations", fontsize=15)
    fig.tight_layout(); fig.savefig(PAN / "figures/cancer_similarity_matrices_final.png", dpi=300)
    fig.savefig(PAN / "figures/cancer_similarity_matrices_final.pdf"); plt.close(fig)
    save_json(PAN / "bridge512_visualization_only_provenance.json", {
        "transformation": "(raw cosine - off-diagonal mean) / off-diagonal population SD; diagonal masked",
        "off_diagonal_mean": float(mu), "off_diagonal_sd": float(sd),
        "visualization_only": True, "raw_matrix_preserved": True,
        "quantitative_conclusions_use_raw_matrix": True})


if __name__ == "__main__":
    main()
