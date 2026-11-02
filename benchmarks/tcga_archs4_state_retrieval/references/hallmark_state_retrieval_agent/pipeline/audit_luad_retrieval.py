#!/usr/bin/env python3
"""Focused frozen-output LUAD retrieval audit (no inference or training)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, WORK, l2_rows, save_json, sha256

OUT = RESULTS / "luad_focused_audit"
TCGA33 = RESULTS / "tcga33_retrieval"


def hit_class(text: str) -> str:
    t = text.lower()
    lung = any(x in t for x in ("lung", "pulmonary", "bronch"))
    if (lung and "adenocarcinoma" in t) or "lung adenocarcinoma" in t or "pulmonary adenocarcinoma" in t:
        return "lung adenocarcinoma"
    if lung and "squamous" in t:
        return "lung squamous carcinoma"
    if lung or "nsclc" in t:
        return "other/unspecified lung context"
    if any(x in t for x in ("cancer", "carcinoma", "tumor", "tumour", "malignan", "sarcoma", "melanoma", "leukemia", "lymphoma")):
        return "other malignant context"
    if any(x in t for x in ("normal", "healthy", "control")):
        return "non-malignant/control context"
    return "ambiguous/other"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(CONFIG["tcga_manifest"]).sort_values("matrix_row").reset_index(drop=True)
    bridge = np.load(CONFIG["tcga_embeddings"], mmap_mode="r")
    hdf = pd.read_parquet(RESULTS / "tcga/hallmark_signatures.parquet")
    probe = np.load(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz", allow_pickle=False)
    names = probe["hallmarks"].astype(str).tolist()
    assert len(manifest) == len(bridge) == len(hdf) == 9816
    assert hdf.sample_id.astype(str).tolist() == manifest.sample_id.astype(str).tolist()
    assert hdf[names].shape[1] == 50

    luad_mask = manifest.cohort.eq("LUAD").to_numpy()
    luad_rows = np.flatnonzero(luad_mask)
    hall = hdf[names].to_numpy(np.float32)
    # Representative is the observed LUAD patient with maximum cosine to the
    # LUAD Hallmark centroid; ties resolve by sample_id. Selection is independent
    # of ARCHS4 retrieval performance.
    centroid_h = hall[luad_rows].mean(0, keepdims=True)
    sims = (l2_rows(hall[luad_rows]) @ l2_rows(centroid_h).T).ravel()
    candidates = pd.DataFrame({"row": luad_rows, "similarity": sims,
                               "sample_id": manifest.iloc[luad_rows].sample_id.astype(str).to_numpy()})
    rep_row = int(candidates.sort_values(["similarity", "sample_id"], ascending=[False, True]).iloc[0].row)

    cache = np.load(WORK / "tcga33_top100_indices_scores.npz")
    ref = pd.read_parquet(RESULTS / "archs4/sample_metadata.parquet")
    href = np.load(WORK / "archs4_hallmark_cosine_index.float32.npy", mmap_mode="r")
    cancers = sorted(pd.read_csv(TCGA33 / "centroid_top100_hits.csv").cancer_type.unique())
    centroid_qi = 9942 + cancers.index("LUAD")

    query_specs = [
        ("representative_patient", rep_row, hall[rep_row], np.asarray(bridge[rep_row], np.float32)),
        ("all_patient_centroid", centroid_qi, centroid_h.ravel(), np.asarray(bridge[luad_rows], np.float32).mean(0)),
    ]
    rows = []
    for query_type, qi, qh, qb in query_specs:
        for method, ikey, skey in [
            ("Bridge-512 cosine", "bridge_indices", "bridge_scores"),
            ("Hallmark-50 cosine", "hallmark_indices", "hallmark_scores"),
        ]:
            for rank, (ri, score) in enumerate(zip(cache[ikey][qi, :5], cache[skey][qi, :5]), 1):
                m = ref.iloc[int(ri)]
                text = " ".join(str(m.get(x, "")) for x in ("title", "source_name_ch1", "characteristics_ch1"))
                rec = {
                    "query_type": query_type, "query_sample_id": manifest.iloc[rep_row].sample_id if query_type == "representative_patient" else None,
                    "query_patient_id": manifest.iloc[rep_row].patient_id if query_type == "representative_patient" else None,
                    "method": method, "rank": rank, "score": float(score), "gsm": m.sample_id,
                    "gse": m.series_id, "title": m.title, "source": m.source_name_ch1,
                    "characteristics": m.characteristics_ch1, "conservative_interpretation": hit_class(text),
                    "dominant_shared_hallmarks": None,
                }
                if method.startswith("Hallmark"):
                    contribution = l2_rows(qh[None, :])[0] * href[int(ri)]
                    order = np.argsort(contribution)[::-1][:8]
                    rec["dominant_shared_hallmarks"] = "; ".join(f"{names[j]}={contribution[j]:.4f}" for j in order)
                rows.append(rec)
    hits = pd.DataFrame(rows)
    hits.to_csv(OUT / "luad_top5_retrieval_audit.csv", index=False)
    hits.to_parquet(OUT / "luad_top5_retrieval_audit.parquet", index=False)

    # Matched heatmap: score every union-of-top-five sample in both spaces and
    # standardize against the corresponding full 510,709-sample score background.
    bref = np.load(WORK / "archs4_bridge_cosine_index.float32.npy", mmap_mode="r")
    heat_rows = []
    for query_type, _, qh, qb in query_specs:
        qhits = hits[hits.query_type.eq(query_type)]
        union_gsms = qhits.sort_values(["method", "rank"]).gsm.drop_duplicates().tolist()
        union_idx = [int(ref.index[ref.sample_id.eq(g)][0]) for g in union_gsms]
        for method, q, rmat in [("Bridge-512", qb, bref), ("Hallmark-50", qh, href)]:
            qn = l2_rows(q[None, :])[0]
            background = np.asarray(rmat @ qn, np.float32)
            mu, sd = float(background.mean()), float(background.std(ddof=0))
            for ri in union_idx:
                score = float(background[ri])
                rank = int(1 + np.count_nonzero(background > score))
                m = ref.iloc[ri]
                text = " ".join(str(m.get(x, "")) for x in ("title", "source_name_ch1", "characteristics_ch1"))
                top5 = bool(((qhits.method.str.startswith(method.split("-")[0])) & qhits.gsm.eq(m.sample_id)).any())
                heat_rows.append({"query_type": query_type, "method": method, "gsm": m.sample_id,
                                  "gse": m.series_id, "interpretation": hit_class(text), "cosine": score,
                                  "background_mean": mu, "background_sd": sd, "z_score": (score - mu) / sd,
                                  "atlas_rank": rank, "percentile": 1.0 - (rank - 1) / len(ref), "method_top5": top5})
    heat = pd.DataFrame(heat_rows)
    heat.to_csv(OUT / "luad_cross_method_heatmap_values.csv", index=False)

    import matplotlib.pyplot as plt
    query_order = ["representative_patient", "all_patient_centroid"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    vmax = max(3.0, float(np.nanpercentile(np.abs(heat.z_score), 98)))
    image = None
    for ax, query_type in zip(axes, query_order):
        hq = heat[heat.query_type.eq(query_type)].copy()
        # Put Bridge-selected rows first and preserve method rank; then Hallmark-only rows.
        top = hits[hits.query_type.eq(query_type)]
        order = top.sort_values(["method", "rank"]).gsm.drop_duplicates().tolist()
        labels = []
        matrix = np.empty((len(order), 2), dtype=float)
        annotations = np.empty((len(order), 2), dtype=object)
        for i, gsm in enumerate(order):
            meta = hq[hq.gsm.eq(gsm)].iloc[0]
            labels.append(f"{gsm} | {meta.gse} | {meta.interpretation}")
            for j, method in enumerate(["Bridge-512", "Hallmark-50"]):
                cell = hq[hq.gsm.eq(gsm) & hq.method.eq(method)].iloc[0]
                matrix[i, j] = cell.z_score
                star = "●" if cell.method_top5 else ""
                annotations[i, j] = f"{cell.z_score:.1f}\n#{cell.atlas_rank}{star}"
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks([0, 1], ["Bridge-512", "Hallmark-50"])
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(2):
                color = "white" if matrix[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=7, color=color)
        title = "Representative TCGA-86-A4JF" if query_type == "representative_patient" else "All 516 LUAD patients: centroid"
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Frozen retrieval representation")
    fig.colorbar(image, ax=axes, shrink=.78, label="Cosine z-score vs all 510,709 ARCHS4 samples")
    fig.suptitle("LUAD retrieval neighborhoods: global transcriptome versus Hallmark state\n● = method-specific top-five hit; # = exact atlas rank", fontsize=14)
    fig.savefig(OUT / "luad_bridge512_vs_hallmark50_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "luad_bridge512_vs_hallmark50_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Complete 50-program profiles for the query and method-selected hits.
    raw_href = np.load(WORK / "archs4_hallmark_signatures.float32.npy", mmap_mode="r")
    centroid_order = np.argsort(np.abs(centroid_h.ravel()))[::-1]
    profile_rows, panel_data = [], []
    for query_type, _, qh, _ in query_specs:
        qhits = hits[hits.query_type.eq(query_type)]
        columns = [("Query", "LUAD query", qh)]
        for method, short in [("Bridge-512 cosine", "B"), ("Hallmark-50 cosine", "H")]:
            for hit in qhits[qhits.method.eq(method)].sort_values("rank").itertuples():
                ri = int(ref.index[ref.sample_id.eq(hit.gsm)][0])
                columns.append((short, f"{short}{hit.rank}\n{hit.gsm}", np.asarray(raw_href[ri], np.float32)))
        matrix = np.column_stack([c[2] for c in columns])[centroid_order]
        panel_data.append((query_type, columns, matrix))
        for ci, (selection, label, _) in enumerate(columns):
            gsm = None if selection == "Query" else label.split("\n", 1)[1]
            for display_rank, j in enumerate(centroid_order, 1):
                profile_rows.append({"query_type": query_type, "selection": selection,
                                     "column_label": label.replace("\n", " "), "gsm": gsm,
                                     "hallmark_display_rank": display_rank, "hallmark": names[j],
                                     "frozen_hallmark_score": float(matrix[display_rank - 1, ci])})
    pd.DataFrame(profile_rows).to_csv(OUT / "luad_hallmark_term_heatmap_values.csv", index=False)

    all_values = np.concatenate([x[2].ravel() for x in panel_data])
    limit = max(1.0, float(np.nanpercentile(np.abs(all_values), 98)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 16), sharey=True, constrained_layout=True)
    image = None
    row_labels = [names[j] for j in centroid_order]
    for ax, (query_type, columns, matrix) in zip(axes, panel_data):
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(columns)), [c[1] for c in columns], rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(50), row_labels, fontsize=8)
        ax.axvline(.5, color="black", lw=1.2)
        ax.axvline(5.5, color="black", lw=1.2)
        ax.text(3, -1.35, "Bridge-512 selected", ha="center", va="bottom", fontsize=10, weight="bold")
        ax.text(8, -1.35, "Hallmark-50 selected", ha="center", va="bottom", fontsize=10, weight="bold")
        title = "Representative TCGA-86-A4JF" if query_type == "representative_patient" else "All 516 LUAD patients: centroid"
        ax.set_title(title, fontsize=13, weight="bold", pad=62)
        ax.set_xlabel("Query and retrieved ARCHS4 samples")
    axes[0].set_ylabel("Hallmark program (ordered by |LUAD centroid score|)")
    fig.colorbar(image, ax=axes, shrink=.75, label="Frozen Hallmark-50 score")
    fig.suptitle("LUAD functional profiles of Bridge-512 versus Hallmark-50 retrievals", fontsize=16, y=1.025)
    fig.savefig(OUT / "luad_hallmark_terms_bridge512_vs_hallmark50.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "luad_hallmark_terms_bridge512_vs_hallmark50.pdf", bbox_inches="tight")
    plt.close(fig)

    composition = (hits.groupby(["query_type", "method", "conservative_interpretation"], as_index=False)
                   .size().rename(columns={"size": "top5_count"}))
    composition.to_csv(OUT / "luad_top5_composition.csv", index=False)
    qrows = []
    for query_type, _, qh, _ in query_specs:
        for rank, j in enumerate(np.argsort(np.abs(qh))[::-1][:10], 1):
            qrows.append({"query_type": query_type, "rank": rank, "hallmark": names[j],
                          "value": float(qh[j]), "direction": "positive" if qh[j] >= 0 else "negative"})
    pd.DataFrame(qrows).to_csv(OUT / "luad_query_dominant_hallmarks.csv", index=False)
    subgroup = pd.DataFrame([{
        "candidate_subgroup": "molecular/clinical LUAD subgroups",
        "status": "not run",
        "reason": "Authoritative local manifest lacks reliable mutation, smoking, stage, and histologic/molecular subtype fields.",
        "available_fields": ", ".join(manifest.columns),
    }])
    subgroup.to_csv(OUT / "subgroup_availability.csv", index=False)
    save_json(OUT / "provenance.json", {
        "frozen_outputs_only": True, "bridge_inference_rerun": False, "hallmark_head_retrained": False,
        "luad_patients": int(len(luad_rows)), "representative_rule": "maximum Hallmark cosine to LUAD Hallmark centroid; sample_id tie-break",
        "representative_matrix_row": rep_row, "representative_sample_id": str(manifest.iloc[rep_row].sample_id),
        "representative_patient_id": str(manifest.iloc[rep_row].patient_id),
        "centroid_rule": "arithmetic mean over all 516 LUAD patients before cosine normalization",
        "top_k": 5, "subgroups_run": False,
        "inputs": {
            "tcga_manifest": CONFIG["tcga_manifest"], "tcga_embeddings": CONFIG["tcga_embeddings"],
            "retrieval_cache": str(WORK / "tcga33_top100_indices_scores.npz"),
            "archs4_metadata": str(RESULTS / "archs4/sample_metadata.parquet"),
            "retrieval_cache_sha256": sha256(WORK / "tcga33_top100_indices_scores.npz"),
        },
    })
    print(hits.to_string(index=False))


if __name__ == "__main__":
    main()
