#!/usr/bin/env python3
"""Held-out OSDR response robustness after controlled library-subspace removal."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "results/task4_response_robustness"
OUT.mkdir(parents=True, exist_ok=True)
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
R3, W3 = T3 / "results", T3 / "work"
KS = (0, 1, 2, 3, 5, 10)
RANDOM_KS = (1, 2, 3, 5, 10)
RANDOM_REPS = 500
SEED = 40241


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def remove(X, B):
    return X if len(B) == 0 else X - (X @ B.T) @ B


def controlled_basis():
    path = ROOT / "work/datasets/chen_2020_tcells"
    m = pd.read_parquet(path / "manifest.parquet").reset_index(drop=True)
    z = np.load(path / "bridgerna_embeddings.npy").astype(float)
    D = []
    for _, g in m.groupby("pair_id", sort=True):
        p = z[g.index[g.library_prep.eq("polyA")]].mean(0)
        r = z[g.index[g.library_prep.eq("ribo")]].mean(0)
        D.append(r-p)
    _, _, vt = np.linalg.svd(np.stack(D), full_matrices=False)
    return vt, m, z


def response(ids, conditions, z, index):
    fi = [index[s] for s, c in zip(ids, conditions) if c == "FLT"]
    gi = [index[s] for s, c in zip(ids, conditions) if c == "GC"]
    return z[fi].mean(0) - z[gi].mean(0)


def build_all_responses(z):
    manifest = pd.read_csv(R3 / "sample_manifest.csv")
    membership = pd.read_csv(R3 / "task3b_contrast_sample_membership.csv")
    index = dict(zip(manifest.sample_id, range(len(manifest))))
    rows, vectors = [], {}
    for cid, g in membership.groupby("contrast_id", sort=False):
        vectors[cid] = response(g.sample_id.tolist(), g.condition.tolist(), z, index)
        rows.append({"contrast_id": cid, "n_FLT": int((g.condition == "FLT").sum()),
                     "n_GC": int((g.condition == "GC").sum())})
    return vectors, pd.DataFrame(rows), manifest


def build_technical_responses(z):
    manifest = pd.read_csv(R3 / "sample_manifest.csv")
    index = dict(zip(manifest.sample_id, range(len(manifest))))
    design = pd.read_csv(R3 / "task3_osd168_technical_replication/technical_response_design.csv")
    vectors = {}
    for x in design.itertuples():
        ids = str(x.samples).split(" | ")
        conditions = ["FLT" if "_FLT_" in s else "GC" if "_GC_" in s else "other" for s in ids]
        vectors[x.representation] = response(ids, conditions, z, index)
    return vectors


COMPARISONS = {
    "RR1 noERCC": ("RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"),
    "RR1 allERCC": ("RR1_OSD48_original_matched", "RR1_OSD168_all_ERCC"),
    "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
    "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
    "OSD168 noERCC/allERCC": ("RR1_OSD168_no-ERCC", "RR1_OSD168_all_ERCC"),
}


def technical_metrics(vectors, k):
    rows = []
    for label, (a, b) in COMPARISONS.items():
        va, vb = vectors[a], vectors[b]
        rows.append({"comparison": label, "removed_components": k,
                     "cosine": cosine(va, vb), "spearman": spearmanr(va, vb).statistic,
                     "norm_a": np.linalg.norm(va), "norm_b": np.linalg.norm(vb)})
    return rows


def cosine_matrix(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T


def mode_metrics(vectors, order, labels, original_matrix, k):
    X = np.stack([vectors[x] for x in order])
    M = cosine_matrix(X)
    same, different = [], []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            (same if labels[i] == labels[j] else different).append(M[i,j])
    dist = np.clip(1-M, 0, 2); np.fill_diagonal(dist, 0)
    clusters = fcluster(linkage(squareform(dist, checks=False), method="average"), 2, criterion="maxclust")
    upper = np.triu_indices(len(X), 1)
    return ({"removed_components": k, "same_mode_mean": np.mean(same), "same_mode_median": np.median(same),
             "different_mode_mean": np.mean(different), "different_mode_median": np.median(different),
             "same_minus_different_mean": np.mean(same)-np.mean(different),
             "silhouette_cosine_fixed_labels": silhouette_score(X, labels, metric="cosine"),
             "cluster_ARI_vs_original_labels": adjusted_rand_score(labels, clusters),
             "matrix_spearman_vs_original": 1.0 if k == 0 else spearmanr(M[upper], original_matrix[upper]).statistic,
             "matrix_pearson_vs_original": 1.0 if k == 0 else pearsonr(M[upper], original_matrix[upper]).statistic}, M)


def response_damage(original, corrected, annotations, k):
    rows = []
    for cid in original:
        a, b = original[cid], corrected[cid]
        osd = annotations.loc[cid, "OSD"]
        mission = annotations.loc[cid, "mission"]
        category = "RR1" if mission == "RR1_NASA" else "RR3" if mission == "RR3" else "other"
        rows.append({"contrast_id": cid, "OSD": osd, "mission": mission, "contrast_group": category,
                     "removed_components": k, "original_corrected_cosine": cosine(a,b),
                     "corrected_original_norm_ratio": np.linalg.norm(b)/np.linalg.norm(a),
                     "original_norm": np.linalg.norm(a), "corrected_norm": np.linalg.norm(b)})
    return rows


def random_controls(original_all, original_tech, order, labels, original_matrix):
    rng = np.random.default_rng(SEED); rows = []
    Xall = np.stack([original_all[x] for x in order])
    for k in RANDOM_KS:
        for rep in range(RANDOM_REPS):
            Q, _ = np.linalg.qr(rng.normal(size=(512, k))); B = Q.T
            allv = dict(zip(order, remove(Xall, B)))
            techv = {name: remove(v[None], B)[0] for name, v in original_tech.items()}
            tm = {x["comparison"]: x for x in technical_metrics(techv, k)}
            mm, _ = mode_metrics(allv, order, labels, original_matrix, k)
            rows.append({"removed_components": k, "replicate": rep,
                         "rr1_cosine": tm["RR1 noERCC"]["cosine"],
                         "rr3_39_cosine": tm["RR3-39"]["cosine"], "rr3_40_cosine": tm["RR3-40"]["cosine"],
                         "same_minus_different_mean": mm["same_minus_different_mean"],
                         "matrix_spearman_vs_original": mm["matrix_spearman_vs_original"]})
    return pd.DataFrame(rows)


def null_summary(null, learned_tech, learned_mode):
    learned_t = learned_tech.set_index(["removed_components","comparison"]).cosine
    learned_m = learned_mode.set_index("removed_components")
    rows = []
    definitions = [("rr1_cosine", "RR1 noERCC"), ("rr3_39_cosine", "RR3-39"), ("rr3_40_cosine", "RR3-40")]
    for k, g in null.groupby("removed_components"):
        for column, comparison in definitions:
            observed = learned_t.loc[(k, comparison)]
            rows.append({"removed_components": k, "metric": column, "controlled_value": observed,
                         "random_mean": g[column].mean(), "random_sd": g[column].std(),
                         "empirical_percentile": np.mean(g[column] <= observed),
                         "one_sided_p_random_at_least_as_high": (1+np.sum(g[column] >= observed))/(1+len(g))})
        for column in ["same_minus_different_mean", "matrix_spearman_vs_original"]:
            observed = learned_m.loc[k, column]
            rows.append({"removed_components": k, "metric": column, "controlled_value": observed,
                         "random_mean": g[column].mean(), "random_sd": g[column].std(),
                         "empirical_percentile": np.mean(g[column] <= observed),
                         "one_sided_p_random_at_least_as_high": (1+np.sum(g[column] >= observed))/(1+len(g))})
    return pd.DataFrame(rows)


def figures(tech, mode, damage, matrices, order, labels, null, absolute):
    palette = {"RR1 noERCC":"#e15759", "RR1 allERCC":"#ff9da7", "RR3-39":"#4e79a7", "RR3-40":"#59a14f", "OSD168 noERCC/allERCC":"#b07aa1"}
    fig, ax = plt.subplots(figsize=(8,4.8), layout="constrained")
    for name, g in tech.groupby("comparison"):
        ax.plot(g.removed_components, g.cosine, marker="o", label=name, color=palette[name])
    ax.axhline(0,color="black",lw=.8); ax.set(xticks=KS, xlabel="Controlled technical PCs removed", ylabel="Replication cosine", title="Held-out OSDR response replication"); ax.legend(fontsize=8, ncol=2)
    for ext in ("png","pdf"): fig.savefig(OUT/f"figure_a_replication_curve.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(14,6),layout="constrained")
    for ax,k,title in zip(axes,[0,5],["Original BridgeRNA","After controlled PC1–5 removal"]):
        im=ax.imshow(matrices[k],vmin=-1,vmax=1,cmap="RdBu_r",aspect="equal")
        ax.set(xticks=[],yticks=np.arange(len(order)),yticklabels=[f"M{m} {c.split('__')[0]}" for c,m in zip(order,labels)],title=title)
        ax.tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=ax,label="Cosine",fraction=.046,pad=.04)
    fig.suptitle("Task 3 response-cosine geometry (fixed ordering and mode labels)")
    for ext in ("png","pdf"): fig.savefig(OUT/f"figure_b_response_matrices.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)

    g=damage[damage.removed_components.eq(5)].sort_values(["contrast_group","original_corrected_cosine"])
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained"); colors=g.contrast_group.map({"RR1":"#e15759","RR3":"#4e79a7","other":"#bab0ac"})
    ax.barh(g.contrast_id,g.original_corrected_cosine,color=colors);ax.set(xlim=(-1,1),xlabel="cos(original Δz, corrected Δz)",title="Response-vector preservation after PC1–5 removal");ax.axvline(0,color="black",lw=.8)
    for ext in ("png","pdf"):fig.savefig(OUT/f"figure_c_response_preservation.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)

    g=null[null.removed_components.eq(5)]; observed=tech.query("removed_components==5 and comparison=='RR1 noERCC'").cosine.iloc[0]
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained");ax.hist(g.rr1_cosine,bins=35,color="#bab0ac");ax.axvline(observed,color="#e15759",lw=2,label=f"Controlled: {observed:.3f}");ax.set(xlabel="RR1 cosine after five dimensions removed",ylabel="Random subspaces",title="Random-subspace specificity control");ax.legend()
    for ext in ("png","pdf"):fig.savefig(OUT/f"figure_d_rr1_random_null.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)

    m5=mode.set_index('removed_components').loc[5]; d5=damage[damage.removed_components.eq(5)]
    vals={"Sample cosine":absolute.loc[5,'median_original_corrected_cosine'],"Top-10 neighbors":absolute.loc[5,'mean_top10_neighbor_overlap'],"Response cosine":d5.original_corrected_cosine.median(),"Response matrix":m5.matrix_spearman_vs_original,"RR3 mean":tech.query("removed_components==5 and comparison in ['RR3-39','RR3-40']").cosine.mean()}
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained");ax.bar(vals.keys(),vals.values(),color=['#f28e2b','#f28e2b','#4e79a7','#4e79a7','#4e79a7']);ax.set(ylim=(0,1),ylabel="Preservation metric",title="Absolute-space versus response-space preservation");ax.tick_params(axis='x',rotation=20)
    for i,v in enumerate(vals.values()):ax.text(i,v+.02,f"{v:.3f}",ha='center')
    for ext in ("png","pdf"):fig.savefig(OUT/f"figure_e_absolute_vs_response.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)


def main():
    basis, controlled_m, controlled_z = controlled_basis()
    sample_z = np.load(W3 / "bridgerna_embeddings.npy").astype(float)
    modes = pd.read_csv(R3 / "task3c_cluster_assignments.csv").sort_values("heatmap_order")
    order = modes.contrast_id.tolist(); labels = modes.geometry_cluster.to_numpy()
    annotations = modes.set_index("contrast_id")
    original_all, counts, _ = build_all_responses(sample_z)
    original_tech = build_technical_responses(sample_z)
    original_matrix = cosine_matrix(np.stack([original_all[x] for x in order]))

    tech_rows, mode_rows, damage_rows, matrices = [], [], [], {}
    for k in KS:
        corrected_z = remove(sample_z, basis[:k])
        allv, _, _ = build_all_responses(corrected_z)
        techv = build_technical_responses(corrected_z)
        tech_rows.extend(technical_metrics(techv, k))
        mm, matrix = mode_metrics(allv, order, labels, original_matrix, k)
        mode_rows.append(mm); matrices[k] = matrix
        damage_rows.extend(response_damage(original_all, allv, annotations, k))
    tech = pd.DataFrame(tech_rows); mode = pd.DataFrame(mode_rows); damage = pd.DataFrame(damage_rows)

    # Verify k=0 against the saved source-of-truth metrics.
    expected = pd.read_csv(R3/"task3_osd168_technical_replication/original_vs_osd168_response_similarity.csv")
    expected_map = {"RR1 noERCC": "RR1_OSD168_no-ERCC", "RR1 allERCC":"RR1_OSD168_all_ERCC",
                    "RR3-39":"C01_OSD168_all_ERCC", "RR3-40":"C02_OSD168_all_ERCC"}
    checks=[]
    for label,name in expected_map.items():
        actual=tech.query("removed_components==0 and comparison==@label").cosine.iloc[0]
        target=expected.loc[expected.OSD168_representation.eq(name),"cosine"].iloc[0]
        checks.append({"comparison":label,"recomputed_cosine":actual,"saved_cosine":target,"absolute_difference":abs(actual-target),"verified":abs(actual-target)<1e-6})
    verification=pd.DataFrame(checks); assert verification.verified.all()

    null = random_controls(original_all, original_tech, order, labels, original_matrix)
    nullsum = null_summary(null, tech, mode)
    selectivity = tech.merge(tech[tech.removed_components.eq(0)][["comparison","cosine"]].rename(columns={"cosine":"original_cosine"}),on="comparison")
    selectivity["delta_cosine"] = selectivity.cosine-selectivity.original_cosine
    selectivity["RR1_noERCC_percentile_among_comparisons"] = selectivity.groupby("removed_components").delta_cosine.rank(pct=True)

    absolute = pd.read_csv(ROOT/"results/task4_followup_controlled_subspace/controlled_geometry_damage.csv").set_index("removed_components")
    tech.to_csv(OUT/"technical_replication_metrics.csv",index=False)
    tech.pivot(index="comparison",columns="removed_components",values="cosine").to_csv(OUT/"technical_replication_cosine_curve.csv")
    tech.pivot(index="comparison",columns="removed_components",values="spearman").to_csv(OUT/"technical_replication_spearman_curve.csv")
    mode.to_csv(OUT/"task3_mode_preservation.csv",index=False)
    damage.to_csv(OUT/"per_contrast_response_damage.csv",index=False)
    damage_summary = damage.groupby(["removed_components","contrast_group"]).agg(
        contrasts=("contrast_id","size"), response_cosine_mean=("original_corrected_cosine","mean"),
        response_cosine_median=("original_corrected_cosine","median"), response_cosine_sd=("original_corrected_cosine","std"),
        norm_ratio_mean=("corrected_original_norm_ratio","mean"), norm_ratio_median=("corrected_original_norm_ratio","median"),
        norm_ratio_sd=("corrected_original_norm_ratio","std")).reset_index()
    damage_summary.to_csv(OUT/"response_damage_summary.csv",index=False)
    selectivity.to_csv(OUT/"technical_replication_selectivity.csv",index=False)
    verification.to_csv(OUT/"original_metric_verification.csv",index=False)
    null.to_parquet(OUT/"random_subspace_metrics.parquet",index=False)
    nullsum.to_csv(OUT/"random_subspace_summary.csv",index=False)
    for k,M in matrices.items():pd.DataFrame(M,index=order,columns=order).to_csv(OUT/f"task3_response_cosine_k{k}.csv")
    figures(tech,mode,damage,matrices,order,labels,null,absolute)

    t5=tech.set_index(["removed_components","comparison"]).cosine; m5=mode.set_index("removed_components").loc[5];d5=damage[damage.removed_components.eq(5)]
    summary=pd.DataFrame([
        ("RR1 cosine original",t5.loc[0,"RR1 noERCC"]),("RR1 cosine PC1-5 removed",t5.loc[5,"RR1 noERCC"]),
        ("RR3-39 cosine original",t5.loc[0,"RR3-39"]),("RR3-39 cosine PC1-5 removed",t5.loc[5,"RR3-39"]),
        ("RR3-40 cosine original",t5.loc[0,"RR3-40"]),("RR3-40 cosine PC1-5 removed",t5.loc[5,"RR3-40"]),
        ("Median response-vector preservation PC1-5",d5.original_corrected_cosine.median()),
        ("Median response norm ratio PC1-5",d5.corrected_original_norm_ratio.median()),
        ("Response-matrix Spearman preservation PC1-5",m5.matrix_spearman_vs_original),
        ("Mode silhouette original",mode.set_index('removed_components').loc[0,'silhouette_cosine_fixed_labels']),
        ("Mode silhouette PC1-5 removed",m5.silhouette_cosine_fixed_labels),
        ("Mode ARI PC1-5 removed",m5.cluster_ARI_vs_original_labels),
        ("Absolute Top-10 neighbor overlap PC1-5",absolute.loc[5,'mean_top10_neighbor_overlap']),
        ("Absolute sample cosine PC1-5",absolute.loc[5,'median_original_corrected_cosine']),
        ("RR1 random-null one-sided p PC1-5",nullsum.query("removed_components==5 and metric=='rr1_cosine'").one_sided_p_random_at_least_as_high.iloc[0]),
    ],columns=['metric','result']);summary.to_csv(OUT/"concise_summary.csv",index=False)
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"controlled_basis_only":True,"OSDR_used_for_fit":False,"models_retrained":False,"contrast_memberships_changed":False,"technical_dimensions":list(KS),"random_replicates_per_k":RANDOM_REPS,"random_seed":SEED,"original_metrics_verified":bool(verification.verified.all())}
    (OUT/"provenance.json").write_text(json.dumps(prov,indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
