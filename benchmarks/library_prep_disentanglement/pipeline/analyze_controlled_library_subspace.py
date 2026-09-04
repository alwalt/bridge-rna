#!/usr/bin/env python3
"""Diagnose whether Task 3 RR1 aligns with controlled PolyA→Ribo effects.

This script is read-only with respect to source data and model artifacts. It
uses cached frozen BridgeRNA embeddings; it does not train FE/RE or BridgeRNA.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "task4_followup_controlled_subspace"
OUT.mkdir(parents=True, exist_ok=True)
TASK3 = REPO / "benchmarks/osdr_batch_effect_representation"
TASK3_RESULTS = TASK3 / "results"
TASK3_WORK = TASK3 / "work"
KS = (1, 2, 5, 10)
RNG_SEED = 41721


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def describe(x: np.ndarray, name: str) -> dict:
    x = np.asarray(x, float)
    return {
        "metric": name,
        "n": int(np.isfinite(x).sum()),
        "mean": float(np.nanmean(x)),
        "median": float(np.nanmedian(x)),
        "sd": float(np.nanstd(x, ddof=1)),
        "q25": float(np.nanquantile(x, .25)),
        "q75": float(np.nanquantile(x, .75)),
        "iqr": float(np.nanquantile(x, .75) - np.nanquantile(x, .25)),
    }


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_parquet(path / "manifest.parquet").reset_index(drop=True)
    embeddings = np.load(path / "bridgerna_embeddings.npy").astype(np.float64)
    if len(manifest) != len(embeddings):
        raise ValueError(f"Manifest/embedding mismatch in {path}")
    return manifest, embeddings


def paired_vectors(manifest: pd.DataFrame, z: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows, vectors = [], []
    for pair_id, group in manifest.groupby("pair_id", sort=True):
        poly = group.index[group.library_prep.eq("polyA")].to_numpy()
        ribo = group.index[group.library_prep.eq("ribo")].to_numpy()
        if not len(poly) or not len(ribo):
            continue
        zp, zr = z[poly].mean(0), z[ribo].mean(0)
        d = zr - zp
        vectors.append(d)
        rows.append({"pair_id": pair_id, "n_polyA": len(poly), "n_ribo": len(ribo),
                     "pair_cosine": cosine(zp, zr), "difference_norm": np.linalg.norm(d)})
    return pd.DataFrame(rows), np.stack(vectors)


def pair_retrieval(manifest: pd.DataFrame, z: np.ndarray) -> dict:
    # Protocol centroids per biological source make this valid for both the
    # one-library T-cell pairs and replicated SRP127360 source groups.
    centroids, labels = [], []
    for (pair_id, prep), g in manifest.groupby(["pair_id", "library_prep"], sort=True):
        centroids.append(z[g.index].mean(0)); labels.append((str(pair_id), prep))
    X = np.stack(centroids); X = X / np.linalg.norm(X, axis=1, keepdims=True)
    ranks = []
    for i, (pair_id, prep) in enumerate(labels):
        target_prep = "ribo" if prep == "polyA" else "polyA"
        candidates = [j for j, (_, p) in enumerate(labels) if p == target_prep]
        sims = X[candidates] @ X[i]
        order = np.argsort(-sims)
        correct = next(k for k, j in enumerate(candidates) if labels[j][0] == pair_id)
        ranks.append(int(np.where(order == correct)[0][0]) + 1)
    ranks = np.asarray(ranks)
    return {"pair_r1": float(np.mean(ranks <= 1)), "pair_r5": float(np.mean(ranks <= 5)),
            "pair_r10": float(np.mean(ranks <= 10)), "pair_mrr": float(np.mean(1 / ranks)),
            "median_rank": float(np.median(ranks))}


def controlled_analysis(manifest: pd.DataFrame, z: np.ndarray):
    pairs, D = paired_vectors(manifest, z)
    mean_direction = D.mean(0); mean_direction /= np.linalg.norm(mean_direction)
    pair_cosines = []
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            pair_cosines.append(cosine(D[i], D[j]))
    pair_cosines = np.asarray(pair_cosines)

    # Uncentered SVD retains the common library displacement and defines the
    # diagnostic removal subspace. Centered SVD is reported separately.
    _, s, vt = np.linalg.svd(D, full_matrices=False)
    uncentered_evr = s**2 / np.sum(s**2)
    Dc = D - D.mean(0)
    _, sc, vtc = np.linalg.svd(Dc, full_matrices=False)
    centered_evr = sc**2 / np.sum(sc**2)
    svd = pd.DataFrame({"component": np.arange(1, len(s)+1),
                        "uncentered_variance_fraction": uncentered_evr,
                        "uncentered_cumulative": np.cumsum(uncentered_evr),
                        "centered_variance_fraction": centered_evr,
                        "centered_cumulative": np.cumsum(centered_evr)})

    loo = []
    for i in range(len(D)):
        direction = np.delete(D, i, axis=0).mean(0)
        loo.append(cosine(D[i], direction))
    pairs["loo_direction_cosine"] = loo

    # Null: preserve each donor's PolyA profile but randomly assign a different
    # donor's Ribo profile, then compute the LOO directional consistency.
    rng = np.random.default_rng(RNG_SEED)
    poly = np.stack([z[g.index[g.library_prep.eq('polyA')]].mean(0)
                     for _, g in manifest.groupby('pair_id', sort=True)])
    ribo = np.stack([z[g.index[g.library_prep.eq('ribo')]].mean(0)
                     for _, g in manifest.groupby('pair_id', sort=True)])
    null_rows = []
    for rep in range(1000):
        perm = rng.permutation(len(ribo))
        while np.any(perm == np.arange(len(ribo))):
            perm = rng.permutation(len(ribo))
        D0 = ribo[perm] - poly
        values = [cosine(D0[i], np.delete(D0, i, axis=0).mean(0)) for i in range(len(D0))]
        null_rows.append({"replicate": rep, "mean_loo_cosine": np.mean(values),
                          "median_loo_cosine": np.median(values)})
    null = pd.DataFrame(null_rows)

    # LOO sample projections, avoiding self-use in direction estimation.
    projections = []
    groups = list(manifest.groupby("pair_id", sort=True))
    for i, (pair_id, g) in enumerate(groups):
        direction = np.delete(D, i, axis=0).mean(0)
        direction /= np.linalg.norm(direction)
        for prep in ("polyA", "ribo"):
            ids = g.index[g.library_prep.eq(prep)]
            projections.append({"pair_id": pair_id, "library_prep": prep,
                                "projection": float(z[ids].mean(0) @ direction)})
    projections = pd.DataFrame(projections)
    wide = projections.pivot(index="pair_id", columns="library_prep", values="projection")
    projection_delta = wide.ribo - wide.polyA

    return pairs, D, mean_direction, vt, svd, pair_cosines, null, projections, projection_delta


def external_analysis(manifest, z, mean_direction, basis):
    groups, D = paired_vectors(manifest, z)
    rows = []
    for row, d in zip(groups.itertuples(), D):
        rec = {"source": row.pair_id, "difference_norm": np.linalg.norm(d),
               "cosine_with_tcell_mean": cosine(d, mean_direction)}
        for k in KS:
            B = basis[:min(k, len(basis))]
            rec[f"fraction_magnitude_squared_in_pc1_{k}"] = float(np.sum((B @ d)**2) / np.dot(d, d))
        rows.append(rec)
    return pd.DataFrame(rows)


def task3_rr1_displacements(mean_direction, basis):
    manifest = pd.read_csv(TASK3_RESULTS / "sample_manifest.csv")
    z = np.load(TASK3_WORK / "bridgerna_embeddings.npy").astype(np.float64)
    index = dict(zip(manifest.sample_id, range(len(manifest))))
    corr = pd.read_csv(TASK3_RESULTS / "task3_osd168_technical_replication/biological_sample_correspondence.csv")
    corr = corr[(corr.RR_mission == "RR1") & corr.exact_animal_match & corr.group.isin(["FLT", "GC"])].copy()
    rows = []
    for tech, q in [("no-ERCC", corr[corr.ERCC_condition.eq("no-ERCC")]),
                    ("ERCC", corr[corr.ERCC_condition.ne("no-ERCC")])]:
        for x in q.itertuples():
            d = z[index[x.OSD168_sample]] - z[index[x.source_sample]]
            rec = {"technical_condition": tech, "animal_id": x.animal_id, "group": x.group,
                   "source_sample": x.source_sample, "OSD168_sample": x.OSD168_sample,
                   "displacement_norm": np.linalg.norm(d), "cosine_with_tcell_mean": cosine(d, mean_direction)}
            for k in KS:
                B = basis[:k]
                frac = float(np.sum((B @ d)**2) / np.dot(d, d))
                rec[f"fraction_magnitude_squared_in_pc1_{k}"] = frac
                rec[f"residual_fraction_pc1_{k}"] = 1-frac
            rows.append(rec)
    return pd.DataFrame(rows)


def project_out(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return X - (X @ basis.T) @ basis


def task3_response_metrics(basis):
    npz = np.load(TASK3_RESULTS / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    vectors = dict(zip(npz["names"], npz["delta_z"].astype(np.float64)))
    comparisons = {"RR1": ("RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"),
                   "RR1_ERCC": ("RR1_OSD48_original_matched", "RR1_OSD168_all_ERCC"),
                   "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
                   "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
                   "ERCC/no-ERCC": ("RR1_OSD168_no-ERCC", "RR1_OSD168_all_ERCC")}
    rows = []
    for k in (0,) + KS:
        B = basis[:k]
        for label, (a, b) in comparisons.items():
            va, vb = vectors[a], vectors[b]
            if k: va, vb = project_out(va[None], B)[0], project_out(vb[None], B)[0]
            rows.append({"comparison": label, "removed_components": k, "cosine": cosine(va, vb),
                         "spearman": spearmanr(va, vb).statistic,
                         "norm_a": np.linalg.norm(va), "norm_b": np.linalg.norm(vb)})
    return pd.DataFrame(rows)


def geometry_damage(manifest, z, basis):
    base = z / np.linalg.norm(z, axis=1, keepdims=True)
    sim0 = base @ base.T; np.fill_diagonal(sim0, -np.inf)
    rows = []
    for k in KS:
        corrected = project_out(z, basis[:k])
        corrn = corrected / np.linalg.norm(corrected, axis=1, keepdims=True)
        sim = corrn @ corrn.T; np.fill_diagonal(sim, -np.inf)
        overlaps = []
        for i in range(len(z)):
            a = set(np.argpartition(-sim0[i], 10)[:10]); b = set(np.argpartition(-sim[i], 10)[:10])
            overlaps.append(len(a & b) / 10)
        geom = np.array([cosine(a, b) for a, b in zip(z, corrected)])
        pair_stats, _ = paired_vectors(manifest, corrected)
        rows.append({"removed_components": k, "mean_original_corrected_cosine": np.mean(geom),
                     "median_original_corrected_cosine": np.median(geom),
                     "mean_top10_neighbor_overlap": np.mean(overlaps),
                     "variance_removed_fraction": float(np.sum(z**2-corrected**2)/np.sum(z**2)),
                     "median_pair_cosine": pair_stats.pair_cosine.median(), **pair_retrieval(manifest, corrected)})
    return pd.DataFrame(rows)


def random_subspace_control(controlled_z, task3_basis_metrics, n_reps=500):
    npz = np.load(TASK3_RESULTS / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    vectors = dict(zip(npz["names"], npz["delta_z"].astype(np.float64)))
    a, b = vectors["RR1_OSD48_original_matched"], vectors["RR1_OSD168_no-ERCC"]
    rng = np.random.default_rng(RNG_SEED + 1); rows = []
    for k in KS:
        for rep in range(n_reps):
            Q, _ = np.linalg.qr(rng.normal(size=(512, k)))
            B = Q.T
            rows.append({"removed_components": k, "replicate": rep,
                         "rr1_cosine": cosine(project_out(a[None], B)[0], project_out(b[None], B)[0])})
    out = pd.DataFrame(rows)
    learned = task3_basis_metrics[task3_basis_metrics.comparison.eq("RR1")].set_index("removed_components").cosine
    summary = out.groupby("removed_components").rr1_cosine.agg(["mean", "median", "std", "min", "max"]).reset_index()
    summary["learned_rr1_cosine"] = summary.removed_components.map(learned)
    summary["fraction_random_at_least_as_improved"] = [
        np.mean(out.loc[out.removed_components.eq(k), "rr1_cosine"] >= learned[k]) for k in summary.removed_components]
    return out, summary


def make_figures(pairs, pairwise, svd, loo_null, external, rr1, response, random_summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    axes[0,0].hist(pairs.difference_norm, bins=12, color="#4c78a8"); axes[0,0].set(title="Controlled displacement magnitude", xlabel="||Ribo − PolyA||")
    axes[0,1].hist(pairwise, bins=20, color="#f58518"); axes[0,1].axvline(np.median(pairwise), color="black", ls="--"); axes[0,1].set(title="Agreement among donor displacements", xlabel="Pairwise cosine")
    axes[1,0].plot(svd.component[:20], svd.uncentered_cumulative[:20], marker="o", label="Uncentered SVD"); axes[1,0].plot(svd.component[:20], svd.centered_cumulative[:20], marker="s", label="Centered PCA"); axes[1,0].set(title="Technical-subspace dimensionality", xlabel="Components", ylabel="Cumulative variance", ylim=(0,1.02)); axes[1,0].legend()
    axes[1,1].hist(loo_null.mean_loo_cosine, bins=30, color="#bab0ac", label="Mismatched null"); axes[1,1].axvline(pairs.loo_direction_cosine.mean(), color="#e45756", lw=2, label="Observed mean"); axes[1,1].set(title="Leave-one-donor-out consistency", xlabel="Mean LOO cosine"); axes[1,1].legend()
    for ext in ("png", "pdf"): fig.savefig(OUT/f"controlled_library_geometry.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = external.source.tolist() + ["RR1\n"+x for x in rr1.technical_condition.unique()]
    vals = external.cosine_with_tcell_mean.tolist() + [rr1.loc[rr1.technical_condition.eq(x), "cosine_with_tcell_mean"].mean() for x in rr1.technical_condition.unique()]
    axes[0].bar(labels, vals, color=["#59a14f"]*len(external)+["#e15759"]*rr1.technical_condition.nunique()); axes[0].axhline(0,color="black",lw=.8); axes[0].set(ylabel="Cosine with controlled mean direction", title="External and RR1 alignment")
    piv = response.pivot(index="comparison", columns="removed_components", values="cosine").loc[["RR1","RR3-39","RR3-40"]]
    for name, row in piv.iterrows(): axes[1].plot(row.index, row.values, marker="o", label=name)
    axes[1].axhline(0,color="black",lw=.8); axes[1].set(xticks=list(piv.columns), xlabel="Controlled PCs removed", ylabel="Response cosine", title="Task 3 diagnostic subspace removal"); axes[1].legend()
    for ext in ("png", "pdf"): fig.savefig(OUT/f"external_rr1_alignment_and_removal.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4), layout="constrained")
    ax.errorbar(random_summary.removed_components, random_summary["mean"], yerr=random_summary["std"], marker="o", capsize=3, label="Random subspace mean ± SD")
    ax.plot(random_summary.removed_components, random_summary.learned_rr1_cosine, marker="s", label="Controlled PolyA/Ribo subspace")
    ax.axhline(response.query("comparison=='RR1' and removed_components==0").cosine.iloc[0], color="black", ls="--", label="Original RR1")
    ax.set(xlabel="Dimensions removed", ylabel="RR1 response cosine", title="Specificity of RR1 change"); ax.legend()
    for ext in ("png", "pdf"): fig.savefig(OUT/f"random_subspace_control.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    chen_path = ROOT / "work/datasets/chen_2020_tcells"
    ext_path = ROOT / "work/datasets/zhao_2018_srp127360"
    chen_m, chen_z = load_dataset(chen_path); ext_m, ext_z = load_dataset(ext_path)
    pairs, D, direction, basis, svd, pairwise, null, projections, projection_delta = controlled_analysis(chen_m, chen_z)
    external = external_analysis(ext_m, ext_z, direction, basis)
    rr1 = task3_rr1_displacements(direction, basis)
    response = task3_response_metrics(basis)
    damage = geometry_damage(chen_m, chen_z, basis)
    random_raw, random_summary = random_subspace_control(chen_z, response)

    pairs.to_csv(OUT/"controlled_pair_metrics.csv", index=False)
    pd.DataFrame({"pairwise_difference_cosine": pairwise}).to_csv(OUT/"controlled_difference_pairwise_cosines.csv", index=False)
    svd.to_csv(OUT/"controlled_difference_svd.csv", index=False)
    null.to_csv(OUT/"loo_mismatched_null.csv", index=False)
    projections.to_csv(OUT/"loo_library_projections.csv", index=False)
    external.to_csv(OUT/"independent_source_alignment.csv", index=False)
    rr1.to_csv(OUT/"rr1_sample_displacement_alignment.csv", index=False)
    response.to_csv(OUT/"task3_response_after_subspace_removal.csv", index=False)
    damage.to_csv(OUT/"controlled_geometry_damage.csv", index=False)
    random_raw.to_parquet(OUT/"random_subspace_rr1_control.parquet", index=False)
    random_summary.to_csv(OUT/"random_subspace_rr1_summary.csv", index=False)

    observed_loo = pairs.loo_direction_cosine.mean()
    null_p = (1 + np.sum(null.mean_loo_cosine >= observed_loo)) / (1 + len(null))
    r = response.set_index(["comparison","removed_components"]).cosine
    summary = pd.DataFrame([
        ("Median controlled PolyA/Ribo pair cosine", pairs.pair_cosine.median()),
        ("Median cosine among d_i", np.median(pairwise)),
        ("Mean LOO technical-direction cosine", observed_loo),
        ("LOO mismatched-null empirical p", null_p),
        ("Uncentered PC1 variance explained", svd.uncentered_variance_fraction.iloc[0]),
        ("Uncentered PC1-5 cumulative variance", svd.uncentered_cumulative.iloc[4]),
        ("Independent blood alignment", external.set_index('source').loc['pooled_blood','cosine_with_tcell_mean']),
        ("Independent colon alignment", external.set_index('source').loc['colon','cosine_with_tcell_mean']),
        ("RR1 sample displacement median cosine with d_library", rr1.cosine_with_tcell_mean.median()),
        ("RR1 sample displacement median variance captured PC1", rr1.fraction_magnitude_squared_in_pc1_1.median()),
        ("RR1 sample displacement median variance captured PC1-5", rr1.fraction_magnitude_squared_in_pc1_5.median()),
        ("Original RR1 response cosine", r.loc['RR1',0]),
        ("RR1 after controlled PC1 removal", r.loc['RR1',1]),
        ("RR1 after controlled PC1-5 removal", r.loc['RR1',5]),
        ("RR3-39 original", r.loc['RR3-39',0]),
        ("RR3-39 after controlled PC1-5 removal", r.loc['RR3-39',5]),
        ("RR3-40 original", r.loc['RR3-40',0]),
        ("RR3-40 after controlled PC1-5 removal", r.loc['RR3-40',5]),
        ("Random PC1-5 RR1 cosine mean", random_summary.set_index('removed_components').loc[5,'mean']),
        ("Random PC1-5 at least as improved fraction", random_summary.set_index('removed_components').loc[5,'fraction_random_at_least_as_improved']),
    ], columns=["metric","result"])
    summary.to_csv(OUT/"concise_summary.csv", index=False)
    distributions = pd.DataFrame([describe(pairs.difference_norm,"difference_norm"), describe(pairs.pair_cosine,"paired_embedding_cosine"), describe(pairwise,"pairwise_difference_cosine"), describe(pairs.loo_direction_cosine,"loo_direction_cosine"), describe(projection_delta,"loo_ribo_minus_polyA_projection")])
    distributions.to_csv(OUT/"controlled_distribution_summary.csv", index=False)
    make_figures(pairs, pairwise, svd, null, external, rr1, response, random_summary)

    # A universal correction requires held-out orientation consistency and
    # acceptable geometry preservation, not merely a large RR1 projection.
    captured = rr1.fraction_magnitude_squared_in_pc1_5.median()
    external_consistent = bool((external.cosine_with_tcell_mean > 0).all())
    damage5 = damage.set_index("removed_components").loc[5]
    acceptable_damage = bool(damage5.variance_removed_fraction < .20 and
                             damage5.mean_top10_neighbor_overlap > .70)
    selective = bool(random_summary.set_index('removed_components').loc[5,'fraction_random_at_least_as_improved'] < .05)
    if observed_loo <= .25 or np.median(pairwise) <= 0:
        conclusion = "D_controlled_effect_not_stable"
    elif captured < .10:
        conclusion = "C_little_overlap"
    elif captured > .50 and external_consistent and selective and acceptable_damage:
        conclusion = "A_strong_alignment"
    else:
        conclusion = "B_partial_context_dependent_overlap"
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "frozen_bridge_only": True,
                  "FE_RE_retrained": False, "OSDR_used_for_fit": False, "controlled_pairs": len(pairs),
                  "permutation_replicates": len(null), "random_subspace_replicates_per_k": 500,
                  "subspace_definition": "right singular vectors of uncentered donor-wise Ribo-minus-PolyA differences",
                  "external_orientation_consistent": external_consistent,
                  "pc1_5_geometry_damage_acceptable": acceptable_damage,
                  "conclusion_class": conclusion}
    (OUT/"provenance.json").write_text(json.dumps(provenance, indent=2))
    print(summary.to_string(index=False)); print(f"\nConclusion class: {conclusion}")


if __name__ == "__main__":
    main()
