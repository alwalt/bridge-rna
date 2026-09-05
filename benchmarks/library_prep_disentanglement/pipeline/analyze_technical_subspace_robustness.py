#!/usr/bin/env python3
"""Donor-resampling robustness of the controlled library-associated subspace.

Uses cached frozen BridgeRNA embeddings only. This script estimates diagnostic
stability; it never corrects or rewrites an embedding.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "results/task4_technical_subspace_robustness"
CONTROLLED = ROOT / "work/datasets/chen_2020_tcells"
TASK3_VECTORS = REPO / "benchmarks/osdr_batch_effect_representation/results/task3_osd168_technical_replication/technical_response_vectors.npz"
PRIOR_RANDOM = ROOT / "results/task4_discrepancy_decomposition/random_subspace_calibration.csv"
KS = (1, 2, 3, 5)
N_BOOT = 1000
N_SPLITS = 250
SEED = 42617


def basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    return vt, s * s / np.sum(s * s)


def score(v: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((b @ v) ** 2) / np.dot(v, v))


def subspace_metrics(b: np.ndarray, reference: np.ndarray) -> dict:
    singular = np.linalg.svd(b @ reference.T, compute_uv=False)
    singular = np.clip(singular, 0, 1)
    angles = np.degrees(np.arccos(singular))
    overlap = float(np.sum(singular ** 2))
    k = len(b)
    return {
        "largest_principal_angle_deg": float(angles.max()),
        "mean_principal_angle_deg": float(angles.mean()),
        "projection_similarity": overlap / k,
        "projection_frobenius_distance": float(np.sqrt(max(0, 2 * k - 2 * overlap))),
        "normalized_projection_distance": float(np.sqrt(max(0, 2 * k - 2 * overlap)) / np.sqrt(2 * k)),
    }


def load_controlled() -> tuple[np.ndarray, list[str]]:
    m = pd.read_parquet(CONTROLLED / "manifest.parquet").reset_index(drop=True)
    z = np.load(CONTROLLED / "bridgerna_embeddings.npy").astype(np.float64)
    assert len(m) == len(z)
    vectors, donors = [], []
    for donor, g in m.groupby("pair_id", sort=True):
        poly = z[g.index[g.library_prep.eq("polyA")]].mean(0)
        ribo = z[g.index[g.library_prep.eq("ribo")]].mean(0)
        vectors.append(ribo - poly); donors.append(str(donor))
    d = np.stack(vectors)
    assert d.shape == (40, 512) and len(set(donors)) == 40
    return d, donors


def load_discrepancies() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    x = np.load(TASK3_VECTORS, allow_pickle=True)
    v = dict(zip(x["names"], x["delta_z"].astype(np.float64)))
    specs = {
        "RR1": ("RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"),
        "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
        "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
    }
    delta, cosines = {}, {}
    for name, (a, b) in specs.items():
        delta[name] = v[a] - v[b]
        cosines[name] = float(np.dot(v[a], v[b]) / (np.linalg.norm(v[a]) * np.linalg.norm(v[b])))
    assert np.isclose(cosines["RR1"], -0.804119, atol=2e-5)
    assert np.isclose(cosines["RR3-39"], 0.790003, atol=2e-5)
    assert np.isclose(cosines["RR3-40"], 0.916649, atol=2e-5)
    return delta, cosines


def summarize(x: pd.DataFrame, group: list[str], value: str) -> pd.DataFrame:
    def one(g):
        a = g[value].to_numpy()
        return pd.Series({"mean": a.mean(), "median": np.median(a), "sd": a.std(ddof=1),
                          "q025": np.quantile(a, .025), "q975": np.quantile(a, .975),
                          "min": a.min(), "max": a.max(), "n": len(a)})
    return x.groupby(group, sort=False).apply(one, include_groups=False).reset_index()


def make_figures(boot_scores, stability, heldout, contributions, original):
    colors = {"RR1": "#CC3311", "RR3-39": "#0077BB", "RR3-40": "#009988"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, k in zip(axes, (1, 5)):
        vals = [boot_scores.loc[(boot_scores.comparison == name) & (boot_scores.components == k),
                                "technical_alignment_score"] for name in colors]
        bp = ax.boxplot(vals, labels=list(colors), showfliers=False, patch_artist=True)
        for box, c in zip(bp["boxes"], colors.values()): box.set_facecolor(c); box.set_alpha(.75)
        for i, name in enumerate(colors, 1): ax.scatter(i, original[(name, k)], marker="D", color="black", s=25, zorder=3)
        ax.set(title=f"k={k}", ylabel="Technical Alignment Score" if k == 1 else "", ylim=(0, 1))
    fig.suptitle("Donor-bootstrap stability of NASA discrepancy alignment\nDiamonds: full 40-donor reference")
    fig.savefig(OUT / "figure_a_bootstrap_alignment.png", dpi=300); fig.savefig(OUT / "figure_a_bootstrap_alignment.pdf"); plt.close(fig)

    ss = summarize(stability, ["components"], "projection_similarity")
    aa = summarize(stability, ["components"], "largest_principal_angle_deg")
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    ax.errorbar(ss.components, ss["median"], yerr=[ss["median"]-ss.q025, ss.q975-ss["median"]], marker="o", capsize=4, color="#4477AA")
    ax.set(xlabel="Reference dimensions (k)", ylabel="Projection-matrix similarity", ylim=(0, 1.02), xticks=KS,
           title="Bootstrap stability of the controlled subspace")
    ax2 = ax.twinx(); ax2.plot(aa.components, aa["median"], marker="s", color="#EE7733"); ax2.set_ylabel("Median largest principal angle (degrees)", color="#EE7733")
    fig.savefig(OUT / "figure_b_subspace_stability.png", dpi=300); fig.savefig(OUT / "figure_b_subspace_stability.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    hs = summarize(heldout.query("validation == 'repeated_32_8'"), ["components"], "technical_alignment_score")
    ax.errorbar(hs.components, hs["median"], yerr=[hs["median"]-hs.q025, hs.q975-hs["median"]], marker="o", capsize=4, color="#228833")
    ax.set(xlabel="Training-reference dimensions (k)", ylabel="Held-out donor Technical Alignment Score", ylim=(0, 1.02), xticks=KS,
           title="Generalization to donors excluded from basis fitting")
    fig.savefig(OUT / "figure_c_heldout_donor_alignment.png", dpi=300); fig.savefig(OUT / "figure_c_heldout_donor_alignment.pdf"); plt.close(fig)

    rr = contributions.query("comparison == 'RR1' and estimate == 'full_reference'").sort_values("component")
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    ax.bar(rr.component.astype(str), rr.component_contribution, color="#66CCEE", label="Incremental component")
    ax.plot(rr.component.astype(str), rr.cumulative_alignment, marker="o", color="#CC3311", label="Cumulative alignment")
    ax.set(xlabel="Uncentered controlled component", ylabel="Fraction of squared RR1 discrepancy", ylim=(0, 1.02),
           title="RR1 alignment beyond the dominant controlled direction")
    ax.legend()
    fig.savefig(OUT / "figure_d_rr1_cumulative_alignment.png", dpi=300); fig.savefig(OUT / "figure_d_rr1_cumulative_alignment.pdf"); plt.close(fig)


def main():
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    d, donors = load_controlled(); delta, replication_cosines = load_discrepancies()
    ref, evr = basis(d)
    rng = np.random.default_rng(SEED)
    print(f"[start] {N_BOOT} bootstraps + {N_SPLITS} 32/8 splits + 40 LODO fits; cached 40x512 input", flush=True)

    original = {(name, k): score(v, ref[:k]) for name, v in delta.items() for k in KS}
    boot_scores, stability, contrib = [], [], []
    for iteration in range(N_BOOT):
        idx = rng.integers(0, len(d), len(d))
        b, _ = basis(d[idx])
        for k in KS:
            stability.append({"iteration": iteration, "components": k, **subspace_metrics(b[:k], ref[:k])})
            for name, v in delta.items():
                boot_scores.append({"validation": "bootstrap", "iteration": iteration, "comparison": name, "components": k,
                                    "technical_alignment_score": score(v, b[:k])})
        for name, v in delta.items():
            running = 0.0
            for j in range(5):
                c = float((v @ b[j]) ** 2 / np.dot(v, v)); running += c
                contrib.append({"estimate": "bootstrap", "iteration": iteration, "comparison": name,
                                "component": j + 1, "component_contribution": c, "cumulative_alignment": running})
        if (iteration + 1) % 200 == 0:
            print(f"[heartbeat] bootstrap {iteration+1}/{N_BOOT} elapsed={time.time()-started:.1f}s", flush=True)
    boot_scores = pd.DataFrame(boot_scores); stability = pd.DataFrame(stability)

    heldout = []
    for split in range(N_SPLITS):
        test = rng.choice(len(d), 8, replace=False); train = np.setdiff1d(np.arange(len(d)), test)
        b, _ = basis(d[train])
        for i in test:
            for k in KS:
                heldout.append({"validation": "repeated_32_8", "split": split, "donor": donors[i], "components": k,
                                "technical_alignment_score": score(d[i], b[:k])})
    for i, donor in enumerate(donors):
        b, _ = basis(np.delete(d, i, axis=0))
        for k in KS:
            heldout.append({"validation": "leave_one_donor_out", "split": i, "donor": donor, "components": k,
                            "technical_alignment_score": score(d[i], b[:k])})
        for name, v in delta.items():
            for k in KS:
                boot_scores.loc[len(boot_scores)] = {"iteration": i, "comparison": name, "components": k,
                                                     "technical_alignment_score": score(v, b[:k]), "validation": "leave_one_donor_out"}
    heldout = pd.DataFrame(heldout)
    full_contrib = []
    for name, v in delta.items():
        cumulative = 0.0
        for j in range(5):
            c = float((v @ ref[j]) ** 2 / np.dot(v, v)); cumulative += c
            full_contrib.append({"estimate": "full_reference", "iteration": np.nan, "comparison": name,
                                 "component": j + 1, "component_contribution": c, "cumulative_alignment": cumulative})
    contributions = pd.concat([pd.DataFrame(full_contrib), pd.DataFrame(contrib)], ignore_index=True)

    bsum = summarize(boot_scores.query("validation == 'bootstrap'"), ["comparison", "components"], "technical_alignment_score")
    bsum["original_full_data_score"] = [original[(r.comparison, r.components)] for r in bsum.itertuples()]
    s_metrics = []
    for metric in ["largest_principal_angle_deg", "mean_principal_angle_deg", "projection_similarity", "projection_frobenius_distance", "normalized_projection_distance"]:
        q = summarize(stability, ["components"], metric); q.insert(1, "metric", metric); s_metrics.append(q)
    ssum = pd.concat(s_metrics, ignore_index=True)
    hsum = summarize(heldout, ["validation", "components"], "technical_alignment_score")
    donor_summary = summarize(heldout.query("validation == 'repeated_32_8'"), ["donor", "components"], "technical_alignment_score")
    ordering = boot_scores.query("validation == 'bootstrap'").pivot(index="iteration", columns=["comparison", "components"], values="technical_alignment_score")
    order_rows = []
    for k in KS:
        ok = (ordering[("RR1", k)] > ordering[("RR3-40", k)]) & (ordering[("RR3-40", k)] > ordering[("RR3-39", k)])
        order_rows.append({"components": k, "fraction_RR1_gt_RR3_40_gt_RR3_39": ok.mean(), "iterations": len(ok)})
    ordering_summary = pd.DataFrame(order_rows)

    # Stability of each secondary contribution is descriptive: PC2-PC5 may
    # rotate/swap, so cumulative subspace stability is the primary inference.
    csum = summarize(contributions.query("estimate == 'bootstrap'"), ["comparison", "component"], "component_contribution")
    csum = csum.merge(pd.DataFrame(full_contrib)[["comparison", "component", "component_contribution"]].rename(columns={"component_contribution": "full_reference_contribution"}), on=["comparison", "component"])

    random = pd.read_csv(PRIOR_RANDOM)
    robust = bsum.merge(ordering_summary, on="components")
    robust = robust.merge(random[["comparison", "components", "empirical_p_one_sided"]], on=["comparison", "components"], how="left")
    robust["original_replication_cosine"] = robust.comparison.map(replication_cosines)

    boot_scores.to_csv(OUT / "bootstrap_profiler_scores.csv", index=False)
    stability.to_csv(OUT / "bootstrap_subspace_stability.csv", index=False)
    heldout.to_csv(OUT / "heldout_donor_alignment.csv", index=False)
    donor_summary.to_csv(OUT / "heldout_donor_summary.csv", index=False)
    contributions.to_csv(OUT / "secondary_pc_contributions.csv", index=False)
    bsum.to_csv(OUT / "bootstrap_profiler_score_summary.csv", index=False)
    ssum.to_csv(OUT / "bootstrap_subspace_stability_summary.csv", index=False)
    hsum.to_csv(OUT / "heldout_donor_alignment_summary.csv", index=False)
    ordering_summary.to_csv(OUT / "bootstrap_ordering_stability.csv", index=False)
    csum.to_csv(OUT / "secondary_pc_contribution_summary.csv", index=False)
    robust.to_csv(OUT / "robustness_summary.csv", index=False)
    make_figures(boot_scores.query("validation == 'bootstrap'"), stability, heldout, contributions, original)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "controlled_donors": 40,
        "embedding_dimensions": 512, "bootstrap_iterations": N_BOOT, "bootstrap_sample_size": 40,
        "bootstrap_sampling": "donors with replacement, intact PolyA/Ribo pairs", "holdout_splits": N_SPLITS,
        "holdout_train_donors": 32, "holdout_test_donors": 8, "leave_one_donor_out": True,
        "components": list(KS), "random_seed": SEED, "svd": "uncentered",
        "reference_evr_first_five": evr[:5].tolist(), "embeddings_recomputed": False,
        "correction_applied": False, "prior_random_subspace_control_reused": str(PRIOR_RANDOM),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\nBootstrap profiler summary\n", bsum.to_string(index=False), flush=True)
    print("\nOrdering stability\n", ordering_summary.to_string(index=False), flush=True)
    print("\nHeld-out donor summary\n", hsum.to_string(index=False), flush=True)
    print(f"[complete] elapsed={time.time()-started:.1f}s outputs={OUT}", flush=True)


if __name__ == "__main__":
    main()
