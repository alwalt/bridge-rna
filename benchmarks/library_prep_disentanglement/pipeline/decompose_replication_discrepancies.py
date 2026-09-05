#!/usr/bin/env python3
"""Diagnostic decomposition of OSDR replication discrepancies.

The reference basis is fitted only to the cached 40-donor same-RNA T-cell
PolyA-to-ribodepletion displacements. No embeddings are corrected or rewritten.
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
OUT = ROOT / "results/task4_discrepancy_decomposition"
TASK3 = REPO / "benchmarks/osdr_batch_effect_representation/results"
KS = (1, 2, 3, 5)
N_RANDOM = 1000
SEED = 42041


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else np.nan


def load_pairs(folder):
    m = pd.read_parquet(folder / "manifest.parquet").reset_index(drop=True)
    z = np.load(folder / "bridgerna_embeddings.npy").astype(float)
    ds, labels = [], []
    for pair, g in m.groupby("pair_id", sort=True):
        p = z[g.index[g.library_prep.eq("polyA")]].mean(0)
        r = z[g.index[g.library_prep.eq("ribo")]].mean(0)
        ds.append(r - p); labels.append(str(pair))
    return np.stack(ds), labels


def load_replications():
    p = TASK3 / "task3_osd168_technical_replication/technical_response_vectors.npz"
    x = np.load(p, allow_pickle=True)
    v = dict(zip(x["names"], x["delta_z"].astype(float)))
    specs = {
        "RR1": ("RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"),
        "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
        "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
    }
    return {name: (v[a], v[b]) for name, (a, b) in specs.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controlled, donors = load_pairs(ROOT / "work/datasets/chen_2020_tcells")
    _, s, vt = np.linalg.svd(controlled, full_matrices=False)  # uncentered by design
    mean = controlled.mean(0); mean /= np.linalg.norm(mean)
    reps = load_replications()
    rng = np.random.default_rng(SEED)
    detail, null = [], []
    deltas = {}
    for name, (original, remeasurement) in reps.items():
        delta = original - remeasurement
        deltas[name] = delta
        for k in KS:
            B = vt[:k]
            parallel = (delta @ B.T) @ B
            residual = delta - parallel
            detail.append({
                "comparison": name, "components": k,
                "original_replication_cosine": cosine(original, remeasurement),
                "original_replication_spearman": spearmanr(original, remeasurement).statistic,
                "discrepancy_norm": np.linalg.norm(delta),
                "parallel_norm": np.linalg.norm(parallel),
                "residual_norm": np.linalg.norm(residual),
                "fraction_squared_aligned": np.dot(parallel, parallel) / np.dot(delta, delta),
                "cosine_with_controlled_mean": cosine(delta, mean),
                "signed_PC1_loading": float(delta @ vt[0]),
                "PC1_alignment": cosine(delta, vt[0]),
            })
            for rep in range(N_RANDOM):
                Q, _ = np.linalg.qr(rng.normal(size=(delta.size, k)))
                projection = Q @ (Q.T @ delta)
                null.append({"comparison": name, "components": k, "replicate": rep,
                             "fraction_squared_aligned": np.dot(projection, projection) / np.dot(delta, delta)})
    detail = pd.DataFrame(detail)
    null = pd.DataFrame(null)
    calibration = []
    for (name, k), g in null.groupby(["comparison", "components"]):
        obs = detail.query("comparison == @name and components == @k").fraction_squared_aligned.iloc[0]
        vals = g.fraction_squared_aligned.to_numpy()
        calibration.append({"comparison": name, "components": k, "observed_fraction": obs,
                            "random_median": np.median(vals), "random_q025": np.quantile(vals, .025),
                            "random_q975": np.quantile(vals, .975),
                            "empirical_percentile": np.mean(vals <= obs),
                            "empirical_p_one_sided": (1 + np.sum(vals >= obs)) / (len(vals) + 1)})
    calibration = pd.DataFrame(calibration)

    compact = detail.pivot(index="comparison", columns="components", values="fraction_squared_aligned")
    compact.columns = [f"fraction_PC1_{k}" if k > 1 else "fraction_PC1" for k in compact.columns]
    base = detail.query("components == 1").set_index("comparison")
    compact = base[["original_replication_cosine", "discrepancy_norm", "PC1_alignment"]].join(compact)
    c5 = calibration.query("components == 5").set_index("comparison")
    compact = compact.join(c5[["empirical_percentile", "empirical_p_one_sided"]].rename(columns={
        "empirical_percentile": "random_percentile_PC1_5", "empirical_p_one_sided": "empirical_p_PC1_5"})).reset_index()

    # Additivity diagnostic. NASA old-to-new is -delta so it has the same
    # directional convention as controlled Ribo-minus-PolyA displacements.
    context = []
    for d, donor in zip(controlled, donors):
        context.append({"source": f"T-cell donor {donor}", "source_class": "T-cell donor",
                        "n_biological_sources": 1, "cosine_with_tcell_mean": cosine(d, mean),
                        "fraction_PC1_5": np.sum((vt[:5] @ d) ** 2) / np.dot(d, d)})
    external, labels = load_pairs(ROOT / "work/datasets/zhao_2018_srp127360")
    for d, label in zip(external, labels):
        context.append({"source": label, "source_class": "held-out blood/colon",
                        "n_biological_sources": 1, "cosine_with_tcell_mean": cosine(d, mean),
                        "fraction_PC1_5": np.sum((vt[:5] @ d) ** 2) / np.dot(d, d)})
    for name, d in deltas.items():
        d = -d
        context.append({"source": f"{name} original-to-remeasurement", "source_class": "NASA discrepancy",
                        "n_biological_sources": np.nan, "cosine_with_tcell_mean": cosine(d, mean),
                        "fraction_PC1_5": np.sum((vt[:5] @ d) ** 2) / np.dot(d, d)})
    context = pd.DataFrame(context)
    context_summary = context.groupby("source_class").agg(
        n_vectors=("source", "size"), mean_cosine=("cosine_with_tcell_mean", "mean"),
        median_cosine=("cosine_with_tcell_mean", "median"),
        median_fraction_PC1_5=("fraction_PC1_5", "median")).reset_index()

    preservation = pd.DataFrame([
        ["OSD-48 C13 vs C14", "polyA fixed", "Mini Cold Bag vs RLT/snap frozen", "different animals", "partial", "no", "no", "Preservation differs within RR1, but biological material is not paired."],
        ["OSD-48 C14 vs OSD-168 RR1", "polyA vs ribodepleted", "source C14 material", "exact animals/RNA-derived material", "held fixed", "library-associated protocol transition", "no", "Library changes with read layout/depth and sequencing workflow; not an isolated causal library effect."],
        ["OSD-168 RR1 no-ERCC vs ERCC", "ribodepleted fixed", "fixed", "same source animals", "no", "no", "no", "Identifies ERCC-associated remeasurement only."],
        ["OSD-137 vs OSD-168 RR3", "ribodepleted fixed", "liquid nitrogen", "exact source material", "no", "no", "no", "Technical resequencing control; no library or preservation contrast."],
        ["Across Lai Polo/OSDR studies", "study-associated", "study-associated", "unpaired", "confounded", "confounded", "not identifiable", "Mission, strain, duration, collection, library, and sequencing differ together."],
    ], columns=["comparison", "library_selection", "preservation", "biological_correspondence",
                "preservation_effect_identifiable", "library_effect_identifiable",
                "interaction_identifiable", "limitation"])

    detail.to_csv(OUT / "discrepancy_decomposition.csv", index=False)
    null.to_parquet(OUT / "random_subspace_null.parquet", index=False)
    calibration.to_csv(OUT / "random_subspace_calibration.csv", index=False)
    compact.to_csv(OUT / "rr1_rr3_compact_comparison.csv", index=False)
    context.to_csv(OUT / "additive_vector_context_alignment.csv", index=False)
    context_summary.to_csv(OUT / "additive_vector_context_summary.csv", index=False)
    preservation.to_csv(OUT / "preservation_library_interaction_feasibility.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    piv = detail.pivot(index="comparison", columns="components", values="fraction_squared_aligned")
    piv.plot.bar(ax=ax, color=["#4477AA", "#66CCEE", "#228833", "#CCBB44"])
    ax.set(ylabel="Fraction of squared discrepancy aligned", xlabel="", ylim=(0, 1),
           title="Alignment with independent T-cell library-associated basis")
    ax.legend(title="Uncentered PCs", ncol=4); fig.savefig(OUT / "discrepancy_alignment.png", dpi=300); fig.savefig(OUT / "discrepancy_alignment.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained", sharey=True)
    for ax, name in zip(axes, reps):
        vals = null.query("comparison == @name and components == 5").fraction_squared_aligned
        obs = calibration.query("comparison == @name and components == 5").observed_fraction.iloc[0]
        ax.hist(vals, bins=30, color="#BBBBBB"); ax.axvline(obs, color="#CC3311", lw=2)
        ax.set(title=name, xlabel="Squared fraction in random/controlled 5-D subspace")
    axes[0].set_ylabel("Random subspaces")
    fig.savefig(OUT / "random_subspace_calibration.png", dpi=300); fig.savefig(OUT / "random_subspace_calibration.pdf")
    plt.close(fig)

    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "random_seed": SEED,
                  "random_subspaces_per_k_per_comparison": N_RANDOM,
                  "basis": "uncentered SVD of 40 same-RNA T-cell Ribo-minus-PolyA displacements",
                  "NASA_discrepancy": "original FLT-GC response minus remeasurement FLT-GC response",
                  "correction_applied": False, "embeddings_modified": False}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\nReplication discrepancy summary\n", compact.to_string(index=False))
    print("\nContext alignment\n", context[context.source_class.ne("T-cell donor")].to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
