#!/usr/bin/env python3
"""Orient existing spaceflight responses in the controlled PolyA/Ribo PC1-2 plane."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_confounding_profiler/polya_ribo_directionality"
FIG = OUT / "figures"
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
R3 = T3 / "results"
CONTROL = HERE / "work/datasets/chen_2020_tcells"
PRESERVATION = HERE / "results/task4_confounding_profiler/rr1_preservation_context"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def controlled_reference():
    manifest = pd.read_parquet(CONTROL / "manifest.parquet").reset_index(drop=True)
    z = np.load(CONTROL / "bridgerna_embeddings.npy").astype(float)
    differences = []
    for _, group in manifest.groupby("pair_id", sort=True):
        poly = z[group.index[group.library_prep.eq("polyA")]].mean(0)
        ribo = z[group.index[group.library_prep.eq("ribo")]].mean(0)
        differences.append(ribo - poly)
    differences = np.stack(differences)
    _, singular, vt = np.linalg.svd(differences, full_matrices=False)
    basis = vt[:2]
    mean_displacement = differences.mean(0)
    oriented_reference = (mean_displacement @ basis.T) @ basis
    unit_reference = oriented_reference / np.linalg.norm(oriented_reference)
    return basis, mean_displacement, oriented_reference, unit_reference, singular**2 / np.sum(singular**2)


def load_responses():
    # Four full biological RR1 strata.
    full = np.load(PRESERVATION / "rr1_response_vectors.npz")
    vectors = {
        "RR1 CASIS 21d": ("OSD-47", "original full stratum", full["C11__OSD-47__RR1-CASIS__21-day"], 2, 2, "PolyA", "on-orbit dissection reported for FLT; Mini Cold Bag"),
        "RR1 CASIS 22d": ("OSD-47", "original full stratum", full["C12__OSD-47__RR1-CASIS__22-day"], 1, 1, "PolyA", "on-orbit dissection reported for FLT; Mini Cold Bag"),
        "RR1 NASA upon-euthanasia": ("OSD-48", "original full stratum", full["C13__OSD-48__RR1-NASA__37-day"], 2, 2, "PolyA", "upon euthanasia; Mini Cold Bag to -80C"),
        "RR1 NASA carcass full": ("OSD-48", "original full stratum", full["C14__OSD-48__RR1-NASA__37-day"], 5, 5, "PolyA", "carcass; RLT homogenized and snap frozen"),
    }
    # Exact matched original/remeasurement response definitions from Task 3.
    archive = np.load(R3 / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    stored = {str(n): v.astype(float) for n, v in zip(archive["names"], archive["delta_z"])}
    vectors.update({
        "RR1 carcass matched|OSD-48": ("OSD-48", "matched original", stored["RR1_OSD48_original_matched"], 4, 5, "PolyA", "carcass; exact animals with OSD-168 counterpart"),
        "RR1 carcass matched|OSD-168": ("OSD-168", "no-ERCC remeasurement", stored["RR1_OSD168_no-ERCC"], 4, 5, "rRNA depletion", "same source animal/liver material; identical RNA aliquot not established"),
        "RR3-39|OSD-137": ("OSD-137", "matched original", stored["C01_OSD137_original_matched"], 2, 2, "rRNA depletion", "liquid nitrogen"),
        "RR3-39|OSD-168": ("OSD-168", "all-ERCC remeasurement", stored["C01_OSD168_all_ERCC"], 2, 2, "rRNA depletion", "same RR3 RNA material supported; ERCC"),
        "RR3-40|OSD-137": ("OSD-137", "matched original", stored["C02_OSD137_original_matched"], 2, 2, "rRNA depletion", "liquid nitrogen"),
        "RR3-40|OSD-168": ("OSD-168", "all-ERCC remeasurement", stored["C02_OSD168_all_ERCC"], 2, 2, "rRNA depletion", "same RR3 RNA material supported; ERCC"),
    })
    return vectors


def analyze(vectors, basis, unit_reference):
    rows = []
    for cohort, (dataset, measurement, response, nf, ng, selection, preservation) in vectors.items():
        coordinates = response @ basis.T
        projected = coordinates @ basis
        occupancy = np.dot(projected, projected) / np.dot(response, response)
        direction = cosine(projected, unit_reference)
        magnitude = float(np.dot(projected, unit_reference))
        # Descriptive visual category only, deliberately not a statistical test.
        label = "weak/no direction" if abs(direction) < .25 else "Ribo-directed" if direction > 0 else "PolyA-directed"
        rows.append({"cohort": cohort.split("|")[0], "dataset": dataset, "measurement": measurement,
                     "FLT_n": nf, "GC_n": ng, "RNA_selection": selection, "preservation": preservation,
                     "PC1_2_occupancy": occupancy, "PolyA_to_Ribo_cosine": direction,
                     "signed_PolyA_to_Ribo_projection": magnitude, "directional_label": label,
                     "PC1_coordinate_arbitrary_sign": coordinates[0], "PC2_coordinate_arbitrary_sign": coordinates[1]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "spaceflight_polya_ribo_directionality.csv", index=False)
    return table


def paired_summary(table):
    pairs = {
        "RR1 carcass": ("RR1 carcass matched", "OSD-48", "OSD-168"),
        "RR3-39": ("RR3-39", "OSD-137", "OSD-168"),
        "RR3-40": ("RR3-40", "OSD-137", "OSD-168"),
    }
    rows = []
    for pair, (cohort, original_dataset, remeasured_dataset) in pairs.items():
        a = table[(table.cohort.eq(cohort)) & table.dataset.eq(original_dataset)].iloc[0]
        b = table[(table.cohort.eq(cohort)) & table.dataset.eq(remeasured_dataset)].iloc[0]
        rows.append({"comparison": pair, "original_dataset": original_dataset, "remeasured_dataset": remeasured_dataset,
                     "original_direction_cosine": a.PolyA_to_Ribo_cosine, "remeasured_direction_cosine": b.PolyA_to_Ribo_cosine,
                     "original_signed_projection": a.signed_PolyA_to_Ribo_projection,
                     "remeasured_signed_projection": b.signed_PolyA_to_Ribo_projection,
                     "original_label": a.directional_label, "remeasured_label": b.directional_label,
                     "direction_switched": np.sign(a.signed_PolyA_to_Ribo_projection) != np.sign(b.signed_PolyA_to_Ribo_projection)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "matched_remeasurement_directionality.csv", index=False)
    return result


def plot_plane(table, reference_coordinates):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    scale = max(np.linalg.norm(table[["PC1_coordinate_arbitrary_sign", "PC2_coordinate_arbitrary_sign"]], axis=1).max(), 1e-8)
    ref = reference_coordinates / np.linalg.norm(reference_coordinates) * scale * .8
    ax.arrow(0, 0, ref[0], ref[1], width=scale*.008, head_width=scale*.06, color="black", length_includes_head=True)
    ax.text(ref[0]*1.05, ref[1]*1.05, "Toward Ribo\n(controlled mean)", fontweight="bold")
    colors = {"RR1 carcass matched": "#E45756", "RR3-39": "#4C78A8", "RR3-40": "#59A14F"}
    for row in table.itertuples():
        x, y = row.PC1_coordinate_arbitrary_sign, row.PC2_coordinate_arbitrary_sign
        color = colors.get(row.cohort, "#9C755F")
        marker = "s" if row.dataset == "OSD-168" else "o"
        ax.scatter(x, y, s=75, marker=marker, color=color, edgecolor="white", linewidth=.7, zorder=3)
        ax.text(x+scale*.018, y+scale*.018, f"{row.cohort}\n{row.dataset}", fontsize=8)
    for cohort in colors:
        g = table[table.cohort.eq(cohort)]
        if len(g) == 2:
            ax.plot(g.PC1_coordinate_arbitrary_sign, g.PC2_coordinate_arbitrary_sign, color=colors[cohort], lw=1.8, alpha=.75)
    ax.axhline(0, color="gray", lw=.7); ax.axvline(0, color="gray", lw=.7)
    ax.set(xlabel="Existing PC1 coordinate (arbitrary SVD sign)", ylabel="Existing PC2 coordinate (arbitrary SVD sign)",
           title="Spaceflight FLT−GC responses in the controlled T-cell PolyA/Ribo-associated PC1–2 plane")
    ax.text(.01, -.12, "Circle: original measurement   Square: OSD-168 remeasurement\nArrow orientation is defined by mean(z_Ribo − z_PolyA), not PCA sign.", transform=ax.transAxes, fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"spaceflight_polya_ribo_directionality_plane.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    b, mean_d, projected_d, unit_d, evr = controlled_reference()
    table = analyze(load_responses(), b, unit_d)
    paired = paired_summary(table)
    plot_plane(table, projected_d @ b.T)
    summary = {"reference": "unchanged uncentered controlled T-cell PolyA/Ribo PC1-2",
               "orientation": "projection of mean(z_Ribo-z_PolyA)", "controlled_donors": 40,
               "PC1_2_controlled_displacement_variance": float(evr[:2].sum()),
               "directional_label_rule": "descriptive only: abs(cosine)<0.25 weak/no direction; otherwise sign defines Ribo/PolyA",
               "interpretation_caveat": "Direction and occupancy indicate relationship to a controlled associated reference, not causality or technical artifact."}
    (OUT / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table.to_string(index=False)); print("\nMatched remeasurements\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
