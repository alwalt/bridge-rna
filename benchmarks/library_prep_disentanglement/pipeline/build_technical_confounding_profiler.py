#!/usr/bin/env python3
"""Build the cached-result BridgeRNA Technical Confounding Profiler prototype."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "task4_confounding_profiler"
FIG = OUT / "figures"
TASK3 = REPO / "benchmarks/osdr_batch_effect_representation/results/task3_osd168_technical_replication/technical_response_vectors.npz"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def category(r):
    if r >= .8: return "Highly reproducible"
    if r >= .5: return "Moderately reproducible"
    if r >= 0: return "Weakly reproducible"
    return "Opposing / reversed response"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    controlled_manifest = pd.read_parquet(ROOT / "work/datasets/chen_2020_tcells/manifest.parquet").reset_index(drop=True)
    controlled_z = np.load(ROOT / "work/datasets/chen_2020_tcells/bridgerna_embeddings.npy").astype(float)
    d = []
    for _, g in controlled_manifest.groupby("pair_id", sort=True):
        p = controlled_z[g.index[g.library_prep.eq("polyA")]].mean(0)
        r = controlled_z[g.index[g.library_prep.eq("ribo")]].mean(0)
        d.append(r - p)
    _, _, vt = np.linalg.svd(np.stack(d), full_matrices=False)
    pc1, pc2 = vt[:2]

    x = np.load(TASK3, allow_pickle=True)
    vectors = dict(zip(x["names"], x["delta_z"].astype(float)))
    specs = {
        "RR1": ("RR1 OSD-48 vs OSD-168", "RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"),
        "RR3-39": ("RR3 39-day OSD-137 vs OSD-168", "C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
        "RR3-40": ("RR3 40-day OSD-137 vs OSD-168", "C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
    }
    boot = pd.read_csv(RESULTS / "task4_technical_subspace_robustness/bootstrap_profiler_score_summary.csv")
    random = pd.read_csv(RESULTS / "task4_discrepancy_decomposition/random_subspace_calibration.csv")
    ig = pd.read_csv(RESULTS / "task4_gene_attribution_diagnostic/technical_replication_ig_comparison.csv").set_index("comparison")
    stability = pd.read_csv(RESULTS / "task4_technical_subspace_robustness/bootstrap_subspace_stability_summary.csv")
    heldout = pd.read_csv(RESULTS / "task4_technical_subspace_robustness/heldout_donor_alignment_summary.csv")
    correction = pd.read_csv(RESULTS / "task4_simple_correction_comparison/svd_correction_curve.csv").set_index("method")

    rows = []
    for key, (label, a, b) in specs.items():
        va, vb = vectors[a], vectors[b]; discrepancy = va - vb; denominator = np.dot(discrepancy, discrepancy)
        t1 = float((discrepancy @ pc1) ** 2 / denominator)
        t2 = float((discrepancy @ pc2) ** 2 / denominator)
        total = float(np.sum((vt[:2] @ discrepancy) ** 2) / denominator)
        assert np.isclose(total, t1 + t2, atol=1e-12)
        bs = boot.query("comparison == @key and components == 2").iloc[0]
        rs = random.query("comparison == @key and components == 2").iloc[0]
        gi = ig.loc[key]
        if key == "RR1":
            pathway = "Shared same-sign genes: small-molecule catabolism, cholesterol and related hepatic metabolism."
            interpretation = "Opposing response; discrepancy is strongly aligned with the controlled reference, with broad influential-gene reweighting rather than simple shared-gene sign reversal."
            overlap = "HIGH"
            overlap_detail = "Removing enough technical-associated structure to resolve the reversal substantially disrupts the broader spaceflight-response organization; technical and biological structure are not cleanly separable."
            protocol = "PolyA, SE 50 bp, HiSeq 3000 (OSD-48) → ribodepletion, PE 150 bp, HiSeq 4000 with no ERCC (OSD-168); UC Davis."
        else:
            pathway = "Reproducible genes: hepatic small-molecule, organic-acid, lipid and broader metabolic programs."
            interpretation = "Response remains reproducible despite detectable alignment of its smaller replication discrepancy with the controlled reference."
            protocol = "Ribodepleted, PE 150 bp, HiSeq 4000 OSD-137 → OSD-168 ERCC remeasurement; UC Davis."
            if key == "RR3-39":
                overlap = "EVIDENT"
                overlap_detail = "Strong PC1–5 removal reduces response reproducibility, while the same removal also reorganizes the broader response geometry."
            else:
                overlap = "LIMITED IN THIS COMPARISON"
                overlap_detail = "The response remains highly reproducible after PC1–5 removal, although the broader reference-removal experiment still shows nonseparability."
        rows.append({
            "Comparison": key, "Comparison Detail": label,
            "Biological System": "Mouse liver", "Perturbation": "Spaceflight",
            "Response": "Flight minus Ground Control", "Comparison Type": "Original measurement versus technical remeasurement",
            "Protocol Context": protocol,
            "Response Reproducibility": cosine(va, vb), "Response Category": category(cosine(va, vb)),
            "Technical Alignment PC1-2": total, "PC1 Contribution": t1, "PC2 Contribution": t2,
            "PC1+PC2 Verification Error": abs(total - t1 - t2),
            "Bootstrap Median": bs["median"], "Bootstrap Mean": bs["mean"], "Bootstrap SD": bs["sd"],
            "Bootstrap 95% Low": bs.q025, "Bootstrap 95% High": bs.q975,
            "Random-Subspace Percentile": rs.empirical_percentile, "Random-Subspace P": rs.empirical_p_one_sided,
            "Attribution Spearman": gi.full_signed_spearman, "Top100 Shared": int(gi.shared),
            "Shared Top100 Sign Agreement": gi.same_sign_fraction,
            "Measurement-Specific Top100 Union": int(2 * (100 - gi.shared)),
            "Biological Overlap": overlap, "Biological Overlap Evidence": overlap_detail,
            "Existing Pathway Summary": pathway, "Overall Interpretation": interpretation,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "profiler_summary.csv", index=False)

    ref = {
        "PC1_projection_similarity_median": float(stability.query("components == 1 and metric == 'projection_similarity'")["median"].iloc[0]),
        "PC1_principal_angle_median_deg": float(stability.query("components == 1 and metric == 'mean_principal_angle_deg'")["median"].iloc[0]),
        "PC1_2_projection_similarity_median": float(stability.query("components == 2 and metric == 'projection_similarity'")["median"].iloc[0]),
        "PC1_2_largest_angle_median_deg": float(stability.query("components == 2 and metric == 'largest_principal_angle_deg'")["median"].iloc[0]),
        "PC1_5_projection_similarity_median": float(stability.query("components == 5 and metric == 'projection_similarity'")["median"].iloc[0]),
        "heldout_32_8_PC1_2_median": float(heldout.query("validation == 'repeated_32_8' and components == 2")["median"].iloc[0]),
        "reference_caveat": "Learned from controlled same-RNA T-cell PolyA/Ribo measurements; cross-tissue universality has not been established.",
    }
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "operational_reference": "uncentered T-cell PolyA/Ribo PC1-2",
               "reference_robustness": ref, "profiles": summary.to_dict(orient="records"),
               "interpretation_limits": ["not a causal fraction", "not pure technical variation", "not batch correction", "residual is not pure biology"]}
    (OUT / "profiler_summary.json").write_text(json.dumps(payload, indent=2))

    # Technical Sensitivity Map: R and T are deliberately not collapsed.
    fig, ax = plt.subplots(figsize=(7.5, 5.8), layout="constrained")
    colors = {"RR1": "#CC3311", "RR3-39": "#0077BB", "RR3-40": "#009988"}
    for _, row in summary.iterrows():
        xval, yval = row["Technical Alignment PC1-2"], row["Response Reproducibility"]
        ax.scatter(xval, yval, s=150, color=colors[row["Comparison"]], edgecolor="white", linewidth=1.2)
        ax.annotate(row["Comparison"], (xval, yval), xytext=(7, 7), textcoords="offset points", fontweight="bold")
    ax.axvline(.5, color="#777777", ls="--", lw=1); ax.axhline(0, color="#777777", ls="--", lw=1)
    ax.set(xlim=(0, 1), ylim=(-1, 1), xlabel="Technical Alignment Score", ylabel="Response Reproducibility",
           title="Technical Sensitivity Map\nResponse reproducibility and technical alignment capture complementary properties")
    fig.savefig(OUT / "technical_sensitivity_map.png", dpi=300); fig.savefig(OUT / "technical_sensitivity_map.pdf")
    fig.savefig(FIG / "technical_sensitivity_map.png", dpi=300); plt.close(fig)

    # Secondary four-panel comparison retained as supporting detail.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    names = summary.Comparison.tolist(); xx = np.arange(len(names)); colors = ["#CC3311", "#0077BB", "#009988"]
    axes[0, 0].bar(xx, summary["Response Reproducibility"], color=colors); axes[0, 0].axhline(0, color="black", lw=.8)
    axes[0, 0].set(xticks=xx, xticklabels=names, ylim=(-1, 1), ylabel="Cosine", title="Response reproducibility (R)")
    axes[0, 1].bar(xx, summary["Technical Alignment PC1-2"], color=colors)
    axes[0, 1].set(xticks=xx, xticklabels=names, ylim=(0, 1), ylabel="Squared fraction", title="Technical alignment (T library)")
    axes[1, 0].bar(xx, summary["PC1 Contribution"], color="#4477AA", label="PC1")
    axes[1, 0].bar(xx, summary["PC2 Contribution"], bottom=summary["PC1 Contribution"], color="#EE7733", label="PC2")
    axes[1, 0].set(xticks=xx, xticklabels=names, ylim=(0, 1), ylabel="Squared fraction", title="Stable-reference contributions"); axes[1, 0].legend()
    axes[1, 1].bar(xx, summary["Attribution Spearman"], color=colors)
    axes[1, 1].set(xticks=xx, xticklabels=names, ylim=(0, 1), ylabel="Spearman", title="Gene-attribution reproducibility")
    fig.suptitle("BridgeRNA Technical Confounding Profiler", fontweight="bold")
    fig.savefig(FIG / "profiler_comparison.png", dpi=300); fig.savefig(FIG / "profiler_comparison.pdf"); plt.close(fig)

    filenames = {"RR1": "rr1_profile.png", "RR3-39": "rr3_39_profile.png", "RR3-40": "rr3_40_profile.png"}
    for row in summary.to_dict(orient="records"):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained"); ax.axis("off")
        text = (f"BRIDGERNA TECHNICAL CONFOUNDING PROFILE\n\n"
                f"RESPONSE CONTEXT\nMouse liver | Spaceflight | Flight minus Ground Control\n{row['Comparison Detail']}\n{row['Protocol Context']}\n\n"
                f"PRIMARY\n\nRESPONSE REPRODUCIBILITY     {row['Response Reproducibility']:.3f}\n{row['Response Category'].upper()}\n\n"
                f"TECHNICAL ALIGNMENT SCORE     {row['Technical Alignment PC1-2']:.3f}\n"
                f"{('EXTREMELY STRONGLY ALIGNED' if row['Technical Alignment PC1-2'] > .9 else 'TECHNICAL-ASSOCIATED STRUCTURE DETECTED')}\n\n"
                f"BIOLOGICAL OVERLAP     {row['Biological Overlap']}\n{row['Biological Overlap Evidence']}\n\n"
                f"OVERALL\n{row['Overall Interpretation']}\n\n"
                "* Reference learned from controlled same-RNA T-cell PolyA/Ribo experiments;\n  cross-tissue universality has not been established.")
        ax.text(.03, .97, text, va="top", ha="left", fontsize=11, linespacing=1.35, wrap=True)
        fig.savefig(FIG / filenames[row["Comparison"]], dpi=300); plt.close(fig)
    print(summary.to_string(index=False)); print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
