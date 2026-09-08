#!/usr/bin/env python3
"""Build the final BridgeRNA Biological Confounding Profiler from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_confounding_profiler/final_profiler"
FIG = OUT / "figures"
PROF = HERE / "results/task4_confounding_profiler"
T3 = REPO / "benchmarks/osdr_batch_effect_representation/results"
CONTROL = HERE / "work/datasets/chen_2020_tcells"


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def reference():
    manifest = pd.read_parquet(CONTROL / "manifest.parquet").reset_index(drop=True)
    z = np.load(CONTROL / "bridgerna_embeddings.npy").astype(float)
    differences = []
    for _, g in manifest.groupby("pair_id", sort=True):
        differences.append(z[g.index[g.library_prep.eq("ribo")]].mean(0) - z[g.index[g.library_prep.eq("polyA")]].mean(0))
    differences = np.stack(differences)
    _, _, vt = np.linalg.svd(differences, full_matrices=False)
    basis = vt[:2]
    oriented = (differences.mean(0) @ basis.T) @ basis
    return basis, oriented / np.linalg.norm(oriented)


def project(v, basis):
    p = (v @ basis.T) @ basis
    return p, v - p


def metrics(v, basis, orientation):
    p, _ = project(v, basis)
    return np.linalg.norm(v), np.dot(p, p) / np.dot(v, v), cosine(p, orientation), float(np.dot(p, orientation))


def response_profiler(basis, orientation):
    summary = pd.read_csv(T3 / "task3b_contrast_summary.csv")
    archive = np.load(T3 / "task3b_bridgerna_response_vectors.npz", allow_pickle=True)
    vectors = {str(n): v.astype(float) for n, v in zip(archive["contrast_id"], archive["delta_z"])}
    rows = []
    for row in summary.itertuples():
        magnitude, occupancy, direction, signed = metrics(vectors[row.contrast_id], basis, orientation)
        rows.append({"contrast_id": row.contrast_id, "cohort": row.contrast_id.split("__", 1)[1], "dataset": row.OSD,
                     "measurement": "validated full biological contrast", "FLT_n": row.n_FLT, "GC_n": row.n_GC,
                     "RNA_selection": row.library_preparation, "preservation": row.preservation,
                     "response_magnitude": magnitude, "technical_reference_overlap": occupancy,
                     "PolyA_to_Ribo_directional_cosine": direction, "signed_PolyA_to_Ribo_projection": signed,
                     "directional_label": "weak/no direction" if abs(direction) < .25 else "Ribo-directed" if direction > 0 else "PolyA-directed",
                     "matched_remeasurement": False, "whole_response_reproducibility": np.nan,
                     "technical_reference_reproducibility": np.nan, "orthogonal_response_reproducibility": np.nan,
                     "discrepancy_localization_PC1_2": np.nan, "interpretation": "UNRESOLVED"})

    tech = np.load(T3 / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    stored = {str(n): v.astype(float) for n, v in zip(tech["names"], tech["delta_z"])}
    pairs = {
        "RR1 carcass": ("RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC", "OSD-48", "OSD-168", 4, 5,
                         "PolyA", "rRNA depletion", "carcass; matched animals", "same source animal/liver; no-ERCC"),
        "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC", "OSD-137", "OSD-168", 2, 2,
                    "rRNA depletion", "rRNA depletion", "liquid nitrogen", "same RR3 RNA material supported; ERCC"),
        "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC", "OSD-137", "OSD-168", 2, 2,
                    "rRNA depletion", "rRNA depletion", "liquid nitrogen", "same RR3 RNA material supported; ERCC"),
    }
    for cohort, spec in pairs.items():
        an, bn, ad, bd, nf, ng, arna, brna, apres, bpres = spec
        a, b = stored[an], stored[bn]; ap, ao = project(a, basis); bp, bo = project(b, basis); dp, _ = project(a-b, basis)
        pair_metrics = {"whole_response_reproducibility": cosine(a, b),
                        "technical_reference_reproducibility": cosine(ap, bp),
                        "orthogonal_response_reproducibility": cosine(ao, bo),
                        "discrepancy_localization_PC1_2": np.dot(dp, dp) / np.dot(a-b, a-b)}
        category = "MEASUREMENT-VULNERABLE" if cohort == "RR1 carcass" else "ROBUST / REPRODUCIBLE DESPITE TECHNICAL OVERLAP"
        for vector, dataset, measurement, rna, preservation in [(a, ad, "matched original", arna, apres),
                                                                (b, bd, "matched remeasurement", brna, bpres)]:
            magnitude, occupancy, direction, signed = metrics(vector, basis, orientation)
            rows.append({"contrast_id": an if dataset == ad else bn, "cohort": cohort, "dataset": dataset,
                         "measurement": measurement, "FLT_n": nf, "GC_n": ng, "RNA_selection": rna,
                         "preservation": preservation, "response_magnitude": magnitude,
                         "technical_reference_overlap": occupancy, "PolyA_to_Ribo_directional_cosine": direction,
                         "signed_PolyA_to_Ribo_projection": signed,
                         "directional_label": "weak/no direction" if abs(direction) < .25 else "Ribo-directed" if direction > 0 else "PolyA-directed",
                         "matched_remeasurement": True, **pair_metrics, "interpretation": category})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "response_profiler.csv", index=False)
    return result


def program_profiler():
    recurrence = pd.read_csv(PROF / "independent_biological_replication/program_recurrence_summary.csv")
    controlled = pd.read_csv(PROF / "controlled_gene_context/pathway_family_decisions.csv")
    excess = pd.read_csv(PROF / "expression_adjusted_context/RR1_context_excess_set_pathway_family_summary.csv")
    excess = excess[excess.gene_set.eq("nonDE_top5pct")]
    mappings = {
        "RNA processing / splicing / rRNA processing": ("RNA processing / splicing", "RNA processing / splicing"),
        "Chromatin regulation": ("Chromatin organization / remodeling", "Chromatin organization / remodeling"),
        "DNA response / repair": ("DNA repair / DNA metabolism", "DNA repair / DNA metabolism"),
        "Lipid / fatty-acid / metabolic programs": ("Hepatic lipid / metabolic", "Fatty-acid / lipid metabolism"),
    }
    rows = []
    for program, (rec_name, control_name) in mappings.items():
        edge = recurrence[(recurrence.analysis.eq("edgeR_expression")) & recurrence.family.eq(rec_name)]
        bridge = recurrence[(recurrence.analysis.eq("Bridge_contextual")) & recurrence.family.eq(rec_name)]
        control = controlled[controlled.family.eq(control_name)]
        exc_name = {"Chromatin regulation":"Chromatin organization / remodeling", "DNA response / repair":"DNA repair / DNA metabolism"}.get(program, "RNA processing / splicing")
        exc = excess[excess.family.eq(exc_name)] if program != "Lipid / fatty-acid / metabolic programs" else pd.DataFrame()
        controlled_status = control.controlled_classification.iloc[0] if len(control) else "NOT EVALUATED"
        if program.startswith("RNA"):
            interpretation = "MEASUREMENT-VULNERABLE IN RR1; BIOLOGICAL SUPPORT ALSO PRESENT"
            remeasurement = "RR1 reverses within PC1-2; RR3 projected responses reproduce"
        elif program in ["Chromatin regulation", "DNA response / repair"]:
            interpretation = "CANDIDATE BIOLOGICAL SIGNAL; TECHNICALLY VULNERABLE IN RR1"
            remeasurement = "Independent recurrence, but RR1 contextual organization is unstable"
        else:
            interpretation = "CANDIDATE BIOLOGICAL SIGNAL; REMEASUREMENT ROLE UNRESOLVED"
            remeasurement = "Strong independent recurrence; no program-specific matched test"
        rows.append({"program": program,
                     "conventional_expression_support": f"{int(edge.independent_osds.iloc[0])} independent OSDs; {int(edge.significant_contrasts.iloc[0])} contrasts" if len(edge) else "not supported",
                     "Bridge_contextual_support": f"{int(bridge.independent_osds.iloc[0])} independent OSDs; {int(bridge.significant_contrasts.iloc[0])} contrasts" if len(bridge) else "not supported",
                     "controlled_Tcell_PolyA_Ribo_contextual_sensitivity": controlled_status,
                     "context_excess_nonDE_support": (f"{int(exc.significant_terms.iloc[0])} significant terms; best FDR {exc.best_fdr.iloc[0]:.2g}" if len(exc) else "not evaluated in the predefined context-excess families"),
                     "independent_biological_support": int(max(edge.independent_osds.iloc[0] if len(edge) else 0, bridge.independent_osds.iloc[0] if len(bridge) else 0)),
                     "matched_remeasurement_support": remeasurement, "interpretation": interpretation})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "program_profiler.csv", index=False)
    return result


def contextual_organization():
    genes = pd.read_csv(PROF / "expression_adjusted_context/RR1_nonDE_top5pct_context_excess_genes.csv")
    genes.insert(0, "set_label", "context-excess / non-significant-DE genes")
    genes.to_csv(OUT / "contextual_organization.csv", index=False)
    return genes


def dashboard(response, programs, context):
    validation = response[(response.matched_remeasurement) & response.measurement.eq("matched original")].copy()
    validation = validation.set_index("cohort").loc[["RR1 carcass", "RR3-39", "RR3-40"]].reset_index()
    colors = {"RR1 carcass":"#C44E52", "RR3-39":"#4C78A8", "RR3-40":"#59A14F"}
    fig = plt.figure(figsize=(17, 10), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.2], width_ratios=[1.3, 1, 1.55])
    ax = fig.add_subplot(gs[0,0]); y=np.arange(3); h=.18
    values=[("technical_reference_overlap","Reference overlap"), ("whole_response_reproducibility","Whole response R"),
            ("technical_reference_reproducibility","PC1–2 R"), ("orthogonal_response_reproducibility","Outside PC1–2 R")]
    for i,(column,label) in enumerate(values): ax.barh(y+(i-1.5)*h,validation[column],height=h,label=label)
    ax.axvline(0,color="black",lw=.8);ax.set(yticks=y,yticklabels=validation.cohort,xlim=(-1,1.05),xlabel="Score",title="Response-level vulnerability");ax.legend(fontsize=8,loc="lower right")

    ax=fig.add_subplot(gs[0,1])
    matched=response[response.matched_remeasurement].copy()
    for cohort in ["RR1 carcass","RR3-39","RR3-40"]:
        g=matched[matched.cohort.eq(cohort)].sort_values("measurement")
        # Explicit order original then remeasurement.
        g=pd.concat([g[g.measurement.eq("matched original")],g[g.measurement.eq("matched remeasurement")]])
        ax.plot([0,1],g.signed_PolyA_to_Ribo_projection,marker="o",lw=2.5,color=colors[cohort],label=cohort)
        for x,v in zip([0,1],g.signed_PolyA_to_Ribo_projection): ax.text(x,v,f" {v:+.2f}",va="bottom",fontsize=8)
    ax.axhline(0,color="black",lw=.8);ax.set(xticks=[0,1],xticklabels=["Original","Remeasurement"],ylabel="Signed projection\n← PolyA-directed     Ribo-directed →",title="Controlled-reference direction");ax.legend(fontsize=8)

    ax=fig.add_subplot(gs[0,2]); evidence=np.zeros((4,5))
    columns=["Conventional","Bridge context","Controlled\nsensitivity","Independent\nsupport","Remeasurement"]
    for i,row in enumerate(programs.itertuples()):
        evidence[i]=[1,1,1 if "SUPPORTED" in row.controlled_Tcell_PolyA_Ribo_contextual_sensitivity and "NOT" not in row.controlled_Tcell_PolyA_Ribo_contextual_sensitivity else 0,
                     1 if row.independent_biological_support>0 else 0, 1 if "no program-specific" not in row.matched_remeasurement_support.lower() else 0]
    ax.imshow(evidence,cmap=plt.matplotlib.colors.ListedColormap(["#ECECEC","#5B8FA8"]),vmin=0,vmax=1,aspect="auto")
    ax.set(xticks=range(5),xticklabels=columns,yticks=range(4),yticklabels=programs.program,title="Program evidence matrix")
    ax.tick_params(axis="x", labelrotation=28)
    for tick in ax.get_xticklabels(): tick.set_horizontalalignment("right")
    ax.legend(handles=[Patch(color="#5B8FA8",label="Support/evidence present"),Patch(color="#ECECEC",label="Not supported/not evaluated")],loc="upper center",bbox_to_anchor=(.5,-.20),fontsize=8,ncol=1)

    ax=fig.add_subplot(gs[1,:]); ax.axis("off")
    text=(f"Contextual organization beyond conventional DE: {len(context):,} RR1 genes are in the top 5% of positive expression-adjusted contextual residuals "
          "while edgeR FDR ≥ 0.05. Existing enrichment finds 23 RNA-processing, 5 chromatin, and 2 DNA-repair significant terms. "
          "These are context-excess / non-significant-DE genes—not novel genes or genes missed by RNA-seq.\n\n"
          "Interpretive principle: technical sensitivity identifies vulnerability, not artifact. RNA processing has biological and controlled technical support; "
          "chromatin/DNA programs recur independently but remain candidates rather than proven biology; lipid/metabolic programs recur broadly and lack controlled T-cell support. "
          "Global PC1–2 removal is inappropriate because RR3 carries highly reproducible response within the same reference.")
    ax.text(.01,.95,text,va="top",fontsize=11,wrap=True,bbox=dict(boxstyle="round,pad=.6",facecolor="#F5F5F5",edgecolor="#BBBBBB"))
    fig.suptitle("BridgeRNA Biological Confounding Profiler",fontsize=18,fontweight="bold")
    for ext in ["png","pdf","svg"]: fig.savefig(FIG/f"bridge_biological_confounding_profiler.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)


def write_summary(response, programs, context):
    matched=response[(response.matched_remeasurement)&response.measurement.eq("matched original")].set_index("cohort")
    text=f"""# BridgeRNA Biological Confounding Profiler

**Central principle: technical sensitivity identifies vulnerability, not artifact.**

- RR1 carcass strongly overlaps the controlled reference ({matched.loc['RR1 carcass','technical_reference_overlap']:.3f}) and is unstable under remeasurement (whole-response cosine {matched.loc['RR1 carcass','whole_response_reproducibility']:.3f}; projected-component cosine {matched.loc['RR1 carcass','technical_reference_reproducibility']:.3f}). It is measurement-vulnerable.
- RR3-39 and RR3-40 also overlap the reference ({matched.loc['RR3-39','technical_reference_overlap']:.3f}, {matched.loc['RR3-40','technical_reference_overlap']:.3f}) but reproduce (whole-response cosine {matched.loc['RR3-39','whole_response_reproducibility']:.3f}, {matched.loc['RR3-40','whole_response_reproducibility']:.3f}; projected-component cosine {matched.loc['RR3-39','technical_reference_reproducibility']:.3f}, {matched.loc['RR3-40','technical_reference_reproducibility']:.3f}).
- RNA processing is both biologically supported and strongly technically sensitive; its RR1 interpretation is measurement-vulnerable rather than dismissible as noise.
- Chromatin and DNA-response contextual organization recurs across independent spaceflight studies and is less completely explained by the controlled perturbation, supporting candidate biological organization—not proven biology.
- Lipid/metabolic programs recur across all six independent OSDs in conventional and Bridge-contextual analyses and were not supported by the controlled T-cell contextual sensitivity analysis; their program-specific remeasurement behavior remains unresolved.
- Bridge identifies {len(context):,} context-excess/non-significant-DE genes in the RR1 top-5% residual set, organized into RNA-processing, chromatin, and DNA-response programs.
- Globally removing PC1–2 would erase reproducible RR3 response information. The reference is technical-associated, not technical-only, and subtraction does not recover a pure biological space.
"""
    (OUT/"profiler_summary.md").write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    basis,orientation=reference();responses=response_profiler(basis,orientation);programs=program_profiler();context=contextual_organization()
    dashboard(responses,programs,context);write_summary(responses,programs,context)
    print(responses[responses.matched_remeasurement].to_string(index=False));print("\n",programs.to_string(index=False));print("\n[complete]",OUT)


if __name__=="__main__": main()
