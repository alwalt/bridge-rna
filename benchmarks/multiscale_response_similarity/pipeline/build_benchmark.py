#!/usr/bin/env python3
"""Integrate existing response results into auditable multiscale profiles."""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/multiscale_response_similarity"
RESULTS = BENCH / "results"
FIG = RESULTS / "figures"
FROZEN = ROOT / "benchmarks/frozen_sample_embedding_readout/results"
EXERCISE = ROOT / "benchmarks/cross_species_exercise_response/results"
TASK4 = ROOT / "benchmarks/library_prep_disentanglement/results"
SCALES = ["expression", "global_bridgerna", "hallmark_program", "attribution", "contextual_graph"]

PATTERN = {
    "GSE108643": "A", "GSE86931": "A", "GSE126962": "A", "GSE132520": "A",
    "GSE71972": "B", "GSE87748": "B", "GSE97718": "B", "GSE151066": "intermediate",
}

def matrix_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str).str.replace(r"^(human|mouse)_", "", regex=True)
    frame.columns = frame.columns.astype(str).str.replace(r"^(human|mouse)_", "", regex=True)
    return frame

def lookup_pairwise(path: Path, metric: str = "cosine") -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path)
    if "metric" in frame: frame = frame[frame.metric.eq(metric)]
    result = {}
    for row in frame.itertuples():
        result[tuple(sorted((str(row.GSE_1), str(row.GSE_2))))] = float(row.similarity)
    return result

def cosine_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(float)
    norms = np.linalg.norm(values, axis=1)
    sim = values @ values.T / np.outer(norms, norms)
    return pd.DataFrame(sim, index=frame.index, columns=frame.index)

def build_rr_profiles() -> pd.DataFrame:
    stress = pd.read_csv(FROZEN / "graph_fingerprint/rr1_rr3_stress/cross_representation_comparison.csv")
    name = {"Raw expression": "expression", "Bridge mean": "global_bridgerna",
            "Hallmark mean+SD": "hallmark_program", "Graph signed-edge response": "contextual_graph"}
    wide = stress[stress.representation.isin(name)].assign(
        scale=lambda x: x.representation.map(name)).pivot(index="comparison", columns="scale", values="similarity")
    attr_true = pd.read_csv(TASK4 / "task4_rr1_rr3_paired_technical_replication/attribution_replication.csv").set_index("cohort")
    attr_pairs = pd.read_csv(TASK4 / "task4_attribution_vs_expression/reproducibility_pair_scores.csv")
    false = attr_pairs[(attr_pairs.a.eq("RR1 OSD48")) & (attr_pairs.b.eq("RR3-39 OSD137"))].attribution_similarity.iloc[0]
    rows = []
    metadata = {
        "RR1": ("technically_sensitive_remeasurement", "RR1 OSD-48 original ↔ OSD-168 remeasurement"),
        "RR3-39": ("true_technical_remeasurement", "RR3-39 OSD-137 original ↔ OSD-168 remeasurement"),
        "RR3-40": ("true_technical_remeasurement", "RR3-40 OSD-137 original ↔ OSD-168 remeasurement"),
        "false_friend": ("unrelated_false_friend", "RR1 original ↔ RR3-39 original"),
    }
    for comparison, (relationship, label) in metadata.items():
        row = {"comparison": label, "short_name": comparison, "domain": "spaceflight", "relationship_type": relationship}
        row.update(wide.loc[comparison].to_dict())
        row["attribution"] = float(false if comparison == "false_friend" else attr_true.loc[comparison, "attribution_cosine"])
        row["attribution_definition"] = "full signed paired-response IG cosine"
        rows.append(row)
    return pd.DataFrame(rows)

def build_exercise_profiles() -> pd.DataFrame:
    contrasts = pd.read_csv(EXERCISE / "response_contrasts.csv")
    ids = contrasts.GSE.astype(str).tolist()
    expression = lookup_pairwise(EXERCISE / "de_response_geometry/de_pairwise_similarities.csv")
    global_bridge = lookup_pairwise(EXERCISE / "de_response_geometry/bridgerna_pairwise_similarities.csv")
    hallmark = cosine_rows(pd.read_parquet(EXERCISE / "hallmark_response_axes/predicted_hallmark_deltas.parquet"))
    graph = matrix_csv(FROZEN / "graph_fingerprint/general_validation/perturbation_graph_response_similarity.csv")
    top = pd.read_parquet(EXERCISE / "latent_axis_attribution/study_top_attributed_genes.parquet")
    attr = top.pivot(index="GSE", columns="gene", values="attribution_change").fillna(0.0)
    attr_sim = cosine_rows(attr)
    rows = []
    for a, b in combinations(ids, 2):
        pa, pb = PATTERN[a], PATTERN[b]
        if pa == pb and pa in {"A", "B"}: rel = "biologically_related_independent"
        elif "intermediate" in {pa, pb}: rel = "ambiguous_intermediate"
        else: rel = "prespecified_cross_pattern_control"
        key = tuple(sorted((a, b)))
        rows.append({
            "comparison": f"{a} ↔ {b}", "short_name": f"{a}↔{b}", "domain": "exercise",
            "relationship_type": rel, "expression": expression.get(key, np.nan),
            "global_bridgerna": global_bridge.get(key, np.nan),
            "hallmark_program": hallmark.loc[a, b], "attribution": attr_sim.loc[a, b],
            "contextual_graph": graph.loc[a, b],
            "attribution_definition": "signed Top-250 IG response cosine (existing truncated rankings)",
        })
    return pd.DataFrame(rows)

def classify(row: pd.Series) -> str:
    vals = row[SCALES].astype(float)
    ranks = row[[f"z_{s}" for s in SCALES]].astype(float)
    if row.relationship_type == "unrelated_false_friend" and vals.global_bridgerna > .75 and vals.contextual_graph < .05:
        return "global_convergence / molecular_divergence"
    if row.relationship_type == "true_technical_remeasurement" and np.nanmin(vals) > 0:
        return "multiscale_agreement"
    if ranks.z_hallmark_program > 0.5 and ranks.z_attribution < 0: return "program_level_convergence"
    if ranks.z_global_bridgerna < 0 and max(ranks.z_attribution, ranks.z_contextual_graph) > 0: return "molecular_agreement / global_distortion"
    return "multiscale_disagreement_or_ambiguous"

def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Percentile normalization is strictly within each domain and scale.
    for domain, idx in out.groupby("domain").groups.items():
        for scale in SCALES:
            values = out.loc[idx, scale]
            valid = values.notna()
            out.loc[np.array(idx)[valid.to_numpy()], f"z_{scale}"] = rankdata(values[valid], method="average") / (valid.sum() + 1) * 2 - 1
    out["interpretation"] = out.apply(classify, axis=1)
    return out

def discrimination(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame.relationship_type.isin(["true_technical_remeasurement", "unrelated_false_friend",
                                                   "biologically_related_independent", "prespecified_cross_pattern_control"])].copy()
    eligible["positive"] = eligible.relationship_type.isin(["true_technical_remeasurement", "biologically_related_independent"]).astype(int)
    rows = []
    score_sets = {s: [f"z_{s}"] for s in SCALES}
    score_sets.update({"global+attribution": ["z_global_bridgerna", "z_attribution"],
                       "global+hallmark": ["z_global_bridgerna", "z_hallmark_program"],
                       "global+graph": ["z_global_bridgerna", "z_contextual_graph"],
                       "expression+attribution": ["z_expression", "z_attribution"],
                       "all_five": [f"z_{s}" for s in SCALES]})
    for label, columns in score_sets.items():
        subset = eligible.dropna(subset=columns)
        score = subset[columns].mean(axis=1)
        rows.append({"score": label, "pairs": len(subset), "positives": int(subset.positive.sum()),
                     "auroc": roc_auc_score(subset.positive, score), "auprc": average_precision_score(subset.positive, score),
                     "scope": "descriptive; RR stress cases and prespecified exercise groupings"})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)

def figures(frame: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    selected = frame[(frame.domain.eq("spaceflight")) | (frame.relationship_type.ne("ambiguous_intermediate"))].copy()
    selected = selected.sort_values(["domain", "relationship_type", "comparison"])
    mat = selected[[f"z_{s}" for s in SCALES]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10, max(5, .34 * len(selected))))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5), ["Expression", "Global", "Hallmark", "Attribution", "Graph"], rotation=30, ha="right")
    ax.set_yticks(range(len(selected)), selected.short_name, fontsize=8)
    ax.set_title("Multiscale response profiles (within-domain percentile scale)")
    fig.colorbar(im, ax=ax, label="Relative similarity within domain")
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"multiscale_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"true_technical_remeasurement": "#238B45", "technically_sensitive_remeasurement": "#D95F0E",
              "unrelated_false_friend": "#6A51A3", "biologically_related_independent": "#3182BD",
              "prespecified_cross_pattern_control": "#969696", "ambiguous_intermediate": "#D9D9D9"}
    for rel, group in frame.groupby("relationship_type"):
        ax.scatter(group.global_bridgerna, group.contextual_graph, label=rel.replace("_", " "), color=colors[rel], s=55, alpha=.85)
    for _, row in frame[frame.domain.eq("spaceflight")].iterrows():
        ax.annotate(row.short_name, (row.global_bridgerna, row.contextual_graph), xytext=(4,4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7); ax.grid(alpha=.2)
    ax.set(xlabel="Global BridgeRNA response cosine", ylabel="Contextual graph response similarity", title="Global versus gene-relational agreement")
    ax.legend(fontsize=7, loc="best"); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"global_vs_graph.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    stress = frame[frame.domain.eq("spaceflight")].copy()
    angles = np.linspace(0, 2 * np.pi, len(SCALES), endpoint=False).tolist(); angles += angles[:1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), subplot_kw={"polar": True})
    for ax, (_, row) in zip(axes.ravel(), stress.iterrows()):
        values = [float(row[f"z_{s}"]) for s in SCALES]; values += values[:1]
        ax.plot(angles, values, color="#2878B5", lw=2); ax.fill(angles, values, color="#2878B5", alpha=.18)
        ax.set_xticks(angles[:-1], ["Expression", "Global", "Hallmark", "Attribution", "Graph"], fontsize=8)
        ax.set_ylim(-1, 1); ax.set_yticks([-.5, 0, .5, 1]); ax.set_yticklabels([])
        ax.set_title(row.short_name, fontsize=10, pad=18)
    fig.suptitle("RR1/RR3 multiscale profiles\n(relative ranks within the four stress comparisons)", y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"rr1_rr3_profiles.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    for sub in ["representations", "pairwise_profiles", "technical_cases", "biological_cases", "false_friends", "discrimination", "figures", "summary"]:
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    profiles = normalize(pd.concat([build_rr_profiles(), build_exercise_profiles()], ignore_index=True))
    profiles.to_csv(RESULTS / "pairwise_profiles/multiscale_pairwise_profiles.csv", index=False)
    profiles[profiles.domain.eq("spaceflight")].to_csv(RESULTS / "technical_cases/rr1_rr3_profiles.csv", index=False)
    profiles[profiles.domain.eq("exercise")].to_csv(RESULTS / "biological_cases/exercise_profiles.csv", index=False)
    profiles[profiles.interpretation.str.contains("global_convergence")].to_csv(RESULTS / "false_friends/multiscale_discordant_pairs.csv", index=False)
    disc = discrimination(profiles); disc.to_csv(RESULTS / "discrimination/descriptive_discrimination.csv", index=False)
    coverage = pd.DataFrame([
        {"dataset": "RR1/RR3", "unit": "response-pair", "profiles": 4, "five_scale_compatible": 4, "role": "held-out stress cases"},
        {"dataset": "exercise", "unit": "response-pair", "profiles": 28, "five_scale_compatible": 28, "role": "prespecified biological relationships"},
        {"dataset": "controlled T-cell PolyA/Ribo", "unit": "single controlled technical response", "profiles": 1, "five_scale_compatible": 0, "role": "technical reference; not a response-pair"},
        {"dataset": "ARCHS4/recount3", "unit": "same-sample processing pair", "profiles": 574, "five_scale_compatible": 0, "role": "sample-level processing control; no treatment-control response"},
        {"dataset": "other Task 3 spaceflight", "unit": "biological response", "profiles": 10, "five_scale_compatible": 0, "role": "missing matched Hallmark/IG/graph components"},
    ])
    coverage.to_csv(RESULTS / "summary/compatibility_inventory.csv", index=False)
    primary = profiles[profiles.domain.eq("spaceflight")][["comparison", "relationship_type", *SCALES, "interpretation"]]
    primary.to_csv(RESULTS / "summary/primary_table.csv", index=False)
    figures(profiles)
    prov = {"source_only": True, "bridge_inference_rerun": False, "scales": SCALES,
            "normalization_for_visualization": "within-domain percentile rank mapped to [-1,1]",
            "raw_values_preserved": True,
            "limitations": ["exercise attribution uses saved signed Top-250 IG vectors", "T-cell and recount3 controls are not response-pair compatible", "discrimination is descriptive due to small/non-independent positive sets"]}
    (RESULTS / "summary/provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(primary.to_string(index=False)); print("\n", disc.to_string(index=False)); print("\nNo BridgeRNA inference was run.")

if __name__ == "__main__": main()
