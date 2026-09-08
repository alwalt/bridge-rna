#!/usr/bin/env python3
"""Build RR1 prep-by-flight interaction diagnostics from existing frozen assets."""
from __future__ import annotations
import json, sys
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom, spearmanr
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/rr1_library_prep_interaction"
OUT, FIG = BENCH / "results", BENCH / "results/figures"
T3 = ROOT / "benchmarks/osdr_batch_effect_representation"
T4 = ROOT / "benchmarks/library_prep_disentanglement"
FR = ROOT / "benchmarks/frozen_sample_embedding_readout"
sys.path.insert(0, str(FR / "pipeline"))
from stress_test_contextual_graphs import G, group_vector  # noqa: E402

def cosine(a, b):
    if sparse.issparse(a) or sparse.issparse(b):
        a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
        den = np.sqrt(a.multiply(a).sum() * b.multiply(b).sum())
        return float(a.multiply(b).sum() / den) if den else np.nan
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else np.nan

def norm(a): return float(np.sqrt(a.multiply(a).sum())) if sparse.issparse(a) else float(np.linalg.norm(a))

def projection_metrics(x, reference):
    if sparse.issparse(x) or sparse.issparse(reference):
        x, reference = sparse.csr_matrix(x), sparse.csr_matrix(reference)
        rr = float(reference.multiply(reference).sum()); coefficient = float(x.multiply(reference).sum() / rr)
        projected = reference * coefficient; residual = x - projected
    else:
        reference = np.asarray(reference, float); unit = reference / np.linalg.norm(reference)
        coefficient = float(np.asarray(x) @ unit); projected = coefficient * unit; residual = np.asarray(x) - projected
    return {"tcell_alignment_cosine": cosine(x, reference), "signed_projection": coefficient,
            "projected_norm": norm(projected), "residual_norm": norm(residual),
            "projected_energy_fraction": (norm(projected) / norm(x)) ** 2 if norm(x) else np.nan}, residual

def groups_and_assets():
    manifest = pd.read_csv(T3 / "results/sample_manifest.csv")
    index = dict(zip(manifest.sample_id.astype(str), range(len(manifest))))
    mapping = pd.read_csv(T4 / "results/task4_rr1_rr3_paired_technical_replication/animal_mapping.csv")
    rr1 = mapping[mapping.cohort.eq("RR1")].copy()
    groups = {}
    for measurement in ["original", "remeasure"]:
        for condition in ["FLT", "GC"]:
            samples = rr1[(rr1.measurement.eq(measurement)) & (rr1.condition.eq(condition))].sample_id
            groups[f"{measurement}_{condition}"] = np.array([index[x] for x in samples], dtype=int)
    arrays = {
        "expression": np.load(T3 / "work/bridgerna_log1p_tpm_inputs.npy", mmap_mode="r"),
        "global": np.load(T3 / "work/bridgerna_embeddings.npy", mmap_mode="r"),
        "l12_mean_sd": np.load(T4 / "work/task4_frozen_readout_selection/task3_layer12_mean_std.float32.npy", mmap_mode="r"),
        "hallmark": np.load(FR / "work/task_agnostic/task3__layer12__hallmark_module_mean_std.float16.npy", mmap_mode="r"),
    }
    return manifest, mapping, groups, arrays

def vector_effects(array, groups):
    means = {name: np.asarray(array[idx], float).mean(0) for name, idx in groups.items()}
    return {"flight_polyA": means["original_FLT"] - means["original_GC"],
            "flight_ribo": means["remeasure_FLT"] - means["remeasure_GC"],
            "prep_FLT": means["remeasure_FLT"] - means["original_FLT"],
            "prep_GC": means["remeasure_GC"] - means["original_GC"],
            "interaction": (means["remeasure_FLT"] - means["original_FLT"]) - (means["remeasure_GC"] - means["original_GC"])}

def exact_interaction_test(array, mapping, sample_index):
    """Exact condition-label permutation over the nine animal-matched prep shifts."""
    original = mapping[(mapping.cohort.eq("RR1")) & mapping.measurement.eq("original")][["animal_id", "condition", "sample_id"]]
    remeasure = mapping[(mapping.cohort.eq("RR1")) & mapping.measurement.eq("remeasure")][["animal_id", "condition", "sample_id"]]
    pairs = original.merge(remeasure, on=["animal_id", "condition"], suffixes=("_polyA", "_ribo"))
    shifts = np.stack([np.asarray(array[sample_index[r.sample_id_ribo]], float) - np.asarray(array[sample_index[r.sample_id_polyA]], float) for r in pairs.itertuples()])
    is_flt = pairs.condition.eq("FLT").to_numpy(); observed = np.linalg.norm(shifts[is_flt].mean(0) - shifts[~is_flt].mean(0))
    null = []
    for chosen in combinations(range(len(pairs)), int(is_flt.sum())):
        mask = np.zeros(len(pairs), dtype=bool); mask[list(chosen)] = True
        null.append(np.linalg.norm(shifts[mask].mean(0) - shifts[~mask].mean(0)))
    null = np.asarray(null)
    return {"interaction_exact_permutation_p": float(np.mean(null >= observed - 1e-12)),
            "interaction_null_permutations": int(len(null)), "interaction_null_median": float(np.median(null))}

def graph_effects(dataset, groups):
    base = FR / f"work/graph_fingerprint/{dataset}"
    neighbors = np.load(base / "neighbors_top20.uint16.npy", mmap_mode="r")
    weights = np.load(base / "weights_top20.float16.npy", mmap_mode="r")
    means = {name: group_vector(idx, neighbors, weights, 10, "union") for name, idx in groups.items()}
    return {"flight_polyA": means["original_FLT"] - means["original_GC"],
            "flight_ribo": means["remeasure_FLT"] - means["remeasure_GC"],
            "prep_FLT": means["remeasure_FLT"] - means["original_FLT"],
            "prep_GC": means["remeasure_GC"] - means["original_GC"],
            "interaction": (means["remeasure_FLT"] - means["original_FLT"]) - (means["remeasure_GC"] - means["original_GC"])}

def tcell_vectors(pca):
    m = pd.read_parquet(T4 / "work/datasets/chen_2020_tcells/manifest.parquet")
    raw = np.load(T4 / "work/datasets/chen_2020_tcells/log1p_tpm.npy", mmap_mode="r")
    emb = np.load(T4 / "work/datasets/chen_2020_tcells/bridgerna_embeddings.npy", mmap_mode="r")
    def delta(a): return np.asarray(a[m.library_prep.eq("ribo")], float).mean(0) - np.asarray(a[m.library_prep.eq("polyA")], float).mean(0)
    result = {"expression": delta(raw), "global": delta(emb), "pca": delta(pca.transform(raw))}
    gm = pd.read_parquet(FR / "work/graph_fingerprint/tcell/manifest.parquet")
    groups = {x: np.flatnonzero(gm.library_prep.eq(x).to_numpy()) for x in ["polyA", "ribo"]}
    base = FR / "work/graph_fingerprint/tcell"
    n = np.load(base / "neighbors_top20.uint16.npy", mmap_mode="r"); w = np.load(base / "weights_top20.float16.npy", mmap_mode="r")
    result["graph"] = group_vector(groups["ribo"], n, w, 10, "union") - group_vector(groups["polyA"], n, w, 10, "union")
    return result

def summarize(name, effects, reference=None):
    interaction = effects["interaction"]
    row = {"representation": name, "flight_response_agreement": cosine(effects["flight_polyA"], effects["flight_ribo"]),
           "prep_FLT_vs_GC_cosine": cosine(effects["prep_FLT"], effects["prep_GC"]),
           **{f"{key}_norm": norm(value) for key, value in effects.items()}}
    row["interaction_to_mean_prep_ratio"] = norm(interaction) / np.mean([norm(effects["prep_FLT"]), norm(effects["prep_GC"])])
    if reference is not None:
        flt, _ = projection_metrics(effects["prep_FLT"], reference); gc, _ = projection_metrics(effects["prep_GC"], reference)
        inter, _ = projection_metrics(interaction, reference)
        row.update({f"prep_FLT_{k}": v for k, v in flt.items()}); row.update({f"prep_GC_{k}": v for k, v in gc.items()})
        row.update({f"interaction_{k}": v for k, v in inter.items()})
        _, poly_res = projection_metrics(effects["flight_polyA"], reference)
        _, ribo_res = projection_metrics(effects["flight_ribo"], reference)
        row["flight_response_agreement_after_removal"] = cosine(poly_res, ribo_res)
    else: row["flight_response_agreement_after_removal"] = np.nan
    return row

def attribution_summary():
    path = T4 / "results/task4_gene_attribution_diagnostic/per_response_gene_rankings.parquet"
    d = pd.read_parquet(path)
    piv = d.pivot(index="response", columns="gene_symbol_human", values="signed_attribution").fillna(0)
    poly, ribo, ref = piv.loc["RR1_OSD48_original_matched"].to_numpy(), piv.loc["RR1_OSD168_no-ERCC"].to_numpy(), piv.loc["controlled_tcell_ribo_minus_polyA"].to_numpy()
    effects = {"flight_polyA": poly, "flight_ribo": ribo, "interaction": ribo-poly,
               "prep_FLT": np.full_like(poly, np.nan), "prep_GC": np.full_like(poly, np.nan)}
    met, _ = projection_metrics(effects["interaction"], ref)
    _, poly_residual = projection_metrics(poly, ref)
    _, ribo_residual = projection_metrics(ribo, ref)
    return {"representation": "attribution", "flight_response_agreement": cosine(poly, ribo),
            "interaction_norm": norm(effects["interaction"]), "interaction_tcell_alignment_cosine": met["tcell_alignment_cosine"],
            "interaction_projected_energy_fraction": met["projected_energy_fraction"],
            "flight_response_agreement_after_removal": np.nan,
            "note": "Signed input-gene IG response cosine; FLT/GC prep shifts and residualization are undefined because IG targets responses, not individual sample states. Do not confuse this with contextual-token PC1-2 projection similarity."}, d

def gene_table(expression_effects, graph_interaction, attribution):
    genes = pd.read_csv(ROOT / "data/ensembl/canonical_genes.csv").sort_values("token_id").reset_index(drop=True)
    out = genes[["gene_symbol"]].copy()
    out["polyA_FLT_minus_GC"] = expression_effects["flight_polyA"]
    out["ribo_FLT_minus_GC"] = expression_effects["flight_ribo"]
    out["expression_interaction"] = expression_effects["interaction"]
    out["sign_flip"] = np.sign(out.polyA_FLT_minus_GC) != np.sign(out.ribo_FLT_minus_GC)
    a = attribution[attribution.response.isin(["RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"])].pivot(index="gene_symbol_human", columns="response", values="signed_attribution")
    out = out.join(a, on="gene_symbol")
    codes, values = graph_interaction.indices, graph_interaction.data.astype(float)
    left, right = codes // G, codes % G
    matrix = sparse.csr_matrix((np.r_[values, values], (np.r_[left, right], np.r_[right, left])), shape=(G, G))
    out["graph_interaction_norm"] = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    out["absolute_interaction_rank"] = out.expression_interaction.abs().rank(ascending=False, method="min").astype(int)
    return out.sort_values("absolute_interaction_rank")

def enrich_interaction(genes):
    selected = set(genes.head(250).gene_symbol.str.upper()); universe = set(genes.gene_symbol.str.upper())
    sets = {}
    for p in [ROOT / "data/gsea/hallmark_gene_sets.json"]:
        sets.update({f"Hallmark: {k}": set(map(str.upper, v)) for k,v in json.loads(p.read_text()).items()})
    base = ROOT / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
    for label, fn in [("GO BP", "GO_Biological_Process_2026.gmt"), ("Reactome", "Reactome_Pathways_2024.gmt")]:
        for line in (base/fn).read_text().splitlines():
            f=line.split("\t"); sets[f"{label}: {f[0]}"]=set(map(str.upper,f[2:]))
    rows=[]
    for term,s in sets.items():
        s &= universe; hit=s&selected
        if len(s)>=5 and hit: rows.append({"term":term,"overlap":len(hit),"set_size":len(s),"p_value":hypergeom.sf(len(hit)-1,len(universe),len(s),len(selected))})
    result=pd.DataFrame(rows); result["fdr"]=multipletests(result.p_value,method="fdr_bh")[1]
    return result.sort_values(["fdr","p_value"])

def main():
    for sub in ["manifest","expression","global","hallmark","attribution","graph","tcell_reference","interaction","handling","figures","summary"]: (OUT/sub).mkdir(parents=True,exist_ok=True)
    manifest,mapping,groups,arrays=groups_and_assets(); mapping.to_csv(OUT/"manifest/exact_animal_mapping.csv",index=False)
    sample_index = dict(zip(manifest.sample_id.astype(str), range(len(manifest))))
    audit=pd.read_parquet(T4/"results/task4_rr1_rr3_sample_pc12_audit/master_sample_level_audit.parquet")
    audit[audit.cohort.str.contains("RR1")].to_csv(OUT/"manifest/rr1_metadata_audit.csv",index=False)
    # PCA is fit once on Task 3 samples, then the T-cell reference is transformed through that fixed basis.
    pca=PCA(n_components=min(arrays["expression"].shape)-1,svd_solver="full").fit(np.asarray(arrays["expression"],float))
    effects={name:vector_effects(array,groups) for name,array in arrays.items()}
    effects["pca"]=vector_effects(pca.transform(np.asarray(arrays["expression"],float)),groups)
    effects["graph"]=graph_effects("task3",groups)
    refs=tcell_vectors(pca)
    rows=[]
    for name in ["expression","pca","global","l12_mean_sd","hallmark","graph"]:
        rows.append(summarize(name,effects[name],refs.get(name)))
    for row in rows:
        if row["representation"] in arrays:
            row.update(exact_interaction_test(arrays[row["representation"]], mapping, sample_index))
        elif row["representation"] == "pca":
            row.update(exact_interaction_test(pca.transform(np.asarray(arrays["expression"],float)), mapping, sample_index))
    attr_row,attr=attribution_summary();rows.append(attr_row)
    summary=pd.DataFrame(rows);summary.to_csv(OUT/"summary/multiscale_interaction_summary.csv",index=False)
    genes=gene_table(effects["expression"],effects["graph"]["interaction"],attr);genes.to_parquet(OUT/"interaction/per_gene_interaction.parquet",index=False)
    genes.head(500).to_csv(OUT/"interaction/top500_interaction_genes.csv",index=False)
    enrich_interaction(genes).to_csv(OUT/"interaction/top250_interaction_enrichment.csv",index=False)
    rin=pd.read_csv(T4/"results/task4_rr1_rr3_sample_pc12_audit/condition_technical_interactions.csv")
    rin.to_csv(OUT/"handling/rin_and_condition_diagnostics.csv",index=False)
    pd.read_csv(T4/"results/task4_confounding_profiler/rr1_preservation_context/rr1_stratum_specific_replication_metrics.csv").to_csv(OUT/"handling/stratum_replication.csv",index=False)
    controls=pd.read_csv(FR/"results/graph_fingerprint/rr1_rr3_stress/cross_representation_comparison.csv")
    controls.to_csv(OUT/"summary/rr1_rr3_multiscale_controls.csv",index=False)
    # Critical decomposition in the controlled T-cell PC1-PC2 plane, already fixed in Task 4.
    cohort=pd.read_csv(T4/"results/task4_rr1_rr3_sample_pc12_audit/cohort_level_audit.csv")
    q=cohort[cohort.cohort.isin(["RR1 carcass","RR1 Ribo remeasurement"])]
    fig,ax=plt.subplots(figsize=(8,7))
    colors={"RR1 carcass":"#4C78A8","RR1 Ribo remeasurement":"#E45756"}
    for _,r in q.iterrows():
        ax.scatter([r.GC_mean_PC1,r.FLT_mean_PC1],[r.GC_mean_PC2,r.FLT_mean_PC2],color=colors[r.cohort],s=65)
        ax.annotate("",xy=(r.FLT_mean_PC1,r.FLT_mean_PC2),xytext=(r.GC_mean_PC1,r.GC_mean_PC2),arrowprops={"arrowstyle":"->","color":colors[r.cohort],"lw":2})
        ax.text(r.FLT_mean_PC1,r.FLT_mean_PC2,r.cohort,fontsize=8)
    ax.set(xlabel="Controlled T-cell PC1 coordinate",ylabel="Controlled T-cell PC2 coordinate",title="RR1 FLT−GC responses in the independent PolyA/Ribo-associated plane");ax.grid(alpha=.2);fig.tight_layout()
    for ext in ["png","pdf"]:fig.savefig(FIG/f"critical_decomposition.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    plot=summary[summary.representation.isin(["expression","pca","global","graph","attribution"])].copy()
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(plot));w=.36
    ax.bar(x-w/2,plot.flight_response_agreement,w,label="Before")
    ax.bar(x+w/2,plot.flight_response_agreement_after_removal,w,label="After T-cell direction removal")
    ax.set_xticks(x,plot.representation,rotation=25,ha="right");ax.axhline(0,color="black",lw=.8);ax.set(ylabel="Within-representation response similarity",title="RR1 response agreement before and after prespecified technical-direction removal");ax.legend();fig.tight_layout()
    for ext in ["png","pdf"]:fig.savefig(FIG/f"before_after_residualization.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)
    protocol=pd.read_csv(T4/"results/task4_confounding_profiler/rr1_preservation_context/rr1_protocol_comparison.csv");protocol.to_csv(OUT/"summary/protocol_comparison.csv",index=False)
    decision={"primary_conclusion":"mixed_protocol_by_flight_interaction_and_deep_RR1_instability","causal_library_claim":False,
              "reason":"FLT and GC protocol displacements differ, while OSD/library/read setup/preservation are structurally confounded",
              "rin_interaction_supported":False,"upon_euthanasia_remeasurement_available":False,"bridge_inference_rerun":False}
    (OUT/"summary/decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    print(summary[["representation","flight_response_agreement","prep_FLT_vs_GC_cosine","interaction_norm","interaction_to_mean_prep_ratio","flight_response_agreement_after_removal"]].to_string(index=False));print(json.dumps(decision,indent=2))

if __name__=="__main__":main()
