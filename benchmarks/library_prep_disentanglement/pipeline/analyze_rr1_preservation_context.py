#!/usr/bin/env python3
"""RR1 preservation-context diagnostic using frozen, existing Task 3/4 assets."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_confounding_profiler/rr1_preservation_context"
FIG = OUT / "figures"
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
R3, W3 = T3 / "results", T3 / "work"
CONTROL = HERE / "work/datasets/chen_2020_tcells"
FULLVOC = HERE / "results/task4_full_vs_bridge_vocab_expression"
BIO = HERE / "results/task4_confounding_profiler/independent_biological_replication"
ROBUST = HERE / "results/task4_response_robustness"
GMT_ROOT = REPO / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMT = {"GO:BP": "GO_Biological_Process_2026.gmt", "KEGG": "KEGG_2026.gmt", "REAC": "Reactome_Pathways_2024.gmt"}
SEED = 47148

import sys
sys.path[:0] = [str(REPO / "benchmarks/tcga_downstream/pipeline"), str(REPO)]
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes

GENES = np.asarray(load_canonical_genes(REPO / "data/ensembl/canonical_genes.csv"))
SPECS = {
    "C11__OSD-47__RR1-CASIS__21-day": "OSD-47 CASIS, 21 day",
    "C12__OSD-47__RR1-CASIS__22-day": "OSD-47 CASIS, 22 day",
    "C13__OSD-48__RR1-NASA__37-day": "OSD-48 NASA, upon euthanasia",
    "C14__OSD-48__RR1-NASA__37-day": "OSD-48 NASA, carcass",
}


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def basis():
    m = pd.read_parquet(CONTROL / "manifest.parquet").reset_index(drop=True)
    z = np.load(CONTROL / "bridgerna_embeddings.npy").astype(float)
    d = []
    for _, g in m.groupby("pair_id", sort=True):
        d.append(z[g.index[g.library_prep.eq("ribo")]].mean(0) - z[g.index[g.library_prep.eq("polyA")]].mean(0))
    _, s, vt = np.linalg.svd(np.stack(d), full_matrices=False)
    return vt, s**2 / np.sum(s**2)


def project(v, b):
    p = (v @ b.T) @ b
    return p, v - p


def load_design():
    membership = pd.read_csv(R3 / "task3b_contrast_sample_membership.csv")
    membership = membership[membership.contrast_id.isin(SPECS)].copy()
    manifest = pd.read_csv(R3 / "sample_manifest.csv")
    membership = membership.merge(manifest, on=["sample_id", "OSD", "condition"], validate="one_to_one")
    idx = dict(zip(manifest.sample_id, range(len(manifest))))
    membership["embedding_index"] = membership.sample_id.map(idx)
    z = np.load(W3 / "bridgerna_embeddings.npy").astype(float)
    x = np.load(W3 / "bridgerna_log1p_tpm_inputs.npy", mmap_mode="r")
    return membership, manifest, x, z


def responses(m, z, b):
    rows, vectors = [], {}
    for cid, g in m.groupby("contrast_id", sort=False):
        f = g.loc[g.condition.eq("FLT"), "embedding_index"].to_numpy(int)
        c = g.loc[g.condition.eq("GC"), "embedding_index"].to_numpy(int)
        v = z[f].mean(0) - z[c].mean(0); p, o = project(v, b)
        vectors[cid] = v
        rows.append({"contrast_id": cid, "label": SPECS[cid], "OSD": g.OSD.iloc[0], "n_FLT": len(f), "n_GC": len(c),
                     "underpowered": len(f) < 2 or len(c) < 2, "total_norm": np.linalg.norm(v),
                     "projected_norm": np.linalg.norm(p), "orthogonal_norm": np.linalg.norm(o),
                     "aligned_fraction": np.dot(p, p) / np.dot(v, v)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rr1_response_decomposition.csv", index=False)
    np.savez_compressed(OUT / "rr1_response_vectors.npz", **vectors)
    return out, vectors


def protocol_audit(m):
    fields = {
        "animal_id": "id.sample name", "group_api": "study.factor value.spaceflight",
        "flight_duration_api": "study.parameter value.duration", "tissue_api": "study.characteristics.material type",
        "sample_preservation_api": "study.parameter value.sample preservation method",
        "carcass_preservation_api": "study.parameter value.carcass preservation method",
        "dissection_condition_api": "study.factor value.dissection condition",
        "library_selection_api": "assay.parameter value.library selection", "library_kit_api": "assay.parameter value.library kit",
        "read_layout_api": "assay.parameter value.library layout", "read_length_api": "assay.parameter value.read length",
        "platform_api": "assay.parameter value.sequencing instrument",
    }
    rows = []
    for osd in ["OSD-47", "OSD-48"]:
        api = pd.read_csv(R3 / f"task3_manifest_api_validation/api_{osd}_sample_metadata.csv")
        # One authoritative RNA-seq assay row per Task 3 sample; exclude proteomics/WGBS rows.
        rna = api[api["investigation.study assays.study assay measurement type"].astype(str).str.contains("transcription profiling", case=False, na=False)].copy()
        if rna.empty:
            rna = api[api["assay.parameter value.library selection"].notna()].copy()
        for row in m[m.OSD.eq(osd)].itertuples():
            hit = rna[rna["id.sample name"].eq(row.sample_id)]
            if hit.empty: hit = rna[rna["assay.sample name"].eq(row.sample_id)]
            if len(hit) != 1: raise ValueError(f"Expected one RNA-seq metadata row for {row.sample_id}, got {len(hit)}")
            a = hit.iloc[0]
            item = {"contrast_id": row.contrast_id, "sample_id": row.sample_id, "condition": row.condition,
                    "mission": row.mission, "strain": row.strain, "sex": row.sex, "age_at_launch": row.age_at_launch,
                    "sequencing_center": row.sequencing_facility}
            item.update({k: a.get(v, np.nan) for k, v in fields.items()})
            if osd == "OSD-47":
                item["dissected_before_freezing"] = "reported for FLT at study-protocol level; GC followed matched processing protocol"
                item["dissection_location"] = "ISS/on orbit for FLT; ground for GC (study-protocol level)"
                item["preservation_context"] = "liver dissected, then frozen in Mini Cold Bag"
            else:
                dis = str(item["dissection_condition_api"])
                item["dissected_before_freezing"] = "yes" if dis == "Upon euthanasia" else "no; recovered from preserved carcass" if dis == "Carcass" else "not reported"
                item["dissection_location"] = "not explicitly reported in sample-level API field"
                item["preservation_context"] = dis
            rows.append(item)
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "rr1_sample_protocol_audit.csv", index=False)
    contrast = (audit.groupby("contrast_id", as_index=False).agg(
        OSD=("sample_id", lambda x: audit.loc[x.index, "contrast_id"].map(lambda y: y.split("__")[1]).iloc[0]),
        samples=("sample_id", "count"), preservation_context=("preservation_context", lambda x: " | ".join(sorted(set(map(str, x))))),
        library_selection=("library_selection_api", lambda x: " | ".join(sorted(set(map(str, x))))),
        library_kit=("library_kit_api", lambda x: " | ".join(sorted(set(map(str, x))))),
        read_layout=("read_layout_api", lambda x: " | ".join(sorted(set(map(str, x))))),
        read_length=("read_length_api", lambda x: " | ".join(sorted(set(map(str, x))))),
        platform=("platform_api", lambda x: " | ".join(sorted(set(map(str, x))))), sequencing_center=("sequencing_center", "first")))
    contrast.to_csv(OUT / "rr1_protocol_comparison.csv", index=False)
    return audit, contrast


def technical_replication(z, b):
    design = pd.read_csv(R3 / "task3_osd168_technical_replication/technical_response_design.csv")
    npz = np.load(R3 / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    stored = {str(n): v.astype(float) for n, v in zip(npz["names"], npz["delta_z"])}
    names = ["RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC", "RR1_OSD168_all_ERCC"]
    missing = [n for n in names if n not in stored]
    if missing: raise KeyError(missing)
    rows = []
    source = stored[names[0]]
    for target in names[1:]:
        other = stored[target]
        sp, so = project(source, b[:2]); tp, to = project(other, b[:2])
        row = {"comparison": f"{names[0]} vs {target}", "direct_technical_replication": True,
               "full_cosine": cosine(source, other), "parallel_cosine": cosine(sp, tp), "orthogonal_cosine": cosine(so, to)}
        for k in [1, 2, 5]: row[f"cosine_after_PC1_{k}"] = cosine(project(source, b[:k])[1], project(other, b[:k])[1])
        rows.append(row)
    # OSD-47 and the OSD-48 upon-euthanasia stratum have no OSD-168 counterpart.
    for label in ["OSD-47 CASIS 21d", "OSD-47 CASIS 22d", "OSD-48 upon-euthanasia 37d"]:
        rows.append({"comparison": label, "direct_technical_replication": False, "reason": "No corresponding OSD-168 biological material"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rr1_technical_replication.csv", index=False)
    return out


def apples_to_apples_component_table(b):
    """Apply the unchanged PC1-2 reference to RR1 and both RR3 replications."""
    archive = np.load(R3 / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    vectors = {str(n): v.astype(float) for n, v in zip(archive["names"], archive["delta_z"])}
    pairs = {
        "RR1": ("RR1_OSD48_original_matched", "RR1_OSD168_no-ERCC"),
        "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
        "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
    }
    rows = []
    for comparison, (a_name, b_name) in pairs.items():
        a, other = vectors[a_name], vectors[b_name]
        ap, ao = project(a, b); bp, bo = project(other, b)
        discrepancy = a - other; dp, _ = project(discrepancy, b)
        rows.append({
            "Metric": "Whole-response cosine", comparison: cosine(a, other)})
        rows.append({
            "Metric": "PC1-2 parallel-component cosine", comparison: cosine(ap, bp)})
        rows.append({
            "Metric": "Outside-PC1-2 cosine", comparison: cosine(ao, bo)})
        rows.append({
            "Metric": "Original response PC1-2 aligned fraction", comparison: np.dot(ap, ap) / np.dot(a, a)})
        rows.append({
            "Metric": "Remeasured response PC1-2 aligned fraction", comparison: np.dot(bp, bp) / np.dot(other, other)})
        rows.append({
            "Metric": "Replication discrepancy captured by PC1-2", comparison: np.dot(dp, dp) / np.dot(discrepancy, discrepancy)})
    # Convert the long temporary rows into one requested Metric x comparison table.
    long = pd.DataFrame(rows).melt("Metric", var_name="Comparison", value_name="Value").dropna()
    table = long.pivot(index="Metric", columns="Comparison", values="Value").reset_index()
    order = ["Whole-response cosine", "PC1-2 parallel-component cosine", "Outside-PC1-2 cosine",
             "Original response PC1-2 aligned fraction", "Remeasured response PC1-2 aligned fraction",
             "Replication discrepancy captured by PC1-2"]
    table["_order"] = table.Metric.map({v: i for i, v in enumerate(order)})
    table = table.sort_values("_order").drop(columns="_order")[["Metric", "RR1", "RR3-39", "RR3-40"]]
    table.to_csv(OUT / "rr1_rr3_component_replication_comparison.csv", index=False)
    return table


def rr1_stratum_replication_audit(b):
    """Trace the published RR1 -0.804 result to animals and preservation strata."""
    correspondence = pd.read_csv(R3 / "task3_osd168_technical_replication/biological_sample_correspondence.csv")
    correspondence = correspondence[(correspondence.source_OSD.eq("OSD-48")) & correspondence.group.isin(["FLT", "GC"])]
    correspondence = correspondence[correspondence.source_sample.notna()].copy()
    api = pd.read_csv(R3 / "task3_manifest_api_validation/api_OSD-48_sample_metadata.csv")
    rna = api[api["investigation.study assays.study assay measurement type"].astype(str).str.contains("transcription profiling", case=False, na=False)]
    preservation = rna[["id.sample name", "study.factor value.dissection condition"]].drop_duplicates("id.sample name")
    rows = correspondence.merge(preservation, left_on="source_sample", right_on="id.sample name", how="left")
    rows = rows.rename(columns={"study.factor value.dissection condition": "OSD48_stratum"})
    rows["used_in_minus_0_804"] = rows.ERCC_condition.eq("no-ERCC")
    rows[["source_sample", "group", "animal_id", "OSD48_stratum", "OSD168_sample", "ERCC_condition",
          "exact_animal_match", "identical_RNA_status", "used_in_minus_0_804"]].sort_values(
              ["ERCC_condition", "group", "animal_id"]).to_csv(OUT / "rr1_osd48_osd168_sample_trace.csv", index=False)

    # Include every OSD-48 biological sample so the unavailable I-stratum is explicit.
    membership = pd.read_csv(R3 / "task3b_contrast_sample_membership.csv")
    membership = membership[membership.contrast_id.str.contains("OSD-48", na=False)].merge(
        preservation, left_on="sample_id", right_on="id.sample name", how="left")
    no_ercc = rows[rows.ERCC_condition.eq("no-ERCC")][["source_sample", "OSD168_sample"]].drop_duplicates("source_sample")
    complete = membership.merge(no_ercc, left_on="sample_id", right_on="source_sample", how="left")
    complete["valid_OSD168_counterpart"] = complete.OSD168_sample.notna()
    complete[["contrast_id", "sample_id", "condition", "study.factor value.dissection condition", "OSD168_sample",
              "valid_OSD168_counterpart"]].to_csv(OUT / "rr1_all_osd48_samples_and_counterparts.csv", index=False)

    comp = apples_to_apples_component_table(b).set_index("Metric")
    metric = pd.DataFrame([
        {"stratum": "carcass (exact matched animals; no-ERCC OSD-168)", "valid": True,
         "OSD48_n_FLT": 4, "OSD48_n_GC": 5, "OSD168_n_FLT": 4, "OSD168_n_GC": 5,
         "whole_cosine": comp.loc["Whole-response cosine", "RR1"],
         "parallel_cosine": comp.loc["PC1-2 parallel-component cosine", "RR1"],
         "outside_PC1_2_cosine": comp.loc["Outside-PC1-2 cosine", "RR1"],
         "OSD48_aligned_fraction": comp.loc["Original response PC1-2 aligned fraction", "RR1"],
         "OSD168_aligned_fraction": comp.loc["Remeasured response PC1-2 aligned fraction", "RR1"],
         "discrepancy_fraction_PC1_2": comp.loc["Replication discrepancy captured by PC1-2", "RR1"],
         "reason": "This is the original -0.804 comparison; M27 was excluded because OSD-168 has no matched M27 profile."},
        {"stratum": "upon euthanasia", "valid": False, "OSD48_n_FLT": 2, "OSD48_n_GC": 2,
         "OSD168_n_FLT": 0, "OSD168_n_GC": 0,
         "reason": "OSD-168 contains no corresponding M21/M22/M31/M32 profiles; component replication metrics are undefined."},
    ])
    metric.to_csv(OUT / "rr1_stratum_specific_replication_metrics.csv", index=False)
    return complete, metric


def random_and_bootstrap(m, z, b, observed, n_random=1000, n_boot=5000):
    rng = np.random.default_rng(SEED); random_rows = []
    for rep in range(n_random):
        q, _ = np.linalg.qr(rng.normal(size=(512, 2))); rb = q.T
        for cid, v in observed.items():
            p, _ = project(v, rb); random_rows.append({"replicate": rep, "contrast_id": cid, "aligned_fraction": np.dot(p,p)/np.dot(v,v)})
    random = pd.DataFrame(random_rows)
    obs = pd.read_csv(OUT / "rr1_response_decomposition.csv").set_index("contrast_id")
    control = []
    for cid, g in random.groupby("contrast_id"):
        value = obs.loc[cid, "aligned_fraction"]
        control.append({"contrast_id": cid, "observed": value, "random_mean": g.aligned_fraction.mean(),
                        "random_95_low": g.aligned_fraction.quantile(.025), "random_95_high": g.aligned_fraction.quantile(.975),
                        "empirical_p": (1 + (g.aligned_fraction >= value).sum()) / (len(g)+1)})
    pd.DataFrame(control).to_csv(OUT / "random_2d_subspace_control.csv", index=False)
    boots = []
    for cid, g in m.groupby("contrast_id"):
        f = g[g.condition.eq("FLT")].embedding_index.to_numpy(int); c = g[g.condition.eq("GC")].embedding_index.to_numpy(int)
        original = observed[cid]
        for rep in range(n_boot):
            v = z[rng.choice(f, len(f), True)].mean(0) - z[rng.choice(c, len(c), True)].mean(0); p, o = project(v, b)
            boots.append({"contrast_id": cid, "replicate": rep, "direction_cosine_to_original": cosine(v, original),
                          "total_norm": np.linalg.norm(v), "projected_norm": np.linalg.norm(p), "orthogonal_norm": np.linalg.norm(o),
                          "aligned_fraction": np.dot(p,p)/max(np.dot(v,v), 1e-12)})
    boot = pd.DataFrame(boots); boot.to_parquet(OUT / "sample_bootstrap.parquet", index=False)
    metrics = ["direction_cosine_to_original", "total_norm", "projected_norm", "orthogonal_norm", "aligned_fraction"]
    summary=[]
    for cid,g in boot.groupby("contrast_id"):
        for metric in metrics:
            x=g[metric]; summary.append({"contrast_id":cid,"metric":metric,"mean":x.mean(),"median":x.median(),"sd":x.std(),"ci_low":x.quantile(.025),"ci_high":x.quantile(.975),"unique_values":x.nunique()})
    pd.DataFrame(summary).to_csv(OUT / "sample_bootstrap_summary.csv", index=False)
    return random, boot


def contextual_components(m, x, b, device):
    cache = OUT / "component_contextual_gene_rankings.parquet"
    if cache.exists(): return pd.read_parquet(cache)
    dev = torch.device(device if torch.cuda.is_available() else "cpu"); model = load_frozen_encoder(dev)
    rows=[]; started=time.monotonic()
    with torch.no_grad():
        for cid,g in m.groupby("contrast_id",sort=False):
            means={}
            for condition in ["FLT","GC"]:
                indices=g[g.condition.eq(condition)].embedding_index.to_numpy(int); acc=None
                for idx in indices:
                    inp=torch.as_tensor(np.array(x[idx],dtype=np.float32,copy=True),device=dev)[None]
                    hidden=model._encode_hidden(inp)[0].cpu().numpy().astype(np.float64)
                    acc=hidden if acc is None else acc+hidden
                means[condition]=acc/len(indices)
            delta=means["FLT"]-means["GC"]
            parallel=np.einsum("gd,kd,kj->gj",delta,b,b); orthogonal=delta-parallel
            for component,array in [("total",delta),("parallel",parallel),("orthogonal",orthogonal)]:
                score=np.linalg.norm(array,axis=1); order=np.argsort(-score); rank=np.empty(len(score),int); rank[order]=np.arange(1,len(score)+1)
                rows.append(pd.DataFrame({"contrast_id":cid,"component":component,"gene_symbol":GENES,"contextual_response_norm":score,"rank":rank}))
            print(f"[context heartbeat] {cid} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
    out=pd.concat(rows,ignore_index=True); out.to_parquet(cache,index=False)
    out.sort_values(["contrast_id","component","rank"]).groupby(["contrast_id","component"],group_keys=False).head(25).to_csv(OUT/"top25_component_contextual_genes.csv",index=False)
    return out


def family(term):
    t=term.upper()
    if "SPLIC" in t or "RNA PROCESS" in t or "RRNA" in t or "RIBOSOM" in t: return "RNA processing/splicing"
    if "CHROMATIN" in t or "NUCLEOSOME" in t or "HISTONE" in t: return "chromatin organization/remodeling"
    if "DNA REPAIR" in t or "DNA DAMAGE" in t or "DNA METABOL" in t: return "DNA repair/DNA-damage response"
    if any(q in t for q in ["LIPID","FATTY ACID","PEROXISOM","CHOLESTEROL","BILE","CATABOL","SMALL MOLECULE","METABOL"]): return "hepatic lipid/fatty-acid/peroxisomal metabolism"
    return "other"


def component_gsea(rankings):
    cache=OUT/"component_contextual_gsea.parquet"
    if cache.exists(): return pd.read_parquet(cache)
    rows=[]; total=rankings.groupby(["contrast_id","component"]).ngroups*3; done=0; started=time.monotonic()
    for (cid,component),g in rankings.groupby(["contrast_id","component"],sort=False):
        rnk=g[["gene_symbol","contextual_response_norm"]].sort_values("contextual_response_norm",ascending=False)
        for source,file in GMT.items():
            r=gp.prerank(rnk=rnk,gene_sets=str(GMT_ROOT/file),min_size=10,max_size=500,permutation_num=1000,threads=8,seed=SEED,outdir=None,verbose=False).res2d
            r=r.rename(columns={"Term":"pathway","NES":"nes","FDR q-val":"fdr","Lead_genes":"leading_edge"});r["contrast_id"]=cid;r["component"]=component;r["source"]=source
            rows.append(r[["contrast_id","component","source","pathway","nes","fdr","leading_edge"]]);done+=1
            elapsed=time.monotonic()-started;print(f"[GSEA heartbeat] {done}/{total} elapsed={elapsed/60:.1f}m eta={elapsed/max(done,1)*(total-done)/60:.1f}m",flush=True)
    out=pd.concat(rows,ignore_index=True);out["family"]=out.pathway.map(family);out.to_parquet(cache,index=False);return out


def conventional_tables():
    primary=pd.read_csv(FULLVOC/"primary_summary.csv"); primary=primary[primary.Study.isin(["OSD-47","OSD-48"])]
    fam=pd.read_csv(FULLVOC/"program_family_summary.csv"); fam=fam[fam.Study.isin(["OSD-47","OSD-48"])]
    primary.to_csv(OUT/"conventional_expression_summary.csv",index=False);fam.to_csv(OUT/"conventional_program_families.csv",index=False)
    path=pd.read_parquet(FULLVOC/"pathway_enrichment_detailed.parquet")
    path=path[path.contrast_id.str.contains("OSD-47|OSD-48",regex=True,na=False)]
    path.to_parquet(OUT/"conventional_pathway_details.parquet",index=False)
    return primary,fam


def figures(decomp, random, boot, enriched):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels=[SPECS[c] for c in SPECS]; vals=decomp.set_index("contrast_id").loc[list(SPECS),"aligned_fraction"]
    bs=boot.groupby("contrast_id").aligned_fraction.quantile([.025,.975]).unstack().reindex(SPECS)
    y=np.arange(4); fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    ax.barh(y,vals,color=["#4C78A8","#9ecae9","#F58518","#E45756"])
    ax.errorbar(vals,y,xerr=np.vstack([vals-bs[.025],bs[.975]-vals]),fmt="none",ecolor="black",capsize=3)
    ax.set(yticks=y,yticklabels=labels,xlim=(0,1),xlabel="Fraction of response energy in controlled PC1–2 subspace",title="RR1 response alignment by preservation context")
    for i,v in enumerate(vals): ax.text(v+.015,i,f"{v:.3f}",va="center")
    for ext in ["png","pdf"]: fig.savefig(FIG/f"rr1_alignment_by_context.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for ax,cid in zip(axes.flat,SPECS):
        n=random[random.contrast_id.eq(cid)].aligned_fraction; obs=decomp.set_index("contrast_id").loc[cid,"aligned_fraction"]
        ax.hist(n,bins=40,color="#72B7B2");ax.axvline(obs,color="#E45756",lw=2);ax.set(title=SPECS[cid],xlabel="Random 2D aligned fraction")
    fig.suptitle("Matched random-subspace controls")
    for ext in ["png","pdf"]: fig.savefig(FIG/f"random_subspace_controls.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)
    families=["RNA processing/splicing","chromatin organization/remodeling","DNA repair/DNA-damage response","hepatic lipid/fatty-acid/peroxisomal metabolism"]
    z=(enriched[enriched.family.isin(families)].sort_values(["contrast_id","component","family","fdr","nes"],ascending=[True,True,True,True,False]).groupby(["contrast_id","component","family"],as_index=False).first())
    z.to_csv(OUT/"component_program_family_summary.csv",index=False)


def main(args):
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    m,manifest,x,z=load_design(); vt,evr=basis(); b=vt[:2]
    audit,protocol=protocol_audit(m);decomp,vectors=responses(m,z,b);tech=technical_replication(z,vt)
    component_comparison=apples_to_apples_component_table(b)
    rr1_trace, rr1_strata=rr1_stratum_replication_audit(b)
    random,boot=random_and_bootstrap(m,z,b,vectors,args.random_replicates,args.bootstrap_replicates)
    rankings=contextual_components(m,x,b,args.device);enriched=component_gsea(rankings)
    conventional,families=conventional_tables();figures(decomp,random,boot,enriched)
    # Direct descriptive comparison; preserve strata instead of pooling.
    direct=decomp.merge(protocol,on="contrast_id",how="left")
    direct.to_csv(OUT/"direct_rr1_context_comparison.csv",index=False)
    c=decomp.set_index("contrast_id").aligned_fraction
    conclusion={"decision":"D. UNDERPOWERED/CONFOUNDED",
      "finding":f"Alignment varies strongly by stratum: OSD-47 21d={c.iloc[0]:.3f}, OSD-47 22d={c.iloc[1]:.3f}, OSD-48 upon-euthanasia={c.iloc[2]:.3f}, OSD-48 carcass={c.iloc[3]:.3f}.",
      "interpretation":"The better-powered carcass stratum is highly aligned and the upon-euthanasia stratum is lower, but OSD-47/48 differ in animal, age, strain, duration, and collection context; the OSD-47 22d result is 1-vs-1. The design cannot isolate preservation from biology or protocol interactions.",
      "reference_caveat":"PC1-2 is a controlled T-cell PolyA/Ribo-associated reference, not a technical-only batch space; aligned fraction is not percent artifact.",
      "PC1_2_controlled_variance":float(evr[:2].sum())}
    (OUT/"decision_summary.json").write_text(json.dumps(conclusion,indent=2)+"\n")
    (OUT/"provenance.json").write_text(json.dumps({"bridge_frozen":True,"embeddings_recomputed":False,"contrasts_redefined":False,"reference_redefined":False,"random_2d_replicates":args.random_replicates,"sample_bootstrap_replicates":args.bootstrap_replicates,"contextual_method":"per-gene L2 norm of mean contextual hidden FLT minus GC; exact decomposition into controlled PC1-2 and orthogonal components","gsea_permutations":1000,"seed":SEED},indent=2)+"\n")
    print(decomp.to_string(index=False));print("\nComponent replication comparison\n",component_comparison.to_string(index=False));print(json.dumps(conclusion,indent=2));print("[complete]",OUT)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",default="cuda:0");p.add_argument("--random-replicates",type=int,default=1000);p.add_argument("--bootstrap-replicates",type=int,default=5000);main(p.parse_args())
