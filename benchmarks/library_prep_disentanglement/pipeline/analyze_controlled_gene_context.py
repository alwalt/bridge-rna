#!/usr/bin/env python3
"""Final Task 4 controlled PolyA/Ribo contextual-gene validation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_confounding_profiler/controlled_gene_context"
FIG = OUT / "figures"
WORK = HERE / "work/task4_controlled_gene_context"
DATA = HERE / "work/datasets/chen_2020_tcells"
RR = HERE / "results/task4_confounding_profiler/contextual_robustness"
GMT_ROOT = REPO / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMT = {"GO:BP": "GO_Biological_Process_2026.gmt", "KEGG": "KEGG_2026.gmt", "REAC": "Reactome_Pathways_2024.gmt"}
SEED = 42731
PERMUTATIONS = 1000

sys.path.insert(0, str(REPO / "benchmarks/tcga_downstream/pipeline"))
sys.path.insert(0, str(REPO))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), eps)


def infer_displacements(device: torch.device, batch_size: int) -> tuple[pd.DataFrame, np.memmap]:
    manifest = pd.read_parquet(DATA / "manifest.parquet").reset_index(drop=True)
    x = np.load(DATA / "log1p_tpm.npy", mmap_mode="r")
    genes = load_canonical_genes(REPO / "data/ensembl/canonical_genes.csv")
    assert x.shape == (80, len(genes)) and manifest.pair_id.nunique() == 40
    pairs = sorted(manifest.pair_id.unique())
    path = WORK / "controlled_context_displacements.float16.dat"
    shape = (len(pairs), len(genes), 512)
    WORK.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == np.prod(shape) * np.dtype("float16").itemsize:
        print(f"[cache] {path}", flush=True)
        return manifest, np.memmap(path, dtype="float16", mode="r", shape=shape)
    model = load_frozen_encoder(device)
    mm = np.memmap(path, dtype="float16", mode="w+", shape=shape)
    started = time.time()
    for j, pair in enumerate(pairs):
        g = manifest[manifest.pair_id.eq(pair)]
        ip = int(g.index[g.library_prep.eq("polyA")][0]); ir = int(g.index[g.library_prep.eq("ribo")][0])
        values = torch.as_tensor(np.asarray(x[[ip, ir]]), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model._encode_hidden(values).float().cpu().numpy()
        mm[j] = (h[1] - h[0]).astype(np.float16); mm.flush()
        print(f"[heartbeat] contextual pairs={j+1}/40 elapsed={(time.time()-started)/60:.1f}m", flush=True)
    return manifest, np.memmap(path, dtype="float16", mode="r", shape=shape)


def summarize_genes(D: np.memmap, genes: list[str], x: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    n, ng, _ = D.shape
    rows, donor_rows = [], []
    for start in range(0, ng, 256):
        stop = min(start + 256, ng); q = np.asarray(D[:, start:stop], dtype=np.float32)
        norms = np.linalg.norm(q, axis=2); total = q.sum(axis=0); mean = total / n
        loo = np.empty((n, stop-start), np.float32)
        for i in range(n): loo[i] = cosine_rows(q[i], (total-q[i])/(n-1))
        mean_mag = norms.mean(0); med_mag = np.median(norms, axis=0); consistency = loo.mean(0)
        prevalence = (loo > 0).mean(0); consensus_strength = np.linalg.norm(mean, axis=1) / np.maximum(mean_mag, 1e-12)
        score = med_mag * np.clip(consistency, 0, None)
        mean_consensus_displacement = mean_mag * consensus_strength
        for k, gene in enumerate(genes[start:stop]):
            rows.append({"gene_symbol": gene, "mean_displacement_magnitude": mean_mag[k], "median_displacement_magnitude": med_mag[k],
                         "loo_directional_consistency": consistency[k], "donor_prevalence_aligned": prevalence[k],
                         "consensus_strength": consensus_strength[k], "mean_consensus_displacement": mean_consensus_displacement[k], "sensitivity_score": score[k],
                         "mean_log1p_tpm": float(x[:, start+k].mean())})
            for i in range(n): donor_rows.append({"donor_index": i, "gene_symbol": gene, "displacement_magnitude": norms[i,k], "loo_directional_cosine": loo[i,k]})
    genes_df = pd.DataFrame(rows).sort_values("sensitivity_score", ascending=False).reset_index(drop=True)
    genes_df["sensitivity_rank"] = np.arange(1, len(genes_df)+1)
    return genes_df, pd.DataFrame(donor_rows)


def stability(donor: pd.DataFrame, genes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mag = donor.pivot(index="donor_index", columns="gene_symbol", values="displacement_magnitude")[genes.gene_symbol].to_numpy()
    loo = donor.pivot(index="donor_index", columns="gene_symbol", values="loo_directional_cosine")[genes.gene_symbol].to_numpy()
    rng = np.random.default_rng(SEED); base = genes.sensitivity_score.to_numpy(); rows = []
    for b in range(250):
        ids = rng.integers(0, 40, 40); score = np.median(mag[ids], axis=0) * np.clip(np.mean(loo[ids], axis=0), 0, None)
        rows.append({"bootstrap": b, "score_spearman": spearmanr(base, score).statistic,
                     "top100_overlap": len(set(np.argsort(-base)[:100]) & set(np.argsort(-score)[:100])),
                     "top500_overlap": len(set(np.argsort(-base)[:500]) & set(np.argsort(-score)[:500]))})
    boot = pd.DataFrame(rows)
    # Sign-flip control, conditional on the observed leave-one-out consensus.
    null = []
    for b in range(1000):
        signs = rng.choice([-1, 1], 40)[:, None]
        null.append({"permutation": b, "median_signed_consistency": float(np.median(np.mean(loo*signs, axis=0)))})
    return boot, pd.DataFrame(null)


def run_gsea(genes: pd.DataFrame) -> pd.DataFrame:
    cache = OUT / "controlled_pathway_enrichment_v2.parquet"
    if cache.exists(): return pd.read_parquet(cache)
    rankings = {
        "magnitude_x_reproducibility": genes[["gene_symbol", "sensitivity_score"]],
        "mean_consensus_displacement": genes[["gene_symbol", "mean_consensus_displacement"]],
    }
    out = []
    for rn, rank in rankings.items():
        value = rank.columns[1]; rnk = rank.sort_values(value, ascending=False)
        for source, file in GMT.items():
            print(f"[GSEA] {rn} {source}", flush=True)
            pre = gp.prerank(rnk=rnk, gene_sets=str(GMT_ROOT/file), min_size=10, max_size=500,
                             permutation_num=PERMUTATIONS, threads=8, seed=SEED, outdir=None, verbose=False)
            z = pre.res2d.rename(columns={"Term":"pathway", "ES":"es", "NES":"nes", "NOM p-val":"nominal_p", "FDR q-val":"fdr", "Lead_genes":"leading_edge"})
            z["ranking"] = rn; z["source"] = source; out.append(z[["ranking","source","pathway","es","nes","nominal_p","fdr","leading_edge"]])
    ans = pd.concat(out, ignore_index=True); ans.to_parquet(cache, index=False); return ans


def pathway_sizes(enr: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    rows=[]
    for source,file in GMT.items():
        for term,members in gp.parser.read_gmt(path=str(GMT_ROOT/file)).items(): rows.append({"source":source,"pathway":term,"represented_genes":len(set(members)&universe)})
    return enr.merge(pd.DataFrame(rows), on=["source","pathway"], how="left")


def family(term: str) -> str:
    t=term.upper()
    if "SPLIC" in t or "RNA PROCESS" in t or "MRNA" in t: return "RNA processing / splicing"
    if "CHROMATIN" in t: return "Chromatin organization / remodeling"
    if "DNA REPAIR" in t or "DNA METABOL" in t: return "DNA repair / DNA metabolism"
    if "FATTY ACID" in t or "PEROX" in t or "LIPID METAB" in t: return "Fatty-acid / lipid metabolism"
    return "Other"


def compare_rr(enr: pd.DataFrame, controlled: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rrgenes = pd.read_parquet(RR/"gene_level_metric_audit.parquet")
    merged = rrgenes.merge(controlled[["gene_symbol","sensitivity_score","sensitivity_rank"]], on="gene_symbol", validate="many_to_one")
    corr=[]; overlaps=[]; M=controlled.gene_symbol.nunique()
    for comp,g in merged.groupby("comparison"):
        for metric in ["normalized_context_discrepancy", "directional_instability"]:
            corr.append({"comparison":comp,"rr_metric":metric,"spearman":spearmanr(g[metric],g.sensitivity_score).statistic,"genes":len(g)})
        for n in [100,250,500,1000]:
            a=set(g.nlargest(n,"normalized_context_discrepancy").gene_symbol); b=set(controlled.head(n).gene_symbol); k=len(a&b)
            overlaps.append({"comparison":comp,"top_n":n,"observed_overlap":k,"expected_overlap":n*n/M,
                             "fold_enrichment":k/(n*n/M),"hypergeom_p":hypergeom.sf(k-1,M,n,n)})
    return merged, pd.DataFrame(corr), pd.DataFrame(overlaps)


def leading_edges(enr: pd.DataFrame, rr: pd.DataFrame, universe: set[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    # Use one representative, strongest positive-NES pathway per family and
    # primary ranking. This avoids inflating overlap by unioning many nested GO
    # terms that largely contain the same genes.
    controlled=enr[enr.ranking.eq("magnitude_x_reproducibility")].copy()
    rr1=rr[(rr.comparison=="RR1")&(rr.ranking.eq("normalized_discrepancy"))].copy()
    sources=[("controlled",controlled), ("RR1",rr1)]
    for origin,z in sources:
        z=z.copy();z["family"]=z.pathway.map(family)
        z=z[(z.family.ne("Other"))&(z.fdr<.05)&(z.nes>0)].sort_values("nes",ascending=False).groupby("family",as_index=False).head(1)
        for r in z.itertuples():
            fam=r.family
            for gene in str(r.leading_edge).split(";") if pd.notna(r.leading_edge) else []:
                if gene in universe: rows.append({"origin":origin,"family":fam,"ranking":r.ranking,"source":r.source,"pathway":r.pathway,"gene_symbol":gene,"nes":r.nes,"fdr":r.fdr})
    le=pd.DataFrame(rows); comparisons=[]; M=len(universe)
    for fam in ["RNA processing / splicing","Chromatin organization / remodeling","DNA repair / DNA metabolism"]:
        a=set(le[(le.origin=="RR1")&(le.family==fam)].gene_symbol); b=set(le[(le.origin=="controlled")&(le.family==fam)].gene_symbol); k=len(a&b)
        comparisons.append({"family":fam,"rr1_leading_edge_genes":len(a),"controlled_leading_edge_genes":len(b),"overlap":k,
                            "expected_overlap":len(a)*len(b)/M,"hypergeom_p":hypergeom.sf(k-1,M,len(a),len(b)) if a and b else np.nan})
    return le,pd.DataFrame(comparisons)


def plots(summary, genes, enrichment, overlaps, concordance):
    colors=["#CC3311","#0077BB","#009988"]
    fig,ax=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    vals=[summary["median_loo_consistency"],summary["fraction_genes_consistency_positive"]]
    names=["Median LOO\ndirectional consistency","Genes aligned in\n>50% donors"]
    b=ax[0].bar(names,vals,color=colors[:2]);ax[0].set_ylim(-.1,1);ax[0].axhline(0,color="black",lw=.8);ax[0].set_title("Controlled PolyA→Ribo gene-context reproducibility")
    for q,v in zip(b,vals):ax[0].text(q.get_x()+q.get_width()/2,v+.025,f"{v:.3f}",ha="center",fontweight="bold")
    ax[1].hist(genes.loo_directional_consistency,bins=40,color="#4477AA");ax[1].set(xlabel="Gene LOO directional consistency",ylabel="Genes",title="Across 40 same-RNA donor pairs")
    fig.savefig(FIG/"controlled_context_reproducibility.png",dpi=300);fig.savefig(FIG/"controlled_context_reproducibility.pdf");plt.close(fig)
    sig=enrichment.query("fdr < .05 and nes > 0").copy();sig["family"]=sig.pathway.map(family);q=sig[sig.family.ne("Other")].sort_values("nes").groupby("family",as_index=False).tail(2).sort_values("nes")
    if len(q):
        fig,ax=plt.subplots(figsize=(9,5.5),layout="constrained");ax.barh(q.pathway,q.nes,color="#4477AA");ax.set(xlabel="NES",title="Controlled PolyA→Ribo-sensitive pathway families");fig.savefig(FIG/"controlled_sensitive_pathways.png",dpi=300);fig.savefig(FIG/"controlled_sensitive_pathways.pdf");plt.close(fig)
    q=overlaps[overlaps.comparison.eq("RR1")]
    fig,ax=plt.subplots(figsize=(7,4.5),layout="constrained");x=np.arange(len(q));w=.36;ax.bar(x-w/2,q.observed_overlap,w,label="Observed",color="#CC3311");ax.bar(x+w/2,q.expected_overlap,w,label="Random expectation",color="#BBBBBB");ax.set(xticks=x,xticklabels=[f"Top {n}" for n in q.top_n],ylabel="Gene overlap",title="RR1 vs controlled gene-level overlap");ax.legend();fig.savefig(FIG/"rr1_controlled_gene_overlap.png",dpi=300);fig.savefig(FIG/"rr1_controlled_gene_overlap.pdf");plt.close(fig)
    if len(concordance):
        q=concordance.melt(id_vars="family",value_vars=["rr1_best_nes","controlled_best_nes"],var_name="analysis",value_name="NES")
        fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained");x=np.arange(q.family.nunique());w=.36
        for j,(label,g) in enumerate(q.groupby("analysis",sort=False)):ax.bar(x+(j-.5)*w,g.NES,w,label=label.replace("_best_nes","").upper())
        ax.set(xticks=x,xticklabels=q.family.drop_duplicates(),ylabel="Best significant NES",title="Predefined RR1 pathway-family concordance");ax.tick_params(axis="x",rotation=15);ax.legend();fig.savefig(FIG/"rr1_controlled_pathway_concordance.png",dpi=300);fig.savefig(FIG/"rr1_controlled_pathway_concordance.pdf");plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");ap.add_argument("--batch-size",type=int,default=2);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);started=time.time();device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    manifest,D=infer_displacements(device,a.batch_size);genes=load_canonical_genes(REPO/"data/ensembl/canonical_genes.csv");x=np.load(DATA/"log1p_tpm.npy",mmap_mode="r")
    gene_path=OUT/"controlled_gene_sensitivity.parquet";donor_path=OUT/"donor_gene_displacement_metrics.parquet"
    if gene_path.exists() and donor_path.exists():
        controlled=pd.read_parquet(gene_path);donor=pd.read_parquet(donor_path)
        if "mean_consensus_displacement" not in controlled:
            controlled["mean_consensus_displacement"]=controlled.mean_displacement_magnitude*controlled.consensus_strength
            controlled.to_parquet(gene_path,index=False)
    else: controlled,donor=summarize_genes(D,genes,x);controlled.to_parquet(gene_path,index=False);donor.to_parquet(donor_path,index=False)
    boot,null=stability(donor,controlled);boot.to_csv(OUT/"bootstrap_ranking_stability.csv",index=False);null.to_csv(OUT/"label_permutation_control.csv",index=False)
    summary={"donors":40,"libraries":80,"genes":15165,"genes_observed":15120,"context_dimensions":512,
             "median_loo_consistency":float(controlled.loo_directional_consistency.median()),
             "fraction_genes_consistency_positive":float((controlled.loo_directional_consistency>0).mean()),
             "fraction_genes_majority_aligned":float((controlled.donor_prevalence_aligned>.5).mean()),
             "bootstrap_median_score_spearman":float(boot.score_spearman.median()),"bootstrap_median_top500_overlap":float(boot.top500_overlap.median()),
             "permutation_p_median_consistency":float((1+(null.median_signed_consistency>=controlled.loo_directional_consistency.median()).sum())/(len(null)+1))}
    enr=pathway_sizes(run_gsea(controlled),set(genes));enr.to_csv(OUT/"controlled_pathway_enrichment.csv",index=False)
    merged,corr,overlaps=compare_rr(enr,controlled);merged.to_parquet(OUT/"rr1_rr3_controlled_gene_comparison.parquet",index=False);corr.to_csv(OUT/"rr1_rr3_rank_correlations.csv",index=False);overlaps.to_csv(OUT/"rr1_rr3_topn_overlap.csv",index=False)
    rr=pd.read_parquet(RR/"gsea_full_results.parquet");rr=pathway_sizes(rr,set(genes));le,lecomp=leading_edges(enr,rr,set(genes));le.to_csv(OUT/"leading_edge_genes.csv",index=False);lecomp.to_csv(OUT/"leading_edge_concordance.csv",index=False)
    families=[]
    for fam in ["RNA processing / splicing","Chromatin organization / remodeling","DNA repair / DNA metabolism","Fatty-acid / lipid metabolism"]:
        c=enr.assign(family=enr.pathway.map(family));r=rr[(rr.comparison=="RR1")].assign(family=rr[rr.comparison=="RR1"].pathway.map(family))
        cs=c[(c.family==fam)&(c.fdr<.05)];rs=r[(r.family==fam)&(r.fdr<.05)]
        families.append({"family":fam,"controlled_significant_terms":cs.pathway.nunique(),"rr1_significant_terms":rs.pathway.nunique(),
                         "controlled_best_nes":cs.nes.max() if len(cs) else np.nan,"controlled_best_fdr":cs.fdr.min() if len(cs) else np.nan,
                         "rr1_best_nes":rs.nes.max() if len(rs) else np.nan,"rr1_best_fdr":rs.fdr.min() if len(rs) else np.nan})
    concordance=pd.DataFrame(families);concordance.to_csv(OUT/"rr1_controlled_pathway_concordance.csv",index=False)
    # Supported only if significant with positive NES in both controlled rankings.
    decisions=[]
    for fam in concordance.family:
        q=enr.assign(family=enr.pathway.map(family));q=q[(q.family==fam)&(q.fdr<.05)&(q.nes>0)]
        nr=q.ranking.nunique();dec="SUPPORTED" if nr==2 else "PARTIALLY SUPPORTED" if nr==1 else "NOT SUPPORTED"
        decisions.append({"family":fam,"controlled_classification":dec})
    decisions=pd.DataFrame(decisions);decisions.to_csv(OUT/"pathway_family_decisions.csv",index=False)
    rr1rho=float(corr.query("comparison=='RR1' and rr_metric=='normalized_context_discrepancy'").spearman.iloc[0]);rr1ov=overlaps[overlaps.comparison.eq("RR1")]
    supported=decisions[decisions.controlled_classification.eq("SUPPORTED")].family.tolist()
    overall="STRONG CONCORDANCE" if len(supported)>=3 and rr1rho>.3 else "PARTIAL CONCORDANCE" if supported or rr1rho>.1 else "SAMPLE-LEVEL CONCORDANCE ONLY"
    final={**summary,"rr1_controlled_gene_rank_spearman":rr1rho,"overall_classification":overall,"supported_families":supported,
           "checkpoint":"model/r7hnr92k/best_model.pt","preprocessing":"counts -> gene-length TPM -> natural log1p; canonical order; absent genes zero-filled",
             "pairing":"40 authoritative same-RNA donor pairs","ranking":"primary: median displacement magnitude * max(mean LOO directional cosine, 0); sensitivity: norm(mean donor displacement)",
           "caveat":"Controlled T-cell library-selection sensitivity is independent evidence, not causal identification of the multi-factor RR1 protocol transition."}
    (OUT/"final_summary.json").write_text(json.dumps(final,indent=2)+"\n");pd.DataFrame([final|{"supported_families":"; ".join(supported)}]).to_csv(OUT/"final_summary.csv",index=False)
    plots(summary,controlled,enr,overlaps,concordance)
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"checkpoint":"model/r7hnr92k/best_model.pt","frozen":True,"donors":40,"libraries":80,"genes":15165,"dimensions":512,"displacement":"Ribo minus PolyA per same-RNA donor","gsea_permutations":PERMUTATIONS,"seed":SEED,"elapsed_minutes":(time.time()-started)/60}
    (OUT/"provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
    print(json.dumps(final,indent=2),flush=True);print(f"[complete] {OUT}",flush=True)


if __name__ == "__main__": main()
