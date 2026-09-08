#!/usr/bin/env python3
"""Decompose the controlled PolyA/Ribo difference basis component by component.

This analysis uses cached frozen BridgeRNA representations only.  Component
labels are exploratory descriptions, not claims that a direction is purely
technical or biological.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, roc_auc_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "results/task4_technical_component_decomposition"
FIG = OUT / "figures"
DATA = ROOT / "work/datasets/chen_2020_tcells"
CTX = ROOT / "work/task4_controlled_gene_context/controlled_context_displacements.float16.dat"
TASK3 = REPO / "benchmarks/osdr_batch_effect_representation/results"
GMT_ROOT = REPO / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMT = {"GO:BP": "GO_Biological_Process_2026.gmt", "KEGG": "KEGG_2026.gmt", "Reactome": "Reactome_Pathways_2024.gmt"}
SEED = 42871
sys.path.insert(0, str(REPO))


def cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def project_out(x, basis):
    return x - (x @ basis.T) @ basis


def load_controlled():
    m = pd.read_parquet(DATA / "manifest.parquet").reset_index(drop=True)
    z = np.load(DATA / "bridgerna_embeddings.npy").astype(float)
    assert len(m) == len(z) == 80 and m.pair_id.nunique() == 40
    poly, ribo, pairs = [], [], []
    for pair, g in m.groupby("pair_id", sort=True):
        ip = g.index[g.library_prep.eq("polyA")]
        ir = g.index[g.library_prep.eq("ribo")]
        assert len(ip) == len(ir) == 1
        pairs.append(str(pair)); poly.append(z[ip[0]]); ribo.append(z[ir[0]])
    return m, z, pairs, np.stack(poly), np.stack(ribo)


def load_responses():
    q = np.load(TASK3 / "task3_osd168_technical_replication/technical_response_vectors.npz", allow_pickle=True)
    tech = dict(zip(q["names"].astype(str), q["delta_z"].astype(float)))
    rep = {
        "RR1": (tech["RR1_OSD48_original_matched"], tech["RR1_OSD168_no-ERCC"]),
        "RR3-39": (tech["C01_OSD137_original_matched"], tech["C01_OSD168_all_ERCC"]),
        "RR3-40": (tech["C02_OSD137_original_matched"], tech["C02_OSD168_all_ERCC"]),
    }
    q = np.load(TASK3 / "task3b_bridgerna_response_vectors.npz", allow_pickle=True)
    return rep, q["contrast_id"].astype(str), q["delta_z"].astype(float)


def pc_metrics(poly, ribo, basis, singular, responses):
    d = ribo - poly
    total = np.sum(singular ** 2)
    rows = []
    for j, v in enumerate(basis, 1):
        donor_scores = d @ v
        if donor_scores.mean() < 0:
            v *= -1; donor_scores *= -1
        # Leave-one-donor-out classification along this fixed component.
        truth, score, pred = [], [], []
        for i in range(len(d)):
            tr = np.arange(len(d)) != i
            p0, r0 = np.mean(poly[tr] @ v), np.mean(ribo[tr] @ v)
            orient = 1 if r0 >= p0 else -1; threshold = (p0 + r0) / 2
            for value, label in ((poly[i] @ v, 0), (ribo[i] @ v, 1)):
                truth.append(label); score.append(orient * value); pred.append(int(orient * (value-threshold) > 0))
        rec = {
            "component": j, "singular_value": singular[j-1],
            "technical_variance_fraction": singular[j-1]**2 / total,
            "technical_cumulative_variance": np.sum(singular[:j]**2) / total,
            "mean_paired_shift": donor_scores.mean(), "sd_paired_shift": donor_scores.std(ddof=1),
            "donor_direction_consistency": np.mean(donor_scores > 0),
            "loo_library_auroc": roc_auc_score(truth, score),
            "loo_library_balanced_accuracy": balanced_accuracy_score(truth, pred),
        }
        for name, pair in responses.items():
            for which, r in zip(("original", "remeasurement"), pair):
                c = cosine(r, v)
                rec[f"{name}_{which}_cosine"] = c
                rec[f"{name}_{which}_abs_cosine"] = abs(c)
                rec[f"{name}_{which}_projection"] = float(r @ v)
                rec[f"{name}_{which}_energy_fraction"] = float((r @ v)**2 / np.dot(r, r))
            discrepancy = pair[0] - pair[1]
            rec[f"{name}_discrepancy_energy_fraction"] = float((discrepancy @ v)**2 / np.dot(discrepancy, discrepancy))
        rows.append(rec)
    return pd.DataFrame(rows), basis


def gene_component_scores(basis):
    cache = OUT / "technical_pc_gene_rankings.parquet"
    if cache.exists():
        print(f"[cache] {cache}", flush=True)
        return pd.read_parquet(cache)
    from src.fm_embed.vocab import load_canonical_genes
    genes = load_canonical_genes(REPO / "data/ensembl/canonical_genes.csv")
    shape = (40, len(genes), 512)
    assert CTX.exists() and CTX.stat().st_size == np.prod(shape) * 2
    mm = np.memmap(CTX, dtype="float16", mode="r", shape=shape)
    rows = []
    for start in range(0, len(genes), 256):
        stop = min(start + 256, len(genes))
        x = np.asarray(mm[:, start:stop], dtype=np.float32)
        projected = np.einsum("dgc,pc->dgp", x, basis, optimize=True)
        mean = projected.mean(0); consistency = np.mean(np.sign(projected) == np.sign(mean)[None], axis=0)
        for gi, gene in enumerate(genes[start:stop]):
            for pj in range(len(basis)):
                rows.append((gene, pj+1, float(mean[gi,pj]), float(abs(mean[gi,pj])), float(consistency[gi,pj])))
        print(f"[heartbeat] contextual projection genes={stop:,}/{len(genes):,}", flush=True)
    out = pd.DataFrame(rows, columns=["gene_symbol","component","mean_signed_projection","abs_mean_projection","donor_sign_consistency"])
    out["component_rank"] = out.groupby("component")["abs_mean_projection"].rank(ascending=False, method="first").astype(int)
    return out


def pathway_enrichment(scores):
    cache = OUT / "pc_pathway_enrichment.parquet"
    if cache.exists():
        print(f"[cache] {cache}", flush=True)
        return pd.read_parquet(cache)
    universe = set(scores.gene_symbol.unique()); M = len(universe); rows = []
    libraries = {source: gp.parser.read_gmt(path=str(GMT_ROOT / file)) for source, file in GMT.items()}
    for pc, g in scores.groupby("component"):
        selected = set(g.nsmallest(250, "component_rank").gene_symbol); N = len(selected)
        pcrows = []
        for source, terms in libraries.items():
            for term, members in terms.items():
                members = set(members) & universe; n = len(members)
                if not 10 <= n <= 500: continue
                overlap = selected & members; k = len(overlap)
                if not k: continue
                p = hypergeom.sf(k-1, M, n, N)
                pcrows.append({"component":pc,"source":source,"pathway":term,"overlap":k,"pathway_size":n,"odds_enrichment":k/N/(n/M),"p_value":p,"genes":";".join(sorted(overlap))})
        q = pd.DataFrame(pcrows)
        if len(q):
            order = np.argsort(q.p_value.to_numpy()); adjusted = np.empty(len(q)); adjusted[order] = np.minimum.accumulate((q.p_value.to_numpy()[order] * len(q) / np.arange(1,len(q)+1))[::-1])[::-1]
            q["fdr"] = np.minimum(adjusted, 1); rows.append(q)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def choose_components(pc, enr):
    k_full = int(np.searchsorted(pc.technical_cumulative_variance, .995) + 1)
    q = pc[pc.component.le(k_full)].copy()
    response_cols = [c for c in q if c.endswith("_abs_cosine")]
    q["max_biological_abs_cosine"] = q[response_cols].max(axis=1)
    significant = set(enr.loc[enr.fdr.lt(.05), "component"]) if len(enr) else set()
    q["has_significant_pathway"] = q.component.isin(significant)
    # Exploratory only: NASA overlap contributes to this label.
    q["candidate_class"] = np.where((q.max_biological_abs_cosine < .10) & ~q.has_significant_pathway,
                                     "predominantly_technical_candidate", "technical_biological_shared_or_uncertain")
    return k_full, q


def pair_retrieval(poly, ribo):
    p = poly / np.linalg.norm(poly, axis=1, keepdims=True); r = ribo / np.linalg.norm(ribo, axis=1, keepdims=True)
    ranks=[]
    for query, db in ((p,r),(r,p)):
        sim=query@db.T
        for i in range(len(query)): ranks.append(int(np.where(np.argsort(-sim[i]) == i)[0][0])+1)
    ranks=np.asarray(ranks)
    return {"pair_r1":np.mean(ranks<=1),"pair_r5":np.mean(ranks<=5),"pair_r10":np.mean(ranks<=10),"pair_mrr":np.mean(1/ranks),"median_rank":np.median(ranks)}


def loo_probe(poly, ribo):
    ys=[]; ps=[]; ss=[]
    for i in range(len(poly)):
        tr=np.arange(len(poly))!=i
        xtr=np.vstack([poly[tr],ribo[tr]]); ytr=np.r_[np.zeros(tr.sum()),np.ones(tr.sum())]
        model=LogisticRegression(max_iter=3000, C=1).fit(xtr,ytr)
        xt=np.vstack([poly[i],ribo[i]]); ys += [0,1]; ps.extend(model.predict(xt)); ss.extend(model.predict_proba(xt)[:,1])
    return {"library_auroc":roc_auc_score(ys,ss),"library_balanced_accuracy":balanced_accuracy_score(ys,ps),"library_macro_f1":f1_score(ys,ps,average="macro")}


def correction_comparison(poly, ribo, basis, responses, ids, allr, k_full, candidates):
    strategies={"uncorrected":basis[:0],"full_technical_correction":basis[:k_full],"selective_correction":basis[[c-1 for c in candidates]] if candidates else basis[:0]}
    mode_path=TASK3/"task3c_cluster_assignments.csv"
    md=pd.read_csv(mode_path)
    mapping=dict(zip(md.contrast_id,md.geometry_cluster))
    assert set(ids) == set(mapping), "Task 3 mode labels must cover the exact 14 contrasts"
    labels=np.array([mapping[x] for x in ids])
    rows=[]
    base_cos=np.array([[cosine(a,b) for b in allr] for a in allr])
    for name,B in strategies.items():
        cp=project_out(poly,B); cr=project_out(ribo,B); row={"strategy":name,"removed_components":";".join(map(str,[c for c in range(1,len(B)+1)] if name!='selective_correction' else candidates)),**loo_probe(cp,cr),**pair_retrieval(cp,cr),"paired_cosine":np.mean([cosine(a,b) for a,b in zip(cp,cr)])}
        corrected=project_out(allr,B); cm=np.array([[cosine(a,b) for b in corrected] for a in corrected])
        tri=np.triu_indices(len(allr),1); row["task3_response_matrix_spearman"]=spearmanr(base_cos[tri],cm[tri]).statistic
        row["mean_response_preservation"]=np.mean([cosine(a,b) for a,b in zip(allr,corrected)])
        if len(set(labels))==2:
            row["fixed_mode_silhouette"]=silhouette_score(corrected,labels,metric="cosine")
            dist=np.clip(1-cm,0,2); np.fill_diagonal(dist,0)
            clusters=fcluster(linkage(squareform(dist,checks=False),method="average"),2,criterion="maxclust")
            row["cluster_ari"]=adjusted_rand_score(labels,clusters)
        for comp,(a,b) in responses.items(): row[f"{comp}_reproducibility"]=cosine(project_out(a[None],B)[0],project_out(b[None],B)[0])
        row["RR3_39_vs_RR3_40_agreement_original"]=cosine(project_out(responses['RR3-39'][0][None],B)[0],project_out(responses['RR3-40'][0][None],B)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def plots(pc, cumulative, correction, enr):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(pc.component,pc.technical_variance_fraction*100,color="#377eb8"); ax.plot(pc.component,pc.technical_cumulative_variance*100,color="#d95f02",marker="o",ms=3); ax.set(xlabel="Uncentered difference PC",ylabel="Technical displacement energy (%)",title="Controlled PolyA→Ribo difference spectrum"); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(FIG/f"technical_difference_spectrum.{ext}",dpi=400); plt.close(fig) if ext=="pdf" else None
    fig,ax=plt.subplots(figsize=(9,5));
    for name,g in cumulative.groupby("response"): ax.plot(g.k,g.cumulative_energy_fraction,label=name)
    ax.plot(pc.component,pc.technical_cumulative_variance,"k--",label="Controlled technical variance"); ax.set(xlabel="PCs included",ylabel="Cumulative squared-magnitude fraction",ylim=(0,1.02),title="Biological-response overlap with the controlled difference basis"); ax.legend(ncol=2,fontsize=8); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(FIG/f"cumulative_response_overlap.{ext}",dpi=400); plt.close(fig) if ext=="pdf" else None
    fig,ax=plt.subplots(figsize=(7,5));
    for r in correction.itertuples(): ax.scatter(r.library_auroc,r.task3_response_matrix_spearman,s=100,label=r.strategy); ax.annotate(r.strategy,(r.library_auroc,r.task3_response_matrix_spearman),xytext=(5,5),textcoords="offset points")
    ax.set(xlabel="Residual PolyA/Ribo AUROC (lower is better)",ylabel="Task 3 response-matrix preservation",title="Technical-removal / biological-preservation tradeoff",xlim=(-.02,1.02),ylim=(-.02,1.02)); fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(FIG/f"correction_tradeoff.{ext}",dpi=400); plt.close(fig) if ext=="pdf" else None
    if len(enr):
        top=enr[enr.fdr.lt(.05)].sort_values(["component","fdr"]).groupby("component").head(2)
        terms=list(dict.fromkeys(top.pathway.tolist()))[:24]; pcs=sorted(top.component.unique())
        mat=np.full((len(terms),len(pcs)),np.nan)
        for i,t in enumerate(terms):
            for j,p in enumerate(pcs):
                z=enr[(enr.pathway==t)&(enr.component==p)];
                if len(z) and z.iloc[0].fdr<.05: mat[i,j]=-np.log10(max(z.iloc[0].fdr,1e-300))
        fig,ax=plt.subplots(figsize=(10,max(5,.3*len(terms)))); im=ax.imshow(mat,aspect="auto",cmap="viridis"); ax.set(xticks=range(len(pcs)),xticklabels=[f"PC{x}" for x in pcs],yticks=range(len(terms)),yticklabels=terms); fig.colorbar(im,ax=ax,label="−log10(FDR)"); ax.set_title("PC × pathway enrichment (Top-250 component genes)"); fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(FIG/f"pc_pathway_enrichment.{ext}",dpi=400,bbox_inches="tight"); plt.close(fig) if ext=="pdf" else None


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
    manifest,z,pairs,poly,ribo=load_controlled(); D=ribo-poly
    _,s,vt=np.linalg.svd(D,full_matrices=False)
    responses,ids,allr=load_responses()
    pc,vt=pc_metrics(poly,ribo,vt,s,responses); pc.to_csv(OUT/"paired_difference_svd.csv",index=False)
    scores=gene_component_scores(vt); scores.to_parquet(OUT/"technical_pc_gene_rankings.parquet",index=False)
    enr=pathway_enrichment(scores); enr.to_parquet(OUT/"pc_pathway_enrichment.parquet",index=False); enr.to_csv(OUT/"pc_pathway_enrichment.csv",index=False)
    k_full,classification=choose_components(pc,enr); classification.to_csv(OUT/"technical_pc_classification.csv",index=False)
    pc=pc.merge(classification[["component","max_biological_abs_cosine","has_significant_pathway","candidate_class"]],on="component",how="left"); pc.to_csv(OUT/"technical_pc_metrics.csv",index=False)
    cumulative=[]
    named={f"{n} {w}":r for n,pair in responses.items() for w,r in zip(("original","remeasurement"),pair)}
    for name,r in named.items():
        vals=(vt@r)**2/np.dot(r,r)
        for k,v in enumerate(np.cumsum(vals),1): cumulative.append({"response":name,"k":k,"cumulative_energy_fraction":v,"technical_cumulative_variance":pc.iloc[k-1].technical_cumulative_variance})
    cumulative=pd.DataFrame(cumulative); cumulative.to_csv(OUT/"cumulative_projection_results.csv",index=False)
    candidates=classification.loc[classification.candidate_class.eq("predominantly_technical_candidate"),"component"].astype(int).tolist()
    correction=correction_comparison(poly,ribo,vt,responses,ids,allr,k_full,candidates); correction.to_csv(OUT/"correction_comparison.csv",index=False)
    plots(pc,cumulative,correction,enr)
    summary={"controlled_pairs":40,"difference_rank":len(s),"full_correction_k":k_full,"full_cumulative_variance":float(pc.iloc[k_full-1].technical_cumulative_variance),"selective_components":candidates,"selection_warning":"Exploratory: independent NASA overlap contributes to the candidate component label; correction performance on NASA is descriptive, not confirmatory.","conclusion":"Protocol-associated components overlap biological response structure; no component is interpreted as purely technical."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (OUT/"provenance.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"controlled_embeddings":str(DATA/"bridgerna_embeddings.npy"),"contextual_cache":str(CTX),"svd":"uncentered SVD of 40 paired Ribo-minus-PolyA vectors","gene_universe":15165,"component_gene_set_size":250,"limitations":[summary["selection_warning"],"Pathway absence is not evidence that a component is purely technical.","The controlled reference is T-cell-specific; cross-tissue universality is unproven."]},indent=2)+"\n")
    print(pd.DataFrame([summary]).to_string(index=False)); print(correction.to_string(index=False)); print(f"[complete] {OUT}",flush=True)


if __name__ == "__main__": main()
