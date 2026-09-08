#!/usr/bin/env python3
"""Scientist-facing utility audit for frozen BridgeRNA contextual gene graphs.

This consumes the exact layer-12 Top-20 graph caches built by the frozen
sample-readout benchmark. It never runs or modifies BridgeRNA.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OLD = REPO / "benchmarks/frozen_sample_embedding_readout"
CACHE = OLD / "work/graph_fingerprint"
EX = REPO / "benchmarks/cross_species_exercise_response"
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
OUT = HERE / "results"
G = 15165
SEED = 20260907


def say(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def load_graphs(name: str):
    root = CACHE / name
    manifest = pd.read_parquet(root / "manifest.parquet")
    nbr = np.load(root / "neighbors_top20.uint16.npy", mmap_mode="r")
    weight = np.load(root / "weights_top20.float16.npy", mmap_mode="r")
    assert nbr.shape == weight.shape == (len(manifest), G, 20)
    return manifest, nbr, weight


def canonical_genes() -> pd.DataFrame:
    d = pd.read_csv(REPO / "data/ensembl/canonical_genes.csv").sort_values("token_id").reset_index(drop=True)
    # Repository token IDs are one-based; array positions are token_id - 1.
    assert len(d) == G and np.array_equal(d.token_id.to_numpy(), np.arange(1, G + 1))
    d["symbol"] = d.gene_symbol.astype(str).str.upper()
    return d


def graph_profile(nbr, weight, k=10, mutual=False):
    src = np.repeat(np.arange(G, dtype=np.int64), k)
    dst = np.asarray(nbr[:, :k], dtype=np.int64).ravel()
    val = np.asarray(weight[:, :k], dtype=np.float32).ravel()
    if mutual:
        sets = [set(map(int, nbr[i, :k])) for i in range(G)]
        keep = np.fromiter((int(i) in sets[int(j)] for i, j in zip(src, dst)), bool, len(src))
        src, dst, val = src[keep], dst[keep], val[keep]
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    code = a * G + b
    order = np.argsort(code)
    code, val = code[order], val[order]
    unique, start = np.unique(code, return_index=True)
    return unique, np.maximum.reduceat(val, start)


def mean_graph(indices, nbr, weight, k=10, mutual=False):
    rows, cols, vals = [], [], []
    for r, i in enumerate(indices):
        c, v = graph_profile(nbr[i], weight[i], k, mutual)
        rows.extend(np.full(len(c), r)); cols.extend(c); vals.extend(v)
    x = csr_matrix((vals, (rows, cols)), shape=(len(indices), G * G))
    return x.mean(axis=0).A1


def sparse_mean_graph(indices, nbr, weight, k=10):
    total = defaultdict(float)
    for i in indices:
        c, v = graph_profile(nbr[i], weight[i], k)
        for a, b in zip(c, v):
            total[int(a)] += float(b) / len(indices)
    codes = np.fromiter(total.keys(), dtype=np.int64)
    vals = np.fromiter(total.values(), dtype=np.float32)
    order = np.argsort(codes)
    return codes[order], vals[order]


def sparse_delta(treat, control):
    d = defaultdict(float)
    for c, v in zip(*treat): d[int(c)] += float(v)
    for c, v in zip(*control): d[int(c)] -= float(v)
    codes = np.array(sorted(d), dtype=np.int64)
    vals = np.array([d[int(c)] for c in codes], dtype=np.float32)
    keep = vals != 0
    return codes[keep], vals[keep]


def sparse_cos(a, b):
    codes = np.union1d(a[0], b[0]); x = np.zeros(len(codes)); y = np.zeros(len(codes))
    x[np.searchsorted(codes, a[0])] = a[1]; y[np.searchsorted(codes, b[0])] = b[1]
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den else np.nan


def pathway_sets():
    out = {"Hallmark": [set(map(str.upper, v)) for v in json.loads((REPO / "data/gsea/hallmark_gene_sets.json").read_text()).values()]}
    root = EX / "results/per_study_ranked_gsea"
    for source, fn in [("GO BP", "GO_Biological_Process_2026.gmt"), ("Reactome", "Reactome_Pathways_2024.gmt")]:
        sets = []
        for line in open(root / fn):
            z = line.rstrip().split("\t"); sets.append(set(map(str.upper, z[2:])))
        out[source] = sets
    return out


def relationship_recovery():
    target = OUT / "relationship_recovery"; target.mkdir(parents=True, exist_ok=True)
    genes = canonical_genes(); symbols = genes.symbol.tolist(); sym_to_i = {g: i for i, g in enumerate(symbols)}
    manifest, nbr, weight = load_graphs("tissue")
    # Deterministic, study-diverse subset: at most one sample per GSE, capped at 600.
    pick = manifest.drop_duplicates("gse").sample(min(600, manifest.gse.nunique()), random_state=SEED).index.to_numpy()
    edge_count, edge_weight = defaultdict(int), defaultdict(float)
    for z, i in enumerate(pick):
        c, v = graph_profile(nbr[i], weight[i], 10)
        for a, b in zip(c, v): edge_count[int(a)] += 1; edge_weight[int(a)] += float(b)
        if (z + 1) % 100 == 0: say(f"relationship graph aggregation {z+1}/{len(pick)}")
    sets = pathway_sets(); rng = np.random.default_rng(SEED); rows = []
    degree = np.zeros(G)
    for code, count in edge_count.items():
        degree[code // G] += count; degree[code % G] += count
    # Expression-level bins and graph-degree bins control the negative sampling.
    x = np.memmap(EX / "work/hallmark_readout/archs4_log1p_tpm.float32.mmap", mode="r", dtype="float32", shape=(40000, G))
    matrix_rows = manifest.loc[pick, "matrix_row"].to_numpy(int)
    X = np.asarray(x[matrix_rows], dtype=np.float32)
    means = X.mean(0); eb = pd.qcut(means, 20, labels=False, duplicates="drop"); db = pd.qcut(degree, 20, labels=False, duplicates="drop")
    for source, coll in sets.items():
        memberships = defaultdict(set)
        for s, members in enumerate(coll):
            for gene in members:
                if gene in sym_to_i: memberships[sym_to_i[gene]].add(s)
        positives = []
        for _ in range(200000):
            i, j = rng.integers(0, G, 2)
            if i != j and memberships[i] & memberships[j]: positives.append((min(i,j), max(i,j)))
            if len(set(positives)) >= 10000: break
        positives = list(set(positives))
        negatives = []
        for i, j in positives:
            candidates = np.flatnonzero((eb == eb[j]) & (db == db[j]))
            for _ in range(30):
                q = int(rng.choice(candidates))
                if q != i and not (memberships[i] & memberships[q]): negatives.append((min(i,q), max(i,q))); break
        pairs = positives + negatives; y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
        bridge = np.array([edge_weight.get(i*G+j, 0.0)/len(pick) for i,j in pairs])
        raw = np.array([np.corrcoef(X[:,i], X[:,j])[0,1] for i,j in pairs]); raw = np.nan_to_num(raw)
        for method, score in [("BridgeRNA contextual graph", bridge), ("Raw expression correlation", raw)]:
            order = np.argsort(-score); rec = {"source":source,"method":method,"pairs":len(y),"positive_pairs":len(positives),"AUROC":roc_auc_score(y,score),"AUPRC":average_precision_score(y,score)}
            for frac in [.01,.05,.10]: rec[f"top_{int(frac*100)}pct_enrichment"] = y[order[:max(1,int(len(y)*frac))]].mean()/y.mean()
            rows.append(rec)
    pd.DataFrame(rows).to_csv(target / "relationship_recovery_summary.csv", index=False)
    pd.DataFrame({"sample_index":pick,"gse":manifest.loc[pick,"gse"].to_numpy()}).to_csv(target / "relationship_sample_manifest.csv",index=False)
    (target / "provenance.json").write_text(json.dumps({"samples":len(pick),"sampling":"one sample per GSE; deterministic cap 600","bridge_score":"mean union-k10 contextual edge weight; absent Top-10 edge scored zero","negative_matching":["expression mean ventile","aggregate graph degree ventile"],"PPI_TF":"not evaluated: no authoritative local resource","PCA_gene_relationship":"not separately evaluated because full-rank gene cosine in PCA is a rotation/re-expression of centered coexpression; raw correlation is the interpretable conventional comparator","seed":SEED},indent=2)+"\n")


def build_effects(name, k=10):
    m, nbr, weight = load_graphs(name); effects = {}; meta = []
    if name == "exercise":
        members = pd.read_parquet(EX / "results/contrast_members.parquet"); loc = dict(zip(m.GSM.astype(str), range(len(m))))
        for cid, q in members.groupby("contrast_id"):
            q=q[q.GSM.astype(str).isin(loc)]; role=q.role.astype(str).str.lower(); ti=q[role.str.contains("post|exercise")].GSM.astype(str).map(loc).to_numpy(int); ci=q[~role.str.contains("post|exercise")].GSM.astype(str).map(loc).to_numpy(int)
            if len(ti) and len(ci):
                effects[cid]=sparse_delta(sparse_mean_graph(ti,nbr,weight,k),sparse_mean_graph(ci,nbr,weight,k)); meta.append({"contrast_id":cid,"dataset":"exercise","species":q.species.iloc[0],"n_treatment":len(ti),"n_control":len(ci)})
    elif name == "task3":
        members=pd.read_csv(T3 / "results/task3b_contrast_sample_membership.csv"); loc=dict(zip(m.sample_id.astype(str),range(len(m))))
        for cid,q in members.groupby("contrast_id"):
            q=q[q.sample_id.astype(str).isin(loc)]; ti=q[q.condition.eq("FLT")].sample_id.map(loc).to_numpy(int); ci=q[q.condition.eq("GC")].sample_id.map(loc).to_numpy(int)
            if len(ti) and len(ci):
                effects[cid]=sparse_delta(sparse_mean_graph(ti,nbr,weight,k),sparse_mean_graph(ci,nbr,weight,k)); meta.append({"contrast_id":cid,"dataset":"spaceflight","species":"mouse","n_treatment":len(ti),"n_control":len(ci)})
    elif name == "tcell":
        loc=np.arange(len(m)); ti=loc[m.library_prep.eq("ribo")]; ci=loc[m.library_prep.eq("polyA")]
        effects["controlled_Tcell_Ribo_minus_PolyA"]=sparse_delta(sparse_mean_graph(ti,nbr,weight,k),sparse_mean_graph(ci,nbr,weight,k));meta.append({"contrast_id":"controlled_Tcell_Ribo_minus_PolyA","dataset":"technical_control","species":"human","n_treatment":len(ti),"n_control":len(ci)})
    return effects,pd.DataFrame(meta)


def edge_tables(effects, metadata, k):
    genes=canonical_genes().symbol.to_numpy(); edge_rows=[]; gene_rows=[]
    for cid,(codes,values) in effects.items():
        ix=np.argsort(-np.abs(values)); top=ix[:20]
        for rank,q in enumerate(top,1):
            a,b=divmod(int(codes[q]),G);edge_rows.append({"contrast_id":cid,"k":k,"rank":rank,"gene_A":genes[a],"gene_B":genes[b],"delta_weight":values[q],"change":"gained/strengthened" if values[q]>0 else "lost/weakened"})
        node=np.zeros(G); np.add.at(node,codes//G,np.abs(values));np.add.at(node,codes%G,np.abs(values))
        signed=np.zeros(G);np.add.at(signed,codes//G,values);np.add.at(signed,codes%G,values)
        for rank,q in enumerate(np.argsort(-node)[:20],1):gene_rows.append({"contrast_id":cid,"k":k,"rank":rank,"gene":genes[q],"neighborhood_turnover":node[q],"signed_strength_change":signed[q]})
    return pd.DataFrame(edge_rows),pd.DataFrame(gene_rows)


def rewiring_and_conservation():
    r2=OUT/"perturbation_rewiring";r3=OUT/"cross_study";r4=OUT/"cross_species";r5=OUT/"attribution_validation"
    for p in [r2,r3,r4,r5]:p.mkdir(parents=True,exist_ok=True)
    all_effects={};all_meta=[]
    for k in [5,10,20]:
        for name in ["exercise","task3","tcell"]:
            eff,meta=build_effects(name,k); edges,genes=edge_tables(eff,meta,k);edges.to_csv(r2/f"{name}_top_edges_k{k}.csv",index=False);genes.to_csv(r2/f"{name}_top_rewired_genes_k{k}.csv",index=False)
            if k==10: all_effects.update(eff);all_meta.append(meta)
    meta=pd.concat(all_meta,ignore_index=True);meta.to_csv(r2/"contrast_manifest.csv",index=False)
    ids=list(all_effects);rows=[]
    for i,a in enumerate(ids):
        for b in ids[i+1:]: rows.append({"contrast_A":a,"contrast_B":b,"weighted_edge_response_cosine":sparse_cos(all_effects[a],all_effects[b])})
    pair=pd.DataFrame(rows);pair.to_csv(r3/"all_pair_graph_response_similarity.csv",index=False)
    exmeta=meta[meta.dataset.eq("exercise")].set_index("contrast_id"); exids=exmeta.index.tolist(); cross=[]
    for h in [x for x in exids if exmeta.loc[x,"species"]=="human"]:
        for m in [x for x in exids if exmeta.loc[x,"species"]=="mouse"]:
            a,b=all_effects[h],all_effects[m];topa=set(a[0][np.argsort(-np.abs(a[1]))[:1000]]);topb=set(b[0][np.argsort(-np.abs(b[1]))[:1000]])
            cross.append({"human_contrast":h,"mouse_contrast":m,"weighted_edge_response_cosine":sparse_cos(a,b),"top1000_edge_jaccard":len(topa&topb)/len(topa|topb)})
    pd.DataFrame(cross).to_csv(r4/"human_mouse_exercise_network_conservation.csv",index=False)
    # Overlap graph-important genes with existing IG and DE, without rerunning either.
    graph=pd.read_csv(r2/"exercise_top_rewired_genes_k10.csv"); ig=pd.read_csv(EX/"results/latent_axis_attribution/axis_consensus_attributed_genes.csv")
    igcols=[c for c in ig.columns if "gene" in c.lower()]; iggenes=set(ig[igcols[0]].astype(str).str.upper()) if igcols else set()
    degenes=set()
    for p in (EX/"results/full_transcriptome_de").glob("*_full_de.parquet"):
        d=pd.read_parquet(p); gc=next((c for c in d if c.lower() in ["gene_symbol","symbol","gene"]),None)
        if gc: degenes.update(d.sort_values(next((c for c in d if c.lower() in ["padj","fdr"]),d.columns[-1])).head(100)[gc].astype(str).str.upper())
    graph["in_existing_IG_results"]=graph.gene.str.upper().isin(iggenes);graph["in_top100_DE_any_exercise_study"]=graph.gene.str.upper().isin(degenes);graph.to_csv(r5/"graph_gene_ig_de_overlap.csv",index=False)
    pd.DataFrame([{"analysis":"graph-guided deletion","status":"not_run","reason":"requires new frozen-model inference; no compatible graph-ranked deletion cache exists"}]).to_csv(r5/"deletion_status.csv",index=False)
    return meta,pair


def figures_and_summary(meta,pair):
    figdir=OUT/"figures";summary=OUT/"summary";figdir.mkdir(parents=True,exist_ok=True);summary.mkdir(parents=True,exist_ok=True)
    rel=pd.read_csv(OUT/"relationship_recovery/relationship_recovery_summary.csv")
    fig,ax=plt.subplots(figsize=(9,4.8)); sources=list(rel.source.unique()); methods=list(rel.method.unique()); x=np.arange(len(sources)); width=.36
    for j,method in enumerate(methods):
        q=rel.set_index(["source","method"]).reindex(pd.MultiIndex.from_product([sources,[method]])).reset_index();ax.bar(x+(j-.5)*width,q.AUPRC,width,label=method)
    ax.set_xticks(x,sources);ax.legend(frameon=False);ax.set_title("Known functional relationship recovery");ax.set_ylabel("AUPRC");fig.tight_layout();fig.savefig(figdir/"relationship_recovery.png",dpi=300);fig.savefig(figdir/"relationship_recovery.pdf");plt.close(fig)
    ex=meta[meta.dataset.eq("exercise")].contrast_id.tolist();mat=pd.DataFrame(np.eye(len(ex)),index=ex,columns=ex)
    lookup={(r.contrast_A,r.contrast_B):r.weighted_edge_response_cosine for r in pair.itertuples()}
    for i,a in enumerate(ex):
        for j,b in enumerate(ex):
            if i!=j:mat.loc[a,b]=lookup.get((a,b),lookup.get((b,a),np.nan))
    mat.to_csv(OUT/"cross_study/exercise_graph_response_matrix.csv")
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(mat.to_numpy(),cmap="coolwarm",vmin=-1,vmax=1);ax.set_xticks(range(len(ex)),ex,rotation=90);ax.set_yticks(range(len(ex)),ex);fig.colorbar(im,ax=ax,label="signed edge-response cosine");ax.set_title("Exercise contextual-network response similarity");fig.tight_layout();fig.savefig(figdir/"exercise_response_similarity.png",dpi=300);fig.savefig(figdir/"exercise_response_similarity.pdf");plt.close(fig)
    # Machine-readable evidence classification; conservative by construction.
    b=rel.pivot(index="source",columns="method",values="AUPRC")
    contextual_wins=int((b["BridgeRNA contextual graph"]>b["Raw expression correlation"]).sum())
    decision="TASK-SPECIFIC NETWORK UTILITY" if contextual_wins not in (0,len(b)) else ("STRONG CONTEXTUAL-NETWORK UTILITY" if contextual_wins==len(b) else "MOSTLY COEXPRESSION RECAPITULATION")
    pd.DataFrame([{"decision":decision,"annotation_sources_contextual_AUPRC_above_raw":contextual_wins,"annotation_sources_total":len(b),"graph_guided_deletion_run":False}]).to_csv(summary/"decision_summary.csv",index=False)
    (summary/"provenance.json").write_text(json.dumps({"checkpoint":"r7hnr92k","encoder":"frozen; no inference in this benchmark","genes":G,"layer":12,"primary_graph":"cosine kNN k=10 union symmetrized weighted","sensitivity_k":[5,20],"source_graph_cache":str(CACHE.relative_to(REPO)),"communities":"not used as primary due prior poor reproducibility","claim_discipline":"contextual relationships, not causal/regulatory edges","limitations":["Top-k graph cache censors non-neighbor contextual cosines","cross-study graph response is exploratory for small curated contrast sets","graph-guided deletion not run"],"seed":SEED},indent=2)+"\n")


def main():
    for p in ["relationship_recovery","perturbation_rewiring","cross_study","cross_species","attribution_validation","figures","summary"]:(OUT/p).mkdir(parents=True,exist_ok=True)
    say("estimated runtime: 4-9 minutes using existing graph caches; no GPU/model inference")
    relationship_recovery();say("Task 1 complete")
    meta,pair=rewiring_and_conservation();say("Tasks 2-5 complete")
    figures_and_summary(meta,pair);say("BENCHMARK COMPLETE")


if __name__ == "__main__": main()
