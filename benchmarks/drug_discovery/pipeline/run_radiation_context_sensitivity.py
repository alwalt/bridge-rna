#!/usr/bin/env python3
"""Radiation context sensitivity using frozen contrast-level rankings only."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "radiation_context_sensitivity"
CANONICAL = Path("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")
EDGE_FILE = HERE.parent / "deweerd_replication/results/evaluation2_expanded/chembl37_broad_drug_target_edges.csv.gz"
RADIATION = ["GSE297090_gamma", "GSE297090_proton", "GSE184119_10Gy",
             "GSE297560_gamma", "GSE297560_proton", "GSE297560_iron", "GSE297560_silicon"]
META = {
    "GSE297090_gamma": ("GSE297090", "intestinal_organoid", "gamma", "1Gy"),
    "GSE297090_proton": ("GSE297090", "intestinal_organoid", "proton", "1Gy"),
    "GSE184119_10Gy": ("GSE184119", "mixed_six_cell_types", "gamma", "10Gy"),
    "GSE297560_gamma": ("GSE297560", "dermal_fibroblast", "gamma", "4Gy"),
    "GSE297560_proton": ("GSE297560", "dermal_fibroblast", "proton", "4Gy"),
    "GSE297560_iron": ("GSE297560", "dermal_fibroblast", "iron", "4Gy"),
    "GSE297560_silicon": ("GSE297560", "dermal_fibroblast", "silicon", "4Gy"),
}
BLOCK_CELL = {"ADSC": "adipose_stromal", "HMEC": "mammary_epithelial", "LEC": "lung_endothelial",
              "NHDF": "dermal_fibroblast", "NHEK": "keratinocyte", "PERI": "pericyte"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def classify(a, b):
    da, ca, ma, _ = META[a]; db, cb, mb, _ = META[b]
    if da == db:
        return 1, "same system, different radiation exposure"
    if ca == cb:
        return 2, "similar cell type, different study/exposure"
    if ma == mb:
        return 3, "different cell type, similar radiation exposure"
    return 4, "different cell type and different radiation exposure"


NULL_CACHE = {}


def matched_draws(gene_set, bins, rng, reps):
    key = (id(bins), tuple(sorted(gene_set)), reps)
    if key in NULL_CACHE:
        return NULL_CACHE[key]
    wanted = bins.loc[list(gene_set)].value_counts().to_dict()
    candidates = {expression_bin: bins.index[bins.eq(expression_bin)].to_numpy()
                  for expression_bin in wanted}
    draws = []
    for _ in range(reps):
        result = []
        for expression_bin, count in wanted.items():
            result.extend(rng.choice(candidates[expression_bin], int(count), replace=False))
        draws.append(set(result))
    NULL_CACHE[key] = draws
    return draws


def pair_metrics(a, b, set_a, set_b, ranks_a, ranks_b, signs_a, signs_b,
                 bins_a, bins_b, rng, reps=2000):
    overlap = set_a & set_b
    union = set_a | set_b
    observed = len(overlap)
    draws_a = matched_draws(set_a, bins_a, rng, reps)
    draws_b = matched_draws(set_b, bins_b, rng, reps)
    null = np.fromiter((len(x & y) for x, y in zip(draws_a, draws_b)), dtype=int, count=reps)
    rho = spearmanr([ranks_a[g] for g in overlap], [ranks_b[g] for g in overlap]).statistic if len(overlap) >= 10 else np.nan
    concordance = np.mean([np.sign(signs_a[g]) == np.sign(signs_b[g]) for g in overlap]) if overlap else np.nan
    return {"overlap_count": observed, "jaccard": observed/max(len(union), 1),
            "overlap_rank_spearman": rho, "sign_concordance": concordance,
            "matched_random_mean_overlap": float(null.mean()),
            "matched_random_sd_overlap": float(null.std(ddof=1)),
            "matched_random_empirical_p": float((1 + np.sum(null >= observed))/(reps+1)),
            "random_replicates": reps}


def enrich(module, universe, targets):
    rows=[]; module=set(module)&universe
    for (did,name), genes in targets.items():
        genes &= universe; k=len(module&genes); K=len(genes); n=len(module); M=len(universe)
        p=float(hypergeom.sf(k-1,M,K,n)); d=M-n-K+k
        odds=k*d/((n-k)*(K-k)) if (n-k) and (K-k) else (np.inf if k else np.nan)
        rows.append({"drug_id":did,"drug_name":name,"target_count":K,"overlap_count":k,
                     "overlap_genes":";".join(sorted(module&genes)),"odds_ratio":odds,"pvalue":p})
    out=pd.DataFrame(rows); out["fdr"]=multipletests(out.pvalue,method="fdr_bh")[1]
    out=out.sort_values(["fdr","pvalue","odds_ratio","drug_id"],ascending=[True,True,False,True]).reset_index(drop=True)
    out["rank"]=np.arange(1,len(out)+1); return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(CANONICAL).gene_symbol.astype(str).tolist(); universe=set(genes)
    bridge_all = pd.read_parquet(RESULTS/"bridge/contrast_gene_scores.parquet")
    bridge_all = bridge_all[bridge_all.contrast_id.isin(RADIATION)]
    tables={}; sets={}; ranks={}; signs={}; eligible={}
    for cid in RADIATION:
        b=bridge_all[bridge_all.contrast_id.eq(cid)].copy().sort_values(["absolute_bridge_score","gene"],ascending=[False,True])
        d=pd.read_csv(RESULTS/f"differential_expression/{cid}.csv.gz").copy()
        d["absolute_score"]=d.t.abs(); d=d.sort_values(["absolute_score","gene"],ascending=[False,True])
        for method,frame,score,absolute,elig in [("Bridge",b,"signed_bridge_score","absolute_bridge_score",b.absolute_bridge_score.gt(0)),
                                                ("DE",d,"t","absolute_score",d.eligible)]:
            frame=frame.loc[elig].copy(); frame["rank"]=np.arange(1,len(frame)+1)
            tables[(method,cid)]=frame; sets[(method,cid,"top500")]=set(frame.head(500).gene)
            ranks[(method,cid)] = frame.set_index("gene")["rank"].to_dict()
            signs[(method,cid)] = frame.set_index("gene")[score].to_dict()
            eligible[(method,cid)] = set(frame.gene)
        sets[("Bridge-only",cid,"partition")]=sets[("Bridge",cid,"top500")]-sets[("DE",cid,"top500")]
        sets[("DE-only",cid,"partition")]=sets[("DE",cid,"top500")]-sets[("Bridge",cid,"top500")]

    # Expression-decile matching from already-frozen prepared inputs.
    bins={}
    for dataset in sorted({META[c][0] for c in RADIATION}):
        x=np.load(HERE/f"work/prepared/{dataset}_log1p_tpm.npy",mmap_mode="r")
        mean=pd.Series(np.asarray(x).mean(0),index=genes)
        detected=mean[mean.gt(0)]
        bins[dataset]=pd.qcut(detected.rank(method="first"),10,labels=False)

    rng=np.random.default_rng(370290)
    pair_rows=[]
    for a,b in itertools.combinations(RADIATION,2):
        distance,label=classify(a,b)
        for analysis,method,set_key in [("Bridge top-500","Bridge","top500"),("DE top-500","DE","top500"),
                                        ("Bridge-only","Bridge-only","partition"),("DE-only","DE-only","partition")]:
            ra_method="Bridge" if method=="Bridge-only" else "DE" if method=="DE-only" else method
            result=pair_metrics(a,b,sets[(method,a,set_key)],sets[(method,b,set_key)],
                                ranks[(ra_method,a)],ranks[(ra_method,b)],signs[(ra_method,a)],signs[(ra_method,b)],
                                bins[META[a][0]],bins[META[b][0]],rng)
            pair_rows.append({"contrast_1":a,"contrast_2":b,"dataset_1":META[a][0],"dataset_2":META[b][0],
                              "cell_context_1":META[a][1],"cell_context_2":META[b][1],
                              "modality_1":META[a][2],"modality_2":META[b][2],
                              "context_distance":distance,"context_class":label,"analysis":analysis,
                              "set_size_1":len(sets[(method,a,set_key)]),"set_size_2":len(sets[(method,b,set_key)]),**result})
    pairwise=pd.DataFrame(pair_rows)
    pairwise["matched_random_fdr"] = pairwise.groupby("analysis")["matched_random_empirical_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    pairwise.to_csv(OUT/"contrast_pair_similarity.csv",index=False)

    # Context-distance decay and Bridge-vs-DE paired stability.
    trend_rows=[]
    for analysis,g in pairwise.groupby("analysis"):
        rho=spearmanr(g.context_distance,g.jaccard).statistic
        null=[]
        for _ in range(10000): null.append(spearmanr(rng.permutation(g.context_distance),g.jaccard).statistic)
        p=(1+sum(abs(x)>=abs(rho)-1e-15 for x in null))/(1+len(null))
        slope=np.polyfit(g.context_distance,g.jaccard,1)[0]
        trend_rows.append({"analysis":analysis,"pairs":len(g),"distance_jaccard_spearman":rho,
                           "permutation_p":p,"linear_slope_per_distance":slope})
    pd.DataFrame(trend_rows).to_csv(OUT/"context_distance_trends.csv",index=False)
    pair_summary=pairwise.groupby(["analysis","context_distance","context_class"],as_index=False).agg(
        pairs=("jaccard","size"),mean_jaccard=("jaccard","mean"),median_jaccard=("jaccard","median"),
        mean_overlap=("overlap_count","mean"),mean_random_p=("matched_random_empirical_p","mean"),
        significant_random_pairs=("matched_random_empirical_p",lambda x:int((x<.05).sum())))
    pair_summary.to_csv(OUT/"context_distance_summary.csv",index=False)

    # Direct, pair-matched Bridge-minus-DE stability comparisons.
    comparison = pairwise[pairwise.analysis.isin(["Bridge top-500", "DE top-500"])].pivot(
        index=["contrast_1", "contrast_2", "context_distance", "context_class"],
        columns="analysis", values="jaccard").reset_index()
    comparison["bridge_minus_de_jaccard"] = comparison["Bridge top-500"] - comparison["DE top-500"]
    stability_rows = []
    for distance, group in [("all", comparison)] + list(comparison.groupby("context_distance")):
        statistic, pvalue = wilcoxon(group["bridge_minus_de_jaccard"], alternative="two-sided")
        stability_rows.append({
            "context_distance": distance,
            "pairs": len(group),
            "mean_bridge_jaccard": group["Bridge top-500"].mean(),
            "mean_de_jaccard": group["DE top-500"].mean(),
            "mean_bridge_minus_de": group["bridge_minus_de_jaccard"].mean(),
            "bridge_higher_pairs": int(group["bridge_minus_de_jaccard"].gt(0).sum()),
            "wilcoxon_statistic": statistic,
            "wilcoxon_pvalue": pvalue,
        })
    pd.DataFrame(stability_rows).to_csv(OUT/"bridge_vs_de_stability.csv",index=False)

    # Frozen GSE184119 block contributions permit Bridge-only cell-type views.
    block=np.load(HERE/"work/bridge/GSE184119_10Gy_block_attribution_contributions.npz")
    block_rows=[]; block_sets={}; block_ranks={}; block_signs={}
    for raw_name,values in zip(block["names"],block["values"]):
        cell=raw_name.strip("()',")
        order=np.lexsort((np.asarray(genes),-np.abs(values)))
        ordered=np.asarray(genes)[order]; cid=f"GSE184119_{cell}_block"
        block_sets[cid]=set(ordered[:500]); block_ranks[cid]={g:i+1 for i,g in enumerate(ordered)}
        block_signs[cid]=dict(zip(genes,values))
        for rank,gene in enumerate(ordered[:500],1):
            block_rows.append({"contrast_id":cid,"dataset":"GSE184119","cell_type":cell,
                               "gene":gene,"rank":rank,"signed_bridge_score":float(values[genes.index(gene)])})
    pd.DataFrame(block_rows).to_csv(OUT/"gse184119_frozen_block_rankings.csv.gz",index=False,compression="gzip")
    block_pair_rows=[]
    # Within GSE184119: same exposure, different cell types (distance 3).
    for a,b in itertools.combinations(block_sets,2):
        result=pair_metrics(a,b,block_sets[a],block_sets[b],block_ranks[a],block_ranks[b],block_signs[a],block_signs[b],
                            bins["GSE184119"],bins["GSE184119"],rng)
        block_pair_rows.append({"contrast_1":a,"contrast_2":b,"cell_context_1":BLOCK_CELL[a.split('_')[1]],
                                "cell_context_2":BLOCK_CELL[b.split('_')[1]],"context_distance":3,
                                "context_class":"different cell type, same 10Gy gamma exposure",**result})
    # Each frozen cell block against organoid and OSDR contrast rankings.
    for block_id in block_sets:
        cell=block_id.split("_")[1]; cell_context=BLOCK_CELL[cell]
        for other in [x for x in RADIATION if x!="GSE184119_10Gy"]:
            other_context=META[other][1]
            if cell_context==other_context: dist=2; label="similar cell type, different study/exposure"
            elif META[other][2]=="gamma": dist=3; label="different cell type, similar radiation exposure"
            else: dist=4; label="different cell type and different radiation exposure"
            result=pair_metrics(block_id,other,block_sets[block_id],sets[("Bridge",other,"top500")],
                                block_ranks[block_id],ranks[("Bridge",other)],block_signs[block_id],signs[("Bridge",other)],
                                bins["GSE184119"],bins[META[other][0]],rng)
            block_pair_rows.append({"contrast_1":block_id,"contrast_2":other,"cell_context_1":cell_context,
                                    "cell_context_2":other_context,"context_distance":dist,"context_class":label,**result})
    block_pairs=pd.DataFrame(block_pair_rows); block_pairs.to_csv(OUT/"bridge_cell_context_similarity.csv",index=False)

    # Fibroblast analysis: frozen NHDF Bridge block vs OSDR; DE is pooled-only.
    fibro=[]; nhdf="GSE184119_NHDF_block"; pooled_de=sets[("DE","GSE184119_10Gy","top500")]
    pooled_de_r=ranks[("DE","GSE184119_10Gy")]; pooled_de_s=signs[("DE","GSE184119_10Gy")]
    for other in [x for x in RADIATION if META[x][0]=="GSE297560"]:
        br=pair_metrics(nhdf,other,block_sets[nhdf],sets[("Bridge",other,"top500")],block_ranks[nhdf],ranks[("Bridge",other)],
                        block_signs[nhdf],signs[("Bridge",other)],bins["GSE184119"],bins["GSE297560"],rng)
        de=pair_metrics("GSE184119_10Gy",other,pooled_de,sets[("DE",other,"top500")],pooled_de_r,ranks[("DE",other)],
                        pooled_de_s,signs[("DE",other)],bins["GSE184119"],bins["GSE297560"],rng)
        for method,result,scope in [("Bridge",br,"NHDF-specific frozen attribution block"),("DE",de,"pooled six-cell common radiation coefficient")]:
            fibro.append({"osdr_contrast":other,"osdr_modality":META[other][2],"method":method,"terrestrial_scope":scope,**result})
    pd.DataFrame(fibro).to_csv(OUT/"fibroblast_comparison.csv",index=False)

    # Core/context-specific sets across seven frozen contrasts.
    core_rows=[]; core_sets={}
    for method in ["Bridge","DE"]:
        counts={}; where={}
        for cid in RADIATION:
            for gene in sets[(method,cid,"top500")]:
                counts[gene]=counts.get(gene,0)+1; where.setdefault(gene,[]).append(cid)
        strict={g for g,n in counts.items() if n==len(RADIATION)}
        majority={g for g,n in counts.items() if n>=4 and len({META[c][0] for c in where[g]})>=2}
        specific={g for g,n in counts.items() if n==1}
        core_sets[(method,"strict_universal")]=strict; core_sets[(method,"majority_cross_study_core")]=majority
        core_sets[(method,"context_specific")]=specific
        for gene,n in counts.items():
            cls="strict_universal" if gene in strict else "majority_cross_study_core" if gene in majority else "context_specific" if gene in specific else "intermediate"
            core_rows.append({"method":method,"gene":gene,"contrast_count":n,
                              "dataset_count":len({META[c][0] for c in where[gene]}),"contrasts":";".join(where[gene]),"classification":cls})
    core_frame = pd.DataFrame(core_rows)
    core_frame.to_csv(OUT/"radiation_core_genes.csv",index=False)

    # Recurrent Bridge-only genes with all requested annotations except pathway,
    # which is joined after the frozen-background pathway query.
    counts={}; where={}
    for cid in RADIATION:
        for gene in sets[("Bridge-only",cid,"partition")]:
            counts[gene]=counts.get(gene,0)+1; where.setdefault(gene,[]).append(cid)
    edges=pd.read_csv(EDGE_FILE); targetable=set(edges.target_gene)
    targets=edges.groupby(["drug_id","drug_name"]).target_gene.agg(set); targets=targets[targets.map(len).gt(10)]
    recurrent_rows=[]
    for gene,n in counts.items():
        if n<2: continue
        cs=where[gene]
        recurrent_rows.append({"gene":gene,"contrast_count":n,"dataset_count":len({META[c][0] for c in cs}),
                               "contrasts":";".join(cs),"appears_in_osd993":any(META[c][0]=="GSE297560" for c in cs),
                               "expanded_chembl_target":gene in targetable,
                               "eligible_drug_count":int(sum(gene in x for x in targets)),
                               "bridge_ranks":";".join(f"{c}:{ranks[('Bridge',c)][gene]}" for c in cs),
                               "bridge_signs":";".join(f"{c}:{np.sign(signs[('Bridge',c)][gene]):+g}" for c in cs)})
    recurrent=pd.DataFrame(recurrent_rows).sort_values(["dataset_count","contrast_count","gene"],ascending=[False,False,True])
    recurrent.to_csv(OUT/"recurrent_bridge_only_genes.csv",index=False)
    recurrent_cross=set(recurrent.loc[recurrent.dataset_count.ge(2),"gene"])

    detail_rows=[]
    for row in core_frame.itertuples():
        cs=str(row.contrasts).split(";")
        method=row.method
        values=[np.sign(signs[(method,c)][row.gene]) for c in cs]
        detail_rows.append({
            **row._asdict(),
            "ranks":";".join(f"{c}:{ranks[(method,c)][row.gene]}" for c in cs),
            "signs":";".join(f"{c}:{np.sign(signs[(method,c)][row.gene]):+g}" for c in cs),
            "majority_sign_fraction":max(values.count(-1),values.count(1))/len(values),
            "expanded_chembl_target":row.gene in targetable,
            "eligible_drug_count":int(sum(row.gene in x for x in targets)),
        })
    pd.DataFrame(detail_rows).to_csv(OUT/"radiation_core_gene_details.csv.gz",index=False,compression="gzip")

    mapping_rows=[]
    annotated_genes=set(core_frame.loc[core_frame.classification.isin(
        ["strict_universal","majority_cross_study_core"]),"gene"]) | set(recurrent_cross)
    for (drug_id,drug_name), drug_targets in targets.items():
        for gene in sorted(drug_targets & annotated_genes):
            mapping_rows.append({"gene":gene,"drug_id":drug_id,"drug_name":drug_name,
                                 "eligible_drug_target_count":len(drug_targets)})
    pd.DataFrame(mapping_rows).to_csv(OUT/"core_expanded_chembl_target_mapping.csv",index=False)

    # Secondary expanded-ChEMBL mapping for core sets.
    drug_parts=[]
    for (method,label),gene_set in core_sets.items():
        if label=="context_specific": continue
        result=enrich(gene_set,universe,targets); result.insert(0,"core_definition",label);result.insert(0,"method",method)
        drug_parts.append(result)
    rr=enrich(recurrent_cross,universe,targets);rr.insert(0,"core_definition","recurrent_cross_study_bridge_only");rr.insert(0,"method","Bridge-only")
    drug_parts.append(rr)
    pd.concat(drug_parts,ignore_index=True).to_csv(OUT/"core_expanded_chembl_enrichment.csv",index=False)

    provenance={"analysis":"radiation context sensitivity from frozen outputs","module_size":500,
                "contrasts":RADIATION,"context_distance_definition":{1:"same system/different exposure",2:"similar cell/different study or exposure",3:"different cell/similar exposure",4:"different cell/different exposure"},
                "random_overlap":"2000 independent expression-decile-matched random sets per pair; fixed seed 370290",
                "gse184119_limitation":"frozen contrast has pooled common 10Gy effect across six cell types; frozen block attribution supports Bridge cell-specific analysis, but no frozen cell-specific DE or 2Gyx5 ranking exists",
                "input_hashes":{"bridge_contrast_scores":sha(RESULTS/"bridge/contrast_gene_scores.parquet"),
                                "gene_modules":sha(RESULTS/"evaluation/gene_modules.parquet"),"expanded_edges":sha(EDGE_FILE)},
                "protected_steps_rerun":{"phase2":False,"expanded_chembl":False,"de":False,"attribution":False,"modules":False}}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__": main()
