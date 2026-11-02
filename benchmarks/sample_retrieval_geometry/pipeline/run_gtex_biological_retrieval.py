#!/usr/bin/env python3
"""Independent GTEx -> ARCHS4 biological retrieval using frozen configurations."""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "gtex_biological_retrieval"
FIG = OUT / "figures"
WORK = HERE / "work"
GTEX_BASE = Path("/home/walt/bridge-rna/benchmarks/mouse_encode/results/task1a_balanced_geometry")
GTEX_MANIFEST = GTEX_BASE / "gtex_human_sample_manifest.parquet"
GTEX_REP_MANIFEST = GTEX_BASE / "cache/representation_manifest.parquet"
GTEX_EMBEDDINGS = GTEX_BASE / "cache/bridgerna_embeddings.npy"
GTEX_PROVENANCE = GTEX_BASE / "provenance.json"
METHODS = ["Raw cosine", "ABTT D=10", "Whitening", "NICDM", "Local Scaling", "CSLS"]
PRIMARY = METHODS[:-1]
REPS = METHODS[:3]
KS = [1, 5, 10, 100]
TOPK = 100
REF_CHUNK = 16384
QUERY_CHUNK = 256
MIN_REFS = 100
MIN_GSES = 5
SEED = 20260923

# Frozen before scoring: anchored whole-field matches only.
ARCHS4_TISSUES = {
    "blood": r"^(whole |peripheral )?blood$|^pbmcs?$",
    "brain": r"^brain( tissue)?$|^cerebral cortex$|^cortex$",
    "liver": r"^liver( tissue)?$", "lung": r"^lung( tissue)?$",
    "skin": r"^skin( tissue)?$", "breast": r"^(breast|mammary gland)( tissue)?$",
    "colon": r"^(colon|colonic mucosa)( tissue)?$",
    "kidney": r"^kidney( tissue)?$|^renal cortex$", "prostate": r"^prostate( tissue)?$",
    "pancreas": r"^pancreas( tissue)?$",
    "adipose": r"^(adipose|adipose tissue|subcutaneous adipose tissue)$",
    "ovary": r"^ovary( tissue)?$", "heart": r"^heart( tissue)?$|^left ventricle$",
    "skeletal_muscle": r"^(skeletal )?muscle( tissue)?$", "spleen": r"^spleen( tissue)?$",
    "testis": r"^(testis|testes)( tissue)?$", "adrenal": r"^adrenal gland( tissue)?$",
    "stomach": r"^stomach( tissue)?$",
}
GTEX_TO_ARCHS4 = {
    "adrenal": "adrenal", "subcutaneous adipose": "adipose",
    "cerebellum": "brain", "cortex": "brain", "frontal cortex": "brain",
    "hippocampus": "brain", "heart": "heart", "liver": "liver", "lung": "lung",
    "mammary gland": "breast", "ovary": "ovary", "pancreas": "pancreas",
    "sigmoid colon": "colon", "skeletal muscle": "skeletal_muscle", "spleen": "spleen",
    "stomach": "stomach", "testis": "testis",
}


def norm_text(x: pd.Series) -> pd.Series:
    return x.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1)


def transform(x: torch.Tensor, method: str, mean: torch.Tensor,
              pcs: torch.Tensor, whitening: torch.Tensor) -> torch.Tensor:
    if method == "Raw cosine": y = x
    elif method == "ABTT D=10":
        y = x - mean; y = y - (y @ pcs.T) @ pcs
    elif method == "Whitening": y = (x - mean) @ whitening
    else: raise ValueError(method)
    return unit(y)


def update_top(values: np.ndarray, indices: np.ndarray, block: np.ndarray,
               ref_start: int, k: int = TOPK) -> tuple[np.ndarray, np.ndarray]:
    idx = np.broadcast_to(np.arange(ref_start, ref_start + block.shape[1], dtype=np.int32), block.shape)
    vv, ii = np.concatenate([values, block], 1), np.concatenate([indices, idx], 1)
    take = np.argpartition(vv, -k, axis=1)[:, -k:]
    return np.take_along_axis(vv, take, 1), np.take_along_axis(ii, take, 1)


def sort_top(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-values, axis=1, kind="stable")
    return np.take_along_axis(values, order, 1), np.take_along_axis(indices, order, 1)


def wilson(success: int, total: int, alpha: float = .05) -> tuple[float, float]:
    if total == 0: return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2); p = success / total; den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return center-half, center+half


def metric_rows(method: str, top_i: np.ndarray, ref_tissue: np.ndarray,
                query_tissue: np.ndarray, eligible: np.ndarray) -> tuple[list[dict], list[dict]]:
    rows, per_query = [], []
    for k in KS:
        labels = ref_tissue[top_i[:, :k]]
        same = labels == query_tissue[:, None]; known = labels != ""
        for i in range(len(top_i)):
            per_query.append({"method": method, "query_index": i, "tissue": query_tissue[i], "k": k,
                              "same_hits": int(same[i].sum()), "labeled_hits": int(known[i].sum()),
                              "unknown_hits": int((~known[i]).sum()),
                              "annotated_precision": same[i].sum()/known[i].sum() if known[i].sum() else np.nan,
                              "absolute_hit_rate": same[i].mean(), "unknown_rate": (~known[i]).mean()})
        s, lab, unk = int(same.sum()), int(known.sum()), int((~known).sum())
        lo, hi = wilson(s, lab)
        rows.append({"method": method, "k": k, "queries": len(top_i), "same_hits": s,
                     "labeled_hits": lab, "unknown_hits": unk,
                     "annotated_precision": s/lab if lab else np.nan,
                     "annotated_precision_ci_low": lo, "annotated_precision_ci_high": hi,
                     "absolute_hit_rate": s/(len(top_i)*k), "unknown_rate": unk/(len(top_i)*k)})
    return rows, per_query


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "run.log").open("w", buffering=1)
    def log(msg: str) -> None:
        line = f"[{pd.Timestamp.now().isoformat()}] {msg}"; print(line, flush=True); logf.write(line + "\n")
    log("Starting frozen GTEx biological retrieval; estimated runtime 10-30 minutes on GPU")
    log(f"Monitor with: tail -f {OUT / 'run.log'}")
    cfg = json.loads((HERE / "config.json").read_text()); inputs = {k: Path(v) for k,v in cfg["inputs"].items()}
    emb_dir = inputs["embedding_dir"]; spec = json.loads((emb_dir / "embedding_manifest.json").read_text())
    locations = pd.read_parquet(emb_dir / "sample_locations.parquet")
    sman = pd.read_parquet(inputs["sample_manifest"], columns=["gsm", "species"])
    human = locations.merge(sman, left_on="geo_accession", right_on="gsm", validate="one_to_one")
    human = human[norm_text(human.species).eq("human")].reset_index(drop=True)
    meta = pd.read_parquet(inputs["sample_metadata"]).drop_duplicates("gsm").set_index("gsm")
    hmeta = meta.reindex(human.geo_accession)
    raw = np.memmap(emb_dir / "sample_embeddings.float16.mmap", mode="r", dtype=np.float16,
                    shape=(spec["total_samples"], spec["embedding_dim"]))
    ref_global = human.global_index.to_numpy(np.int64); nref, dim = len(human), int(spec["embedding_dim"])

    gtex = pd.read_parquet(GTEX_MANIFEST).reset_index(drop=True)
    repman = pd.read_parquet(GTEX_REP_MANIFEST)
    all_gtex_emb = np.load(GTEX_EMBEDDINGS, mmap_mode="r")
    assert repman.iloc[:len(gtex)].sample_id.reset_index(drop=True).equals(gtex.sample_id)
    gtex_emb = np.asarray(all_gtex_emb[:len(gtex)], np.float32)
    gtex["donor_id"] = gtex.sample_id.str.extract(r"^(GTEX-[^-]+)", expand=False)
    gtex["archs4_tissue"] = gtex.tissue.map(GTEX_TO_ARCHS4)
    assert nref == 510709 and dim == 512 and len(gtex) == 6314 and gtex.archs4_tissue.notna().all()

    # Conservative ARCHS4 tissue annotation fixed before retrieval.
    source = norm_text(hmeta.source_name_ch1).fillna("")
    ref_tissue_all = np.full(nref, "", object)
    for label, pattern in ARCHS4_TISSUES.items():
        mask = source.str.match(pattern, na=False).to_numpy()
        assert not np.any((ref_tissue_all != "") & mask), f"overlapping tissue regex: {label}"
        ref_tissue_all[mask] = label
    ref_gse_all = hmeta.series_id.fillna("").astype(str).to_numpy()

    # Exclude explicit GTEx-origin ARCHS4 records before any scoring.
    leak_text = (hmeta.title.fillna("").astype(str) + " " + hmeta.characteristics_ch1.fillna("").astype(str))
    leak = leak_text.str.contains(r"GTEX[- _]|Genotype-Tissue Expression|\bGTEx\b", case=False, regex=True, na=False).to_numpy()
    keep_ref = ~leak
    ref_indices = np.flatnonzero(keep_ref); ref_tissue = ref_tissue_all[keep_ref]; ref_gse = ref_gse_all[keep_ref]
    source_kept = hmeta.source_name_ch1.fillna("").astype(str).to_numpy()[keep_ref]
    gsm_kept = human.geo_accession.to_numpy()[keep_ref]
    labeled = ref_tissue != ""
    tissue_qc = (pd.DataFrame({"tissue": ref_tissue[labeled], "gse": ref_gse[labeled]})
                 .groupby("tissue").agg(references=("tissue","size"), gses=("gse","nunique")).reset_index())
    tissue_qc["labeled_prevalence"] = tissue_qc.references / labeled.sum()
    query_counts = gtex.groupby("archs4_tissue").size().rename("gtex_queries")
    harmon = pd.DataFrame({"gtex_tissue": sorted(gtex.tissue.unique())})
    harmon["archs4_tissue"] = harmon.gtex_tissue.map(GTEX_TO_ARCHS4)
    harmon = harmon.merge(gtex.groupby("tissue").size().rename("gtex_queries"), left_on="gtex_tissue", right_index=True)
    harmon = harmon.merge(tissue_qc, left_on="archs4_tissue", right_on="tissue", how="left").drop(columns="tissue")
    harmon["eligible"] = harmon.references.ge(MIN_REFS) & harmon.gses.ge(MIN_GSES)
    eligible_tissues = set(harmon.loc[harmon.eligible, "archs4_tissue"])
    eligible_query = gtex.archs4_tissue.isin(eligible_tissues).to_numpy()
    qmeta = gtex.loc[eligible_query].reset_index(drop=True); query = gtex_emb[eligible_query]
    query_tissue = qmeta.archs4_tissue.to_numpy()

    # Cohort and annotation QC outputs precede retrieval.
    gtex_qc = (gtex.groupby(["tissue", "source_tissue"], dropna=False)
               .agg(samples=("sample_id","size"), donors=("donor_id","nunique")).reset_index())
    gtex_qc["sample_fraction"] = gtex_qc.samples / len(gtex)
    gtex_qc.to_csv(OUT / "gtex_cohort_qc.csv", index=False)
    pd.DataFrame([{"samples":len(gtex),"unique_donors":gtex.donor_id.nunique(),
                   "broad_tissues":gtex.tissue.nunique(),"detailed_tissues":gtex.source_tissue.nunique(),
                   "missing_broad_labels":gtex.tissue.isna().sum(),"missing_detailed_labels":gtex.source_tissue.isna().sum(),
                   "largest_tissue_fraction":gtex.tissue.value_counts(normalize=True).iloc[0],
                   "eligible_queries":len(qmeta),"eligible_harmonized_tissues":len(eligible_tissues)}]).to_csv(OUT/"gtex_cohort_summary.csv",index=False)
    tissue_qc.to_csv(OUT / "archs4_tissue_annotation_qc.csv", index=False)
    pd.DataFrame([{"references_before_exclusion":nref,"gtEx_origin_excluded":int(leak.sum()),
                   "references_after_exclusion":len(ref_indices),"confidently_labeled":int(labeled.sum()),
                   "unlabeled":int((~labeled).sum()),"coverage":labeled.mean(),
                   "labeled_tissues":tissue_qc.tissue.nunique(),"largest_labeled_tissue_fraction":tissue_qc.labeled_prevalence.max()}]).to_csv(
                       OUT/"archs4_annotation_summary.csv",index=False)
    harmon.to_csv(OUT / "tissue_harmonization.csv", index=False)
    pd.DataFrame({"gsm":human.geo_accession[leak].to_numpy(),"gse":ref_gse_all[leak],
                  "title":hmeta.title.fillna("").astype(str).to_numpy()[leak],
                  "source_name_ch1":hmeta.source_name_ch1.fillna("").astype(str).to_numpy()[leak],
                  "reason":"explicit GTEx identifier/origin in title or characteristics"}).to_csv(OUT/"gtex_origin_archs4_exclusions.csv",index=False)
    log(f"QC: GTEx={len(gtex):,} ({len(qmeta):,} eligible); references={len(ref_indices):,}; leakage exclusions={leak.sum():,}; labeled coverage={labeled.mean():.2%}")

    # Frozen transformations from the previous completed stage.
    params = np.load(WORK / "literature_postprocessing_transform.npz")
    mean_np, pcs_np, white_np = params["mean"], params["pcs10"], params["whitening"]
    frozen = json.loads((HERE/"results/literature_postprocessing/frozen_for_gtex.json").read_text())
    assert frozen["frozen_before_gtex"] and len(pcs_np) == 10 and white_np.shape == (512,512)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    mean = torch.from_numpy(mean_np).to(device); pcs = torch.from_numpy(pcs_np).to(device); whitening = torch.from_numpy(white_np).to(device)
    qtrans = {m: transform(torch.from_numpy(query).to(device),m,mean,pcs,whitening) for m in REPS}

    # Pass 1: raw Top-100 plus reference/query cross-domain k=10 statistics.
    nq = len(query); topv = {m:np.full((nq,TOPK),-np.inf,np.float32) for m in METHODS}
    topi = {m:np.full((nq,TOPK),-1,np.int32) for m in METHODS}
    ref_mean10 = np.empty(len(ref_indices),np.float32); ref_kth10=np.empty(len(ref_indices),np.float32)
    for rs in range(0,len(ref_indices),REF_CHUNK):
        re=min(len(ref_indices),rs+REF_CHUNK); rt=transform(torch.from_numpy(np.asarray(raw[ref_global[ref_indices[rs:re]]],np.float32)).to(device),"Raw cosine",mean,pcs,whitening)
        rtop=np.full((10,re-rs),-np.inf,np.float32)
        for qs in range(0,nq,QUERY_CHUNK):
            qe=min(nq,qs+QUERY_CHUNK); block=(qtrans["Raw cosine"][qs:qe]@rt.T).cpu().numpy()
            topv["Raw cosine"][qs:qe],topi["Raw cosine"][qs:qe]=update_top(topv["Raw cosine"][qs:qe],topi["Raw cosine"][qs:qe],block,rs)
            merged=np.concatenate([rtop,block],axis=0); rtop=np.partition(merged,-10,axis=0)[-10:]
        ref_mean10[rs:re]=rtop.mean(0); ref_kth10[rs:re]=rtop.min(0)
        log(f"raw/local-density pass {re:,}/{len(ref_indices):,}")
    topv["Raw cosine"],topi["Raw cosine"]=sort_top(topv["Raw cosine"],topi["Raw cosine"])
    q_mean10=topv["Raw cosine"][:,:10].mean(1); q_kth10=topv["Raw cosine"][:,9]

    # Pass 2: transformed cosine and frozen local-score methods.
    for rs in range(0,len(ref_indices),REF_CHUNK):
        re=min(len(ref_indices),rs+REF_CHUNK); x=torch.from_numpy(np.asarray(raw[ref_global[ref_indices[rs:re]]],np.float32)).to(device)
        rtrans={m:transform(x,m,mean,pcs,whitening) for m in REPS}
        for qs in range(0,nq,QUERY_CHUNK):
            qe=min(nq,qs+QUERY_CHUNK)
            raw_block=(qtrans["Raw cosine"][qs:qe]@rtrans["Raw cosine"].T).cpu().numpy()
            blocks={
                "ABTT D=10":(qtrans["ABTT D=10"][qs:qe]@rtrans["ABTT D=10"].T).cpu().numpy(),
                "Whitening":(qtrans["Whitening"][qs:qe]@rtrans["Whitening"].T).cpu().numpy(),
                "CSLS":2*raw_block-q_mean10[qs:qe,None]-ref_mean10[None,rs:re],
                "Local Scaling":-(np.maximum(1-raw_block,np.finfo(np.float32).eps)**2)/(
                    np.maximum(1-q_kth10[qs:qe,None],np.finfo(np.float32).eps)*np.maximum(1-ref_kth10[None,rs:re],np.finfo(np.float32).eps)),
                "NICDM":-np.maximum(1-raw_block,np.finfo(np.float32).eps)/np.sqrt(
                    np.maximum(1-topv["Raw cosine"][qs:qe,:10],np.finfo(np.float32).eps).mean(1)[:,None]*
                    np.maximum(1-ref_mean10[None,rs:re],np.finfo(np.float32).eps)),
            }
            for m,block in blocks.items(): topv[m][qs:qe],topi[m][qs:qe]=update_top(topv[m][qs:qe],topi[m][qs:qe],block,rs)
        log(f"transformed/local retrieval {re:,}/{len(ref_indices):,}")
    for m in METHODS[1:]: topv[m],topi[m]=sort_top(topv[m],topi[m])
    np.savez_compressed(WORK/"gtex_archs4_top100.npz",**{f"values_{i}":topv[m] for i,m in enumerate(METHODS)},**{f"indices_{i}":topi[m] for i,m in enumerate(METHODS)})

    # Metrics, per-query, per-tissue, enrichment, and tissue macro.
    micro_rows=[]; pq_rows=[]
    prevalence=tissue_qc.set_index("tissue").labeled_prevalence.to_dict()
    for m in METHODS:
        arows,prows=metric_rows(m,topi[m],ref_tissue,query_tissue,eligible_query)
        micro_rows.extend(arows); pq_rows.extend(prows)
    perq=pd.DataFrame(pq_rows); micro=pd.DataFrame(micro_rows)
    pt=(perq.groupby(["method","tissue","k"],observed=True)
        .agg(queries=("query_index","size"),same_hits=("same_hits","sum"),labeled_hits=("labeled_hits","sum"),unknown_hits=("unknown_hits","sum")).reset_index())
    pt["annotated_precision"]=pt.same_hits/pt.labeled_hits
    pt["absolute_hit_rate"]=pt.same_hits/(pt.queries*pt.k); pt["unknown_rate"]=pt.unknown_hits/(pt.queries*pt.k)
    pt["background_prevalence"]=pt.tissue.map(prevalence); pt["fold_enrichment"]=pt.annotated_precision/pt.background_prevalence
    cis=np.array([wilson(int(s),int(n)) for s,n in zip(pt.same_hits,pt.labeled_hits)])
    pt["precision_ci_low"],pt["precision_ci_high"]=cis[:,0],cis[:,1]
    pt["enrichment_ci_low"]=pt.precision_ci_low/pt.background_prevalence; pt["enrichment_ci_high"]=pt.precision_ci_high/pt.background_prevalence
    macro=(pt.groupby(["method","k"],observed=True)
           .agg(tissues=("tissue","nunique"),annotated_precision=("annotated_precision","mean"),
                absolute_hit_rate=("absolute_hit_rate","mean"),unknown_rate=("unknown_rate","mean"),fold_enrichment=("fold_enrichment","mean")).reset_index())
    micro.to_csv(OUT/"retrieval_metrics_micro.csv",index=False); perq.to_parquet(OUT/"per_query_metrics.parquet",index=False)
    pt.to_csv(OUT/"per_tissue_metrics.csv",index=False); macro.to_csv(OUT/"retrieval_metrics_tissue_macro.csv",index=False)

    # Confusions for labeled retrieved neighbors only.
    tissues=sorted(eligible_tissues)
    for m in REPS:
        for k in [1,10]:
            labels=ref_tissue[topi[m][:,:k]]; rows=[]
            for qt,arr in zip(query_tissue,labels):
                rows.extend((qt,x) for x in arr if x)
            cm=pd.crosstab(pd.Series([x[0] for x in rows],name="query_tissue"),pd.Series([x[1] for x in rows],name="retrieved_tissue")).reindex(index=tissues,columns=tissues,fill_value=0)
            cm.to_csv(OUT/f"confusion_{m.lower().replace(' ','_').replace('=','')}_top{k}.csv")

    # Density and score distributions.
    density_rows=[]
    for m in REPS:
        for i in range(nq): density_rows.append({"method":m,"query_index":i,"tissue":query_tissue[i],"mean_top10_score":topv[m][i,:10].mean()})
    density=pd.DataFrame(density_rows); density.to_csv(OUT/"density_by_query.csv",index=False)
    density.groupby(["method","tissue"]).mean_top10_score.agg(["count","mean","median","std"]).reset_index().to_csv(OUT/"density_by_tissue.csv",index=False)
    score_rows=[]
    for m in METHODS:
        for k in KS:
            vals=topv[m][:,:k]
            for t in sorted(eligible_tissues):
                mask=query_tissue==t
                score_rows.append({"method":m,"tissue":t,"k":k,"n_scores":int(mask.sum()*k),"mean":vals[mask].mean(),"median":np.median(vals[mask]),"p5":np.quantile(vals[mask],.05),"p95":np.quantile(vals[mask],.95)})
    pd.DataFrame(score_rows).to_csv(OUT/"score_distributions_by_tissue.csv",index=False)
    raw_random=np.sort(np.load(WORK/"anisotropy_random_cosines.npz")["rep_0"])
    raw_labels=ref_tissue[topi["Raw cosine"]]
    simrows=[]
    for status,mask in [("same_tissue",raw_labels==query_tissue[:,None]),("different_tissue",(raw_labels!="")&(raw_labels!=query_tissue[:,None])),("unknown",raw_labels=="")]:
        vals=topv["Raw cosine"][mask]
        simrows.append({"status":status,"n":len(vals),"mean":vals.mean(),"median":np.median(vals),"p5":np.quantile(vals,.05),"p95":np.quantile(vals,.95),"median_random_pair_percentile":np.searchsorted(raw_random,np.median(vals),side="right")/len(raw_random)})
    pd.DataFrame(simrows).to_csv(OUT/"raw_cosine_by_tissue_match_status.csv",index=False)

    # GTEx -> GTEx control: self and same donor excluded, authoritative detailed labels.
    control_rows=[]; control_top={}
    donors=qmeta.donor_id.to_numpy(); detail=qmeta.tissue.to_numpy()
    for m in REPS:
        qt=qtrans[m]; cv=np.full((nq,TOPK),-np.inf,np.float32); ci=np.full((nq,TOPK),-1,np.int32)
        for qs in range(0,nq,QUERY_CHUNK):
            qe=min(nq,qs+QUERY_CHUNK); block=(qt[qs:qe]@qt.T).cpu().numpy()
            block[donors[qs:qe,None]==donors[None,:]]=-np.inf
            take=np.argpartition(block,-TOPK,axis=1)[:,-TOPK:]
            vals=np.take_along_axis(block,take,1); order=np.argsort(-vals,axis=1)
            cv[qs:qe]=np.take_along_axis(vals,order,1); ci[qs:qe]=np.take_along_axis(take,order,1)
        control_top[m]=ci
        for k in KS:
            hit=detail[ci[:,:k]]==detail[:,None]
            by=[hit[detail==t].mean() for t in np.unique(detail)]
            control_rows.append({"method":m,"k":k,"queries":nq,"micro_precision":hit.mean(),"tissue_macro_precision":np.mean(by),"tissues":len(by)})
    pd.DataFrame(control_rows).to_csv(OUT/"gtex_to_gtex_control.csv",index=False)

    # Deterministic examples: high/median/low raw tissue P@10, representative query nearest tissue median.
    rawpt=pt[(pt.method.eq("Raw cosine"))&pt.k.eq(10)].sort_values(["annotated_precision","tissue"])
    selected=[rawpt.iloc[-1].tissue,rawpt.iloc[(len(rawpt)-1)//2].tissue,rawpt.iloc[0].tissue]
    examples=[]
    for category,t in zip(["high","median","low"],selected):
        candidate=perq[(perq.method.eq("Raw cosine"))&perq.k.eq(10)&perq.tissue.eq(t)].copy()
        target=candidate.annotated_precision.median(); qi=int(candidate.assign(delta=(candidate.annotated_precision-target).abs()).sort_values(["delta","query_index"]).iloc[0].query_index)
        for m in REPS:
            for rank,(ri,score) in enumerate(zip(topi[m][qi,:10],topv[m][qi,:10]),1):
                raw_score=float((qtrans["Raw cosine"][qi:qi+1]@transform(torch.from_numpy(np.asarray(raw[ref_global[ref_indices[ri:ri+1]]],np.float32)).to(device),"Raw cosine",mean,pcs,whitening).T).item())
                examples.append({"selection":category,"query_index":qi,"gtex_sample":qmeta.sample_id[qi],"query_tissue":t,"method":m,"rank":rank,"gsm":gsm_kept[ri],"gse":ref_gse[ri],"native_score":score,"raw_cosine":raw_score,"archs4_tissue":ref_tissue[ri] or "unknown","source_name_ch1":source_kept[ri]})
    pd.DataFrame(examples).to_csv(OUT/"representative_retrieval_examples.csv",index=False)
    # Frequent unlabeled top hits are reported, never remapped post hoc.
    unl=[]
    for m in PRIMARY:
        inds=topi[m][:,:10].ravel(); labels=ref_tissue[inds]; vals=pd.Series(source_kept[inds][labels==""]).value_counts().head(30)
        unl.extend({"method":m,"source_name_ch1":s,"top10_occurrences":int(c)} for s,c in vals.items())
    pd.DataFrame(unl).to_csv(OUT/"frequent_unlabeled_top_hits.csv",index=False)

    # Central comparison to the established technical-identity benchmark.
    lit=pd.read_csv(HERE/"results/literature_postprocessing/main_comparison.csv").set_index("method")
    comp=[]
    for m in PRIMARY:
        mi=micro[(micro.method.eq(m))&micro.k.eq(10)].iloc[0]; ma=macro[(macro.method.eq(m))&macro.k.eq(10)].iloc[0]
        comp.append({"method":m,"recount3_exact_r_at_10":lit.loc[m,"r_at_10"],"recount3_macro_r_at_10":lit.loc[m,"macro_r_at_10"],"gtex_archs4_tissue_p_at_10":mi.annotated_precision,"gtex_tissue_macro_p_at_10":ma.annotated_precision,"tissue_macro_enrichment_at_10":ma.fold_enrichment,"absolute_hit_at_10":mi.absolute_hit_rate,"unknown_at_10":mi.unknown_rate})
    comparison=pd.DataFrame(comp); comparison.to_csv(OUT/"technical_identity_vs_biological_retrieval.csv",index=False)

    # Figures (compact, publication-facing).
    def save(fig,stem):
        fig.tight_layout(); fig.savefig(FIG/f"{stem}.png",dpi=220); fig.savefig(FIG/f"{stem}.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5)); c=gtex.groupby("source_tissue").size().sort_values(); ax.barh(c.index,c.values); ax.set(xlabel="GTEx samples",title="GTEx query composition (authoritative detailed tissue)"); save(fig,"gtex_tissue_composition")
    fig,ax=plt.subplots(figsize=(10,5)); q=tissue_qc.sort_values("references"); ax.barh(q.tissue,q.references); ax.set(xlabel="Confidently labeled ARCHS4 references",title=f"Conservative ARCHS4 tissue coverage: {labeled.mean():.1%}"); save(fig,"archs4_tissue_coverage")
    for table,field,stem,title in [(micro,"annotated_precision","micro_precision_by_k","GTEx→ARCHS4 annotated precision"),(macro,"annotated_precision","tissue_macro_precision_by_k","GTEx→ARCHS4 tissue-macro precision")]:
        fig,ax=plt.subplots(figsize=(10,5));
        for m in PRIMARY: d=table[table.method.eq(m)]; ax.plot(d.k,d[field],marker="o",label=m)
        ax.set(xscale="log",xticks=KS,xticklabels=KS,ylim=(0,1),xlabel="K",ylabel="Annotated precision",title=title); ax.legend(); save(fig,stem)
    heat=pt[pt.k.eq(10)].pivot(index="tissue",columns="method",values="annotated_precision")[PRIMARY]
    fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(heat,cmap="viridis",vmin=0,vmax=1,aspect="auto"); ax.set_xticks(range(len(PRIMARY)),PRIMARY,rotation=30,ha="right"); ax.set_yticks(range(len(heat)),heat.index); fig.colorbar(im,ax=ax,label="Annotated P@10"); ax.set_title("Per-tissue GTEx→ARCHS4 retrieval"); save(fig,"per_tissue_p10_heatmap")
    cm=pd.read_csv(OUT/"confusion_whitening_top10.csv",index_col=0); cmn=cm.div(cm.sum(1).replace(0,np.nan),axis=0)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1); ax.set_xticks(range(len(cm)),cm.columns,rotation=45,ha="right"); ax.set_yticks(range(len(cm)),cm.index); fig.colorbar(im,ax=ax,label="Row fraction"); ax.set(title="Whitening Top-10 labeled-tissue confusion",xlabel="Retrieved ARCHS4 tissue",ylabel="GTEx tissue"); save(fig,"tissue_confusion_whitening_top10")
    enr=pt[pt.k.eq(10)].pivot(index="tissue",columns="method",values="fold_enrichment")[PRIMARY]
    fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(np.log2(enr),cmap="RdBu_r",aspect="auto"); ax.set_xticks(range(len(PRIMARY)),PRIMARY,rotation=30,ha="right"); ax.set_yticks(range(len(enr)),enr.index); fig.colorbar(im,ax=ax,label="log2 fold enrichment"); ax.set_title("Tissue enrichment over labeled-corpus prevalence"); save(fig,"tissue_enrichment")
    dd=density.groupby(["method","tissue"]).mean_top10_score.mean().unstack(0)[REPS]
    fig,ax=plt.subplots(figsize=(10,7)); im=ax.imshow(dd,cmap="magma",aspect="auto"); ax.set_xticks(range(3),REPS,rotation=25,ha="right"); ax.set_yticks(range(len(dd)),dd.index); fig.colorbar(im,ax=ax,label="Mean Top-10 native cosine"); ax.set_title("Local density by GTEx tissue"); save(fig,"local_density_by_tissue")
    fig,ax=plt.subplots(figsize=(9,5));
    for status,mask in [("same tissue",raw_labels==query_tissue[:,None]),("different tissue",(raw_labels!="")&(raw_labels!=query_tissue[:,None]))]: ax.hist(topv["Raw cosine"][mask],bins=100,density=True,histtype="step",lw=1.5,label=status)
    ax.set(xlabel="Raw Bridge cosine",ylabel="Density",title="Raw cosine among retrieved labeled ARCHS4 neighbors"); ax.legend(); save(fig,"raw_same_vs_different_cosine")
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(PRIMARY)); w=.35; ax.bar(x-w/2,comparison.recount3_exact_r_at_10,w,label="recount3 exact R@10"); ax.bar(x+w/2,comparison.gtex_tissue_macro_p_at_10,w,label="GTEx tissue-macro P@10"); ax.set_xticks(x,PRIMARY,rotation=25,ha="right"); ax.set_ylim(0,1); ax.set_title("Technical identity versus biological retrieval"); ax.legend(); save(fig,"technical_vs_biological")
    ctrl=pd.read_csv(OUT/"gtex_to_gtex_control.csv"); fig,ax=plt.subplots(figsize=(9,5));
    for m in REPS: d=ctrl[ctrl.method.eq(m)]; ax.plot(d.k,d.tissue_macro_precision,marker="o",label=m)
    ax.set(xscale="log",xticks=KS,xticklabels=KS,ylim=(0,1),xlabel="K",ylabel="Tissue-macro precision",title="GTEx→GTEx control (same donor excluded)"); ax.legend(); save(fig,"gtex_to_gtex_control")

    runtime=time.time()-started
    config_out={"seed":SEED,"methods":METHODS,"primary_methods":PRIMARY,"k":KS,"minimum_archs4_references":MIN_REFS,"minimum_archs4_gses":MIN_GSES,"archs4_tissue_patterns":ARCHS4_TISSUES,"gtex_to_archs4_mapping":GTEX_TO_ARCHS4,"reference_exclusion":"explicit GTEx identifier/origin in title or characteristics","reference_fit_only":True}
    (OUT/"config.json").write_text(json.dumps(config_out,indent=2)+"\n")
    provenance={"created_at":pd.Timestamp.now(tz="UTC").isoformat(),"runtime_seconds":runtime,"peak_rss_gb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,"bridge_frozen":True,"gtex_not_used_for_fitting":True,"gtex_samples":len(gtex),"eligible_queries":nq,"references_after_exclusion":len(ref_indices),"checkpoint":"model/r7hnr92k/best_model.pt","preprocessing":"GENCODE v49 counts -> TPM -> natural log1p; canonical 15,165 genes","assets":{str(p):{"bytes":p.stat().st_size,"sha256":sha256(p) if p.stat().st_size<100_000_000 else None} for p in [GTEX_MANIFEST,GTEX_REP_MANIFEST,GTEX_EMBEDDINGS,GTEX_PROVENANCE,WORK/"literature_postprocessing_transform.npz"]},"limitations":["ARCHS4 tissue labels use conservative anchored source_name_ch1 mappings and leave unknowns explicit.","GTEx-origin exclusion uses explicit metadata identifiers/origin statements; latent unmarked duplicates cannot be ruled out.","Tissue labels do not evaluate disease, treatment, cell type, or state.","GTEx-to-GTEx is a standardized-dataset control, not independent-study validation."]}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    log(f"COMPLETE runtime={runtime:.1f}s peak_RSS={provenance['peak_rss_gb']:.2f}GB")
    logf.close()


if __name__ == "__main__": main()
