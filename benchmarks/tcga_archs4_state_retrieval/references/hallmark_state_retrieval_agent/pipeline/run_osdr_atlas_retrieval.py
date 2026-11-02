#!/usr/bin/env python3
"""Retrieve all eligible OSDR samples and FLT/GC centroids against ARCHS4."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from common import RESULTS, WORK, l2_rows, save_json, sha256

OUT = RESULTS / "osdr_atlas"
ATLAS_WORK = WORK / "osdr_atlas"
TOP = 100
KS = (5, 10, 50, 100)
SEED = 20260925

TISSUE_ALIASES = {
    "liver": r"liver|hepatic|hepatocyte", "kidney": r"kidney|renal", "eye": r"eye|retina|retinal|ocular",
    "muscle": r"muscle|soleus|gastrocnemius|quadriceps|tibialis|digitorum", "skin": r"skin|cutaneous|dermal",
    "spleen": r"spleen|splenic", "thymus": r"thymus|thymic", "lung": r"lung|pulmonary|bronch",
    "colon": r"colon|colonic|colorectal", "adrenal": r"adrenal", "adipose": r"adipose|fat tissue",
    "brain": r"brain|cerebell|hippocamp|cerebr", "heart": r"heart|cardiac|ventric",
    "bone_marrow": r"bone marrow|marrow", "mammary": r"mammary|breast", "bone": r"bone|mandible",
    "blood": r"blood|leukocyte|lymphocyte|monocyte|neutrophil", "cells": r"cell line|cultured cells|cells"
}
CATEGORY_REGEX = {
    "cancer": r"cancer|carcinoma|tumou?r|neoplasm|leukemia|lymphoma|sarcoma|melanoma|glioma",
    "infection_inflammation": r"infect|sepsis|inflam|cytokine|lipopolysaccharide|\blps\b|virus|viral|bacter",
    "oxidative_stress": r"oxidative|reactive oxygen|\bros\b|antioxid|peroxide",
    "mitochondrial": r"mitochond|electron transport|oxidative phosphorylation",
    "hypoxia": r"hypoxi|ischemi|anoxi|low oxygen",
    "fibrosis_remodeling": r"fibro|cirrho|remodel|wound|injury",
    "metabolic_disease": r"fatty liver|steatos|obes|diabet|metabolic|insulin|nafld|mafld",
    "radiation": r"radiation|irradiat|x-ray|gamma ray",
    "exercise_disuse": r"exercise|training|immobil|bed rest|unload|hindlimb",
    "drug_perturbation": r"treated|treatment|drug|inhibitor|agonist|antagonist|knockdown|knockout|overexpress"
}


def log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat(); print(f"[{stamp}] {message}", flush=True)
    with (OUT / "retrieval_run.log").open("a") as stream: stream.write(f"[{stamp}] {message}\n")


def tissue_key(value: str) -> str:
    value = str(value).lower()
    for key, pattern in TISSUE_ALIASES.items():
        if re.search(pattern, value): return key
    return "other"


def top_mask(x: np.ndarray, k: int = 10) -> np.ndarray:
    idx = np.argpartition(np.abs(x), -k, axis=1)[:, -k:]
    mask = np.zeros(x.shape, bool); mask[np.arange(len(x))[:, None], idx] = True
    return mask


def exact_topk_gpu(queries: np.ndarray, reference_path: Path, dim: int, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.load(reference_path, mmap_mode="r"); assert reference.shape == (510_709, dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rt = torch.from_numpy(np.asarray(reference)).to(device)
    indices = np.empty((len(queries), TOP), np.int32); values = np.empty((len(queries), TOP), np.float32)
    quantiles = np.empty((len(queries), 5), np.float32); qlevels = (0.5, 0.9, 0.95, 0.99, 0.999)
    started = time.time(); qn = l2_rows(queries)
    for left in range(0, len(qn), 32):
        right = min(left + 32, len(qn)); score = torch.from_numpy(qn[left:right]).to(device) @ rt.T
        val, idx = torch.topk(score, TOP, dim=1, sorted=True)
        indices[left:right] = idx.cpu().numpy(); values[left:right] = val.cpu().numpy()
        quantiles[left:right] = np.quantile(score.float().cpu().numpy(), qlevels, axis=1).T
        if right % 256 == 0 or right == len(qn): log(f"{label}: {right}/{len(qn)} elapsed={time.time()-started:.1f}s")
    del rt; torch.cuda.empty_cache() if device.type == "cuda" else None
    return indices, values, quantiles


def overlap_topk_gpu(queries: np.ndarray, ref: np.ndarray, cosine_scores: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Signed top-10 overlap is an inner product in a 100-D sign-expanded binary space.
    rm = top_mask(ref, 10); rcode = np.concatenate([rm & (ref >= 0), rm & (ref < 0)], axis=1).astype(np.float32)
    qm = top_mask(queries, 10); qcode = np.concatenate([qm & (queries >= 0), qm & (queries < 0)], axis=1).astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); rt = torch.from_numpy(rcode).to(device); rcos = torch.from_numpy(l2_rows(ref)).to(device)
    indices = np.empty((len(queries), TOP), np.int32); values = np.empty((len(queries), TOP), np.float32)
    quantiles = np.empty((len(queries), 5), np.float32); levels = (0.5, 0.9, 0.95, 0.99, 0.999)
    for left in range(0, len(queries), 32):
        right = min(left + 32, len(queries)); score = (torch.from_numpy(qcode[left:right]).to(device) @ rt.T) / 10
        # Tiny cosine tie-break preserves overlap as the reported score.
        tie = torch.from_numpy(l2_rows(queries[left:right])).to(device) @ rcos.T
        _, idx = torch.topk(score + tie * 1e-6, TOP, dim=1, sorted=True)
        indices[left:right] = idx.cpu().numpy(); values[left:right] = torch.gather(score, 1, idx).cpu().numpy()
        quantiles[left:right] = np.quantile(score.cpu().numpy(), levels, axis=1).T
        if right % 256 == 0 or right == len(queries): log(f"signed top10 overlap: {right}/{len(queries)}")
    del rt, rcos; torch.cuda.empty_cache() if device.type == "cuda" else None
    return indices, values, quantiles


def signal_coverage(q: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    qm = top_mask(q[None], k)[0]; rm = top_mask(rows, k)
    return (rm[:, qm] & (np.sign(rows[:, qm]) == np.sign(q[qm])[None])).sum(1) / k


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p); ranked = np.empty(len(p), float); ranked[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    return np.minimum(ranked, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); (OUT / "retrieval_run.log").write_text("")
    started = time.time(); rng = np.random.default_rng(SEED)
    manifest = pd.read_parquet(OUT / "eligible_sample_manifest.parquet").sort_values("matrix_row").reset_index(drop=True)
    bridge = np.load(ATLAS_WORK / "bridge_embeddings.float32.npy", mmap_mode="r"); hallmark = np.load(ATLAS_WORK / "hallmark_profiles.float32.npy", mmap_mode="r")
    names = np.load(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz", allow_pickle=False)["hallmarks"].astype(str).tolist()
    assert len(manifest) == len(bridge) == len(hallmark) == 1471
    centmeta = manifest.groupby(["group_id", "condition"], sort=True).first().reset_index()
    centmeta["sample_id"] = centmeta.group_id + "_" + centmeta.condition + "_CENTROID"
    cb = np.vstack([np.asarray(bridge)[(manifest.group_id.eq(r.group_id) & manifest.condition.eq(r.condition))].mean(0) for r in centmeta.itertuples()])
    ch = np.vstack([np.asarray(hallmark)[(manifest.group_id.eq(r.group_id) & manifest.condition.eq(r.condition))].mean(0) for r in centmeta.itertuples()])
    querymeta = pd.concat([manifest.assign(query_type="individual"), centmeta.assign(query_type="centroid")], ignore_index=True, sort=False)
    allb = np.vstack([np.asarray(bridge), cb]); allh = np.vstack([np.asarray(hallmark), ch])
    log(f"retrieval start: {len(manifest)} individuals + {len(centmeta)} centroids; expected 2-6 minutes")
    refh = np.load(WORK / "archs4_hallmark_signatures.float32.npy", mmap_mode="r")
    cache_path = ATLAS_WORK / "retrieval_top100.npz"
    background_path = ATLAS_WORK / "retrieval_background_quantiles.npz"
    if cache_path.exists() and background_path.exists():
        cache=np.load(cache_path); bgcache=np.load(background_path)
        bi,bs,hi,hs,oi,oscore=(cache[x] for x in ("bridge_indices","bridge_scores","hallmark_indices","hallmark_scores","overlap_indices","overlap_scores"))
        bq,hq,oq=(bgcache[x] for x in ("bridge","hallmark","overlap")); log("reused saved exact retrieval rankings and backgrounds")
    else:
        bi, bs, bq = exact_topk_gpu(allb, WORK / "archs4_bridge_cosine_index.float32.npy", 512, "Bridge-512")
        hi, hs, hq = exact_topk_gpu(allh, WORK / "archs4_hallmark_cosine_index.float32.npy", 50, "Hallmark-50")
        oi, oscore, oq = overlap_topk_gpu(allh, np.asarray(refh))
        np.savez_compressed(cache_path, bridge_indices=bi, bridge_scores=bs,
                            hallmark_indices=hi, hallmark_scores=hs, overlap_indices=oi, overlap_scores=oscore)
        np.savez_compressed(background_path,bridge=bq,hallmark=hq,overlap=oq)

    refmeta = pd.read_parquet(RESULTS / "archs4/sample_metadata.parquet")
    text = refmeta[["title", "source_name_ch1", "characteristics_ch1"]].fillna("").agg(" ".join, axis=1).str.lower()
    ref_tissue = {key: text.str.contains(pattern, regex=True).to_numpy() for key, pattern in TISSUE_ALIASES.items()}
    categories = {key: text.str.contains(pattern, regex=True).to_numpy() for key, pattern in CATEGORY_REGEX.items()}
    ref_hnorm = l2_rows(np.asarray(refh)); query_hnorm = l2_rows(allh)
    rows = []; methods = [("Bridge-512 cosine", bi, bs, bq), ("Hallmark-50 cosine", hi, hs, hq), ("Hallmark signed top10 overlap", oi, oscore, oq)]
    for method, indices, scores, quant in methods:
        for qi in range(len(querymeta)):
            q = querymeta.iloc[qi]; tk = tissue_key(q.tissue); cov5 = signal_coverage(allh[qi], np.asarray(refh)[indices[qi]], 5); cov10 = signal_coverage(allh[qi], np.asarray(refh)[indices[qi]], 10)
            dominant = np.argsort(np.abs(allh[qi]))[::-1][:10]
            for rank, (ri, score) in enumerate(zip(indices[qi], scores[qi]), 1):
                r = refmeta.iloc[int(ri)]; contrib = query_hnorm[qi] * ref_hnorm[int(ri)]; topc = np.argsort(contrib)[::-1][:6]
                rec = {"query_index": qi, "query_type": q.query_type, "group_id": q.group_id, "condition": q.condition,
                       "osdr_sample_id": q.sample_id, "study": q.study, "tissue": q.tissue, "tissue_key": tk,
                       "method": method, "rank": rank, "score": float(score), "rank_percentile": 1-(rank-.5)/len(refmeta),
                       "reference_row": int(ri), "gsm": r.sample_id, "gse": r.series_id, "title": r.title,
                       "source_name": r.source_name_ch1, "characteristics": r.characteristics_ch1,
                       "same_tissue": bool(ref_tissue.get(tk, np.zeros(len(refmeta), bool))[ri]), "same_organism": False,
                       "signal_coverage_top5": float(cov5[rank-1]), "signal_coverage_top10": float(cov10[rank-1]),
                       "dominant_query_hallmarks": "; ".join(f"{names[j]}={'+' if allh[qi,j]>=0 else '-'}" for j in dominant),
                       "dominant_shared_hallmarks": "; ".join(f"{names[j]}={contrib[j]:.4f}" for j in topc)}
                for category, flag in categories.items(): rec[category] = bool(flag[ri])
                rows.append(rec)
    hits = pd.DataFrame(rows); hits.to_parquet(OUT / "individual_and_centroid_top100.parquet", index=False)
    hits[hits["rank"].le(10)].to_csv(OUT / "individual_and_centroid_top10.csv", index=False)
    hits[(hits.query_type.eq("centroid")) & hits["rank"].le(5)].to_csv(OUT / "centroid_top5_biological_audit.csv", index=False)
    bg = []
    for method, quant in [("Bridge-512 cosine", bq), ("Hallmark-50 cosine", hq), ("Hallmark signed top10 overlap", oq)]:
        for qi, vals in enumerate(quant):
            for level, value in zip((.5,.9,.95,.99,.999), vals): bg.append({"query_index":qi,"method":method,"quantile":level,"score":value})
    pd.DataFrame(bg).to_csv(OUT / "full_archs4_score_background.csv", index=False)

    # Replicate consistency within each condition-specific biological group.
    consistency = []
    individual = hits[hits.query_type.eq("individual")]
    for (group_id, condition, method), block in individual.groupby(["group_id", "condition", "method"]):
        sample_groups = [g.sort_values("rank") for _, g in block.groupby("osdr_sample_id")]
        for k in (10, 50, 100):
            nj, gj, rc = [], [], []
            for a in range(len(sample_groups)):
                for b in range(a+1, len(sample_groups)):
                    aa, bb = sample_groups[a].head(k), sample_groups[b].head(k); sa, sb = set(aa.reference_row), set(bb.reference_row)
                    ga, gb = set(aa.gse.dropna()), set(bb.gse.dropna()); nj.append(len(sa&sb)/len(sa|sb)); gj.append(len(ga&gb)/len(ga|gb) if ga|gb else np.nan)
                    union = list(sa|sb); ra = {x:r for x,r in zip(aa.reference_row,aa["rank"])}; rb={x:r for x,r in zip(bb.reference_row,bb["rank"])}
                    rc.append(spearmanr([ra.get(x,k+1) for x in union],[rb.get(x,k+1) for x in union]).statistic)
            z = block[block["rank"].le(k)]; gse_counts = z.gse.fillna("MISSING").value_counts()
            consistency.append({"group_id":group_id,"condition":condition,"method":method,"k":k,"replicates":len(sample_groups),
                "mean_neighbor_jaccard":np.nanmean(nj),"mean_gse_jaccard":np.nanmean(gj),"mean_rank_spearman":np.nanmean(rc),
                "same_tissue_fraction":z.same_tissue.mean(),"signal_coverage_top10":z.signal_coverage_top10.mean(),
                "unique_gse":z.gse.nunique(),"largest_gse_fraction":gse_counts.iloc[0]/len(z)})
    pd.DataFrame(consistency).to_csv(OUT / "replicate_consistency.csv", index=False)

    # Differential reciprocal-rank enrichment with label permutation within each valid group.
    diff_rows = []; permutations = 1000
    for (group_id, method), block in individual.groupby(["group_id", "method"]):
        sample_labels = block[["osdr_sample_id","condition"]].drop_duplicates().set_index("osdr_sample_id").condition
        samples = sample_labels.index.tolist(); labels = sample_labels.to_numpy(); nflt=(labels=="FLT").sum()
        candidates = block.reference_row.value_counts().head(300).index.to_numpy(); mat=np.zeros((len(samples),len(candidates)),np.float32); smap={s:i for i,s in enumerate(samples)}; cmap={r:i for i,r in enumerate(candidates)}
        for r in block.itertuples():
            if r.reference_row in cmap: mat[smap[r.osdr_sample_id],cmap[r.reference_row]]=1/np.log2(r.rank+1)
        observed=mat[labels=="FLT"].mean(0)-mat[labels=="GC"].mean(0); perm=np.empty((permutations,len(candidates)),np.float32)
        for pi in range(permutations):
            take=rng.permutation(len(samples)); flt=take[:nflt]; gc=take[nflt:]; perm[pi]=mat[flt].mean(0)-mat[gc].mean(0)
        p=(1+(np.abs(perm)>=np.abs(observed)).sum(0))/(permutations+1); qv=bh(p)
        for ri,delta,pv,qvalue in zip(candidates,observed,p,qv):
            rr=refmeta.iloc[int(ri)]; sub=block[block.reference_row.eq(ri)]
            diff_rows.append({"group_id":group_id,"method":method,"reference_row":ri,"gsm":rr.sample_id,"gse":rr.series_id,
                "title":rr.title,"source_name":rr.source_name_ch1,"characteristics":rr.characteristics_ch1,"rrf_delta_flt_minus_gc":delta,
                "permutation_p":pv,"bh_q":qvalue,"flt_replicate_prevalence":sub[sub.condition.eq("FLT")].osdr_sample_id.nunique()/nflt,
                "gc_replicate_prevalence":sub[sub.condition.eq("GC")].osdr_sample_id.nunique()/(len(samples)-nflt)})
    diff=pd.DataFrame(diff_rows); diff.to_parquet(OUT / "flt_vs_gc_differential_retrieval.parquet",index=False)
    diff[(diff.bh_q<=.1)&(diff.rrf_delta_flt_minus_gc>0)].to_csv(OUT / "flt_enriched_archs4_hits.csv",index=False)

    # GSE-level differential summaries and cross-study recurrence.
    gse = individual.dropna(subset=["gse"]).copy(); gse["rrf"]=1/np.log2(gse["rank"]+1)
    gs = gse.groupby(["group_id","condition","method","gse","osdr_sample_id"],as_index=False).rrf.max()
    gd=gs.groupby(["group_id","condition","method","gse"]).agg(mean_rrf=("rrf","mean"),replicate_prevalence=("osdr_sample_id","nunique")).reset_index()
    wide=gd.pivot_table(index=["group_id","method","gse"],columns="condition",values=["mean_rrf","replicate_prevalence"],fill_value=0).reset_index(); wide.columns=['_'.join(x).strip('_') if isinstance(x,tuple) else x for x in wide.columns]
    wide["mean_rrf_delta_flt_minus_gc"]=wide.get("mean_rrf_FLT",0)-wide.get("mean_rrf_GC",0); wide.to_csv(OUT/"flt_vs_gc_gse_enrichment.csv",index=False)
    recurrence=wide.groupby(["method","gse"],as_index=False).agg(groups=("group_id","nunique"),mean_delta=("mean_rrf_delta_flt_minus_gc","mean"),flt_groups=("mean_rrf_delta_flt_minus_gc",lambda x:int((x>0).sum())),gc_groups=("mean_rrf_delta_flt_minus_gc",lambda x:int((x<0).sum()))).sort_values(["flt_groups","mean_delta"],ascending=False)
    recurrence.to_csv(OUT/"cross_study_recurring_gse.csv",index=False)

    # Hub frequency and leave-one-dominant-Hallmark-out sensitivity for centroid cosine hits.
    hub=[]
    for (method,condition), block in hits[hits["rank"].le(10)].groupby(["method","condition"]):
        for entity,count in block.gse.fillna("MISSING").value_counts().head(30).items(): hub.append({"method":method,"condition":condition,"gse":entity,"occurrences":count,"fraction":count/len(block)})
    pd.DataFrame(hub).to_csv(OUT/"archs4_hub_frequency.csv",index=False)
    loho=[]; hindex=np.load(WORK/"archs4_hallmark_cosine_index.float32.npy",mmap_mode="r")
    for ci,q in enumerate(ch):
        qi=len(manifest)+ci; dominant=int(np.argmax(np.abs(q))); modified=q.copy(); modified[dominant]=0; score=np.asarray(hindex@l2_rows(modified[None])[0]); order=np.argsort(score)[::-1]; ranks=np.empty(len(order),np.int32); ranks[order]=np.arange(1,len(order)+1)
        for ri in hi[qi,:5]: loho.append({"group_id":centmeta.iloc[ci].group_id,"condition":centmeta.iloc[ci].condition,"gsm":refmeta.iloc[int(ri)].sample_id,"left_out_hallmark":names[dominant],"original_rank":int(np.where(hi[qi]==ri)[0][0]+1),"loho_rank":int(ranks[int(ri)]),"survives_top100":bool(ranks[int(ri)]<=100)})
    pd.DataFrame(loho).to_csv(OUT/"hallmark_leave_one_dominant_out.csv",index=False)
    save_json(OUT/"retrieval_provenance.json",{"individual_queries":len(manifest),"centroid_queries":len(centmeta),"reference_samples":len(refmeta),"top_k":TOP,"methods":[x[0] for x in methods],"label_permutations":permutations,"seed":SEED,"bridge_inference_repeated":False,"hallmark_retrained":False,"runtime_seconds":time.time()-started,"limitations":["ARCHS4 is human while OSDR is mouse; same-organism score is structurally false.","Automated metadata categories are regex flags, not biological adjudication.","Permutation tests operate within exact matched group and on the 300 most recurrent top-100 candidates."]})
    log(f"COMPLETE runtime={time.time()-started:.1f}s")


if __name__ == "__main__": main()
