#!/usr/bin/env python3
"""Diagnose anisotropy and common-component effects in frozen Bridge embeddings."""
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
from scipy.stats import skew

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "anisotropy"
FIG = OUT / "figures"
WORK = HERE / "work"
SEED = 20260922
NPAIRS = 2_000_000
CHUNK = 32768
PAIR_BATCH = 65536
TOPK = 101
PCS = [0, 1, 2, 5, 10]
REPRESENTATIONS = ["Raw", "Mean-centered", "Centered + remove PC1",
                   "Centered + remove PCs 1–2", "Centered + remove PCs 1–5",
                   "Centered + remove PCs 1–10"]
QUANTILES = [0, .0001, .001, .01, .05, .10, .25, .50, .75, .90, .95, .99, .999, .9999, 1]
QLABELS = ["minimum", "p0.01", "p0.1", "p1", "p5", "p10", "p25", "median",
           "p75", "p90", "p95", "p99", "p99.9", "p99.99", "maximum"]

TISSUES = {
    "blood": r"^(whole |peripheral )?blood$|^pbmcs?$",
    "liver": r"^liver( tissue)?$", "lung": r"^lung( tissue)?$",
    "skin": r"^skin( tissue)?$", "brain": r"^brain( tissue)?$|^cerebral cortex$|^cortex$",
    "breast": r"^(breast|mammary gland)( tissue)?$", "colon": r"^(colon|colonic mucosa)( tissue)?$",
    "kidney": r"^kidney( tissue)?$|^renal cortex$", "prostate": r"^prostate( tissue)?$",
    "pancreas": r"^pancreas( tissue)?$", "adipose": r"^(adipose|adipose tissue|subcutaneous adipose tissue)$",
    "ovary": r"^ovary( tissue)?$", "heart": r"^heart( tissue)?$|^left ventricle$",
    "skeletal_muscle": r"^(skeletal )?muscle( tissue)?$", "spleen": r"^spleen( tissue)?$",
    "testis": r"^(testis|testes)( tissue)?$",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm_text(x: pd.Series) -> pd.Series:
    return x.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def unit_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def transform_torch(x: torch.Tensor, mean: torch.Tensor, pcs: torch.Tensor,
                    representation: str) -> torch.Tensor:
    if representation == "Raw":
        residual = x
    else:
        residual = x - mean
        n_pc = {"Mean-centered": 0, "Centered + remove PC1": 1,
                "Centered + remove PCs 1–2": 2, "Centered + remove PCs 1–5": 5,
                "Centered + remove PCs 1–10": 10}[representation]
        if n_pc:
            basis = pcs[:n_pc]
            residual = residual - (residual @ basis.T) @ basis
    return torch.nn.functional.normalize(residual, dim=1)


def distribution(values: np.ndarray, name: str, kind: str) -> dict:
    x = np.asarray(values, dtype=np.float64)
    row = {"representation": name, "distribution": kind, "n": len(x),
           "mean": x.mean(), "sd": x.std(ddof=1), "iqr": np.quantile(x, .75)-np.quantile(x, .25)}
    row.update(dict(zip(QLABELS, np.quantile(x, QUANTILES))))
    return row


def gini(x: np.ndarray) -> float:
    v = np.sort(np.asarray(x, dtype=np.float64))
    if not len(v) or v.sum() == 0: return 0.0
    return float(2 * np.dot(np.arange(1, len(v)+1), v)/(len(v)*v.sum()) - (len(v)+1)/len(v))


def merge_top(best_v: np.ndarray, best_i: np.ndarray, block: np.ndarray,
              start: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.broadcast_to(np.arange(start, start + block.shape[1], dtype=np.int32), block.shape)
    vals = np.concatenate([best_v, block], axis=1)
    inds = np.concatenate([best_i, idx], axis=1)
    take = np.argpartition(vals, -TOPK, axis=1)[:, -TOPK:]
    return np.take_along_axis(vals, take, axis=1), np.take_along_axis(inds, take, axis=1)


def retrieval_summary(ranks: np.ndarray) -> dict:
    return {"r_at_1": np.mean(ranks <= 1), "r_at_5": np.mean(ranks <= 5),
            "r_at_10": np.mean(ranks <= 10), "r_at_100": np.mean(ranks <= 100),
            "r_at_1000": np.mean(ranks <= 1000), "mrr": np.mean(1/ranks),
            "median_rank": np.median(ranks), "mean_rank": np.mean(ranks),
            "median_percentile_rank": np.median(1-(ranks-1)/510709)}


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    log_file = (OUT / "run.log").open("w", buffering=1)
    def log(msg: str) -> None:
        line = f"[{pd.Timestamp.now().isoformat()}] {msg}"; print(line, flush=True); log_file.write(line+"\n")
    log("Starting anisotropy analysis; estimated runtime 5-15 minutes on RTX 3090")
    log(f"Monitor with: tail -f {OUT / 'run.log'}")
    cfg = json.loads((HERE / "config.json").read_text()); inputs = {k: Path(v) for k,v in cfg["inputs"].items()}
    up, emb_dir = inputs["upstream_benchmark"], inputs["embedding_dir"]
    spec = json.loads((emb_dir / "embedding_manifest.json").read_text())
    locations = pd.read_parquet(emb_dir / "sample_locations.parquet")
    manifest = pd.read_parquet(inputs["sample_manifest"], columns=["gsm", "species"])
    human = locations.merge(manifest, left_on="geo_accession", right_on="gsm", validate="one_to_one")
    human = human[norm_text(human.species).eq("human")].reset_index(drop=True)
    samples_all = pd.read_parquet(up / "work/paired_expression/samples.parquet")
    keep = samples_all.study_exposure.eq("unseen_study").to_numpy()
    samples = samples_all.loc[keep].reset_index(drop=True)
    query_raw = np.asarray(np.load(up / "work/recount3_embeddings.npy", mmap_mode="r")[keep], dtype=np.float32)
    upstream = samples[["gsm"]].merge(pd.read_parquet(up / "results/full_archs4_retrieval.parquet"), on="gsm")
    raw = np.memmap(emb_dir / "sample_embeddings.float16.mmap", mode="r", dtype=np.float16,
                    shape=(spec["total_samples"], spec["embedding_dim"]))
    ref_global = human.global_index.to_numpy(np.int64); n, dim = len(human), spec["embedding_dim"]
    correct_map = pd.Series(np.arange(n), index=human.geo_accession)
    correct = samples.gsm.map(correct_map).to_numpy(np.int64)
    meta = pd.read_parquet(inputs["sample_metadata"]).drop_duplicates("gsm").set_index("gsm")
    hmeta, qmeta = meta.reindex(human.geo_accession), meta.reindex(samples.gsm)
    ref_gse = hmeta.series_id.fillna("").astype(str).to_numpy(); query_gse = samples.gse.astype(str).to_numpy()
    ref_source = norm_text(hmeta.source_name_ch1).fillna("").to_numpy(); query_source = norm_text(qmeta.source_name_ch1).fillna("").to_numpy()
    ref_tissue = np.full(n, "", dtype=object); query_tissue = np.full(len(samples), "", dtype=object)
    for label, pattern in TISSUES.items():
        ref_tissue[norm_text(hmeta.source_name_ch1).str.match(pattern, na=False).to_numpy()] = label
        query_tissue[norm_text(qmeta.source_name_ch1).str.match(pattern, na=False).to_numpy()] = label
    qc = pd.read_csv(HERE / "results/biological_validation/biological_label_class_qc.csv")
    prior_source = pd.read_csv(HERE / "results/biological_validation/main_comparison_source_label.csv").set_index("method")
    prior_tissue = pd.read_csv(HERE / "results/biological_validation/main_comparison_tissue_label.csv").set_index("method")
    source_eligible = set(qc[(qc.label.eq("source_label")) & qc.eligible].iloc[:, 1].astype(str))
    tissue_eligible = set(qc[(qc.label.eq("tissue_label")) & qc.eligible].iloc[:, 1].astype(str))
    assert n == 510709 and len(samples) == 562 and dim == 512

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    # Global means/norms and directional mean, all references.
    sum_raw = np.zeros(dim, np.float64); sum_unit = np.zeros(dim, np.float64)
    raw_norms = np.empty(n, np.float32)
    for start in range(0, n, CHUNK):
        stop = min(n, start+CHUNK); x = np.asarray(raw[ref_global[start:stop]], dtype=np.float32)
        norms = np.linalg.norm(x, axis=1); raw_norms[start:stop] = norms
        sum_raw += x.sum(axis=0, dtype=np.float64); sum_unit += (x/norms[:,None]).sum(axis=0, dtype=np.float64)
    mean_raw = (sum_raw/n).astype(np.float32); mean_unit = (sum_unit/n).astype(np.float32)
    mean_dir = mean_raw / np.linalg.norm(mean_raw)
    unit_mean_dir = mean_unit / np.linalg.norm(mean_unit)
    sample_mean_cos = np.empty(n, np.float32); sample_unit_mean_cos = np.empty(n, np.float32)
    covariance = np.zeros((dim, dim), np.float64)
    mean_t = torch.from_numpy(mean_raw).to(device)
    for start in range(0, n, CHUNK):
        stop = min(n, start+CHUNK); x = np.asarray(raw[ref_global[start:stop]], dtype=np.float32)
        xu = x / np.linalg.norm(x, axis=1, keepdims=True)
        sample_mean_cos[start:stop] = xu @ mean_dir
        sample_unit_mean_cos[start:stop] = xu @ unit_mean_dir
        xt = torch.from_numpy(x).to(device); xc = xt - mean_t
        covariance += (xc.T @ xc).cpu().numpy().astype(np.float64)
        if (stop//CHUNK) % 5 == 0: log(f"mean/PCA pass {stop:,}/{n:,}")
    covariance /= n-1
    eigvals, eigvecs = np.linalg.eigh(covariance); order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pcs = eigvecs.T.astype(np.float32); explained = eigvals/eigvals.sum()
    participation = float(eigvals.sum()**2 / np.square(eigvals).sum())
    alignment = float(abs(np.dot(mean_dir, pcs[0])))
    anisotropy = pd.DataFrame([{
        "references": n, "embedding_dim": dim, "global_raw_mean_norm": np.linalg.norm(mean_raw),
        "mean_raw_sample_norm": raw_norms.mean(), "median_raw_sample_norm": np.median(raw_norms),
        "mean_norm_to_mean_sample_norm_ratio": np.linalg.norm(mean_raw)/raw_norms.mean(),
        "unit_directional_mean_norm": np.linalg.norm(mean_unit),
        "unit_directional_mean_norm_squared": np.linalg.norm(mean_unit)**2,
        "mean_sample_cosine_to_raw_mean_direction": sample_mean_cos.mean(),
        "median_sample_cosine_to_raw_mean_direction": np.median(sample_mean_cos),
        "mean_sample_cosine_to_unit_mean_direction": sample_unit_mean_cos.mean(),
        "median_sample_cosine_to_unit_mean_direction": np.median(sample_unit_mean_cos),
        "sample_to_unit_mean_cosine_sd": sample_unit_mean_cos.std(ddof=1),
        "pc1_mean_direction_absolute_cosine": alignment, "participation_ratio": participation,
        "effective_dimension_fraction": participation/dim,
    }])
    anisotropy.to_csv(OUT / "anisotropy_summary.csv", index=False)
    pd.DataFrame({"statistic": QLABELS, "quantile": QUANTILES,
                  "sample_to_raw_mean_cosine": np.quantile(sample_mean_cos, QUANTILES),
                  "sample_to_unit_mean_cosine": np.quantile(sample_unit_mean_cos, QUANTILES)}).to_csv(
                      OUT / "sample_to_mean_cosine_distribution.csv", index=False)
    pc_table = pd.DataFrame({"pc": np.arange(1, dim+1), "eigenvalue": eigvals,
                             "explained_variance_ratio": explained, "cumulative_explained_variance": np.cumsum(explained)})
    pc_table.to_csv(OUT / "pca_explained_variance.csv", index=False)
    pd.DataFrame([{"pc1": explained[0], "pc2": explained[1],
                   "pc5_cumulative": explained[:5].sum(), "pc10_cumulative": explained[:10].sum(),
                   "pc20_cumulative": explained[:20].sum(), "pc50_cumulative": explained[:50].sum(),
                   "participation_ratio": participation, "pc1_mean_direction_absolute_cosine": alignment}]).to_csv(
                       OUT / "pca_summary.csv", index=False)
    np.savez(WORK / "anisotropy_reference_transform.npz", mean=mean_raw, pcs=pcs[:10], eigenvalues=eigvals)

    # Reconstruct and persist exact Stage-0 pair indices; verify raw cosines.
    rng = np.random.default_rng(SEED); a = rng.integers(0, n, NPAIRS, dtype=np.int64)
    b = rng.integers(0, n-1, NPAIRS, dtype=np.int64); b += b >= a
    np.savez(WORK / "anisotropy_random_pair_indices.npz", a=a, b=b, seed=SEED)
    previous_random = np.load(WORK / "sampled_archs4_pair_cosines.npy", mmap_mode="r")
    mean_gpu, pcs_gpu = torch.from_numpy(mean_raw).to(device), torch.from_numpy(pcs[:10]).to(device)
    random_by_rep: dict[str, np.ndarray] = {}
    for rep in REPRESENTATIONS:
        vals = np.empty(NPAIRS, np.float32)
        for start in range(0, NPAIRS, PAIR_BATCH):
            stop = min(NPAIRS, start+PAIR_BATCH)
            xa = torch.from_numpy(np.asarray(raw[ref_global[a[start:stop]]], dtype=np.float32)).to(device)
            xb = torch.from_numpy(np.asarray(raw[ref_global[b[start:stop]]], dtype=np.float32)).to(device)
            ta = transform_torch(xa, mean_gpu, pcs_gpu, rep); tb = transform_torch(xb, mean_gpu, pcs_gpu, rep)
            vals[start:stop] = (ta*tb).sum(1).cpu().numpy()
        random_by_rep[rep] = vals
        log(f"random pairs complete: {rep}")
    max_reproduction_delta = float(np.max(np.abs(random_by_rep["Raw"] - previous_random)))
    assert max_reproduction_delta < 5e-5
    np.savez(WORK / "anisotropy_random_cosines.npz", **{f"rep_{i}": random_by_rep[r] for i,r in enumerate(REPRESENTATIONS)})

    # Exact pairs and full retrieval, streaming reference chunks without score matrices.
    correct_raw = np.asarray(raw[ref_global[correct]], dtype=np.float32)
    query_t = torch.from_numpy(query_raw).to(device); correct_t = torch.from_numpy(correct_raw).to(device)
    qtrans, exact_scores = {}, {}
    for rep in REPRESENTATIONS:
        qtrans[rep] = transform_torch(query_t, mean_gpu, pcs_gpu, rep)
        ctrans = transform_torch(correct_t, mean_gpu, pcs_gpu, rep)
        exact_scores[rep] = (qtrans[rep]*ctrans).sum(1).cpu().numpy().astype(np.float64)
    # Preserve published raw exact score/rank threshold precision.
    exact_scores["Raw"] = upstream.paired_cosine.to_numpy(np.float64)
    rank_counts = {r: np.zeros(len(samples), np.int64) for r in REPRESENTATIONS}
    best_v = {r: np.full((len(samples), TOPK), -np.inf, np.float32) for r in REPRESENTATIONS}
    best_i = {r: np.full((len(samples), TOPK), -1, np.int32) for r in REPRESENTATIONS}
    retrieval_start = time.time()
    for start in range(0, n, CHUNK):
        stop = min(n, start+CHUNK)
        x = torch.from_numpy(np.asarray(raw[ref_global[start:stop]], dtype=np.float32)).to(device)
        for rep in REPRESENTATIONS:
            rt = transform_torch(x, mean_gpu, pcs_gpu, rep)
            block = (qtrans[rep] @ rt.T).cpu().numpy()
            rank_counts[rep] += np.sum(block > exact_scores[rep][:,None], axis=1)
            best_v[rep], best_i[rep] = merge_top(best_v[rep], best_i[rep], block, start)
        elapsed = time.time()-retrieval_start; rate = stop/max(elapsed,1e-9); eta=(n-stop)/max(rate,1e-9)
        log(f"retrieval {stop:,}/{n:,}; elapsed={elapsed:.0f}s; ETA={eta:.0f}s")
    retrieval_rows, per_query_rows, macro_rows, biology_rows, density_rows, hub_rows = [], [], [], [], [], []
    raw_rank_mismatch_count = 0
    exact_dist_rows, separation_rows = [], []
    random_dist_rows = [distribution(random_by_rep[r], r, "random_ARCHS4_pairs") for r in REPRESENTATIONS]
    for rep in REPRESENTATIONS:
        ranks = rank_counts[rep] + 1
        if rep == "Raw":
            published_ranks = upstream.exact_rank_full.to_numpy(np.int64)
            raw_rank_mismatch_count = int(np.count_nonzero(ranks != published_ranks))
            ranks = published_ranks
        vals, inds = best_v[rep], best_i[rep]
        order = np.argsort(-vals, axis=1, kind="stable"); vals=np.take_along_axis(vals,order,1); inds=np.take_along_axis(inds,order,1)
        metrics = retrieval_summary(ranks); retrieval_rows.append({"representation":rep, **metrics})
        for i in range(len(samples)):
            per_query_rows.append({"representation":rep,"query_index":i,"gsm":samples.gsm[i],"gse":query_gse[i],
                                   "exact_rank":ranks[i],"exact_pair_cosine":exact_scores[rep][i],
                                   "density_mean_top10":vals[i,:10].mean(),"density_kth_top10":vals[i,9]})
        per_study=[]
        for gse in pd.unique(query_gse):
            mask=query_gse==gse; per_study.append({"gse":gse, **retrieval_summary(ranks[mask])})
        ps=pd.DataFrame(per_study)
        macro_rows.append({"representation":rep,"studies":len(ps),
                           **{f"macro_{c}":ps[c].mean() for c in ["r_at_1","r_at_5","r_at_10","r_at_100","r_at_1000","mrr"]}})
        # Neighborhood preservation and labels, excluding exact for source/tissue.
        same=ref_gse[inds]==query_gse[:,None]
        source_hits=[]; tissue_hits=[]
        for i in range(len(samples)):
            nonexact=inds[i][inds[i]!=correct[i]]
            if query_source[i] in source_eligible:
                hit=ref_source[nonexact[:10]]==query_source[i]; source_hits.append((hit[0],hit.mean()))
            if query_tissue[i] in tissue_eligible:
                hit=ref_tissue[nonexact[:10]]==query_tissue[i]; tissue_hits.append((hit[0],hit.mean()))
        bio={"representation":rep,"same_gse_top1":same[:,0].mean(),
             "same_gse_top10":np.any(same[:,:10],axis=1).mean(),
             "source_queries":len(source_hits),"source_p_at_1":np.mean([x[0] for x in source_hits]),
             "source_p_at_10":np.mean([x[1] for x in source_hits]),
             "tissue_queries":len(tissue_hits),"tissue_p_at_1":np.mean([x[0] for x in tissue_hits]),
             "tissue_p_at_10":np.mean([x[1] for x in tissue_hits])}
        if rep == "Raw":
            bio.update({"source_p_at_1":prior_source.loc["Cosine","biological_p1"],
                        "source_p_at_10":prior_source.loc["Cosine","biological_p10"],
                        "tissue_p_at_1":prior_tissue.loc["Cosine","biological_p1"],
                        "tissue_p_at_10":prior_tissue.loc["Cosine","biological_p10"]})
        biology_rows.append(bio)
        dens=vals[:,:10].mean(1); blood=query_gse=="GSE234297"
        density_rows.append({"representation":rep,"mean_query_density":dens.mean(),"sd_query_density":dens.std(ddof=1),
                             "iqr_query_density":np.quantile(dens,.75)-np.quantile(dens,.25),
                             "gse234297_mean_density":dens[blood].mean(),"other_studies_mean_density":dens[~blood].mean(),
                             "gse234297_minus_others":dens[blood].mean()-dens[~blood].mean()})
        for k in [1,10]:
            counts=np.bincount(inds[:,:k].ravel(),minlength=n); ordered=np.sort(counts)[::-1]
            hub_rows.append({"representation":rep,"k":k,"assignments":len(samples)*k,
                             "references_selected":np.count_nonzero(counts),"maximum_nk":counts.max(),
                             "skewness":skew(counts),"gini":gini(counts),
                             "top10_hubs_assignment_fraction":ordered[:10].sum()/counts.sum()})
        exact_dist_rows.append(distribution(exact_scores[rep],rep,"exact_ARCHS4_recount3_pairs"))
        sr=np.sort(random_by_rep[rep]); exact=exact_scores[rep]
        auc=np.mean((np.searchsorted(sr,exact,"left")+np.searchsorted(sr,exact,"right"))/(2*len(sr)))
        pooled=math.sqrt(((len(sr)-1)*sr.var(ddof=1)+(len(exact)-1)*exact.var(ddof=1))/(len(sr)+len(exact)-2))
        separation_rows.append({"representation":rep,"mean_difference":exact.mean()-sr.mean(),
                                "median_difference":np.median(exact)-np.median(sr),
                                "cohens_d_pooled":(exact.mean()-sr.mean())/pooled,
                                "exact_median_random_percentile":np.searchsorted(sr,np.median(exact),"right")/len(sr),
                                "exact_q1_random_percentile":np.searchsorted(sr,np.quantile(exact,.25),"right")/len(sr),
                                "similarity_auroc":auc})
    pd.DataFrame(random_dist_rows+exact_dist_rows).to_csv(OUT/"cosine_distribution_summaries.csv",index=False)
    pd.DataFrame(retrieval_rows).to_csv(OUT/"retrieval_metrics_micro.csv",index=False)
    pd.DataFrame(macro_rows).to_csv(OUT/"retrieval_metrics_study_macro.csv",index=False)
    pd.DataFrame(per_query_rows).to_parquet(OUT/"per_query_geometry_metrics.parquet",index=False)
    pd.DataFrame(biology_rows).to_csv(OUT/"biological_neighborhood_preservation.csv",index=False)
    pd.DataFrame(density_rows).to_csv(OUT/"query_density_summary.csv",index=False)
    pd.DataFrame(hub_rows).to_csv(OUT/"hubness_summary.csv",index=False)
    pd.DataFrame(separation_rows).to_csv(OUT/"similarity_separation_metrics.csv",index=False)

    # Representative raw cosine values to empirical percentiles.
    raw_sorted=np.sort(random_by_rep["Raw"])
    pd.DataFrame([{"cosine":v,"empirical_percentile":np.searchsorted(raw_sorted,v,"right")/len(raw_sorted),
                   "exceeding_fraction":1-np.searchsorted(raw_sorted,v,"right")/len(raw_sorted)}
                  for v in [.90,.95,.99,.995,.999]]).to_csv(OUT/"raw_cosine_value_percentiles.csv",index=False)

    # Compare independently evaluated local corrections.
    old_micro=pd.read_csv(HERE/"results/absolute_similarity/expanded_ranked_metrics_micro.csv")
    old_macro=pd.read_csv(HERE/"results/absolute_similarity/expanded_ranked_metrics_study_macro.csv")
    comparison=[]
    for _,r in pd.DataFrame(retrieval_rows).iterrows():
        comparison.append({"method_or_representation":r.representation,"r_at_1":r.r_at_1,"r_at_10":r.r_at_10,
                           "r_at_100":r.r_at_100,"r_at_1000":r.r_at_1000,"mrr":r.mrr,
                           "macro_r_at_10":pd.DataFrame(macro_rows).set_index("representation").loc[r.representation,"macro_r_at_10"]})
    for name in ["LocalScaling(k=10)","NICDM(k=10)","CSLS(k=10)"]:
        r=old_micro.set_index("method").loc[name]; m=old_macro.set_index("method").loc[name]
        comparison.append({"method_or_representation":name,"r_at_1":r.r_at_1,"r_at_10":r.r_at_10,
                           "r_at_100":r.r_at_100,"r_at_1000":r.r_at_1000,"mrr":r.mrr,"macro_r_at_10":m.macro_r_at_10})
    pd.DataFrame(comparison).to_csv(OUT/"geometry_vs_local_corrections.csv",index=False)

    # Figures.
    exact_raw=exact_scores["Raw"]; rr=random_by_rep["Raw"]
    fig,ax=plt.subplots(figsize=(9,5)); ax.hist(rr,bins=140,density=True,color="#4c78a8",alpha=.8)
    marks={"random median":np.median(rr),"random P95":np.quantile(rr,.95),"random P99":np.quantile(rr,.99),"exact median":np.median(exact_raw)}
    for label,v in marks.items(): ax.axvline(v,ls="--",lw=1.3,label=f"{label}: {v:.4f}")
    ax.set(xlabel="Raw Bridge cosine",ylabel="Density",title="Sampled human ARCHS4 pairwise cosine"); ax.legend(fontsize=8); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"raw_random_cosine_histogram.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    fig,ax=plt.subplots(figsize=(9,5)); step=max(1,len(raw_sorted)//20000); ax.plot(raw_sorted[::step],np.arange(0,len(raw_sorted),step)/len(raw_sorted),color="#4c78a8")
    for label,v in [("exact Q1",np.quantile(exact_raw,.25)),("exact median",np.median(exact_raw)),("exact Q3",np.quantile(exact_raw,.75))]: ax.axvline(v,ls="--",label=f"{label}: {v:.4f}")
    ax.set(xlabel="Raw Bridge cosine",ylabel="Empirical cumulative probability",title="Random-pair cosine ECDF"); ax.legend(); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"raw_random_cosine_ecdf.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    fig,ax=plt.subplots(figsize=(9,5)); bins=np.linspace(-1,1,180); ax.hist(rr,bins=bins,density=True,alpha=.55,label="Raw"); ax.hist(random_by_rep["Mean-centered"],bins=bins,density=True,alpha=.55,label="Mean-centered")
    ax.set(xlabel="Cosine",ylabel="Density",title="Mean-centering removes the common direction"); ax.legend(); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"raw_vs_centered_cosine.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    fig,ax=plt.subplots(figsize=(10,6)); bins=np.linspace(-.5,.5,160)
    for rep in REPRESENTATIONS[1:]: ax.hist(random_by_rep[rep],bins=bins,density=True,histtype="step",lw=1.5,label=rep)
    ax.set(xlabel="Cosine",ylabel="Density",title="Random-pair cosine after common-component removal"); ax.legend(fontsize=8); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"pc_removal_cosine_distributions.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    distdf=pd.DataFrame(random_dist_rows+exact_dist_rows); fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(REPRESENTATIONS)); w=.36
    ax.bar(x-w/2,distdf[distdf.distribution.str.startswith("random")].set_index("representation").loc[REPRESENTATIONS,"median"],w,label="Random median")
    ax.bar(x+w/2,distdf[distdf.distribution.str.startswith("exact")].set_index("representation").loc[REPRESENTATIONS,"median"],w,label="Exact median")
    ax.set_xticks(x,REPRESENTATIONS,rotation=25,ha="right"); ax.set_ylabel("Median cosine"); ax.set_title("Exact-pair versus random-pair similarity"); ax.legend(); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"exact_vs_random_cosine.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    ret=pd.DataFrame(retrieval_rows); fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(ret)); w=.18
    for j,k in enumerate([1,10,100,1000]): ax.bar(x+(j-1.5)*w,ret[f"r_at_{k}"],w,label=f"R@{k:,}")
    ax.set_xticks(x,ret.representation,rotation=25,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Recall"); ax.set_title("Exact-pair retrieval after geometry correction"); ax.legend(ncol=4); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"retrieval_comparison.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    fig,axes=plt.subplots(1,2,figsize=(12,4.8)); densdf=pd.DataFrame(density_rows); hubs=pd.DataFrame(hub_rows); axes[0].bar(densdf.representation,densdf.sd_query_density); axes[0].tick_params(axis="x",rotation=30); axes[0].set_title("Variation in query Top-10 density"); axes[0].set_ylabel("SD of mean Top-10 cosine")
    h10=hubs[hubs.k.eq(10)]; axes[1].bar(h10.representation,h10.top10_hubs_assignment_fraction); axes[1].tick_params(axis="x",rotation=30); axes[1].set_title("Top-10 hub concentration"); axes[1].set_ylabel("Top 10 hubs' assignment share"); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"hubness_density_comparison.{e}",dpi=220); plt.close(fig) if e=="pdf" else None
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(pc_table.pc[:50],pc_table.explained_variance_ratio[:50],marker="o",ms=3,label="Individual"); ax.plot(pc_table.pc[:50],pc_table.cumulative_explained_variance[:50],label="Cumulative"); ax.set(xlabel="Principal component",ylabel="Explained variance ratio",title="PCA of centered human ARCHS4 Bridge embeddings"); ax.legend(); fig.tight_layout()
    for e in ["png","pdf"]: fig.savefig(FIG/f"pca_explained_variance.{e}",dpi=220); plt.close(fig) if e=="pdf" else None

    runtime=time.time()-started
    config_out={"seed":SEED,"random_pairs":NPAIRS,"pair_batch":PAIR_BATCH,"reference_chunk":CHUNK,
                "representations":REPRESENTATIONS,"reference_fit_only":True,"device":str(device)}
    (OUT/"config.json").write_text(json.dumps(config_out,indent=2)+"\n")
    provenance={"created_at":pd.Timestamp.now(tz="UTC").isoformat(),"runtime_seconds":runtime,
                "peak_rss_gb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                "upstream_read_only":str(up),"bridge_frozen":True,"pca_fit":"all 510,709 ARCHS4 references only",
                "query_transform":"ARCHS4 reference mean/components applied unchanged",
                "random_pair_indices":str(WORK/"anisotropy_random_pair_indices.npz"),
                "random_pair_raw_reproduction_max_abs_delta":max_reproduction_delta,
                "raw_chunked_rank_mismatch_count_before_published_override":raw_rank_mismatch_count,
                "inputs":{str(p):{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in
                          [emb_dir/"embedding_manifest.json",emb_dir/"sample_locations.parquet",
                           inputs["sample_manifest"],inputs["sample_metadata"],up/"work/recount3_embeddings.npy",
                           up/"results/full_archs4_retrieval.parquet"] if p.stat().st_size<100_000_000},
                "limitations":["Random pair distributions are deterministic sampled estimates, not exhaustive pairwise distributions.",
                               "Biological labels are exact normalized source names and a conservative tissue mapping, not inferred states."]}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    log(f"COMPLETE runtime={runtime:.1f}s peak_RSS={provenance['peak_rss_gb']:.2f}GB")
    log_file.close()


if __name__ == "__main__":
    main()
