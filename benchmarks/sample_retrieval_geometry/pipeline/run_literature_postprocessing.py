#!/usr/bin/env python3
"""Literature-grounded post-processing comparison for frozen Bridge embeddings."""
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
from scipy.special import logsumexp
from scipy.stats import skew

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "literature_postprocessing"
FIG = OUT / "figures"
WORK = HERE / "work"
CHUNK = 32768
PAIR_BATCH = 65536
TOPK = 101
METHODS = ["Raw cosine", "Mean-centered", "ABTT D=10", "Whitening",
           "NICDM", "Local Scaling", "CSLS"]
REPS = ["Raw cosine", "Mean-centered", "ABTT D=10", "Whitening"]
Q = [0, .01, .05, .25, .5, .75, .95, .99, .999, 1]
QL = ["minimum", "p1", "p5", "p25", "median", "p75", "p95", "p99", "p99.9", "maximum"]
TISSUES = {
    "blood": r"^(whole |peripheral )?blood$|^pbmcs?$", "liver": r"^liver( tissue)?$",
    "lung": r"^lung( tissue)?$", "skin": r"^skin( tissue)?$",
    "brain": r"^brain( tissue)?$|^cerebral cortex$|^cortex$",
    "breast": r"^(breast|mammary gland)( tissue)?$",
    "colon": r"^(colon|colonic mucosa)( tissue)?$", "kidney": r"^kidney( tissue)?$|^renal cortex$",
    "prostate": r"^prostate( tissue)?$", "pancreas": r"^pancreas( tissue)?$",
    "adipose": r"^(adipose|adipose tissue|subcutaneous adipose tissue)$",
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


def distribution(x: np.ndarray, method: str, kind: str) -> dict:
    x = np.asarray(x, np.float64)
    row = {"method": method, "distribution": kind, "n": len(x), "mean": x.mean(),
           "sd": x.std(ddof=1), "iqr": np.quantile(x, .75) - np.quantile(x, .25)}
    row.update(dict(zip(QL, np.quantile(x, Q))))
    return row


def gini(x: np.ndarray) -> float:
    v = np.sort(np.asarray(x, np.float64))
    if not len(v) or v.sum() == 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(v) + 1), v) / (len(v) * v.sum()) - (len(v) + 1) / len(v))


def merge_top(best_v: np.ndarray, best_i: np.ndarray, block: np.ndarray,
              start: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.broadcast_to(np.arange(start, start + block.shape[1], dtype=np.int32), block.shape)
    vals, inds = np.concatenate([best_v, block], 1), np.concatenate([best_i, idx], 1)
    take = np.argpartition(vals, -TOPK, axis=1)[:, -TOPK:]
    return np.take_along_axis(vals, take, 1), np.take_along_axis(inds, take, 1)


def retrieval_summary(ranks: np.ndarray) -> dict:
    return {"r_at_1": np.mean(ranks <= 1), "r_at_5": np.mean(ranks <= 5),
            "r_at_10": np.mean(ranks <= 10), "r_at_100": np.mean(ranks <= 100),
            "r_at_1000": np.mean(ranks <= 1000), "mrr": np.mean(1 / ranks),
            "median_rank": np.median(ranks),
            "median_percentile_rank": np.median(1 - (ranks - 1) / 510709)}


def transform(x: torch.Tensor, mean: torch.Tensor, pcs10: torch.Tensor,
              whitening: torch.Tensor, method: str, normalize: bool = True) -> torch.Tensor:
    if method == "Raw cosine":
        y = x
    elif method == "Mean-centered":
        y = x - mean
    elif method == "ABTT D=10":
        y = x - mean
        y = y - (y @ pcs10.T) @ pcs10
    elif method == "Whitening":
        y = (x - mean) @ whitening
    else:
        raise ValueError(method)
    return torch.nn.functional.normalize(y, dim=1) if normalize else y


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    log_handle = (OUT / "run.log").open("w", buffering=1)
    def log(message: str) -> None:
        line = f"[{pd.Timestamp.now().isoformat()}] {message}"; print(line, flush=True); log_handle.write(line + "\n")
    log("Starting literature post-processing comparison; estimated runtime 5-15 minutes on GPU")
    log(f"Monitor with: tail -f {OUT / 'run.log'}")

    cfg = json.loads((HERE / "config.json").read_text()); inputs = {k: Path(v) for k, v in cfg["inputs"].items()}
    up, emb_dir = inputs["upstream_benchmark"], inputs["embedding_dir"]
    spec = json.loads((emb_dir / "embedding_manifest.json").read_text())
    locations = pd.read_parquet(emb_dir / "sample_locations.parquet")
    manifest = pd.read_parquet(inputs["sample_manifest"], columns=["gsm", "species"])
    human = locations.merge(manifest, left_on="geo_accession", right_on="gsm", validate="one_to_one")
    human = human[norm_text(human.species).eq("human")].reset_index(drop=True)
    all_samples = pd.read_parquet(up / "work/paired_expression/samples.parquet")
    keep = all_samples.study_exposure.eq("unseen_study").to_numpy()
    samples = all_samples.loc[keep].reset_index(drop=True)
    query_raw = np.asarray(np.load(up / "work/recount3_embeddings.npy", mmap_mode="r")[keep], np.float32)
    upstream = samples[["gsm"]].merge(pd.read_parquet(up / "results/full_archs4_retrieval.parquet"), on="gsm")
    raw = np.memmap(emb_dir / "sample_embeddings.float16.mmap", mode="r", dtype=np.float16,
                    shape=(spec["total_samples"], spec["embedding_dim"]))
    ref_global = human.global_index.to_numpy(np.int64); n, dim = len(human), int(spec["embedding_dim"])
    correct = samples.gsm.map(pd.Series(np.arange(n), index=human.geo_accession)).to_numpy(np.int64)
    correct_raw = np.asarray(raw[ref_global[correct]], np.float32)
    pair_idx = np.load(WORK / "anisotropy_random_pair_indices.npz"); a, b = pair_idx["a"], pair_idx["b"]
    cached_random = np.load(WORK / "anisotropy_random_cosines.npz")
    assert n == 510709 and dim == 512 and len(samples) == 562 and len(a) == 2_000_000
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Fit BERT-whitening exactly as x_tilde=(x-mu) U Lambda^-1/2. Covariance uses 1/N.
    sum_raw = np.zeros(dim, np.float64)
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        sum_raw += np.asarray(raw[ref_global[start:stop]], np.float32).sum(0, dtype=np.float64)
    mean = (sum_raw / n).astype(np.float32)
    mean_t = torch.from_numpy(mean).to(device)
    covariance = np.zeros((dim, dim), np.float64)
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        x = torch.from_numpy(np.asarray(raw[ref_global[start:stop]], np.float32)).to(device) - mean_t
        covariance += (x.T @ x).cpu().numpy().astype(np.float64)
        if stop % (CHUNK * 5) < CHUNK: log(f"covariance fit {stop:,}/{n:,}")
    covariance /= n
    eigvals, eigvecs = np.linalg.eigh(covariance); order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    floor = float(eigvals[0] * 1e-12)
    floored = np.maximum(eigvals, floor)
    whitening = (eigvecs * (1 / np.sqrt(floored))[None, :]).astype(np.float32)
    pcs10 = eigvecs[:, :10].T.astype(np.float32)
    np.savez(WORK / "literature_postprocessing_transform.npz", mean=mean, pcs10=pcs10,
             covariance_eigenvalues=eigvals, whitening=whitening, eigenvalue_floor=floor)
    mean_gpu, pcs_gpu, white_gpu = mean_t, torch.from_numpy(pcs10).to(device), torch.from_numpy(whitening).to(device)

    # Same random pairs for every representation.
    random_values = {"Raw cosine": cached_random["rep_0"], "Mean-centered": cached_random["rep_1"],
                     "ABTT D=10": cached_random["rep_5"]}
    white_random = np.empty(len(a), np.float32)
    for start in range(0, len(a), PAIR_BATCH):
        stop = min(len(a), start + PAIR_BATCH)
        xa = torch.from_numpy(np.asarray(raw[ref_global[a[start:stop]]], np.float32)).to(device)
        xb = torch.from_numpy(np.asarray(raw[ref_global[b[start:stop]]], np.float32)).to(device)
        ta = transform(xa, mean_gpu, pcs_gpu, white_gpu, "Whitening")
        tb = transform(xb, mean_gpu, pcs_gpu, white_gpu, "Whitening")
        white_random[start:stop] = (ta * tb).sum(1).cpu().numpy()
    random_values["Whitening"] = white_random
    np.save(WORK / "literature_postprocessing_whitening_random_cosines.npy", white_random)

    # Exact-pair alignment.
    qraw_t, craw_t = torch.from_numpy(query_raw).to(device), torch.from_numpy(correct_raw).to(device)
    qtrans, exact = {}, {}
    for method in REPS:
        qtrans[method] = transform(qraw_t, mean_gpu, pcs_gpu, white_gpu, method)
        ct = transform(craw_t, mean_gpu, pcs_gpu, white_gpu, method)
        exact[method] = (qtrans[method] * ct).sum(1).cpu().numpy().astype(np.float64)
    exact["Raw cosine"] = upstream.paired_cosine.to_numpy(np.float64)

    # Full-corpus retrieval is needed only for whitening. Existing representations are reused.
    white_counts = np.zeros(len(samples), np.int64)
    best_v = np.full((len(samples), TOPK), -np.inf, np.float32)
    best_i = np.full((len(samples), TOPK), -1, np.int32)
    # Geometry summaries on normalized references and numerical whitening verification.
    sums = {m: np.zeros(dim, np.float64) for m in REPS}
    sums_outer = {m: np.zeros((dim, dim), np.float64) for m in REPS}
    white_pre_sum = np.zeros(dim, np.float64); white_pre_outer = np.zeros((dim, dim), np.float64)
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        x = torch.from_numpy(np.asarray(raw[ref_global[start:stop]], np.float32)).to(device)
        for method in REPS:
            yt = transform(x, mean_gpu, pcs_gpu, white_gpu, method)
            y = yt.cpu().numpy().astype(np.float64)
            sums[method] += y.sum(0); sums_outer[method] += y.T @ y
            if method == "Whitening":
                block = (qtrans[method] @ yt.T).cpu().numpy()
                white_counts += np.sum(block > exact[method][:, None], axis=1)
                best_v, best_i = merge_top(best_v, best_i, block, start)
        wp = transform(x, mean_gpu, pcs_gpu, white_gpu, "Whitening", normalize=False).cpu().numpy().astype(np.float64)
        white_pre_sum += wp.sum(0); white_pre_outer += wp.T @ wp
        elapsed = time.time() - started; log(f"geometry/retrieval {stop:,}/{n:,}; elapsed={elapsed:.0f}s")
    white_ranks = white_counts + 1
    order_top = np.argsort(-best_v, axis=1, kind="stable")
    best_v, best_i = np.take_along_axis(best_v, order_top, 1), np.take_along_axis(best_i, order_top, 1)

    # Geometry/eigenvalue summaries on L2-normalized representations.
    geom_rows, spectrum_rows = [], []
    for method in REPS:
        mu = sums[method] / n
        cov = (sums_outer[method] / n) - np.outer(mu, mu)
        ev = np.linalg.eigvalsh(cov)[::-1]; pr = ev.sum() ** 2 / np.square(ev).sum()
        geom_rows.append({"method": method, "mean_direction_magnitude": np.linalg.norm(mu),
                          "mean_direction_magnitude_squared": np.dot(mu, mu),
                          "covariance_max_eigenvalue": ev[0], "covariance_min_eigenvalue": ev[-1],
                          "participation_ratio": pr, "effective_dimension_fraction": pr / dim})
        for i, value in enumerate(ev):
            spectrum_rows.append({"method": method, "component": i + 1, "eigenvalue": value,
                                  "explained_variance_ratio": value / ev.sum()})
    geom = pd.DataFrame(geom_rows); spectrum = pd.DataFrame(spectrum_rows)
    geom.to_csv(OUT / "representation_geometry_summary.csv", index=False)
    spectrum.to_csv(OUT / "representation_covariance_spectra.csv", index=False)
    wp_mu = white_pre_sum / n; wp_cov = white_pre_outer / n - np.outer(wp_mu, wp_mu)
    wp_ev = np.linalg.eigvalsh(wp_cov)[::-1]
    pd.DataFrame({"component": np.arange(1, dim + 1), "pre_whitening_eigenvalue": eigvals,
                  "post_whitening_eigenvalue": wp_ev}).to_csv(OUT / "whitening_covariance_spectrum.csv", index=False)
    pd.DataFrame([{"dimensions_retained": dim, "eigenvalue_floor": floor,
                   "eigenvalues_affected_by_floor": int(np.sum(eigvals < floor)),
                   "pre_min_eigenvalue": eigvals[-1], "pre_max_eigenvalue": eigvals[0],
                   "pre_condition_number": eigvals[0] / eigvals[-1],
                   "pre_participation_ratio": eigvals.sum() ** 2 / np.square(eigvals).sum(),
                   "post_min_eigenvalue": wp_ev[-1], "post_max_eigenvalue": wp_ev[0],
                   "post_condition_number": wp_ev[0] / wp_ev[-1],
                   "post_participation_ratio": wp_ev.sum() ** 2 / np.square(wp_ev).sum(),
                   "post_max_abs_eigenvalue_minus_one": np.max(np.abs(wp_ev - 1))}]).to_csv(
                       OUT / "whitening_diagnostics.csv", index=False)

    # Distributions, alignment, uniformity, and separation.
    dist_rows, diagnostic_rows = [], []
    for method in REPS:
        rv, ex = random_values[method], exact[method]
        dist_rows.extend([distribution(rv, method, "random_ARCHS4_pairs"),
                          distribution(ex, method, "exact_ARCHS4_recount3_pairs")])
        sr = np.sort(rv)
        auc = np.mean((np.searchsorted(sr, ex, "left") + np.searchsorted(sr, ex, "right")) / (2 * len(sr)))
        diagnostic_rows.append({
            "method": method,
            "alignment": np.mean(2 - 2 * ex),
            "uniformity": logsumexp(-4 + 4 * rv) - math.log(len(rv)),
            "exact_pair_auroc": auc,
            "exact_median_random_percentile": np.searchsorted(sr, np.median(ex), "right") / len(sr),
            "exact_q1_random_percentile": np.searchsorted(sr, np.quantile(ex, .25), "right") / len(sr),
        })
    pd.DataFrame(dist_rows).to_csv(OUT / "cosine_distribution_summaries.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(OUT / "alignment_uniformity.csv", index=False)

    # Combine newly computed whitening retrieval with previously frozen results.
    old_micro = pd.read_csv(HERE / "results/anisotropy/retrieval_metrics_micro.csv").set_index("representation")
    old_macro = pd.read_csv(HERE / "results/anisotropy/retrieval_metrics_study_macro.csv").set_index("representation")
    local_micro = pd.read_csv(HERE / "results/absolute_similarity/expanded_ranked_metrics_micro.csv").set_index("method")
    local_macro = pd.read_csv(HERE / "results/absolute_similarity/expanded_ranked_metrics_study_macro.csv").set_index("method")
    map_rep = {"Raw cosine": "Raw", "Mean-centered": "Mean-centered", "ABTT D=10": "Centered + remove PCs 1–10"}
    micro_rows, macro_rows = [], []
    for method, old in map_rep.items():
        micro_rows.append({"method": method, "type": "Representation", **old_micro.loc[old].to_dict()})
        macro_rows.append({"method": method, "type": "Representation", **old_macro.loc[old].to_dict()})
    micro_rows.append({"method": "Whitening", "type": "Representation", **retrieval_summary(white_ranks)})
    study_rows = []
    for gse in pd.unique(samples.gse.astype(str)):
        study_rows.append(retrieval_summary(white_ranks[samples.gse.astype(str).eq(gse).to_numpy()]))
    st = pd.DataFrame(study_rows)
    macro_rows.append({"method": "Whitening", "type": "Representation", "studies": len(st),
                       **{f"macro_{c}": st[c].mean() for c in ["r_at_1", "r_at_5", "r_at_10", "r_at_100", "r_at_1000", "mrr"]}})
    local_map = {"NICDM": "NICDM(k=10)", "Local Scaling": "LocalScaling(k=10)", "CSLS": "CSLS(k=10)"}
    for method, old in local_map.items():
        micro_rows.append({"method": method, "type": "Retrieval", **local_micro.loc[old].to_dict()})
        macro_rows.append({"method": method, "type": "Retrieval", **local_macro.loc[old].to_dict()})
    micro, macro = pd.DataFrame(micro_rows), pd.DataFrame(macro_rows)
    micro.to_csv(OUT / "retrieval_metrics_micro.csv", index=False)
    macro.to_csv(OUT / "retrieval_metrics_study_macro.csv", index=False)

    # Biological preservation. Existing exact-label metrics are authoritative; whitening uses same definitions.
    meta = pd.read_parquet(inputs["sample_metadata"]).drop_duplicates("gsm").set_index("gsm")
    hmeta, qmeta = meta.reindex(human.geo_accession), meta.reindex(samples.gsm)
    ref_gse = hmeta.series_id.fillna("").astype(str).to_numpy(); query_gse = samples.gse.astype(str).to_numpy()
    ref_source = norm_text(hmeta.source_name_ch1).fillna("").to_numpy()
    query_source = norm_text(qmeta.source_name_ch1).fillna("").to_numpy()
    ref_tissue = np.full(n, "", object); query_tissue = np.full(len(samples), "", object)
    for label, pattern in TISSUES.items():
        ref_tissue[norm_text(hmeta.source_name_ch1).str.match(pattern, na=False).to_numpy()] = label
        query_tissue[norm_text(qmeta.source_name_ch1).str.match(pattern, na=False).to_numpy()] = label
    qc = pd.read_csv(HERE / "results/biological_validation/biological_label_class_qc.csv")
    source_eligible = set(qc[(qc.label.eq("source_label")) & qc.eligible].iloc[:, 1].astype(str))
    tissue_eligible = set(qc[(qc.label.eq("tissue_label")) & qc.eligible].iloc[:, 1].astype(str))
    old_bio = pd.read_csv(HERE / "results/anisotropy/biological_neighborhood_preservation.csv").set_index("representation")
    source_table = pd.read_csv(HERE / "results/biological_validation/main_comparison_source_label.csv").set_index("method")
    tissue_table = pd.read_csv(HERE / "results/biological_validation/main_comparison_tissue_label.csv").set_index("method")
    neighbors = pd.read_parquet(HERE / "results/biological_validation/biological_top10_neighbors.parquet")
    ranked_comparison = pd.read_csv(HERE / "results/ranked_method_comparison.csv").set_index("method")
    # Reconstruct only Top-10 identities for the three local-score methods from the
    # retained cosine cache. This avoids rerunning embedding search while making
    # same-GSE Top-10 directly comparable (the exact counterpart remains eligible).
    scores = np.memmap(WORK / "query_corpus_cosine.float32.mmap", mode="r", dtype=np.float32,
                       shape=(len(samples), n))
    q_top = np.partition(scores, -10, axis=1)[:, -10:].copy()
    q_mean, q_kth = q_top.mean(1), q_top.min(1)
    ref_mean = np.empty(n, np.float32); ref_kth = np.empty(n, np.float32)
    local_best_v = {m: np.full((len(samples), 10), -np.inf, np.float32) for m in local_map}
    local_best_i = {m: np.full((len(samples), 10), -1, np.int32) for m in local_map}
    eps = np.finfo(np.float32).eps
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK); block = np.asarray(scores[:, start:stop])
        rtop = np.partition(block, -10, axis=0)[-10:, :]
        ref_mean[start:stop], ref_kth[start:stop] = rtop.mean(0), rtop.min(0)
        local_blocks = {
            "CSLS": 2 * block - q_mean[:, None] - ref_mean[start:stop][None, :],
            "Local Scaling": -(np.maximum(1 - block, eps) ** 2) /
                             (np.maximum(1 - q_kth, eps)[:, None] * np.maximum(1 - ref_kth[start:stop], eps)[None, :]),
            "NICDM": -np.maximum(1 - block, eps) /
                     np.sqrt(np.maximum(1 - q_top, eps).mean(1)[:, None] *
                             np.maximum(1 - rtop, eps).mean(0)[None, :]),
        }
        for method, local_block in local_blocks.items():
            local_best_v[method], local_best_i[method] = merge_top(local_best_v[method], local_best_i[method], local_block, start)
            # Keep only ten entries even though merge_top uses the global TOPK.
            order = np.argsort(-local_best_v[method], axis=1, kind="stable")[:, :10]
            local_best_v[method] = np.take_along_axis(local_best_v[method], order, 1)
            local_best_i[method] = np.take_along_axis(local_best_i[method], order, 1)
    bio_rows = []
    for method, old in map_rep.items():
        bio_rows.append({"method": method, **old_bio.loc[old].to_dict()})
    same = ref_gse[best_i] == query_gse[:, None]
    source_hits, tissue_hits = [], []
    for i in range(len(samples)):
        nonexact = best_i[i][best_i[i] != correct[i]]
        if query_source[i] in source_eligible:
            hit = ref_source[nonexact[:10]] == query_source[i]; source_hits.append((hit[0], hit.mean()))
        if query_tissue[i] in tissue_eligible:
            hit = ref_tissue[nonexact[:10]] == query_tissue[i]; tissue_hits.append((hit[0], hit.mean()))
    bio_rows.append({"method": "Whitening", "same_gse_top1": same[:, 0].mean(),
                     "same_gse_top10": np.any(same[:, :10], 1).mean(), "source_queries": len(source_hits),
                     "source_p_at_1": np.mean([x[0] for x in source_hits]), "source_p_at_10": np.mean([x[1] for x in source_hits]),
                     "tissue_queries": len(tissue_hits), "tissue_p_at_1": np.mean([x[0] for x in tissue_hits]),
                     "tissue_p_at_10": np.mean([x[1] for x in tissue_hits])})
    for method, old in local_map.items():
        s, t = source_table.loc[old], tissue_table.loc[old]
        inds = local_best_i[method]
        sg = np.any(ref_gse[inds] == query_gse[:, None], axis=1).mean()
        bio_rows.append({"method": method, "same_gse_top1": ranked_comparison.loc[old, "same_gse_r1"], "same_gse_top10": sg,
                         "source_queries": 286, "source_p_at_1": s.biological_p1, "source_p_at_10": s.biological_p10,
                         "tissue_queries": 162, "tissue_p_at_1": t.biological_p1, "tissue_p_at_10": t.biological_p10})
    biology = pd.DataFrame(bio_rows)
    biology.to_csv(OUT / "biological_neighborhood_preservation.csv", index=False)

    # Hubness: reuse all existing methods and compute whitening identically.
    prior_hub = pd.read_csv(HERE / "results/hubness_summary.csv")
    hub_rows = []
    for method, old in [("Raw cosine", "Cosine"), ("NICDM", "NICDM(k=10)"),
                        ("Local Scaling", "LocalScaling(k=10)"), ("CSLS", "CSLS(k=10)")]:
        for k in (1, 10):
            r = prior_hub[(prior_hub.method.eq(old)) & prior_hub.k.eq(k)].iloc[0]
            hub_rows.append({"method": method, "k": k, "assignments": r.assignments,
                             "references_selected": r.references_selected, "maximum_nk": r.max_nk,
                             "skewness": r.skewness, "gini": r.gini,
                             "top10_hubs_assignment_fraction": r.top_10_hubs_assignment_fraction})
    anis_hub = pd.read_csv(HERE / "results/anisotropy/hubness_summary.csv")
    for method, old in [("Mean-centered", "Mean-centered"), ("ABTT D=10", "Centered + remove PCs 1–10")]:
        for _, r in anis_hub[anis_hub.representation.eq(old)].iterrows():
            hub_rows.append({"method": method, **{c: r[c] for c in ["k", "assignments", "references_selected", "maximum_nk", "skewness", "gini", "top10_hubs_assignment_fraction"]}})
    for k in (1, 10):
        counts = np.bincount(best_i[:, :k].ravel(), minlength=n); ordered = np.sort(counts)[::-1]
        hub_rows.append({"method": "Whitening", "k": k, "assignments": len(samples) * k,
                         "references_selected": np.count_nonzero(counts), "maximum_nk": counts.max(),
                         "skewness": skew(counts), "gini": gini(counts),
                         "top10_hubs_assignment_fraction": ordered[:10].sum() / counts.sum()})
    hubs = pd.DataFrame(hub_rows)
    hubs.to_csv(OUT / "hubness_summary.csv", index=False)

    # Main table keeps diagnostics blank for retrieval-score corrections.
    main = micro.merge(macro[["method", "macro_r_at_10"]], on="method", how="left")
    main = main.merge(diagnostics[["method", "alignment", "uniformity"]], on="method", how="left")
    main = main[["method", "type", "r_at_1", "r_at_10", "r_at_100", "r_at_1000", "mrr",
                 "macro_r_at_10", "alignment", "uniformity"]]
    main.to_csv(OUT / "main_comparison.csv", index=False)

    # Figures.
    colors = dict(zip(REPS, ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]))
    fig, ax = plt.subplots(figsize=(10, 5.5)); bins = np.linspace(-1, 1, 220)
    for method in REPS: ax.hist(random_values[method], bins=bins, density=True, histtype="step", lw=1.7, label=method, color=colors[method])
    ax.set(xlabel="Cosine similarity", ylabel="Density", title="Random human ARCHS4 pairs after post-processing"); ax.legend(); savefig(fig, "random_pair_distributions")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8));
    for ax, method in zip(axes.ravel(), REPS):
        ax.hist(random_values[method], bins=160, density=True, alpha=.65, label="Random ARCHS4 pairs", color="#4c78a8")
        ax.hist(exact[method], bins=50, density=True, alpha=.55, label="Exact cross-pipeline pairs", color="#e45756")
        ax.set_title(method); ax.set_xlabel("Cosine"); ax.set_ylabel("Density")
    axes[0, 0].legend(fontsize=8); savefig(fig, "exact_vs_random_distributions")
    fig, ax = plt.subplots(figsize=(7, 5.5));
    for _, r in diagnostics.iterrows(): ax.scatter(r.uniformity, r.alignment, s=80, label=r.method)
    ax.set(xlabel="Uniformity (lower / more negative is more uniform)", ylabel="Alignment (lower is closer)", title="SimCSE alignment–uniformity diagnostic"); ax.legend(); savefig(fig, "alignment_uniformity")
    for table, macro_mode, stem, title in [(micro, False, "retrieval_micro", "Micro exact-pair retrieval"), (macro, True, "retrieval_study_macro", "Study-macro exact-pair retrieval")]:
        fig, ax = plt.subplots(figsize=(11, 5.5)); x = np.arange(len(table)); width = .18
        prefix = "macro_" if macro_mode else ""
        for j, k in enumerate([1, 10, 100, 1000]): ax.bar(x + (j - 1.5) * width, table[f"{prefix}r_at_{k}"], width, label=f"R@{k:,}")
        ax.set_xticks(x, table.method, rotation=25, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Recall"); ax.set_title(title); ax.legend(ncol=4); savefig(fig, stem)
    fig, ax = plt.subplots(figsize=(11, 5.5)); bcols = ["same_gse_top10", "source_p_at_10", "tissue_p_at_10"]
    x = np.arange(len(biology)); width = .24
    for j, col in enumerate(bcols): ax.bar(x + (j - 1) * width, biology[col], width, label=col.replace("_", " "))
    ax.set_xticks(x, biology.method, rotation=25, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Fraction / precision"); ax.set_title("Biological-neighborhood preservation (tissue is sensitivity-only)"); ax.legend(); savefig(fig, "biological_preservation")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5));
    for ax, k in zip(axes, [1, 10]):
        d = hubs[hubs.k.eq(k)]; ax.bar(d.method, d.top10_hubs_assignment_fraction); ax.tick_params(axis="x", rotation=30); ax.set_title(f"Top-{k} hub concentration"); ax.set_ylabel("Top 10 hubs' assignment share")
    savefig(fig, "hubness_comparison")
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.semilogy(np.arange(1, dim + 1), eigvals, label="Before whitening"); ax.semilogy(np.arange(1, dim + 1), wp_ev, label="After whitening"); ax.set(xlabel="Covariance component", ylabel="Eigenvalue (log scale)", title="BERT-whitening equalizes covariance variance"); ax.legend(); savefig(fig, "whitening_covariance_spectrum")

    # Fully specified, immutable candidate configurations for independent GTEx evaluation.
    frozen = {
        "frozen_before_gtex": True, "reference_corpus": "510,709 human ARCHS4 embeddings", "query_data_examined": False,
        "candidates": {
            "Raw cosine": {"transform": "L2 normalize original embedding", "ranking": "cosine"},
            "ABTT D=10": {"fit": "ARCHS4 references only", "center": True, "removed_components": list(range(1, 11)), "ranking": "cosine"},
            "Whitening": {"fit": "ARCHS4 references only", "covariance_denominator": "N", "dimensions": 512,
                           "eigendecomposition": "Sigma=U diag(lambda) U^T; W=U diag(max(lambda,lambda_floor)^-1/2)",
                           "eigenvalue_floor_relative_to_lambda_max": 1e-12, "eigenvalues_affected_by_floor": int(np.sum(eigvals < floor)),
                           "post_transform": "L2 normalization", "ranking": "cosine",
                           "parameter_file": str(WORK / "literature_postprocessing_transform.npz")},
            "NICDM": {"k": 10, "configuration": "existing frozen Stage-1 formulation"},
            "Local Scaling": {"k": 10, "configuration": "existing frozen Stage-1 formulation"},
        },
        "supplemental_not_advanced_as_primary": {"Mean-centered": {}, "CSLS": {"k": 10}},
    }
    (OUT / "frozen_for_gtex.json").write_text(json.dumps(frozen, indent=2) + "\n")
    literature = {
        "ABTT": {"citation": "Mu and Viswanath, ICLR 2018", "url": "https://openreview.net/forum?id=HkuGJ3kCb",
                 "formulation": "subtract reference mean, then subtract projections onto reference PCs 1-10", "D": 10, "tuned_here": False},
        "BERT_whitening": {"citation": "Su et al., 2021", "url": "https://arxiv.org/abs/2103.15316",
                           "formulation": "(x-mu) U Lambda^-1/2", "dimensions": 512, "covariance": "maximum-likelihood (1/N)"},
        "alignment_uniformity": {"citation": "Wang and Isola, ICML 2020; used diagnostically by SimCSE",
                                 "urls": ["https://proceedings.mlr.press/v119/wang20k.html", "https://arxiv.org/abs/2104.08821"],
                                 "alignment": "mean ||f(x)-f(x+)||_2^2", "uniformity": "log mean exp(-2||f(x)-f(y)||_2^2)"},
    }
    (OUT / "literature_method_provenance.json").write_text(json.dumps(literature, indent=2) + "\n")
    runtime = time.time() - started
    provenance = {"created_at": pd.Timestamp.now(tz="UTC").isoformat(), "runtime_seconds": runtime,
                  "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                  "bridge_frozen": True, "upstream_read_only": str(up), "reference_fit_only": True,
                  "random_pairs": len(a), "random_pair_seed": int(pair_idx["seed"]),
                  "raw_centered_abtt_reused_from": str(HERE / "results/anisotropy"),
                  "retrieval_corrections_reused_from": str(HERE / "results/absolute_similarity"),
                  "whitening_parameter_sha256": sha256(WORK / "literature_postprocessing_transform.npz"),
                  "limitations": ["Random-pair quantities are deterministic sampled estimates.",
                                  "Source is exact normalized source_name_ch1; tissue is a limited blood/brain-dominated sensitivity analysis.",
                                  "Recount3 was not used to tune ABTT D or whitening regularization.",
                                  "GTEx was not inspected."]}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"COMPLETE runtime={runtime:.1f}s peak_RSS={provenance['peak_rss_gb']:.2f}GB")
    log_handle.close()


if __name__ == "__main__":
    main()
