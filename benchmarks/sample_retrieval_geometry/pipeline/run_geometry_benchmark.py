#!/usr/bin/env python3
"""Frozen Bridge sample-retrieval geometry benchmark (Stages 0 and 1 only).

Large arrays are written only to work/. Compact tables, provenance, logs, and
figures are written to results/. Upstream assets are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import skew

HERE = Path(__file__).resolve().parents[1]
QUANTILES = [0, .0001, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
QLABELS = ["minimum", "p0.01", "p0.1", "p1", "p5", "p25", "median", "p75", "p95", "p99", "p99.9", "maximum"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="Rebuild regenerable score memmap")
    return p.parse_args()


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", buffering=1)

    def __call__(self, message: str):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self):
        self.handle.close()


@contextmanager
def timed(name: str, timings: dict[str, float], log: Logger):
    start = time.perf_counter()
    log(f"START {name}")
    yield
    timings[name] = time.perf_counter() - start
    log(f"DONE {name}: {timings[name]:.1f}s")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding")
    return x / norms


def quantile_frame(values: np.ndarray, distribution: str, estimated: bool) -> pd.DataFrame:
    q = np.quantile(values, QUANTILES)
    return pd.DataFrame({"distribution": distribution, "statistic": QLABELS,
                         "quantile": QUANTILES, "cosine": q, "estimated": estimated,
                         "observations": values.size})


def normalized_text(x: pd.Series) -> pd.Series:
    return x.astype("string").str.strip().str.lower()


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum())) - (len(x) + 1) / len(x))


def topk_indices(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


def ranked_metrics(scores: np.ndarray, correct: np.ndarray, ref_gse: np.ndarray,
                   query_gse: np.ndarray, ref_source: np.ndarray, query_source: np.ndarray,
                   method: str, runtime: float, paired_override: np.ndarray | None = None
                   ) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    n, m = scores.shape
    paired = (scores[np.arange(n), correct].astype(np.float64) if paired_override is None
              else np.asarray(paired_override, dtype=np.float64))
    # Competition rank is stable under exact ties and avoids sorting 510k values/query.
    ranks = 1 + np.sum(scores > paired[:, None], axis=1)
    raw_idx, raw_vals = topk_indices(scores, 101)
    top_idx = np.empty((n, 100), dtype=raw_idx.dtype)
    top_vals = np.empty((n, 100), dtype=np.float64)
    for i in range(n):
        keep = raw_idx[i] != correct[i]
        ci = np.r_[raw_idx[i, keep], correct[i]]
        cv = np.r_[raw_vals[i, keep], paired[i]]
        order = np.argsort(-cv, kind="stable")[:100]
        top_idx[i], top_vals[i] = ci[order], cv[order]
    top_gse = ref_gse[top_idx]
    same_gse = top_gse == query_gse[:, None]
    query_source = np.asarray(query_source)
    known_q = ~pd.isna(query_source) & (query_source != "")
    same_source = ref_source[top_idx] == query_source[:, None]
    source_r1 = float(np.mean(same_source[known_q, 0])) if known_q.any() else np.nan
    source_r5 = float(np.mean(np.any(same_source[known_q, :5], axis=1))) if known_q.any() else np.nan
    metric = {
        "method": method, "ranking_defined": True, "queries": n, "references": m,
        "exact_r1": np.mean(ranks <= 1), "exact_r5": np.mean(ranks <= 5),
        "exact_r10": np.mean(ranks <= 10), "exact_r100": np.mean(ranks <= 100),
        "mrr": np.mean(1 / ranks), "median_rank": np.median(ranks),
        "median_percentile_rank": np.median(1 - (ranks - 1) / m),
        "same_gse_r1": np.mean(same_gse[:, 0]), "same_gse_r5": np.mean(np.any(same_gse[:, :5], axis=1)),
        "source_r1_known": source_r1, "source_r5_known": source_r5,
        "runtime_seconds": runtime,
    }
    detail = pd.DataFrame({"method": method, "query_index": np.arange(n), "exact_rank": ranks,
                           "paired_score": paired, "top1_reference_index": top_idx[:, 0],
                           "top1_score": top_vals[:, 0], "top1_same_gse": same_gse[:, 0],
                           "top5_same_gse": np.any(same_gse[:, :5], axis=1)})
    return metric, detail, top_idx, top_vals


def concentration_rows(top_idx: np.ndarray, method: str, nref: int,
                       ks=(1, 5, 10, 100)) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, all_counts = [], []
    for k in ks:
        counts = np.bincount(top_idx[:, :k].ravel(), minlength=nref)
        nonzero = counts[counts > 0]
        ordered = np.sort(counts)[::-1]
        summary.append({"method": method, "k": k, "assignments": len(top_idx) * k,
                        "references_selected": int((counts > 0).sum()), "max_nk": int(counts.max()),
                        "mean_nk": float(counts.mean()), "sd_nk": float(counts.std()),
                        "skewness": float(skew(counts)), "gini": gini(counts),
                        "top_1pct_assignment_fraction": float(ordered[:max(1, math.ceil(.01 * len(counts)))].sum() / counts.sum()),
                        "top_10_hubs_assignment_fraction": float(ordered[:10].sum() / counts.sum()),
                        "nonzero_median_nk": float(np.median(nonzero)) if len(nonzero) else 0.0})
        all_counts.append(pd.DataFrame({"method": method, "k": k, "reference_index": np.arange(len(counts)), "n_k": counts}))
    return pd.DataFrame(summary), pd.concat(all_counts, ignore_index=True)


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    if args.device:
        cfg["device"] = args.device
    results, work = HERE / "results", HERE / "work"
    figures = results / "figures"
    results.mkdir(exist_ok=True); work.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    log = Logger(results / "run.log")
    timings: dict[str, float] = {}
    started = time.time()
    log("Estimated runtime: 10-30 minutes on one RTX 3090; peak RAM ~5 GB; work cache ~1.2 GB")
    log(f"Monitoring command: tail -f {results / 'run.log'}")

    inputs = {k: Path(v) for k, v in cfg["inputs"].items()}
    up = inputs["upstream_benchmark"]
    emb_dir = inputs["embedding_dir"]
    required = [up / "work/recount3_embeddings.npy", up / "work/paired_expression/samples.parquet",
                up / "results/full_archs4_retrieval.parquet", up / "results/benchmark_summary.csv",
                up / "results/full_archs4_top10_neighbors.parquet",
                emb_dir / "embedding_manifest.json", emb_dir / "sample_locations.parquet",
                inputs["sample_manifest"], inputs["sample_metadata"], inputs["canonical_genes"]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    with timed("load_and_validate", timings, log):
        samples_all = pd.read_parquet(up / "work/paired_expression/samples.parquet")
        keep = samples_all.study_exposure.eq("unseen_study").to_numpy()
        samples = samples_all.loc[keep].reset_index(drop=True)
        queries = unit_rows(np.load(up / "work/recount3_embeddings.npy", mmap_mode="r")[keep])
        upstream_full = pd.read_parquet(up / "results/full_archs4_retrieval.parquet")
        upstream_full = samples[["gsm"]].merge(upstream_full, on="gsm", validate="one_to_one")
        upstream_neighbors = pd.read_parquet(up / "results/full_archs4_top10_neighbors.parquet")
        spec = json.loads((emb_dir / "embedding_manifest.json").read_text())
        locations = pd.read_parquet(emb_dir / "sample_locations.parquet")
        manifest = pd.read_parquet(inputs["sample_manifest"], columns=["gsm", "species"])
        human = locations.merge(manifest, left_on="geo_accession", right_on="gsm", validate="one_to_one")
        human = human[normalized_text(human.species).eq("human")].reset_index(drop=True)
        assert len(samples) == len(queries) == 562
        assert len(human) == 510709
        assert samples.gsm.is_unique and human.geo_accession.is_unique
        correct_lookup = pd.Series(np.arange(len(human)), index=human.geo_accession)
        correct = samples.gsm.map(correct_lookup).to_numpy()
        assert np.all(pd.notna(correct)); correct = correct.astype(np.int64)
        raw = np.memmap(emb_dir / f"sample_embeddings.{spec.get('embedding_dtype', 'float16')}.mmap",
                        mode="r", dtype=np.dtype(spec.get("embedding_dtype", "float16")),
                        shape=(spec["total_samples"], spec["embedding_dim"]))
        metadata = pd.read_parquet(inputs["sample_metadata"]).drop_duplicates("gsm").set_index("gsm")
        hmeta = metadata.reindex(human.geo_accession)
        qmeta = metadata.reindex(samples.gsm)
        ref_gse = hmeta.series_id.fillna("").astype(str).to_numpy()
        query_gse = samples.gse.fillna("").astype(str).to_numpy()
        ref_source = normalized_text(hmeta.source_name_ch1).fillna("").to_numpy()
        query_source = normalized_text(qmeta.source_name_ch1).fillna("")
        assert np.array_equal(human.iloc[correct].geo_accession.to_numpy(), samples.gsm.to_numpy())
        canonical_gene_count = len(pd.read_csv(inputs["canonical_genes"]))
        assert canonical_gene_count == 15165
        log(f"Validated {len(samples)} queries, {len(human):,} human references, {canonical_gene_count:,} genes")

    score_path = work / "query_corpus_cosine.float32.mmap"
    score_shape = (len(queries), len(human))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if args.force and score_path.exists():
        score_path.unlink()
    if not score_path.exists():
        with timed("exact_query_corpus_cosine", timings, log):
            scores = np.memmap(score_path, mode="w+", dtype=np.float32, shape=score_shape)
            tq = torch.from_numpy(queries).to(device)
            chunk = cfg["reference_chunk_size"]
            for start in range(0, len(human), chunk):
                stop = min(start + chunk, len(human))
                x = unit_rows(np.asarray(raw[human.global_index.iloc[start:stop].to_numpy()], dtype=np.float32))
                tx = torch.from_numpy(x).to(device)
                scores[:, start:stop] = (tq @ tx.T).cpu().numpy()
                elapsed = time.perf_counter() - (started - time.time() + time.perf_counter()) if False else time.time() - started
                rate = stop / max(elapsed, 1e-9)
                eta = (len(human) - stop) / max(rate, 1e-9)
                log(f"cosine {stop:,}/{len(human):,}; elapsed={elapsed:.0f}s; rate={rate:,.0f} refs/s; ETA={eta:.0f}s")
            scores.flush()
            del tq
    else:
        log(f"Reusing regenerable score cache {score_path}")
    scores = np.memmap(score_path, mode="r", dtype=np.float32, shape=score_shape)

    with timed("stage0_ranges", timings, log):
        # Exact quantiles over all 287M query/reference similarities.
        range_frames = [quantile_frame(scores, "query_to_corpus_all", False)]
        cosine_top_idx, cosine_top_vals = topk_indices(scores, 100)
        range_frames += [quantile_frame(cosine_top_vals[:, 0], "query_top1", False),
                         quantile_frame(cosine_top_vals[:, 9], "query_top10_threshold", False),
                         quantile_frame(cosine_top_vals[:, 99], "query_top100_threshold", False),
                         quantile_frame(upstream_full.paired_cosine.to_numpy(), "exact_cross_pipeline_pair", False)]
        rng = np.random.default_rng(cfg["random_seed"])
        npairs = cfg["pairwise_sample_pairs"]
        a = rng.integers(0, len(human), npairs, dtype=np.int64)
        b = rng.integers(0, len(human) - 1, npairs, dtype=np.int64)
        b += b >= a
        pair_cos = np.empty(npairs, dtype=np.float32)
        batch = 65536
        for start in range(0, npairs, batch):
            stop = min(start + batch, npairs)
            va = unit_rows(np.asarray(raw[human.global_index.iloc[a[start:stop]].to_numpy()], dtype=np.float32))
            vb = unit_rows(np.asarray(raw[human.global_index.iloc[b[start:stop]].to_numpy()], dtype=np.float32))
            pair_cos[start:stop] = np.einsum("ij,ij->i", va, vb)
        range_frames.append(quantile_frame(pair_cos, "archs4_random_pairs", True))
        pd.concat(range_frames, ignore_index=True).to_csv(results / "cosine_distributions.csv", index=False)
        lowest = np.argsort(pair_cos)[:20]
        low = pd.DataFrame({"sample_index_a": a[lowest], "sample_index_b": b[lowest], "cosine": pair_cos[lowest]})
        for side, inds in [("a", a[lowest]), ("b", b[lowest])]:
            rows = hmeta.iloc[inds]
            low[f"gsm_{side}"] = human.geo_accession.iloc[inds].to_numpy()
            low[f"gse_{side}"] = rows.series_id.to_numpy()
            low[f"source_{side}"] = rows.source_name_ch1.to_numpy()
            low[f"title_{side}"] = rows.title.to_numpy()
            low[f"characteristics_{side}"] = rows.characteristics_ch1.to_numpy()
        low.to_csv(results / "lowest_sampled_pairs.csv", index=False)
        np.save(work / "sampled_archs4_pair_cosines.npy", pair_cos)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(pair_cos, bins=100, density=True, color="#4C78A8", alpha=.85)
        markers = {"median random pair": np.median(pair_cos),
                   "median exact pair": np.median(upstream_full.paired_cosine),
                   "median query Top-1": np.median(cosine_top_vals[:, 0])}
        for label, value in markers.items():
            ax.axvline(value, linestyle="--", linewidth=1.5, label=f"{label}: {value:.4f}")
        ax.set(xlabel="Cosine similarity", ylabel="Density", title="Frozen Bridge human ARCHS4 geometry")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figures / "archs4_pairwise_cosine.png", dpi=200)
        fig.savefig(figures / "archs4_pairwise_cosine.pdf")
        plt.close(fig)

    with timed("reference_density_probe", timings, log):
        rng = np.random.default_rng(cfg["random_seed"] + 1)
        probe_idx = np.sort(rng.choice(len(human), cfg["reference_density_probe_samples"], replace=False))
        probe = unit_rows(np.asarray(raw[human.global_index.iloc[probe_idx].to_numpy()], dtype=np.float32))
        best = np.full((len(probe), 101), -np.inf, dtype=np.float32)
        tq = torch.from_numpy(probe).to(device)
        chunk = cfg["reference_chunk_size"]
        for start in range(0, len(human), chunk):
            stop = min(start + chunk, len(human))
            x = unit_rows(np.asarray(raw[human.global_index.iloc[start:stop].to_numpy()], dtype=np.float32))
            block = (tq @ torch.from_numpy(x).to(device).T).cpu().numpy()
            merged = np.concatenate([best, block], axis=1)
            best = np.partition(merged, -101, axis=1)[:, -101:]
            log(f"reference density {stop:,}/{len(human):,}")
        best.sort(axis=1); best = best[:, ::-1]
        # First hit is self; report non-self neighbors.
        drows = []
        for k in cfg["density_k"]:
            vals = best[:, 1:k + 1]
            for i in range(len(probe)):
                drows.append({"reference_index": probe_idx[i], "gsm": human.geo_accession.iloc[probe_idx[i]], "k": k,
                              "mean_cosine": vals[i].mean(), "median_cosine": np.median(vals[i]), "kth_cosine": vals[i, -1]})
        pd.DataFrame(drows).to_csv(results / "reference_density_probe.csv", index=False)

    metrics, details, top_by_method = [], [], {}
    t0 = time.perf_counter()
    metric, detail, ti, tv = ranked_metrics(scores, correct, ref_gse, query_gse, ref_source, query_source,
                                             "Cosine", time.perf_counter() - t0,
                                             upstream_full.paired_cosine.to_numpy())
    cosine_delta = np.abs(scores[np.arange(len(samples)), correct] - upstream_full.paired_cosine.to_numpy())
    assert cosine_delta.max() < 5e-6
    assert np.array_equal(detail.exact_rank.to_numpy(), upstream_full.exact_rank_full.to_numpy())
    log("Cosine verification: all 562 exact ranks reproduce the upstream source of truth")
    # The published upstream ranks are the requested baseline source of truth.
    upstream_ranks = upstream_full.exact_rank_full.to_numpy(dtype=np.int64)
    detail["exact_rank"] = upstream_ranks
    metric.update({"exact_r1": np.mean(upstream_ranks <= 1), "exact_r5": np.mean(upstream_ranks <= 5),
                   "exact_r10": np.mean(upstream_ranks <= 10), "exact_r100": np.mean(upstream_ranks <= 100),
                   "mrr": np.mean(1 / upstream_ranks), "median_rank": np.median(upstream_ranks),
                   "median_percentile_rank": np.median(1 - (upstream_ranks - 1) / len(human)),
                   "same_gse_r1": upstream_full.top1_same_gse_full.mean(),
                   "same_gse_r5": upstream_full.top5_same_gse_full.mean(),
                   "baseline_provenance": str(up / "results/full_archs4_retrieval.parquet")})
    upstream_top10 = (upstream_neighbors.sort_values(["query_gsm", "rank"])
                      .pivot(index="query_gsm", columns="rank", values="neighbor_gsm")
                      .reindex(samples.gsm).to_numpy())
    upstream_top10_idx = np.vectorize(correct_lookup.__getitem__)(upstream_top10).astype(np.int64)
    cosine_retrieval_idx = np.empty_like(ti)
    for i in range(len(samples)):
        tail = ti[i, ~np.isin(ti[i], upstream_top10_idx[i])][:90]
        cosine_retrieval_idx[i] = np.r_[upstream_top10_idx[i], tail]
    upstream_top1 = (upstream_neighbors[upstream_neighbors["rank"].eq(1)]
                     .set_index("query_gsm").reindex(samples.gsm))
    detail["top1_reference_index"] = upstream_top10_idx[:, 0]
    detail["top1_score"] = upstream_top1.cosine_similarity.to_numpy()
    detail["top1_same_gse"] = upstream_full.top1_same_gse_full.to_numpy()
    detail["top5_same_gse"] = upstream_full.top5_same_gse_full.to_numpy()
    metrics.append(metric); details.append(detail); top_by_method["Cosine"] = cosine_retrieval_idx

    # Cross-domain neighborhood statistics shared by CSLS, local scaling, NICDM, reciprocal kNN.
    primary_k = cfg["primary_k"]
    query_top100_idx, query_top100_vals = cosine_retrieval_idx, tv
    ref_query_top_by_k = {}
    for reciprocal_k in cfg["reciprocal_k"]:
        ref_top = np.argpartition(scores, -reciprocal_k, axis=0)[-reciprocal_k:, :]
        for i, ref_i in enumerate(correct):
            column = scores[:, ref_i].astype(np.float64)
            column[i] = upstream_full.paired_cosine.iloc[i]
            ref_top[:, ref_i] = np.argsort(column, kind="stable")[-reciprocal_k:]
        ref_query_top_by_k[reciprocal_k] = ref_top
    ref_topk_vals = np.partition(scores, -primary_k, axis=0)[-primary_k:, :]
    rq = query_top100_vals[:, :primary_k].mean(axis=1)
    rr = ref_topk_vals.mean(axis=0)
    eps = np.finfo(np.float32).eps

    for k in cfg["sensitivity_k"]:
        t0 = time.perf_counter()
        qmean = query_top100_vals[:, :k].mean(axis=1)
        rmean = np.partition(scores, -k, axis=0)[-k:, :].mean(axis=0)
        csls = 2 * scores - qmean[:, None] - rmean[None, :]
        name = f"CSLS(k={k})"
        metric, detail, ti, tv = ranked_metrics(csls, correct, ref_gse, query_gse, ref_source, query_source,
                                                name, time.perf_counter() - t0,
                                                2 * upstream_full.paired_cosine.to_numpy() - qmean - rmean[correct])
        metrics.append(metric); details.append(detail); top_by_method[name] = ti
        del csls

    with timed("local_scaling", timings, log):
        t0 = time.perf_counter()
        distance = np.maximum(1 - scores, eps)
        sigma_q = np.maximum(1 - query_top100_vals[:, primary_k - 1], eps)
        sigma_r = np.maximum(1 - np.partition(scores, -primary_k, axis=0)[-primary_k, :], eps)
        # Zelnik-Manor & Perona self-tuning affinity, using cosine distance.
        local_score = -(distance * distance) / (sigma_q[:, None] * sigma_r[None, :])
        metric, detail, ti, tv = ranked_metrics(local_score, correct, ref_gse, query_gse, ref_source, query_source,
                                                "LocalScaling(k=10)", time.perf_counter() - t0,
                                                -((1-upstream_full.paired_cosine.to_numpy())**2) /
                                                (sigma_q * sigma_r[correct]))
        metrics.append(metric); details.append(detail); top_by_method["LocalScaling(k=10)"] = ti
        del local_score

    with timed("nicdm", timings, log):
        t0 = time.perf_counter()
        mean_dq = np.maximum((1 - query_top100_vals[:, :primary_k]).mean(axis=1), eps)
        mean_dr = np.maximum((1 - ref_topk_vals).mean(axis=0), eps)
        nicdm_score = -distance / np.sqrt(mean_dq[:, None] * mean_dr[None, :])
        metric, detail, ti, tv = ranked_metrics(nicdm_score, correct, ref_gse, query_gse, ref_source, query_source,
                                                "NICDM(k=10)", time.perf_counter() - t0,
                                                -(1-upstream_full.paired_cosine.to_numpy()) /
                                                np.sqrt(mean_dq * mean_dr[correct]))
        metrics.append(metric); details.append(detail); top_by_method["NICDM(k=10)"] = ti
        del nicdm_score, distance

    # Empirical global null is a monotone transform and must preserve ranks.
    with timed("empirical_null", timings, log):
        empirical_start = time.perf_counter()
        null = np.sort(pair_cos[:cfg["empirical_null_observations"]])
        exact_scores = upstream_full.paired_cosine.to_numpy()
        exceed = len(null) - np.searchsorted(null, exact_scores, side="left")
        empirical_p = (exceed + 1) / (len(null) + 1)
        bh_q = empirical_p * len(human)
        calibration = pd.DataFrame({"query_index": np.arange(len(samples)), "gsm": samples.gsm,
                                    "density_k10": query_top100_vals[:, :10].mean(axis=1),
                                    "exact_cosine": exact_scores, "empirical_p": empirical_p,
                                    "bonferroni_p": np.minimum(1, bh_q)})
        calibration["density_stratum"] = pd.qcut(calibration.density_k10, 3, labels=["sparse", "intermediate", "dense"])
        calibration.to_csv(results / "empirical_null_calibration.csv", index=False)
        base = dict(metrics[0]); base["method"] = "Empirical-null cosine (global)"
        base["runtime_seconds"] = time.perf_counter() - empirical_start
        metrics.append(base)
        ed = details[0].copy(); ed["method"] = base["method"]; details.append(ed)
        top_by_method[base["method"]] = top_by_method["Cosine"]

    # Mutual kNN: exact query/reference bipartite reciprocity, evaluated as sets.
    set_rows = []
    for k in cfg["reciprocal_k"]:
        qtop = query_top100_idx[:, :k]
        ref_query_order = ref_query_top_by_k[k]
        exact_hits = same_gse_hits = returned = 0
        per_query = []
        for qi in range(len(samples)):
            reciprocal = [r for r in qtop[qi] if np.any(ref_query_order[:, r] == qi)]
            hit = correct[qi] in reciprocal
            gse_hit = any(ref_gse[r] == query_gse[qi] for r in reciprocal)
            exact_hits += hit; same_gse_hits += gse_hit; returned += len(reciprocal)
            per_query.append({"method": f"Mutual-kNN(k={k})", "query_index": qi, "returned": len(reciprocal),
                              "exact_in_set": hit, "same_gse_in_set": gse_hit})
        set_rows.append({"method": f"Mutual-kNN(k={k})", "ranking_defined": False, "k": k,
                         "exact_set_recall": exact_hits / len(samples), "same_gse_set_recall": same_gse_hits / len(samples),
                         "mean_returned": returned / len(samples), "empty_fraction": np.mean([x["returned"] == 0 for x in per_query])})
        details.append(pd.DataFrame(per_query))

    # Cross-domain SNN: Jaccard between each query's k nearest query contexts
    # (including itself) and each reference's k nearest queries. This is a
    # neighborhood-set score with many ties, so report maximum-score sets only.
    with timed("snn", timings, log):
        k = cfg["snn_k"]
        qq = queries @ queries.T
        q_context = np.argpartition(qq, -k, axis=1)[:, -k:]
        r_context = np.argpartition(scores, -k, axis=0)[-k:, :].T
        snn_rows = []
        for qi in range(len(samples)):
            context_mask = np.zeros(len(samples), dtype=bool)
            context_mask[q_context[qi]] = True
            overlap = context_mask[r_context].sum(axis=1)
            jac = overlap / (2 * k - overlap)
            best_score = jac.max(); best_refs = np.flatnonzero(jac == best_score)
            snn_rows.append({"method": f"SNN-Jaccard(k={k})", "query_index": qi,
                             "max_jaccard": best_score, "returned": len(best_refs),
                             "exact_in_set": correct[qi] in best_refs,
                             "same_gse_in_set": np.any(ref_gse[best_refs] == query_gse[qi])})
        sd = pd.DataFrame(snn_rows); details.append(sd)
        set_rows.append({"method": f"SNN-Jaccard(k={k})", "ranking_defined": False, "k": k,
                         "exact_set_recall": sd.exact_in_set.mean(), "same_gse_set_recall": sd.same_gse_in_set.mean(),
                         "mean_returned": sd.returned.mean(), "empty_fraction": 0.0})
    pd.DataFrame(set_rows).to_csv(results / "set_retrieval_metrics.csv", index=False)

    with timed("summaries_and_diagnostics", timings, log):
        metric_df = pd.DataFrame(metrics)
        metric_df.to_csv(results / "ranked_method_comparison.csv", index=False)
        rank_detail = pd.concat([d for d in details if "exact_rank" in d.columns], ignore_index=True)
        rank_detail = rank_detail.merge(samples[["gsm", "gse"]], left_on="query_index", right_index=True, validate="many_to_one")
        density = query_top100_vals[:, :10].mean(axis=1)
        strata = pd.qcut(density, 3, labels=["sparse", "intermediate", "dense"])
        rank_detail["density_k10"] = density[rank_detail.query_index]
        rank_detail["density_stratum"] = np.asarray(strata)[rank_detail.query_index]
        rank_detail.to_parquet(results / "per_query_ranked_results.parquet", index=False)
        ds = (rank_detail.groupby(["method", "density_stratum"], observed=True)
              .agg(queries=("exact_rank", "size"), exact_r1=("exact_rank", lambda x: np.mean(x <= 1)),
                   exact_r5=("exact_rank", lambda x: np.mean(x <= 5)), exact_r10=("exact_rank", lambda x: np.mean(x <= 10)),
                   exact_r100=("exact_rank", lambda x: np.mean(x <= 100)), mrr=("exact_rank", lambda x: np.mean(1 / x)),
                   median_rank=("exact_rank", "median"), same_gse_r1=("top1_same_gse", "mean"))
              .reset_index())
        ds.to_csv(results / "density_stratified_metrics.csv", index=False)

        hub_summaries, hub_counts = [], []
        for method, ti in top_by_method.items():
            hs, hc = concentration_rows(ti, method, len(human))
            hub_summaries.append(hs); hub_counts.append(hc)
        pd.concat(hub_summaries, ignore_index=True).to_csv(results / "hubness_summary.csv", index=False)
        cosine_counts = np.bincount(cosine_top_idx[:, :100].ravel(), minlength=len(human))
        hub_order = np.argsort(cosine_counts)[::-1][:50]
        hubs = pd.DataFrame({"reference_index": hub_order, "n_100": cosine_counts[hub_order],
                             "gsm": human.geo_accession.iloc[hub_order].to_numpy(),
                             "gse": ref_gse[hub_order], "source": hmeta.source_name_ch1.iloc[hub_order].to_numpy(),
                             "title": hmeta.title.iloc[hub_order].to_numpy(),
                             "characteristics": hmeta.characteristics_ch1.iloc[hub_order].to_numpy()})
        for k in [1, 5, 10]:
            c = np.bincount(cosine_top_idx[:, :k].ravel(), minlength=len(human))
            hubs[f"n_{k}"] = c[hub_order]
        hubs.to_csv(results / "top_cosine_hubs.csv", index=False)

        cosine_detail = rank_detail[rank_detail.method.eq("Cosine")].set_index("query_index")
        diagnostics = []
        for method in metric_df.method:
            if method == "Cosine": continue
            alt = rank_detail[rank_detail.method.eq(method)].set_index("query_index")
            if len(alt) != len(samples): continue
            change = cosine_detail.exact_rank - alt.exact_rank
            cats = pd.Series("essentially_unchanged", index=change.index)
            cats[change > 0] = "improved"; cats[change < 0] = "degraded"
            selected = pd.unique(np.r_[change.nlargest(10).index, change.nsmallest(10).index,
                                       change[change == 0].index[:10], change[strata == "dense"].nsmallest(10).index])
            for qi in selected:
                oldr = int(cosine_detail.loc[qi].exact_rank); newr = int(alt.loc[qi].exact_rank)
                old_top = int(cosine_detail.loc[qi].top1_reference_index); new_top = int(alt.loc[qi].top1_reference_index)
                diagnostics.append({"method": method, "criterion": cats.loc[qi], "rank_change_cosine_minus_alternative": oldr-newr,
                    "query_gsm": samples.gsm.iloc[qi], "query_gse": query_gse[qi], "correct_archs4_gsm": samples.gsm.iloc[qi],
                    "cosine_rank": oldr, "alternative_rank": newr, "paired_cosine": upstream_full.paired_cosine.iloc[qi],
                    "local_density_k10": density[qi], "density_stratum": str(strata[qi]),
                    "original_top1_gsm": human.geo_accession.iloc[old_top], "corrected_top1_gsm": human.geo_accession.iloc[new_top],
                    "original_top1_same_gse": ref_gse[old_top] == query_gse[qi],
                    "corrected_top1_same_gse": ref_gse[new_top] == query_gse[qi],
                    "query_source": qmeta.source_name_ch1.iloc[qi], "original_top1_source": hmeta.source_name_ch1.iloc[old_top],
                    "corrected_top1_source": hmeta.source_name_ch1.iloc[new_top],
                    "original_top1_hub_n100": cosine_counts[old_top], "corrected_top1_hub_n100": cosine_counts[new_top]})
        pd.DataFrame(diagnostics).to_csv(results / "rank_change_diagnostics.csv", index=False)

        qdensity = []
        for k in cfg["density_k"]:
            vals = query_top100_vals[:, :k]
            qdensity.append(pd.DataFrame({"query_gsm": samples.gsm, "query_gse": query_gse, "k": k,
                                          "mean_cosine": vals.mean(axis=1), "median_cosine": np.median(vals, axis=1),
                                          "kth_cosine": vals[:, -1], "cosine_exact_rank": cosine_detail.exact_rank.to_numpy()}))
        pd.concat(qdensity, ignore_index=True).to_csv(results / "query_local_density.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot = pd.concat(qdensity); plot = plot[plot.k.eq(10)]
        ax.scatter(plot.mean_cosine, np.log10(plot.cosine_exact_rank), alpha=.6, s=18)
        ax.set(xlabel="Mean cosine to 10 nearest references", ylabel="log10 exact-pair rank",
               title="Exact retrieval degrades in dense neighborhoods")
        fig.tight_layout(); fig.savefig(figures / "density_vs_cosine_rank.png", dpi=200); fig.savefig(figures / "density_vs_cosine_rank.pdf"); plt.close(fig)

    runtime_total = time.time() - started
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    runtime_rows = [{"stage": k, "runtime_seconds": v} for k, v in timings.items()]
    runtime_rows.append({"stage": "total", "runtime_seconds": runtime_total})
    pd.DataFrame(runtime_rows).to_csv(results / "runtime.csv", index=False)
    provenance = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "benchmark": "sample_retrieval_geometry",
        "stages": [0, 1], "future_representation_learning_run": False, "random_seed": cfg["random_seed"],
        "queries": len(samples), "references": len(human), "embedding_dim": queries.shape[1],
        "canonical_gene_count": canonical_gene_count, "device": str(device), "torch_version": torch.__version__,
        "configuration": cfg, "resolved_inputs": {k: str(v.resolve()) for k, v in inputs.items()},
        "upstream_imported_results": [str((up / "results/benchmark_summary.csv").resolve()), str((up / "results/full_archs4_retrieval.parquet").resolve())],
        "input_sha256": {str(p.resolve()): sha256(p) for p in required if p.stat().st_size < 100_000_000},
        "score_cache": str(score_path.resolve()), "score_cache_bytes": score_path.stat().st_size,
        "peak_rss_gb": peak_gb, "runtime_seconds": runtime_total,
        "limitations": ["ARCHS4 pairwise distribution and minimum are deterministic sampled estimates, not the exact global minimum.",
                        "Reference local-density distribution uses a deterministic probe subset.",
                        "Source agreement is exact normalized source_name_ch1 agreement, not inferred tissue/disease/treatment.",
                        "Global empirical-null calibration is monotone in cosine and therefore cannot change ranks."]
    }
    (results / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"COMPLETE total={runtime_total:.1f}s peak_RSS={peak_gb:.2f}GB")
    log.close()


if __name__ == "__main__":
    main()
