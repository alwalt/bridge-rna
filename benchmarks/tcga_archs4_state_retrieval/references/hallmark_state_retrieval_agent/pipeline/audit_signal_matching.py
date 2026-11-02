#!/usr/bin/env python3
"""Frozen-output audit of dominant-signal versus whole-vector Hallmark retrieval."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, WORK, l2_rows, save_json, sha256
from run_tcga33_retrieval import RULES, annotate, load_queries


OUT = RESULTS / "signal_matching_audit"
KS = (5, 10, 50)
TOP = 50
SEED = 20260924
DEEP_CANCERS = ("UVM", "CHOL", "LUAD", "PCPG", "OV", "THCA", "KICH")
METHODS = (
    "whole_vector_cosine",
    "signed_absolute_rank",
    "signed_top5_overlap",
    "signed_top10_overlap",
    "query_weighted_top10",
    "asymmetric_containment_top10",
)


def log(message: str) -> None:
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    print(f"[{stamp}] {message}", flush=True)
    with (OUT / "run.log").open("a") as stream:
        stream.write(f"[{stamp}] {message}\n")


def absolute_rank_percentile(x: np.ndarray) -> np.ndarray:
    """Within-row absolute ranks in [1/d, 1], where 1 is strongest."""
    order = np.argsort(np.abs(x), axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[np.arange(len(x))[:, None], order] = np.arange(1, x.shape[1] + 1, dtype=np.float32)
    return ranks / x.shape[1]


def strongest_mask(x: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(np.abs(x), -k, axis=1)[:, -k:]
    mask = np.zeros(x.shape, dtype=bool)
    mask[np.arange(len(x))[:, None], idx] = True
    return mask


def score_vector(method: str, q: np.ndarray, ref: np.ndarray, ref_rank: np.ndarray,
                 ref_mask5: np.ndarray, ref_mask10: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float32)
    qrank = absolute_rank_percentile(q[None])[0]
    qsign = np.sign(q)
    rsign = np.sign(ref)
    if method == "whole_vector_cosine":
        return l2_rows(ref) @ l2_rows(q[None])[0]
    if method == "signed_absolute_rank":
        qr = qsign * qrank
        rr = rsign * ref_rank
        return l2_rows(rr) @ (qr / np.linalg.norm(qr))
    if method.startswith("signed_top"):
        k = 5 if "top5" in method else 10
        qm = strongest_mask(q[None], k)[0]
        rm = ref_mask5 if k == 5 else ref_mask10
        return ((rm & qm[None] & (rsign == qsign[None])).sum(1) / k).astype(np.float32)
    if method == "query_weighted_top10":
        qm = strongest_mask(q[None], 10)[0]
        weights = np.abs(q[qm]); weights /= weights.sum()
        # Query-weighted signed magnitude agreement on query-selected coordinates only.
        qscale = np.maximum(np.abs(q[qm]), 1e-8)
        ratio = np.clip(np.abs(ref[:, qm]) / qscale[None], 0, 1)
        agreement = np.where(rsign[:, qm] == qsign[qm], ratio, -ratio)
        return (agreement * weights[None]).sum(1).astype(np.float32)
    if method == "asymmetric_containment_top10":
        qm = strongest_mask(q[None], 10)[0]
        # Rank containment ignores all non-query coordinates. Matching strong reference
        # signals receive full credit; mismatches receive only a 0.25 penalty.
        strength = np.minimum(ref_rank[:, qm] / qrank[qm][None], 1.0)
        signed = np.where(rsign[:, qm] == qsign[qm], strength, -0.25 * strength)
        return signed.mean(1).astype(np.float32)
    raise KeyError(method)


def top_indices(scores: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    # Cosine is a deterministic secondary key for discrete overlap scores.
    candidates = np.argpartition(scores, -TOP)[-TOP:]
    return candidates[np.lexsort((-cosine[candidates], -scores[candidates]))]


def signal_coverage(q: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    qm = strongest_mask(q[None], k)[0]
    rm = strongest_mask(rows, k)
    return (rm[:, qm] & (np.sign(rows[:, qm]) == np.sign(q[qm])[None])).sum(1) / k


def named_signals(q: np.ndarray, ref: np.ndarray, names: list[str], k: int = 10) -> tuple[str, str]:
    qi = np.argsort(np.abs(q))[::-1][:k]
    dominant = "; ".join(f"{names[j]}={'+' if q[j] >= 0 else '-'}{abs(q[j]):.3f}" for j in qi)
    ri = set(np.argsort(np.abs(ref))[::-1][:k])
    matched = [j for j in qi if j in ri and np.sign(q[j]) == np.sign(ref[j])]
    return dominant, "; ".join(f"{names[j]}={'+' if q[j] >= 0 else '-'}" for j in matched)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "run.log").write_text("")
    started = time.time()
    probe = np.load(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz", allow_pickle=False)
    names = probe["hallmarks"].astype(str).tolist()
    qmeta, _, hallmarks, _ = load_queries(probe, names)
    cancers = sorted(RULES)
    queries = np.vstack([hallmarks[qmeta.cancer_type.eq(c)].mean(0) for c in cancers]).astype(np.float32)
    ref = np.load(WORK / "archs4_hallmark_signatures.float32.npy", mmap_mode="r")
    assert ref.shape == (510_709, 50) and queries.shape == (33, 50)
    ref = np.asarray(ref, np.float32)
    meta = pd.read_parquet(RESULTS / "archs4/sample_metadata.parquet")
    _, exact, tissue, any_exact, _, malignant, _ = annotate(meta)
    ref_rank = absolute_rank_percentile(ref)
    ref_mask5, ref_mask10 = strongest_mask(ref, 5), strongest_mask(ref, 10)
    ref_cos = l2_rows(ref)
    bridge_cache = np.load(WORK / "tcga33_top100_indices_scores.npz")
    bridge_idx = bridge_cache["bridge_indices"][-33:, :TOP]
    bridge_scores = bridge_cache["bridge_scores"][-33:, :TOP]
    log(f"startup: 33 centroids x {len(METHODS)} methods against {len(ref):,} rows; expected <2 minutes")

    rankings: dict[tuple[str, str], np.ndarray] = {}
    scores_by_key: dict[tuple[str, str], np.ndarray] = {}
    null_rows, hit_rows = [], []
    for qi, cancer in enumerate(cancers):
        cosine = ref_cos @ l2_rows(queries[qi:qi + 1])[0]
        for method in METHODS:
            scores = score_vector(method, queries[qi], ref, ref_rank, ref_mask5, ref_mask10)
            ix = top_indices(scores, cosine)
            rankings[(cancer, method)] = ix
            scores_by_key[(cancer, method)] = scores[ix]
            # The background is the complete ARCHS4 database, not a sampled null.
            null = scores
            for quantile in (0.5, 0.9, 0.95, 0.99, 0.999):
                null_rows.append({"cancer_type": cancer, "method": method, "quantile": quantile, "null_score": float(np.quantile(null, quantile))})
            null_rows.append({"cancer_type": cancer, "method": method, "quantile": 1.0, "null_score": float(null.max())})
            cov5, cov10 = signal_coverage(queries[qi], ref[ix], 5), signal_coverage(queries[qi], ref[ix], 10)
            for rank, (ri, score, c5, c10) in enumerate(zip(ix, scores[ix], cov5, cov10), 1):
                record = meta.iloc[int(ri)]
                related = bool(tissue[cancer][ri] and any_exact[ri] and not exact[cancer][ri])
                hit_rows.append({"cancer_type": cancer, "method": method, "rank": rank, "reference_row": int(ri),
                                 "gsm": record.sample_id, "gse": record.series_id, "score": float(score),
                                 "null_percentile": float((null <= score).mean()), "signal_coverage_top5": float(c5),
                                 "signal_coverage_top10": float(c10), "exact_disease": bool(exact[cancer][ri]),
                                 "tissue_organ": bool(tissue[cancer][ri]), "related_cancer": related,
                                 "malignancy": bool(malignant[ri]), "title": record.title,
                                 "source_name": record.source_name_ch1, "characteristics": record.characteristics_ch1})
        if (qi + 1) % 5 == 0 or qi == 32:
            log(f"scored {qi + 1}/33 centroids; elapsed={time.time() - started:.1f}s")

    hits = pd.DataFrame(hit_rows)
    pd.DataFrame(null_rows).to_csv(OUT / "null_score_distributions.csv", index=False)
    hits.to_parquet(OUT / "all_top50_hits.parquet", index=False)

    # Add the global Bridge reference using the same metadata and signal-coverage audit.
    bridge_rows = []
    for qi, cancer in enumerate(cancers):
        ix = bridge_idx[qi]
        c5, c10 = signal_coverage(queries[qi], ref[ix], 5), signal_coverage(queries[qi], ref[ix], 10)
        for rank, (ri, score, v5, v10) in enumerate(zip(ix, bridge_scores[qi], c5, c10), 1):
            r = meta.iloc[int(ri)]
            bridge_rows.append({"cancer_type": cancer, "method": "Bridge-512 reference", "rank": rank,
                                "reference_row": int(ri), "gsm": r.sample_id, "gse": r.series_id, "score": float(score),
                                "null_percentile": np.nan, "signal_coverage_top5": float(v5), "signal_coverage_top10": float(v10),
                                "exact_disease": bool(exact[cancer][ri]), "tissue_organ": bool(tissue[cancer][ri]),
                                "related_cancer": bool(tissue[cancer][ri] and any_exact[ri] and not exact[cancer][ri]),
                                "malignancy": bool(malignant[ri]), "title": r.title, "source_name": r.source_name_ch1,
                                "characteristics": r.characteristics_ch1})
    all_hits = pd.concat([hits, pd.DataFrame(bridge_rows)], ignore_index=True)
    all_hits.to_csv(OUT / "all_methods_top50_hits.csv", index=False)

    metric_rows = []
    for (cancer, method), group in all_hits.groupby(["cancer_type", "method"], sort=False):
        bridge_set = set(bridge_idx[cancers.index(cancer)])
        for k in KS:
            g = group.nsmallest(k, "rank")
            counts = g.gse.fillna("MISSING").value_counts()
            metric_rows.append({"cancer_type": cancer, "method": method, "k": k,
                                "exact_disease_precision": g.exact_disease.mean(), "tissue_precision": g.tissue_organ.mean(),
                                "related_cancer_precision": g.related_cancer.mean(), "malignancy_precision": g.malignancy.mean(),
                                "signal_coverage_top5": g.signal_coverage_top5.mean(), "signal_coverage_top10": g.signal_coverage_top10.mean(),
                                "unique_studies": int(g.gse.nunique(dropna=True)), "largest_study_fraction": counts.iloc[0] / k,
                                "bridge_overlap_fraction": len(set(g.reference_row) & bridge_set) / k})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "per_cancer_method_metrics.csv", index=False)
    macro = metrics.groupby(["method", "k"], as_index=False).agg(
        exact_disease_precision=("exact_disease_precision", "mean"), tissue_precision=("tissue_precision", "mean"),
        related_cancer_precision=("related_cancer_precision", "mean"), malignancy_precision=("malignancy_precision", "mean"),
        signal_coverage_top5=("signal_coverage_top5", "mean"), signal_coverage_top10=("signal_coverage_top10", "mean"),
        unique_studies=("unique_studies", "mean"), largest_study_fraction=("largest_study_fraction", "mean"),
        bridge_overlap_fraction=("bridge_overlap_fraction", "mean"))
    macro.to_csv(OUT / "macro_method_comparison.csv", index=False)

    # Hub audit across top-10 lists: repeated samples/studies can otherwise mimic success.
    hub_rows = []
    for method, group in all_hits[all_hits["rank"].le(10)].groupby("method"):
        for field in ("gsm", "gse"):
            for value, count in Counter(group[field].fillna("MISSING")).most_common(20):
                hub_rows.append({"method": method, "entity_type": field, "entity": value, "top10_occurrences": count,
                                 "fraction_of_330_slots": count / 330})
    pd.DataFrame(hub_rows).to_csv(OUT / "hub_frequency.csv", index=False)

    # Deep-inspection table with explicit signed programs and conservative automated interpretation.
    deep = all_hits[all_hits.cancer_type.isin(DEEP_CANCERS) & all_hits["rank"].le(5)].copy()
    dominant, matched, labels = [], [], []
    for row in deep.itertuples():
        qi = cancers.index(row.cancer_type)
        d, m = named_signals(queries[qi], ref[row.reference_row], names)
        dominant.append(d); matched.append(m)
        if row.exact_disease: label = "same identity"
        elif row.related_cancer: label = "related same-tissue cancer"
        elif row.tissue_organ: label = "same tissue/context"
        elif row.signal_coverage_top10 >= 0.5: label = "candidate cross-context functional analog"
        else: label = "unresolved/weak functional match"
        labels.append(label)
    deep["query_dominant_signed_hallmarks"] = dominant
    deep["matched_top10_signed_hallmarks"] = matched
    deep["interpretation"] = labels
    deep.to_csv(OUT / "deep_inspection_top5.csv", index=False)

    # Ubiquity audit: which signed Hallmarks dominate reference samples and retrieved slots.
    ref_top1 = np.argmax(np.abs(ref), axis=1)
    ubiq = pd.DataFrame({"hallmark": names, "archs4_top1_fraction": np.bincount(ref_top1, minlength=50) / len(ref)})
    for method in METHODS:
        idx = np.concatenate([rankings[(c, method)][:10] for c in cancers])
        ubiq[method + "_top10_fraction"] = np.bincount(ref_top1[idx], minlength=50) / len(idx)
    ubiq.to_csv(OUT / "hallmark_ubiquity_audit.csv", index=False)

    # Publication figures: metric comparison and per-cancer signal-coverage heatmap.
    p10 = macro[macro.k.eq(10)].set_index("method").loc[list(METHODS) + ["Bridge-512 reference"]]
    cols = ["exact_disease_precision", "tissue_precision", "signal_coverage_top5", "signal_coverage_top10"]
    ax = p10[cols].plot.bar(figsize=(12, 6), color=["#264653", "#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set(ylabel="Macro mean at top 10", xlabel="", ylim=(0, 1), title="TCGA centroid retrieval: metadata and dominant-signal coverage")
    ax.legend(["Exact disease", "Tissue", "Signal coverage@5", "Signal coverage@10"], frameon=False, ncol=2)
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(OUT / "figures/method_comparison.png", dpi=240); plt.savefig(OUT / "figures/method_comparison.pdf"); plt.close()
    heat = metrics[(metrics.k.eq(10)) & metrics.method.isin(METHODS)].pivot(index="cancer_type", columns="method", values="signal_coverage_top10").reindex(index=cancers, columns=METHODS)
    fig, ax = plt.subplots(figsize=(12, 11)); im = ax.imshow(heat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(METHODS)), [m.replace("_", " ") for m in METHODS], rotation=28, ha="right"); ax.set_yticks(range(33), cancers)
    fig.colorbar(im, ax=ax, label="Mean top-10 signed-signal coverage"); ax.set_title("Dominant Hallmark signal reproduced by retrieved ARCHS4 samples")
    fig.tight_layout(); fig.savefig(OUT / "figures/per_cancer_signal_coverage.png", dpi=240); fig.savefig(OUT / "figures/per_cancer_signal_coverage.pdf"); plt.close(fig)

    save_json(OUT / "provenance.json", {"frozen_outputs_only": True, "bridge_inference": False, "head_retrained": False,
        "queries": 33, "reference_samples": len(ref), "neighbors": list(KS), "seed": SEED, "background_samples_per_query": len(ref),
        "methods": list(METHODS), "score_definitions": {
            "whole_vector_cosine": "cosine across all 50 signed Hallmark values",
            "signed_absolute_rank": "cosine of sign(value) times within-sample absolute-rank percentile",
            "signed_topK_overlap": "fraction of query top-K absolute Hallmarks also reference top-K with matching sign; cosine breaks ties",
            "query_weighted_top10": "absolute-query-weighted signed magnitude agreement on only query top 10; reference/query magnitude ratio capped at one",
            "asymmetric_containment_top10": "mean query-top10 signed rank containment; extras ignored and sign mismatches penalized by 0.25"},
        "inputs": {"hallmark_matrix": str(WORK / "archs4_hallmark_signatures.float32.npy"),
                   "probe_sha256": sha256(RESULTS / "frozen/frozen_gtex_hallmark_probe.npz"),
                   "bridge_rank_cache": str(WORK / "tcga33_top100_indices_scores.npz")},
        "runtime_seconds": time.time() - started,
        "limitations": ["Metadata labels are conservative regex annotations of incomplete free text.",
                        "Cross-context functional analog is an automated signal-coverage flag requiring biological review.",
                        "Background distributions use every ARCHS4 sample and therefore reflect ARCHS4 study composition rather than a study-balanced null."]})
    log(f"COMPLETE runtime={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
