#!/usr/bin/env python3
"""Absolute Bridge similarity versus retrieval uniqueness for the 562 pairs.

Uses only saved Stage 0/1 ranks, the existing query/corpus cosine memmap, and
the existing deterministic ARCHS4 random-pair sample. No embeddings or search
scores are regenerated.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "absolute_similarity"
FIG = OUT / "figures"
NREF = 510_709
NQUERY = 562
METHODS = ["Cosine", "LocalScaling(k=10)", "NICDM(k=10)", "CSLS(k=10)"]
RANK_LABELS = ["Rank 1", "Rank 2–5", "Rank 6–10", "Rank 11–100",
               "Rank 101–1,000", "Rank >1,000"]
RANK_BINS = [0, 1, 5, 10, 100, 1000, np.inf]
CUTOFFS = [1, 5, 10, 100, 1000]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_group(rank: pd.Series) -> pd.Categorical:
    return pd.cut(rank, bins=RANK_BINS, labels=RANK_LABELS, ordered=True)


def values_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {key: (0 if key == "n" else np.nan) for key in
                ["n", "mean", "median", "sd", "q1", "q3", "minimum", "maximum",
                 "p1", "p5", "p95", "p99"]}
    return {
        "n": len(values), "mean": values.mean(), "median": np.median(values),
        "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
        "q1": np.quantile(values, .25), "q3": np.quantile(values, .75),
        "minimum": values.min(), "maximum": values.max(),
        "p1": np.quantile(values, .01), "p5": np.quantile(values, .05),
        "p95": np.quantile(values, .95), "p99": np.quantile(values, .99),
    }


def retrieval_metrics(frame: pd.DataFrame) -> dict[str, float]:
    rank = frame.exact_rank.to_numpy(float)
    row = {f"r_at_{k}": np.mean(rank <= k) for k in CUTOFFS}
    row.update({"queries": len(rank), "mrr": np.mean(1 / rank),
                "median_rank": np.median(rank), "mean_rank": np.mean(rank),
                "median_percentile_rank": np.median(1 - (rank - 1) / NREF)})
    return row


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    log_handle = (OUT / "run.log").open("w", buffering=1)
    def log(message: str) -> None:
        line = f"[{pd.Timestamp.now().isoformat()}] {message}"
        print(line, flush=True); log_handle.write(line + "\n")
    log("Starting cached absolute-similarity extension; estimated runtime <1 minute")
    log(f"Monitor with: tail -f {OUT / 'run.log'}")

    ranked_path = HERE / "results/per_query_ranked_results.parquet"
    query_path = HERE / "results/biological_validation/query_composition_and_density.parquet"
    score_path = HERE / "work/query_corpus_cosine.float32.mmap"
    random_path = HERE / "work/sampled_archs4_pair_cosines.npy"
    ranked = pd.read_parquet(ranked_path)
    query = pd.read_parquet(query_path).reset_index(names="query_index")
    ranked = ranked[ranked.method.isin(METHODS)].copy()
    assert ranked.groupby("method").size().eq(NQUERY).all()
    cosine = (ranked[ranked.method.eq("Cosine")]
              .merge(query[["query_index", "source_name_ch1", "source_label"]],
                     on="query_index", validate="one_to_one")
              .sort_values("query_index").reset_index(drop=True))
    cosine["rank_group"] = rank_group(cosine.exact_rank)

    # 1: exact Bridge cosine by prespecified cosine-rank bins.
    by_rank = []
    for label in RANK_LABELS:
        vals = cosine.loc[cosine.rank_group.eq(label), "paired_score"].to_numpy()
        by_rank.append({"rank_group": label, **values_summary(vals)})
    by_rank.append({"rank_group": "All 562", **values_summary(cosine.paired_score.to_numpy())})
    pd.DataFrame(by_rank).to_csv(OUT / "exact_pair_cosine_by_rank.csv", index=False)

    # 2: study-level paired cosine and macro summaries.
    study = (cosine.groupby("gse", observed=True).paired_score
             .agg(queries="size", mean_paired_cosine="mean", median_paired_cosine="median",
                  sd_paired_cosine="std", minimum_paired_cosine="min", maximum_paired_cosine="max")
             .reset_index())
    study.to_csv(OUT / "study_paired_cosine.csv", index=False)
    sample_stats = values_summary(cosine.paired_score.to_numpy())
    macro_summary = pd.DataFrame([{
        "samples": NQUERY, "studies": len(study),
        "sample_level_mean_cosine": sample_stats["mean"],
        "sample_level_median_cosine": sample_stats["median"],
        "study_macro_mean_cosine": study.mean_paired_cosine.mean(),
        "study_macro_median_cosine": study.median_paired_cosine.mean(),
        "median_across_study_means": study.mean_paired_cosine.median(),
        "median_across_study_medians": study.median_paired_cosine.median(),
        "sd_across_study_means": study.mean_paired_cosine.std(ddof=1),
        "minimum_study_mean": study.mean_paired_cosine.min(),
        "maximum_study_mean": study.mean_paired_cosine.max(),
    }])
    macro_summary.to_csv(OUT / "paired_cosine_micro_macro_summary.csv", index=False)

    # 3/9: expanded micro and study-macro retrieval metrics.
    micro_rows, per_study_rows = [], []
    for method, sub in ranked.groupby("method", sort=False):
        micro_rows.append({"method": method, **retrieval_metrics(sub)})
        for gse, group in sub.groupby("gse", observed=True):
            per_study_rows.append({"method": method, "gse": gse, **retrieval_metrics(group)})
    micro = pd.DataFrame(micro_rows)
    per_study = pd.DataFrame(per_study_rows)
    macro_rows = []
    for method, sub in per_study.groupby("method", sort=False):
        row = {"method": method, "studies": len(sub), "queries": int(sub.queries.sum())}
        for metric in [f"r_at_{k}" for k in CUTOFFS] + ["mrr", "median_rank", "mean_rank",
                                                             "median_percentile_rank"]:
            row[f"macro_{metric}"] = sub[metric].mean()
            row[f"study_sd_{metric}"] = sub[metric].std(ddof=1)
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)
    micro.to_csv(OUT / "expanded_ranked_metrics_micro.csv", index=False)
    per_study.to_csv(OUT / "expanded_ranked_metrics_per_study.csv", index=False)
    macro.to_csv(OUT / "expanded_ranked_metrics_study_macro.csv", index=False)
    pd.DataFrame([{"cutoff": k, "references": NREF, "search_fraction": k / NREF,
                   "search_percent": 100 * k / NREF, "display": f"{k:,} / {NREF:,}"}
                  for k in CUTOFFS]).to_csv(OUT / "retrieval_cutoff_search_fractions.csv", index=False)

    main_rows = []
    for metric, label in [("r_at_1", "R@1"), ("r_at_5", "R@5"), ("r_at_10", "R@10"),
                          ("r_at_100", "R@100"), ("r_at_1000", "R@1,000"),
                          ("mrr", "MRR"), ("median_rank", "Median rank"),
                          ("mean_rank", "Mean rank")]:
        row = {"metric": label}
        for _, item in micro.iterrows(): row[item.method] = item[metric]
        main_rows.append(row)
    for metric, label in [("macro_r_at_10", "Study-macro R@10"),
                          ("macro_r_at_100", "Study-macro R@100"),
                          ("macro_r_at_1000", "Study-macro R@1,000")]:
        row = {"metric": label}
        for _, item in macro.iterrows(): row[item.method] = item[metric]
        main_rows.append(row)
    pd.DataFrame(main_rows).to_csv(OUT / "main_retrieval_summary.csv", index=False)

    # 5: empirical placement against the existing deterministic random-pair sample.
    random_pair = np.load(random_path).astype(np.float64, copy=False)
    exact_pair = cosine.paired_score.to_numpy(float)
    dist = pd.DataFrame([
        {"distribution": "Random ARCHS4 pairs", **values_summary(random_pair)},
        {"distribution": "Exact ARCHS4-recount3 pairs", **values_summary(exact_pair)},
    ])
    dist.to_csv(OUT / "random_vs_exact_cosine.csv", index=False)
    sorted_random = np.sort(random_pair)
    thresholds = {"exact_minimum": exact_pair.min(), "exact_q1": np.quantile(exact_pair, .25),
                  "exact_median": np.median(exact_pair)}
    empirical = []
    for name, threshold in thresholds.items():
        count = np.searchsorted(sorted_random, threshold, side="right")
        empirical.append({"exact_statistic": name, "cosine": threshold,
                          "random_pairs_at_or_below": int(count),
                          "random_pair_empirical_percentile": count / len(sorted_random),
                          "random_pairs_exceeding_fraction": (len(sorted_random) - count) / len(sorted_random)})
    pd.DataFrame(empirical).to_csv(OUT / "exact_cosine_random_empirical_percentiles.csv", index=False)

    # 6: top-1 margin, top-10 context, and exact-threshold neighborhood counts.
    score = np.memmap(score_path, dtype=np.float32, mode="r", shape=(NQUERY, NREF))
    context_rows = []
    for i, row in enumerate(score):
        top10 = np.partition(row, -10)[-10:].astype(np.float64)
        exact = float(exact_pair[i])
        saved_top1 = float(cosine.loc[i, "top1_score"])
        top1 = max(saved_top1, exact)  # reconcile sub-micro saved/cached precision
        context_rows.append({
            "query_index": i, "gsm": cosine.loc[i, "gsm"], "gse": cosine.loc[i, "gse"],
            "source": cosine.loc[i, "source_name_ch1"], "exact_rank": int(cosine.loc[i, "exact_rank"]),
            "rank_group": str(cosine.loc[i, "rank_group"]), "exact_pair_cosine": exact,
            "top1_cosine": top1,
            "top1_minus_exact_cosine": top1 - exact,
            "raw_saved_top1_minus_exact_cosine": saved_top1 - exact,
            "top10_mean_cosine": top10.mean(), "top10_median_cosine": np.median(top10),
            "top10_sd_cosine": top10.std(ddof=1), "top10_minimum_cosine": top10.min(),
            "top10_maximum_cosine": top10.max(),
            # This preserves the upstream diagnostic definition: every sample
            # at least exact-delta, including closer samples and the exact pair.
            "references_at_least_exact_minus_0_0001": int(np.count_nonzero(row >= exact - .0001)),
            "references_at_least_exact_minus_0_001": int(np.count_nonzero(row >= exact - .001)),
            "references_at_least_exact_minus_0_005": int(np.count_nonzero(row >= exact - .005)),
        })
    context = pd.DataFrame(context_rows)
    context.to_parquet(OUT / "nearest_neighbor_context.parquet", index=False)
    margin_rows, tie_rows = [], []
    for label in RANK_LABELS + ["All 562"]:
        sub = context if label == "All 562" else context[context.rank_group.eq(label)]
        margin_rows.append({"rank_group": label, **values_summary(sub.top1_minus_exact_cosine.to_numpy())})
        for column, delta in [("references_at_least_exact_minus_0_0001", .0001),
                              ("references_at_least_exact_minus_0_001", .001),
                              ("references_at_least_exact_minus_0_005", .005)]:
            vals = sub[column].to_numpy(float)
            tie_rows.append({"rank_group": label, "delta": delta, **values_summary(vals)})
    pd.DataFrame(margin_rows).to_csv(OUT / "top1_minus_exact_margin_by_rank.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(OUT / "near_exact_neighbor_counts_by_rank.csv", index=False)

    # 7: objective examples and frequency at fixed interpretable cosine thresholds.
    examples = []
    for threshold in [100, 1000]:
        eligible = context[context.exact_rank.gt(threshold)].sort_values(
            ["exact_pair_cosine", "exact_rank", "gsm"], ascending=[False, False, True]).head(5)
        for order, (_, row) in enumerate(eligible.iterrows(), 1):
            examples.append({"criterion": f"exact_rank_gt_{threshold}", "selection_order": order,
                             **row[["gsm", "gse", "source", "exact_rank", "exact_pair_cosine",
                                    "top1_cosine", "top1_minus_exact_cosine",
                                    "references_at_least_exact_minus_0_001",
                                    "references_at_least_exact_minus_0_005"]].to_dict()})
    pd.DataFrame(examples).to_csv(OUT / "high_similarity_poor_rank_examples.csv", index=False)
    frequency = []
    for rank_threshold in [100, 1000]:
        for cosine_threshold in [.99, .995, .999]:
            hit = context.exact_rank.gt(rank_threshold) & context.exact_pair_cosine.ge(cosine_threshold)
            frequency.append({"rank_threshold": rank_threshold, "cosine_threshold": cosine_threshold,
                              "queries": int(hit.sum()), "fraction_all_queries": hit.mean(),
                              "fraction_among_poor_rank": hit.sum() / context.exact_rank.gt(rank_threshold).sum()})
    pd.DataFrame(frequency).to_csv(OUT / "high_similarity_poor_rank_frequency.csv", index=False)

    # 8: each method's native correct-pair score, stratified by its own rank.
    bridge_cosine = cosine.set_index("query_index").paired_score
    native_rows = []
    for method, sub in ranked.groupby("method", sort=False):
        sub = sub.copy(); sub["rank_group"] = rank_group(sub.exact_rank)
        sub["bridge_exact_pair_cosine"] = sub.query_index.map(bridge_cosine)
        for label in RANK_LABELS:
            part = sub[sub.rank_group.eq(label)]
            native_rows.append({"method": method, "rank_group": label})
            native_rows[-1].update({f"native_score_{k}": v for k, v in values_summary(part.paired_score).items()})
            native_rows[-1].update({f"bridge_cosine_{k}": v for k, v in values_summary(part.bridge_exact_pair_cosine).items()})
    pd.DataFrame(native_rows).to_csv(OUT / "method_native_score_by_rank.csv", index=False)

    # 4 plus contextual figures.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(np.log10(context.exact_rank), context.exact_pair_cosine, s=24, alpha=.65,
               color="#35618d", edgecolor="none")
    for boundary in [1, 10, 100, 1000]:
        ax.axvline(np.log10(boundary), color="#b5483a", lw=1, ls="--")
        ax.text(np.log10(boundary), ax.get_ylim()[0], f" {boundary:,}", rotation=90,
                va="bottom", ha="left", fontsize=9, color="#8c3227")
    ax.set_xlabel("log10(exact ARCHS4 counterpart rank)")
    ax.set_ylabel("Exact ARCHS4–recount3 Bridge cosine")
    ax.set_title("High representation agreement can coexist with poor retrieval rank")
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"exact_cosine_vs_rank.{ext}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(random_pair.min(), exact_pair.min()), 1, 120)
    ax.hist(random_pair, bins=bins, density=True, alpha=.55, color="#777777", label="Random ARCHS4 pairs")
    ax.hist(exact_pair, bins=bins, density=True, alpha=.65, color="#2a9d8f", label="Exact cross-pipeline pairs")
    ax.set_yscale("log"); ax.set_xlabel("Bridge cosine"); ax.set_ylabel("Density (log scale)")
    ax.set_title("Exact counterparts occupy the extreme similarity tail")
    ax.legend(frameon=False); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"random_vs_exact_cosine.{ext}", dpi=220)
    plt.close(fig)

    tie_summary = pd.DataFrame(tie_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot = tie_summary[tie_summary.rank_group.ne("All 562")]
    for delta, sub in plot.groupby("delta"):
        ax.plot(sub.rank_group, sub["median"], marker="o", label=f"exact − {delta:g}")
    ax.set_yscale("log"); ax.set_ylabel("Median references at or above threshold")
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    ax.set_title("Poor ranks occur in increasingly crowded near-exact neighborhoods")
    ax.legend(frameon=False); fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"near_exact_neighbors_by_rank.{ext}", dpi=220)
    plt.close(fig)

    provenance = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "queries": NQUERY,
        "references": NREF, "bridge_frozen": True, "embeddings_generated": False,
        "expensive_search_repeated": False,
        "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in [ranked_path, query_path, random_path]},
        "score_cache": {"path": str(score_path), "bytes": score_path.stat().st_size,
                        "use": "row-wise top-10 context and threshold counts only"},
        "rank_bins": RANK_LABELS, "cutoffs": CUTOFFS,
        "near_neighbor_definition": "score >= exact_pair_cosine - delta; includes closer samples and the exact pair",
        "top1_margin_precision": "Top1 is max(saved top1, saved exact cosine), clamping sub-micro negative differences caused by saved/cached precision.",
        "method_score_note": "Only Cosine paired_score is Bridge cosine; other paired_score values are native ranking criteria.",
        "elapsed_seconds": time.time() - start,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"Completed extension in {time.time() - start:.1f}s")
    log_handle.close()


if __name__ == "__main__":
    main()
