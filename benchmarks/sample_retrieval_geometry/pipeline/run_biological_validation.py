#!/usr/bin/env python3
"""Stages 1A-1H: composition-controlled biological retrieval validation.

Reads the existing Stage 0/1 score cache and shared read-only metadata. It does
not regenerate embeddings or mutate upstream assets.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import entropy
from statsmodels.discrete.conditional_models import ConditionalLogit

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "biological_validation"
FIG = OUT / "figures"
WORK = HERE / "work"
SEED = 20260922
MIN_REFS = 50
MIN_STUDIES = 3
TOP_K = 100

# Reused verbatim from the repository's deliberately conservative exact
# source-name mappings. These are anchored mappings, not title parsing.
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


def norm(x: pd.Series) -> pd.Series:
    return x.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def effective_n(counts: pd.Series) -> tuple[float, float]:
    p = counts.to_numpy(float) / counts.sum()
    return float(np.exp(entropy(p))), float(1 / np.sum(p * p))


def top_indices(score: np.ndarray, excluded: list[np.ndarray], k=TOP_K) -> np.ndarray:
    out = np.empty((score.shape[0], k), dtype=np.int64)
    for i in range(score.shape[0]):
        row = score[i].copy()
        row[excluded[i]] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        out[i] = idx[np.argsort(-row[idx], kind="stable")]
    return out


def label_metrics(top: np.ndarray, qlabels: np.ndarray, rlabels: np.ndarray,
                  rgse: np.ndarray, qgse: np.ndarray, label: str, method: str,
                  scope: str, eligible: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in np.flatnonzero(eligible):
        hits = rlabels[top[i]] == qlabels[i]
        allowed = np.ones(len(rlabels), dtype=bool)
        if scope == "cross_study": allowed &= rgse != qgse[i]
        total = int(np.sum(allowed & (rlabels == qlabels[i])))
        prevalence = total / max(1, allowed.sum())
        precisions = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        ap100 = float(np.sum(precisions * hits) / min(max(total, 1), TOP_K))
        row = {"label": label, "method": method, "scope": scope, "query_index": i,
               "query_label": qlabels[i], "label_references": total, "label_prevalence": prevalence,
               "ap_at_100": ap100}
        for k in [1, 5, 10]:
            pk = float(hits[:k].mean())
            row[f"precision_at_{k}"] = pk
            row[f"recall_at_{k}"] = float(hits[:k].sum() / max(total, 1))
            row[f"enrichment_at_{k}"] = pk / prevalence if prevalence > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_biology(perq: pd.DataFrame) -> pd.DataFrame:
    keys = ["label", "method", "scope"]
    aggs = {"queries": ("query_index", "size"), "p_at_1": ("precision_at_1", "mean"),
            "p_at_5": ("precision_at_5", "mean"), "p_at_10": ("precision_at_10", "mean"),
            "map_at_100": ("ap_at_100", "mean"),
            "median_enrichment_at_10": ("enrichment_at_10", "median")}
    return perq.groupby(keys, observed=True).agg(**aggs).reset_index()


def conditional_effect(y, x, groups, model_name):
    frame = pd.DataFrame({"y": y, "x": x, "group": groups}).dropna()
    varying = frame.groupby("group").y.transform("nunique").gt(1)
    frame = frame[varying]
    try:
        fit = ConditionalLogit(frame.y, frame[["x"]], groups=frame.group).fit(disp=False)
        beta, se = fit.params.iloc[0], fit.bse.iloc[0]
        return {"model": model_name, "samples": len(frame), "strata": frame.group.nunique(),
                "odds_ratio_per_density_sd": math.exp(beta),
                "ci_low": math.exp(beta - 1.96 * se), "ci_high": math.exp(beta + 1.96 * se),
                "p_value": fit.pvalues.iloc[0]}
    except Exception as exc:
        return {"model": model_name, "samples": len(frame), "strata": frame.group.nunique(),
                "odds_ratio_per_density_sd": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "error": str(exc)}


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    log_handle = (OUT / "run.log").open("w", buffering=1)
    def log(message):
        line = f"[{pd.Timestamp.now().isoformat()}] {message}"
        print(line, flush=True); log_handle.write(line + "\n")
    log("Starting Stage 1A-1H biological validation; estimated runtime 2-5 minutes")
    log(f"Monitor with: tail -f {OUT / 'run.log'}")
    cfg = json.loads((HERE / "config.json").read_text())
    inputs = {k: Path(v) for k, v in cfg["inputs"].items()}
    up = inputs["upstream_benchmark"]
    all_samples = pd.read_parquet(up / "work/paired_expression/samples.parquet")
    keep_queries = all_samples.study_exposure.eq("unseen_study").to_numpy()
    samples = all_samples.loc[keep_queries].reset_index(drop=True)
    upstream = samples[["gsm"]].merge(pd.read_parquet(up / "results/full_archs4_retrieval.parquet"), on="gsm")
    meta = pd.read_parquet(inputs["sample_metadata"])
    locations = pd.read_parquet(inputs["embedding_dir"] / "sample_locations.parquet")
    manifest = pd.read_parquet(inputs["sample_manifest"], columns=["gsm", "species"])
    human = locations.merge(manifest, left_on="geo_accession", right_on="gsm", validate="one_to_one")
    human = human[norm(human.species).eq("human")].reset_index(drop=True)
    hmeta = meta.set_index("gsm").reindex(human.geo_accession).reset_index(drop=True)
    hmeta.insert(0, "gsm", human.geo_accession.to_numpy())
    hmeta["source_label"] = norm(hmeta.source_name_ch1)
    hmeta["tissue_label"] = pd.NA
    tissue_match_count = np.zeros(len(hmeta), dtype=np.int8)
    for label, pattern in TISSUES.items():
        hit = hmeta.source_label.str.match(pattern, na=False)
        tissue_match_count += hit.to_numpy(dtype=np.int8)
        hmeta.loc[hit, "tissue_label"] = label
    assert tissue_match_count.max() <= 1, "Conservative tissue rules overlap"
    qmeta = samples[["gsm", "gse"]].merge(hmeta, on="gsm", validate="one_to_one")
    correct_lookup = pd.Series(np.arange(len(human)), index=human.geo_accession)
    correct = samples.gsm.map(correct_lookup).to_numpy(np.int64)
    assert len(samples) == 562 and len(human) == 510709

    density = pd.read_csv(RESULTS / "query_local_density.csv")
    density = density[density.k.eq(10)].set_index("query_gsm").reindex(samples.gsm)
    qa = qmeta.copy()
    qa["local_density_k10"] = density.mean_cosine.to_numpy()
    qa["density_stratum"] = pd.qcut(qa.local_density_k10, 3, labels=["sparse", "intermediate", "dense"])
    qa["exact_rank"] = upstream.exact_rank_full.to_numpy()
    qa["exact_failure"] = qa.exact_rank.gt(1)
    qa.to_parquet(OUT / "query_composition_and_density.parquet", index=False)

    # 1A: composition tables and reliable field availability.
    comp_rows = []
    for field in ["gse", "source_name_ch1", "source_label", "tissue_label", "organism_ch1",
                  "library_strategy", "library_source", "platform_id", "instrument_model", "molecule_ch1"]:
        counts = qa[field].fillna("<missing>").value_counts()
        for value, count in counts.items():
            comp_rows.append({"field": field, "value": value, "samples": count, "fraction": count / len(qa)})
    pd.DataFrame(comp_rows).to_csv(OUT / "composition_counts.csv", index=False)
    study_counts = qa.gse.value_counts(); source_counts = qa.source_label.value_counts()
    study_shannon, study_simpson = effective_n(study_counts)
    source_shannon, source_simpson = effective_n(source_counts)
    composition_summary = pd.DataFrame([
        {"dimension": "study", "classes": study_counts.size, "largest": study_counts.index[0],
         "largest_fraction": study_counts.iloc[0] / len(qa), "top5_fraction": study_counts.iloc[:5].sum() / len(qa),
         "effective_n_shannon": study_shannon, "effective_n_inverse_simpson": study_simpson},
        {"dimension": "normalized_source", "classes": source_counts.size, "largest": source_counts.index[0],
         "largest_fraction": source_counts.iloc[0] / len(qa), "top5_fraction": source_counts.iloc[:5].sum() / len(qa),
         "effective_n_shannon": source_shannon, "effective_n_inverse_simpson": source_simpson},
    ])
    composition_summary.to_csv(OUT / "composition_summary.csv", index=False)
    availability = pd.DataFrame([
        ["study", True, "ARCHS4 series_id / frozen pair manifest", "structured"],
        ["biological_source", True, "ARCHS4 source_name_ch1", "structured but submitter-defined and heterogeneous"],
        ["tissue", True, "existing conservative anchored source-name mapping", "limited coverage; not a curated ontology"],
        ["cell_type", False, "none corpus-wide", "present only inside free-text characteristics for some samples"],
        ["cell_line", False, "none corpus-wide", "present only inside free-text characteristics for some samples"],
        ["primary_vs_cultured", False, "none corpus-wide", "cannot determine without free-text inference"],
        ["disease_state", False, "none corpus-wide", "present only inside free-text characteristics for some samples"],
        ["treatment_condition", False, "none corpus-wide", "present only inside free-text characteristics for some samples"],
        ["organism", True, "ARCHS4 organism_ch1", "all evaluation samples are human"],
        ["library_strategy", True, "ARCHS4 library_strategy", "all queries are RNA-Seq"],
        ["library_source", True, "ARCHS4 library_source", "all queries are transcriptomic"],
        ["platform", True, "ARCHS4 platform_id", "structured technical label"],
        ["instrument", True, "ARCHS4 instrument_model", "structured technical label"],
    ], columns=["field", "usable", "provenance", "limitation"])
    availability.to_csv(OUT / "metadata_field_availability.csv", index=False)
    group_density = (qa.groupby(["density_stratum", "gse"], observed=True).size().rename("samples").reset_index()
                     .merge(qa.groupby("gse").local_density_k10.mean().rename("mean_density"), on="gse"))
    group_density.to_csv(OUT / "study_density_composition.csv", index=False)

    # 1B: density/failure effects, within-stratum controls, exclusions.
    z = (qa.local_density_k10 - qa.local_density_k10.mean()) / qa.local_density_k10.std()
    glm = sm.GLM(qa.exact_failure.astype(int), sm.add_constant(z), family=sm.families.Binomial()).fit()
    beta, se = glm.params.iloc[1], glm.bse.iloc[1]
    effects = [{"model": "unadjusted logistic", "samples": len(qa), "strata": np.nan,
                "odds_ratio_per_density_sd": math.exp(beta), "ci_low": math.exp(beta-1.96*se),
                "ci_high": math.exp(beta+1.96*se), "p_value": glm.pvalues.iloc[1]}]
    effects += [conditional_effect(qa.exact_failure.astype(int), z, qa.gse, "study fixed effects"),
                conditional_effect(qa.exact_failure.astype(int), z, qa.source_label, "source fixed effects")]
    extremes = qa.density_stratum.ne("intermediate")
    ex = qa.loc[extremes].copy(); dense_indicator = ex.density_stratum.eq("dense").astype(int)
    dense_fit = sm.GLM(ex.exact_failure.astype(int), sm.add_constant(dense_indicator), family=sm.families.Binomial()).fit()
    dbeta, dse = dense_fit.params.iloc[1], dense_fit.bse.iloc[1]
    effects.append({"model": "dense vs sparse unadjusted", "samples": len(ex), "strata": np.nan,
                    "odds_ratio_per_density_sd": math.exp(dbeta), "ci_low": math.exp(dbeta-1.96*dse),
                    "ci_high": math.exp(dbeta+1.96*dse), "p_value": dense_fit.pvalues.iloc[1]})
    effects += [conditional_effect(ex.exact_failure.astype(int), dense_indicator, ex.gse,
                                   "dense vs sparse, study fixed effects"),
                conditional_effect(ex.exact_failure.astype(int), dense_indicator, ex.source_label,
                                   "dense vs sparse, source fixed effects")]
    pd.DataFrame(effects).to_csv(OUT / "density_failure_models.csv", index=False)
    sensitivity = []
    largest_study, largest_source = study_counts.index[0], source_counts.index[0]
    masks = {"all": np.ones(len(qa), bool), "exclude_largest_study": qa.gse.ne(largest_study).to_numpy(),
             "exclude_largest_source": qa.source_label.ne(largest_source).to_numpy()}
    for name, mask in masks.items():
        sub = qa.loc[mask]
        rates = sub.groupby("density_stratum", observed=True).exact_failure.mean()
        sensitivity.append({"analysis": name, "samples": len(sub), "sparse_failure": rates.get("sparse"),
                            "intermediate_failure": rates.get("intermediate"), "dense_failure": rates.get("dense"),
                            "dense_minus_sparse": rates.get("dense") - rates.get("sparse")})
    pd.DataFrame(sensitivity).to_csv(OUT / "density_exclusion_sensitivity.csv", index=False)
    within = []
    for field in ["gse", "source_label"]:
        for value, sub in qa.groupby(field):
            if len(sub) < 20: continue
            lo, hi = sub.local_density_k10.quantile([1/3, 2/3])
            low = sub[sub.local_density_k10 <= lo].exact_failure.mean()
            high = sub[sub.local_density_k10 >= hi].exact_failure.mean()
            within.append({"field": field, "value": value, "samples": len(sub),
                           "low_density_failure": low, "high_density_failure": high,
                           "high_minus_low": high-low})
    pd.DataFrame(within).to_csv(OUT / "within_context_density_effects.csv", index=False)

    # 1C: label QC with prespecified support criteria.
    qc_rows, eligible_by_label = [], {}
    for label_col in ["source_label", "tissue_label"]:
        stats = hmeta.dropna(subset=[label_col]).groupby(label_col).agg(
            references=("gsm", "size"), studies=("series_id", "nunique"))
        stats["eligible"] = (stats.references >= MIN_REFS) & (stats.studies >= MIN_STUDIES)
        eligible_labels = set(stats.index[stats.eligible]); eligible_by_label[label_col] = eligible_labels
        qcounts = qa[label_col].value_counts()
        for value, row in stats.iterrows():
            qc_rows.append({"label": label_col, "class": value, "references": row.references,
                            "studies": row.studies, "queries": qcounts.get(value, 0), "eligible": row.eligible})
    qc = pd.DataFrame(qc_rows); qc.to_csv(OUT / "biological_label_class_qc.csv", index=False)
    label_summary = []
    for label_col in ["source_label", "tissue_label"]:
        labeled_ref = hmeta[label_col].notna(); labeled_q = qa[label_col].notna()
        elig = qa[label_col].isin(eligible_by_label[label_col])
        counts = qa.loc[elig, label_col].value_counts()
        label_summary.append({"label": label_col, "labeled_queries": labeled_q.sum(),
                              "eligible_queries": elig.sum(), "labeled_references": labeled_ref.sum(),
                              "eligible_references": hmeta[label_col].isin(eligible_by_label[label_col]).sum(),
                              "eligible_classes_corpus": len(eligible_by_label[label_col]),
                              "classes_in_eligible_queries": counts.size,
                              "ambiguous_references": 0,
                              "largest_query_class_fraction": counts.iloc[0]/counts.sum() if len(counts) else np.nan,
                              "query_missing_fraction": 1-labeled_q.mean(), "reference_missing_fraction": 1-labeled_ref.mean()})
    pd.DataFrame(label_summary).to_csv(OUT / "biological_label_summary.csv", index=False)

    # 1D-1G: recompute only method scores/top lists from existing score cache.
    scores = np.memmap(WORK / "query_corpus_cosine.float32.mmap", mode="r", dtype=np.float32,
                       shape=(len(qa), len(human)))
    cosine_top = np.partition(scores, -100, axis=1)[:, -100:]
    cosine_top.sort(axis=1); cosine_top = cosine_top[:, ::-1]
    k = 10
    qmean = cosine_top[:, :k].mean(axis=1)
    rtop = np.partition(scores, -k, axis=0)[-k:, :]
    rmean = rtop.mean(axis=0)
    sigma_q = np.maximum(1-cosine_top[:, k-1], np.finfo(np.float32).eps)
    sigma_r = np.maximum(1-np.partition(scores, -k, axis=0)[-k, :], np.finfo(np.float32).eps)
    mean_dq = np.maximum((1-cosine_top[:, :k]).mean(axis=1), np.finfo(np.float32).eps)
    mean_dr = np.maximum((1-rtop).mean(axis=0), np.finfo(np.float32).eps)
    query_gses = set(qa.gse.astype(str))
    gse_lists = {g: [] for g in query_gses}
    for idx, value in enumerate(hmeta.series_id.astype(str)):
        for g in value.split(","):
            if g in gse_lists: gse_lists[g].append(idx)
    same_gse_indices = {g: np.asarray(v, dtype=np.int64) for g, v in gse_lists.items()}
    full_excluded = [np.array([correct[i]]) for i in range(len(qa))]
    cross_excluded = [same_gse_indices.get(qa.gse.iloc[i], np.array([], dtype=int)) for i in range(len(qa))]
    rgse = hmeta.series_id.astype(str).to_numpy(); qgse = qa.gse.astype(str).to_numpy()
    perq_frames, diversity_rows, neighbor_rows, method_top = [], [], [], {}
    method_scores = {
        "Cosine": scores,
        "CSLS(k=10)": 2*scores - qmean[:, None] - rmean[None, :],
    }
    distance = np.maximum(1-scores, np.finfo(np.float32).eps)
    method_scores["LocalScaling(k=10)"] = -(distance*distance)/(sigma_q[:, None]*sigma_r[None, :])
    method_scores["NICDM(k=10)"] = -distance/np.sqrt(mean_dq[:, None]*mean_dr[None, :])
    for method, score in method_scores.items():
        log(f"Biological retrieval: {method}; elapsed={time.time()-start:.0f}s")
        full_top = top_indices(score, full_excluded)
        cross_top = top_indices(score, cross_excluded)
        method_top[method] = (full_top, cross_top)
        assert np.all([correct[i] not in full_top[i] for i in range(len(qa))])
        assert np.all([not np.any(np.isin(cross_top[i], cross_excluded[i])) for i in range(len(qa))])
        for label_col in ["source_label", "tissue_label"]:
            qlabels = qa[label_col].astype("string").fillna("").to_numpy()
            rlabels = hmeta[label_col].astype("string").fillna("").to_numpy()
            eligible = qa[label_col].isin(eligible_by_label[label_col]).to_numpy()
            perq_frames.append(label_metrics(full_top, qlabels, rlabels, rgse, qgse, label_col, method, "full_no_exact", eligible))
            perq_frames.append(label_metrics(cross_top, qlabels, rlabels, rgse, qgse, label_col, method, "cross_study", eligible))
        for scope, top in [("full_no_exact", full_top), ("cross_study", cross_top)]:
            for i in range(len(qa)):
                idx = top[i, :10]; gs = pd.Series(rgse[idx]); ls = pd.Series(hmeta.source_label.iloc[idx].to_numpy())
                diversity_rows.append({"method": method, "scope": scope, "query_index": i,
                    "unique_gse_top10": gs.nunique(), "unique_source_top10": ls.nunique(),
                    "largest_gse_fraction_top10": gs.value_counts().iloc[0]/10,
                    "largest_source_fraction_top10": ls.value_counts().iloc[0]/10})
                for rank, ref_i in enumerate(idx, 1):
                    neighbor_rows.append({"method": method, "scope": scope, "query_index": i,
                        "query_gsm": qa.gsm.iloc[i], "query_gse": qgse[i], "rank": rank,
                        "neighbor_gsm": hmeta.gsm.iloc[ref_i], "neighbor_gse": rgse[ref_i],
                        "neighbor_source": hmeta.source_name_ch1.iloc[ref_i],
                        "neighbor_source_label": hmeta.source_label.iloc[ref_i],
                        "neighbor_tissue_label": hmeta.tissue_label.iloc[ref_i]})
        if method != "Cosine": del score
    perq = pd.concat(perq_frames, ignore_index=True)
    perq = perq.merge(qa[["density_stratum"]], left_on="query_index", right_index=True)
    perq.to_parquet(OUT / "biological_retrieval_per_query.parquet", index=False)
    summary = summarize_biology(perq); summary.to_csv(OUT / "biological_retrieval_summary.csv", index=False)
    exact = pd.read_csv(RESULTS / "ranked_method_comparison.csv")
    exact = exact[exact.method.isin(method_scores)].set_index("method")
    hubs = pd.read_csv(RESULTS / "hubness_summary.csv")
    hubs = hubs[hubs.k.eq(100) & hubs.method.isin(method_scores)].set_index("method")
    for label_col in ["source_label", "tissue_label"]:
        full = summary[(summary.label.eq(label_col)) & summary.scope.eq("full_no_exact")].set_index("method")
        cross = summary[(summary.label.eq(label_col)) & summary.scope.eq("cross_study")].set_index("method")
        table = pd.DataFrame(index=exact.index)
        table["exact_r1"] = exact.exact_r1; table["exact_r10"] = exact.exact_r10
        table["biological_p1"] = full.p_at_1; table["biological_p10"] = full.p_at_10
        table["cross_study_biological_p1"] = cross.p_at_1
        table["cross_study_biological_p10"] = cross.p_at_10
        table["hubness_max_n100"] = hubs.max_nk; table["hubness_gini_n100"] = hubs.gini
        table.reset_index().to_csv(OUT / f"main_comparison_{label_col}.csv", index=False)
    density_bio = (perq.groupby(["label", "method", "scope", "density_stratum"], observed=True)
                   .agg(queries=("query_index", "size"), p_at_1=("precision_at_1", "mean"),
                        p_at_10=("precision_at_10", "mean"), map_at_100=("ap_at_100", "mean"),
                        median_enrichment_at_10=("enrichment_at_10", "median")).reset_index())
    density_bio.to_csv(OUT / "density_stratified_biological_retrieval.csv", index=False)
    diversity = pd.DataFrame(diversity_rows)
    diversity.to_csv(OUT / "retrieval_diversity_per_query.csv", index=False)
    pd.DataFrame(neighbor_rows).to_parquet(OUT / "biological_top10_neighbors.parquet", index=False)
    diversity.groupby(["method", "scope"]).agg(
        unique_gse_top10=("unique_gse_top10", "mean"), unique_source_top10=("unique_source_top10", "mean"),
        largest_gse_fraction_top10=("largest_gse_fraction_top10", "mean"),
        largest_source_fraction_top10=("largest_source_fraction_top10", "mean")).reset_index().to_csv(
            OUT / "retrieval_diversity_summary.csv", index=False)

    # Set methods at k=10. Biological precision is over the returned set.
    cosine_full = method_top["Cosine"][0]
    ref_topq = np.argpartition(scores, -10, axis=0)[-10:, :]
    qq = np.asarray(np.load(up / "work/recount3_embeddings.npy", mmap_mode="r")[keep_queries], dtype=np.float32)
    qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    qqsim = qq @ qq.T; qcontext = np.argpartition(qqsim, -10, axis=1)[:, -10:]
    rcontext = ref_topq.T
    set_rows = []
    for i in range(len(qa)):
        mutual = np.array([r for r in cosine_full[i, :10] if np.any(ref_topq[:, r] == i)], dtype=int)
        mask = np.zeros(len(qa), bool); mask[qcontext[i]] = True
        overlap = mask[rcontext].sum(axis=1); jac = overlap/(20-overlap); snn = np.flatnonzero(jac == jac.max())
        for method, refs in [("Mutual-kNN(k=10)", mutual), ("SNN-Jaccard(k=10)", snn)]:
            refs = refs[refs != correct[i]]
            for scope in ["full_no_exact", "cross_study"]:
                use = refs if scope == "full_no_exact" else refs[rgse[refs] != qgse[i]]
                for label_col in ["source_label", "tissue_label"]:
                    if qa[label_col].iloc[i] not in eligible_by_label[label_col]: continue
                    retrieved = hmeta[label_col].iloc[use].astype("string").fillna("").to_numpy()
                    hit = retrieved == str(qa[label_col].iloc[i])
                    set_rows.append({"method": method, "scope": scope, "label": label_col, "query_index": i,
                                     "returned": len(use), "same_label_precision": hit.mean() if len(hit) else np.nan,
                                     "same_label_any": hit.any() if len(hit) else False})
    setbio = pd.DataFrame(set_rows); setbio.to_csv(OUT / "set_method_biological_retrieval.csv", index=False)
    setbio.groupby(["label", "method", "scope"]).agg(queries=("query_index", "size"),
        mean_returned=("returned", "mean"), nonempty_fraction=("returned", lambda x:(x>0).mean()),
        mean_set_precision=("same_label_precision", "mean"), any_same_label=("same_label_any", "mean")).reset_index().to_csv(
            OUT / "set_method_biological_summary.csv", index=False)

    # Figures.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    study_counts.head(12).sort_values().plot.barh(ax=axes[0], color="#4C78A8", title="Query composition by study")
    source_counts.head(12).sort_values().plot.barh(ax=axes[1], color="#F58518", title="Query composition by source")
    for ax in axes: ax.set_xlabel("Queries"); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "query_composition.png", dpi=220); fig.savefig(FIG / "query_composition.pdf"); plt.close(fig)

    major = study_counts[study_counts >= 12].index
    plot = qa[qa.gse.isin(major)].copy(); order = plot.groupby("gse").local_density_k10.median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 5)); ax.boxplot([plot.loc[plot.gse.eq(g), "local_density_k10"] for g in order], tick_labels=order, vert=True, showfliers=False)
    ax.tick_params(axis="x", rotation=65); ax.set(ylabel="Mean cosine to 10 nearest references", title="Local density varies across studies")
    fig.tight_layout(); fig.savefig(FIG / "density_by_study.png", dpi=220); fig.savefig(FIG / "density_by_study.pdf"); plt.close(fig)

    major_sources = source_counts[source_counts >= 12].index
    plot = qa[qa.source_label.isin(major_sources)].copy()
    order = plot.groupby("source_label").local_density_k10.median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 5)); ax.boxplot([plot.loc[plot.source_label.eq(g), "local_density_k10"] for g in order], tick_labels=order, vert=True, showfliers=False)
    ax.tick_params(axis="x", rotation=65); ax.set(ylabel="Mean cosine to 10 nearest references", title="Local density varies across biological sources")
    fig.tight_layout(); fig.savefig(FIG / "density_by_source.png", dpi=220); fig.savefig(FIG / "density_by_source.pdf"); plt.close(fig)

    eff = pd.DataFrame(effects).dropna(subset=["odds_ratio_per_density_sd"])
    fig, ax = plt.subplots(figsize=(7, 3.8)); y=np.arange(len(eff)); x=eff.odds_ratio_per_density_sd.to_numpy();
    ax.errorbar(x, y, xerr=[x-eff.ci_low, eff.ci_high-x], fmt="o", color="#E45756", capsize=4)
    ax.axvline(1, color="black", ls="--"); ax.set_yticks(y, eff.model); ax.set(xlabel="Failure odds ratio per density SD", title="Density effect after composition controls")
    fig.tight_layout(); fig.savefig(FIG / "density_failure_effects.png", dpi=220); fig.savefig(FIG / "density_failure_effects.pdf"); plt.close(fig)

    source_sum = summary[summary.label.eq("source_label")]
    methods = ["Cosine", "CSLS(k=10)", "LocalScaling(k=10)", "NICDM(k=10)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True); width=.18; x=np.arange(2)
    for j, method in enumerate(methods):
        sub=source_sum[source_sum.method.eq(method)].set_index("scope").reindex(["full_no_exact","cross_study"])
        for ax, metric, title in zip(axes,["p_at_1","p_at_10"],["Source P@1","Source P@10"]):
            ax.bar(x+(j-1.5)*width,sub[metric],width,label=method)
            ax.set(xticks=x,xticklabels=["Full, exact excluded","Cross-study"],ylabel="Precision",title=title,ylim=(0,1))
    axes[0].legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(FIG/"biological_retrieval_comparison.png",dpi=220); fig.savefig(FIG/"biological_retrieval_comparison.pdf");plt.close(fig)

    db = density_bio[(density_bio.label.eq("source_label")) & density_bio.method.isin(methods)]
    fig, axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True); strata=["sparse","intermediate","dense"]
    for j,method in enumerate(methods):
        for ax,scope,title in zip(axes,["full_no_exact","cross_study"],["Full corpus","Cross-study"]):
            sub=db[(db.method.eq(method))&(db.scope.eq(scope))].set_index("density_stratum").reindex(strata)
            ax.plot(strata,sub.p_at_10,marker="o",label=method);ax.set(title=title,ylabel="Source P@10",ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(FIG/"density_stratified_biological_retrieval.png",dpi=220);fig.savefig(FIG/"density_stratified_biological_retrieval.pdf");plt.close(fig)

    divsum=pd.read_csv(OUT/"retrieval_diversity_summary.csv");divsum=divsum[divsum.method.isin(methods)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));
    for scope,marker in [("full_no_exact","o"),("cross_study","s")]:
        sub=divsum[divsum.scope.eq(scope)];axes[0].plot(sub.method,sub.unique_gse_top10,marker=marker,label=scope);axes[1].plot(sub.method,sub.largest_gse_fraction_top10,marker=marker,label=scope)
    axes[0].set(ylabel="Mean unique GSEs in Top-10",title="Study diversity");axes[1].set(ylabel="Mean largest-GSE fraction",title="Study redundancy")
    for ax in axes:ax.tick_params(axis="x",rotation=30);ax.grid(alpha=.2)
    axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(FIG/"retrieval_diversity.png",dpi=220);fig.savefig(FIG/"retrieval_diversity.pdf");plt.close(fig)

    provenance = {"created_at": pd.Timestamp.now().isoformat(), "stage": "1A-1H", "bridge_frozen": True,
        "representation_learning": False, "queries": len(qa), "references": len(human),
        "minimum_reference_support": MIN_REFS, "minimum_study_support": MIN_STUDIES,
        "exact_gsm_excluded_from_biological_retrieval": True,
        "tissue_mapping_provenance": str((HERE.parents[1]/"benchmarks/library_prep_disentanglement/pipeline/analyze_pca_vs_bridgerna_generalization.py").resolve()),
        "inputs": {k:str(v.resolve()) for k,v in inputs.items()}, "runtime_seconds": time.time()-start,
        "limitations": ["source_name_ch1 is structured but submitter-defined and heterogeneous",
                        "conservative tissue mapping covers only blood and brain among eligible queries and is blood-dominated",
                        "cell type, cell line, disease, treatment, and primary/cultured status were not inferred from free text"]}
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    log(f"Completed biological validation in {time.time()-start:.1f}s: {OUT}")
    log_handle.close()


if __name__ == "__main__":
    main()
