#!/usr/bin/env python3
"""Audit paired-recount3 discovery and rebalance saved retrieval results.

This is a metadata-only analysis. It reads the upstream benchmark and its
restart cache without changing either, and it never downloads expression,
regenerates embeddings, or changes Bridge exposure labels.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "recount3_audit"
FIG = OUT / "figures"
SEED = 20260923
METHODS = ["Cosine", "CSLS(k=10)", "LocalScaling(k=10)", "NICDM(k=10)"]
GSM_RE = re.compile(r"GSM[0-9]+", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def studies(frame: pd.DataFrame, column: str = "gse") -> int:
    return frame[column].dropna().astype(str).nunique()


def metric_row(frame: pd.DataFrame) -> dict[str, float]:
    rank = frame["exact_rank"].to_numpy(float)
    return {
        "queries": len(frame), "r_at_1": np.mean(rank <= 1),
        "r_at_5": np.mean(rank <= 5), "r_at_10": np.mean(rank <= 10),
        "r_at_100": np.mean(rank <= 100), "mrr": np.mean(1 / rank),
    }


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def scan_cached_metadata(cache: Path, target_gsms: set[str], log) -> pd.DataFrame:
    """Repeat the upstream all-character-column GSM lookup on cached files only."""
    rows: list[dict[str, str]] = []
    paths = sorted(cache.glob("*.gz"))
    for file_number, path in enumerate(paths, 1):
        project_match = re.search(r"\.([DES]RP[0-9]+)\.MD\.gz$", path.name)
        project = project_match.group(1) if project_match else path.name
        try:
            with gzip.open(path, "rt", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                run_idx = header.index("external_id") if "external_id" in header else None
                for fields in reader:
                    text = " | ".join(fields)
                    hits = {x.upper() for x in GSM_RE.findall(text)} & target_gsms
                    if not hits:
                        continue
                    run = fields[run_idx] if run_idx is not None and run_idx < len(fields) else ""
                    if not re.fullmatch(r"(?:SRR|ERR|DRR)[0-9]+", run):
                        run = ""
                    rows.extend({"gsm": gsm, "recount3_project": project,
                                 "recount3_run_id": run} for gsm in hits)
        except (OSError, EOFError, csv.Error, StopIteration) as exc:
            rows.append({"gsm": "", "recount3_project": project,
                         "recount3_run_id": "", "scan_error": type(exc).__name__})
        if file_number % 1000 == 0:
            log(f"Cached metadata inventory {file_number:,}/{len(paths):,} projects")
    hits = pd.DataFrame(rows)
    if hits.empty:
        return pd.DataFrame(columns=["gsm", "recount3_project", "recount3_run_ids",
                                     "run_count", "project_count"])
    hits = hits[hits.gsm.ne("")].drop_duplicates()
    return (hits.groupby("gsm", as_index=False)
            .agg(recount3_project=("recount3_project", lambda x: ",".join(sorted(set(x)))),
                 recount3_run_ids=("recount3_run_id", lambda x: ",".join(sorted({v for v in x if v}))),
                 run_count=("recount3_run_id", lambda x: len({v for v in x if v})),
                 project_count=("recount3_project", "nunique")))


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    log_handle = (OUT / "run.log").open("w", buffering=1)
    def log(message: str) -> None:
        line = f"[{pd.Timestamp.now().isoformat()}] {message}"
        print(line, flush=True); log_handle.write(line + "\n")

    log("Starting metadata-only recount3 provenance/balancing audit; estimate 1-3 minutes")
    log(f"Monitor with: tail -f {OUT / 'run.log'}")
    config = json.loads((HERE / "config.json").read_text())
    inputs = {k: Path(v) for k, v in config["inputs"].items()}
    upstream = inputs["upstream_benchmark"]
    manifest_path = inputs["sample_manifest"]
    metadata_path = inputs["sample_metadata"]
    candidate_path = upstream / "work/candidate_samples.parquet"
    matches_path = upstream / "work/recount3_matches.parquet"
    final_path = upstream / "work/final_pairs.parquet"
    failures_path = upstream / "work/recount3_matches_metadata_failures.csv"
    cache = upstream / "work/recount3_metadata_cache"

    manifest = pd.read_parquet(manifest_path)
    candidates = pd.read_parquet(candidate_path)
    matches = pd.read_parquet(matches_path)
    final = pd.read_parquet(final_path)
    processed = pd.read_parquet(upstream / "work/paired_expression/samples.parquet")
    failures = pd.read_csv(failures_path)
    meta = pd.read_parquet(metadata_path)
    log(f"Loaded {len(candidates):,} frozen candidates, {len(final):,} final pairs")

    # The tables, rather than the later-edited config, are authoritative for the run.
    exposure = (candidates.groupby(["cohort", "study_exposure"], observed=True)
                .agg(bridge_candidates=("gsm", "nunique"), candidate_studies=("gse", "nunique"))
                .reset_index())
    matched_counts = (matches.groupby(["cohort", "study_exposure", "match_status"], observed=True)
                      .agg(samples=("gsm", "nunique"), studies=("gse", "nunique"))
                      .reset_index())
    exposure.to_csv(OUT / "candidate_exposure_inventory.csv", index=False)
    matched_counts.to_csv(OUT / "match_status_inventory.csv", index=False)
    (final.groupby(["cohort", "study_exposure"], observed=True)
     .agg(paired_samples=("gsm", "nunique"), studies=("gse", "nunique"))
     .reset_index().to_csv(OUT / "paired_exposure_inventory.csv", index=False))

    human_unseen = manifest[manifest.species.eq("human") & manifest.split.eq("unseen")].copy()
    single = human_unseen[human_unseen.mapping_status.eq("mapped_single")]
    read_projects = len(list(cache.glob("*.gz")))
    project_total = read_projects + failures.project.nunique()
    funnel = pd.DataFrame([
        ["Bridge human samples", manifest[manifest.species.eq("human")].sample_id.nunique(),
         manifest.loc[manifest.species.eq("human"), "gse_candidates_str"].nunique(), np.nan,
         "Frozen master manifest"],
        ["Predefined unseen split", human_unseen.sample_id.nunique(),
         human_unseen.gse_candidates_str.nunique(),
         manifest[manifest.species.eq("human")].sample_id.nunique() - human_unseen.sample_id.nunique(),
         "split == unseen"],
        ["Single-GSE eligible candidates", single.sample_id.nunique(),
         single.gse_candidates_str.nunique(), len(human_unseen) - len(single),
         "mapping_status == mapped_single; both exposure cohorts"],
        ["Frozen discovery candidates", candidates.gsm.nunique(), studies(candidates),
         len(single) - candidates.gsm.nunique(), "All eligible rows retained; no subsampling"],
        ["Any recount3 metadata match", matches[~matches.match_status.eq("not_found")].gsm.nunique(),
         studies(matches[~matches.match_status.eq("not_found")]),
         candidates.gsm.nunique() - matches[~matches.match_status.eq("not_found")].gsm.nunique(),
         f"Catalog-wide scan: {read_projects:,}/{project_total:,} project metadata files read"],
        ["Unambiguous project with run(s)", matches[ matches.match_status.eq("matched")].gsm.nunique(),
         studies(matches[matches.match_status.eq("matched")]), 1,
         "One multiple-project GSM excluded; no matched_no_run rows"],
        ["Expression downloaded / vocabulary compatible", processed.gsm.nunique(), studies(processed), 0,
         "All 574 selected pairs processed on 15,165 canonical genes; no sample QC exclusion"],
        ["Published strict unseen-study benchmark", final[final.study_exposure.eq("unseen_study")].gsm.nunique(),
         studies(final[final.study_exposure.eq("unseen_study")]),
         final[final.study_exposure.ne("unseen_study")].gsm.nunique(),
         "Notebook/report filter; 12 unseen-sample/seen-study pairs excluded from headline"],
    ], columns=["step", "samples", "studies", "removed_from_prior", "reason"])
    funnel.to_csv(OUT / "original_cohort_funnel.csv", index=False)

    project_coverage = pd.DataFrame([
        {"status": "metadata_read_and_cached", "projects": read_projects,
         "fraction": read_projects / project_total},
        {"status": "download_or_read_failed", "projects": failures.project.nunique(),
         "fraction": failures.project.nunique() / project_total},
    ])
    project_coverage.to_csv(OUT / "recount3_project_scan_coverage.csv", index=False)

    # Inventory every predefined human evaluation sample against the preserved
    # metadata snapshot, including mapped-multiple rows excluded by the original protocol.
    cached_hits = scan_cached_metadata(cache, set(human_unseen.gsm.str.upper()), log)
    inventory = human_unseen.merge(cached_hits, on="gsm", how="left", validate="one_to_one")
    inventory["cached_metadata_hit"] = inventory.recount3_project.notna()
    inventory["has_run"] = inventory.run_count.fillna(0).gt(0)
    inventory["single_project"] = inventory.project_count.fillna(0).eq(1)
    inventory["existing_upstream_pair"] = inventory.gsm.isin(final.gsm)
    inventory["existing_published_pair"] = inventory.gsm.isin(
        final.loc[final.study_exposure.eq("unseen_study"), "gsm"])
    inventory["candidate_additional_pair"] = (inventory.cached_metadata_hit & inventory.has_run &
                                               inventory.single_project & ~inventory.existing_upstream_pair)
    inventory.to_parquet(OUT / "evaluation_recount3_cached_overlap.parquet", index=False)

    cohort_inventory = (human_unseen.groupby(["study_exposure", "mapping_status"], observed=True)
                        .agg(bridge_samples=("gsm", "nunique"),
                             gse_labels=("gse_candidates_str", "nunique"))
                        .reset_index())
    ov = (inventory.groupby(["study_exposure", "mapping_status"], observed=True)
          .agg(cached_metadata_hits=("cached_metadata_hit", "sum"),
               unambiguous_cached_candidates=("single_project", lambda x: int((x & inventory.loc[x.index, "has_run"]).sum())),
               existing_upstream_pairs=("existing_upstream_pair", "sum"),
               additional_candidates=("candidate_additional_pair", "sum"))
          .reset_index())
    cohort_inventory = cohort_inventory.merge(ov, on=["study_exposure", "mapping_status"], how="left")
    cohort_inventory.to_csv(OUT / "bridge_evaluation_cohort_inventory.csv", index=False)

    additional = inventory[inventory.candidate_additional_pair].copy()
    if len(additional):
        additional = additional.merge(meta, on="gsm", how="left", suffixes=("", "_metadata"))
        additional.to_csv(OUT / "candidate_additional_pairs.csv", index=False)
    else:
        pd.DataFrame(columns=list(inventory.columns)).to_csv(OUT / "candidate_additional_pairs.csv", index=False)

    published_gses = set(final.loc[final.study_exposure.eq("unseen_study"), "gse"])
    diversity_rows = []
    for exposure_name, sub in additional.groupby("study_exposure", observed=True):
        component_gses = {part for label in sub.gse_candidates_str.dropna().astype(str)
                          for part in label.split(";")}
        source_counts = sub.get("source_name_ch1", pd.Series(dtype=str)).fillna("<missing>").value_counts()
        diversity_rows.append({
            "study_exposure": exposure_name, "candidate_samples": len(sub),
            "composite_gse_labels": sub.gse_candidates_str.nunique(),
            "component_gses": len(component_gses),
            "component_gses_new_vs_published_562": len(component_gses - published_gses),
            "component_gses_overlapping_published_562": len(component_gses & published_gses),
            "largest_composite_label_fraction": sub.gse_candidates_str.value_counts().iloc[0] / len(sub),
            "largest_source": source_counts.index[0] if len(source_counts) else "<missing>",
            "largest_source_fraction": source_counts.iloc[0] / len(sub) if len(source_counts) else np.nan,
        })
    pd.DataFrame(diversity_rows).to_csv(OUT / "candidate_diversity_summary.csv", index=False)

    expansion_rows = []
    for exposure_name, sub in inventory.groupby("study_exposure", observed=True):
        existing = sub.existing_published_pair.sum()
        add = sub.candidate_additional_pair.sum()
        add_rows = sub[sub.candidate_additional_pair]
        expansion_rows.append({
            "cohort": exposure_name, "bridge_samples": len(sub),
            "existing_published_pairs": int(existing), "existing_upstream_pairs": int(sub.existing_upstream_pair.sum()),
            "additional_cached_candidates": int(add),
            "additional_candidate_gse_labels": int(add_rows.gse_candidates_str.nunique()),
            "largest_candidate_gse_fraction": (add_rows.gse_candidates_str.value_counts().iloc[0] / add
                                                if add else np.nan),
            "mapped_single_additional": int((add_rows.mapping_status == "mapped_single").sum()),
            "mapped_multiple_additional": int((add_rows.mapping_status == "mapped_multiple").sum()),
        })
    pd.DataFrame(expansion_rows).to_csv(OUT / "expansion_inventory.csv", index=False)
    log(f"Cached overlap scan found {len(additional):,} unprocessed unambiguous candidates")

    # Reuse saved exact ranks for study-macro and deterministic study caps.
    ranks = pd.read_parquet(HERE / "results/per_query_ranked_results.parquet")
    ranks = ranks[ranks.method.isin(METHODS)].copy()
    query = pd.read_parquet(HERE / "results/biological_validation/query_composition_and_density.parquet")
    composition_rows = []
    for field in ["gse", "source_label", "density_stratum", "library_strategy", "library_source",
                  "platform_id", "instrument_model"]:
        for value, count in query[field].value_counts(dropna=False).items():
            composition_rows.append({"field": field, "value": value, "samples": count,
                                     "fraction": count / len(query)})
    pd.DataFrame(composition_rows).to_csv(OUT / "current_562_composition.csv", index=False)
    ranks = ranks.drop(columns=["gsm", "gse", "density_stratum"], errors="ignore").merge(
        query[["gsm", "gse", "source_label", "density_stratum", "local_density_k10"]].reset_index(names="query_index"),
        on="query_index", validate="many_to_one")
    assert ranks.groupby("method").size().eq(562).all()

    per_study_rows = []
    for (method, gse), sub in ranks.groupby(["method", "gse"], observed=True):
        per_study_rows.append({"method": method, "gse": gse, **metric_row(sub)})
    per_study = pd.DataFrame(per_study_rows)
    per_study.to_csv(OUT / "study_level_retrieval_metrics.csv", index=False)
    rng = np.random.default_rng(SEED)
    macro_rows = []
    for method, sub in per_study.groupby("method", sort=False):
        row = {"method": method, "studies": len(sub), "queries": int(sub.queries.sum())}
        for metric in ["r_at_1", "r_at_5", "r_at_10", "r_at_100", "mrr"]:
            values = sub[metric].to_numpy(float)
            low, high = bootstrap_mean_ci(values, rng)
            row[f"macro_{metric}"] = values.mean()
            row[f"study_sd_{metric}"] = values.std(ddof=1)
            row[f"ci_low_{metric}"] = low; row[f"ci_high_{metric}"] = high
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)
    macro.to_csv(OUT / "study_macro_retrieval_metrics.csv", index=False)

    # Stable hash ordering avoids accidental dependence on input order.
    query = query.assign(_score=query.gsm.map(lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest()))
    cap_membership = []
    capped_metrics = []
    capped_composition = []
    for cap in [5, 10]:
        selected = (query.sort_values("_score").groupby("gse", sort=False, group_keys=False)
                    .head(cap).index.to_numpy())
        selected_query = query.loc[selected]
        cap_membership.extend({"cap": cap, "query_index": int(i), "gsm": query.loc[i, "gsm"],
                               "gse": query.loc[i, "gse"]} for i in selected)
        for method, sub in ranks[ranks.query_index.isin(selected)].groupby("method", sort=False):
            capped_metrics.append({"cap": cap, "method": method, "studies": sub.gse.nunique(),
                                   **metric_row(sub)})
        for dimension in ["gse", "source_label", "density_stratum"]:
            for value, count in selected_query[dimension].value_counts(dropna=False).items():
                capped_composition.append({"cap": cap, "dimension": dimension, "value": value,
                                           "samples": count, "fraction": count / len(selected_query)})
    pd.DataFrame(cap_membership).to_csv(OUT / "capped_study_membership.csv", index=False)
    pd.DataFrame(capped_metrics).to_csv(OUT / "capped_study_retrieval_metrics.csv", index=False)
    pd.DataFrame(capped_composition).to_csv(OUT / "capped_study_composition.csv", index=False)

    # Density summaries under macro weighting, caps, and dominant-study exclusion.
    density_rows = []
    cosine = ranks[ranks.method.eq("Cosine")].copy()
    analyses = {"all_micro": cosine,
                "leave_GSE234297_out": cosine[cosine.gse.ne("GSE234297")]}
    for cap in [5, 10]:
        member = {x["query_index"] for x in cap_membership if x["cap"] == cap}
        analyses[f"study_cap_{cap}"] = cosine[cosine.query_index.isin(member)]
    for name, sub in analyses.items():
        for stratum, part in sub.groupby("density_stratum", observed=True):
            density_rows.append({"analysis": name, "weighting": "query", "density_stratum": stratum,
                                 "queries": len(part), "studies": part.gse.nunique(),
                                 "r_at_1": np.mean(part.exact_rank <= 1),
                                 "failure_rate": np.mean(part.exact_rank > 1),
                                 "median_rank": np.median(part.exact_rank)})
    for stratum, part in cosine.groupby("density_stratum", observed=True):
        per = part.groupby("gse").exact_rank.agg(
            queries="size", r_at_1=lambda x: np.mean(x <= 1),
            failure_rate=lambda x: np.mean(x > 1), median_rank="median").reset_index()
        density_rows.append({"analysis": "all_study_macro", "weighting": "study",
                             "density_stratum": stratum, "queries": int(per.queries.sum()),
                             "studies": len(per), "r_at_1": per.r_at_1.mean(),
                             "failure_rate": per.failure_rate.mean(), "median_rank": per.median_rank.mean()})
    pd.DataFrame(density_rows).to_csv(OUT / "balanced_density_summary.csv", index=False)

    # Figures use only saved/new compact summaries.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(macro)); width = 0.18
    for j, metric in enumerate(["macro_r_at_1", "macro_r_at_5", "macro_r_at_10", "macro_r_at_100"]):
        ax.bar(x + (j - 1.5) * width, macro[metric], width, label=metric.replace("macro_", "").upper())
    ax.set_xticks(x, macro.method, rotation=18, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("Study-macro recall"); ax.legend(ncol=4, frameon=False)
    ax.set_title("Exact-pair retrieval with equal weight per GSE")
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"study_macro_retrieval.{ext}", dpi=220)
    plt.close(fig)

    dens = pd.DataFrame(density_rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot = dens.pivot(index="analysis", columns="density_stratum", values="failure_rate")
    pivot = pivot.reindex(columns=["sparse", "intermediate", "dense"])
    pivot.plot.bar(ax=ax, color=["#4c78a8", "#f2cf5b", "#e45756"])
    ax.set_ylabel("Cosine exact-pair failure rate"); ax.set_xlabel("")
    ax.set_title("Density association under study-balanced sensitivity analyses")
    ax.legend(title="Density", frameon=False); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"balanced_density_sensitivity.{ext}", dpi=220)
    plt.close(fig)

    provenance = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "random_seed": SEED,
        "analysis": "metadata-only provenance audit, cached overlap inventory, and saved-rank balancing",
        "inputs": {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in
                   [manifest_path, metadata_path, candidate_path, matches_path, final_path,
                    failures_path, HERE / "results/per_query_ranked_results.parquet",
                    HERE / "results/biological_validation/query_composition_and_density.parquet"]},
        "upstream_read_only": str(upstream), "cached_projects_read": read_projects,
        "cached_projects_failed_in_original_run": failures.project.nunique(),
        "limitations": [
            "The current upstream config differs from the preserved candidate/final tables; tables describe the executed run.",
            "Cached metadata overlap is an inventory, not proof of current recount3 availability.",
            "The original run failed to download/read 604 project metadata files, so its not_found statuses are not fully auditable.",
            "Mapped-multiple GSE samples were outside the original single-GSE eligibility protocol.",
            "No candidate expression was downloaded and no embedding was generated.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log(f"Completed audit in {time.time() - started:.1f}s")
    log_handle.close()


if __name__ == "__main__":
    main()
