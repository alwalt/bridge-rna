#!/usr/bin/env python3
"""Rank genes by marginal utility for predicting gene variation across samples.

This is intentionally separate from the existing within-sample whole-profile
ranking. Discovery uses one sample per unseen ARCHS4 study and evaluates 500-
and 1,000-visible-gene randomized panels without selecting a final Top-N panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from common import CONFIG, REPO_ROOT, RESULTS, WORK
from rank_informative_genes import build_design, gene_scores, seeded_rng

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct


VISIBLE_COUNTS = (500, 1000)
PROBE_FOLDS = 4
PANELS_PER_FOLD = 24
SAMPLES_PER_SPECIES = 500


def digest(values: pd.Series) -> str:
    text = "".join(f"{value}\n" for value in values.astype(str))
    return hashlib.sha256(text.encode()).hexdigest()


def load_balanced_cohort(species: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load the frozen one-sample-per-study discovery cohort and expression."""
    selected = pd.read_parquet(RESULTS / f"definitive_{species}_ranking_samples.parquet")
    if len(selected) != SAMPLES_PER_SPECIES:
        raise ValueError(f"{species}: expected {SAMPLES_PER_SPECIES} frozen samples, found {len(selected)}")
    if selected.sample_id.duplicated().any() or selected.gse_candidates_str.duplicated().any():
        raise ValueError(f"{species}: cohort is not one unique sample per unique study")
    valid = (selected.split.eq("unseen") & selected.study_exposure.eq("unseen_study")
             & selected.mapping_status.eq("mapped_single"))
    if not valid.all():
        raise ValueError(f"{species}: cohort violates unseen-sample/unseen-study criteria")
    full = np.load(WORK / f"{species}_information_discovery_log1p_tpm.npy", mmap_mode="r")
    matrix = np.asarray(full[selected.row_index.to_numpy(int)], dtype=np.float32)
    genes = pd.read_parquet(WORK / f"{species}_information_discovery_genes.parquet")
    if "model_index" not in genes and "native_index" in genes:
        genes = genes.rename(columns={"native_index": "model_index"})
    return matrix, genes, selected


def across_sample_gene_metrics(truth: np.ndarray, prediction: np.ndarray,
                               probe: np.ndarray, block_size: int = 512) -> tuple[float, float, float]:
    """Average per-gene metrics, where each gene is evaluated across samples."""
    pearson_blocks, spearman_blocks, mse_blocks = [], [], []
    for start in range(0, len(probe), block_size):
        idx = probe[start:start + block_size]
        left = np.asarray(truth[:, idx], dtype=np.float64)
        right = np.asarray(prediction[:, idx], dtype=np.float64)
        lc, rc = left - left.mean(0), right - right.mean(0)
        denom = np.sqrt((lc * lc).sum(0) * (rc * rc).sum(0))
        pearson = np.divide((lc * rc).sum(0), denom,
                            out=np.full(len(idx), np.nan), where=denom > 0)
        lr, rr = rankdata(left, axis=0), rankdata(right, axis=0)
        lrc, rrc = lr - lr.mean(0), rr - rr.mean(0)
        rank_denom = np.sqrt((lrc * lrc).sum(0) * (rrc * rrc).sum(0))
        spearman = np.divide((lrc * rrc).sum(0), rank_denom,
                             out=np.full(len(idx), np.nan), where=rank_denom > 0)
        pearson_blocks.append(pearson); spearman_blocks.append(spearman)
        mse_blocks.append(np.mean((left - right) ** 2, axis=0))
    return (float(np.nanmean(np.concatenate(pearson_blocks))),
            float(np.nanmean(np.concatenate(spearman_blocks))),
            float(np.nanmean(np.concatenate(mse_blocks))))


def run_size(species: str, matrix: np.ndarray, genes: pd.DataFrame, model: torch.nn.Module,
             device: torch.device, visible_count: int) -> None:
    eligible = np.flatnonzero(genes.observed.to_numpy(bool))
    design, panels = build_design(eligible, PROBE_FOLDS, PANELS_PER_FOLD, visible_count)
    fold_members = np.array_split(seeded_rng("probe-folds").permutation(eligible), PROBE_FOLDS)
    for condition in design.itertuples(index=False):
        visible = panels[(condition.fold, condition.replicate)]
        if np.intersect1d(fold_members[condition.fold], visible).size:
            raise AssertionError("Probe genes leaked into a visible panel")
    cache = WORK / "gene_inference_conditions" / species / f"visible_{visible_count}"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in design.itertuples(index=False):
        path = cache / f"fold{condition.fold:02d}_panel{condition.replicate:03d}.parquet"
        probe = np.sort(fold_members[condition.fold])
        if path.is_file():
            result = pd.read_parquet(path).iloc[0].to_dict()
            print(f"[{species} visible={visible_count}] reuse {path.name}", flush=True)
        else:
            visible = panels[(condition.fold, condition.replicate)]
            prediction = reconstruct(model,
                mask_except_panel(matrix, visible, float(CONFIG["mask_token"])),
                device, int(CONFIG["batch_size"]),
                f"gene-inference {species} visible={visible_count} "
                f"fold={condition.fold + 1}/{PROBE_FOLDS} panel={condition.replicate + 1}/{PANELS_PER_FOLD}")
            pearson, spearman, mse = across_sample_gene_metrics(matrix, prediction, probe)
            result = {"fold": condition.fold, "replicate": condition.replicate,
                      "visible_genes": visible_count, "pearson": pearson,
                      "spearman": spearman, "mse": mse, "samples": len(matrix),
                      "probe_genes": len(probe)}
            pd.DataFrame([result]).to_parquet(path, index=False)
        result["probe_indices"] = tuple(map(int, probe)); rows.append(result)
    condition_results = pd.DataFrame(rows)
    condition_results.drop(columns="probe_indices").to_parquet(
        RESULTS / f"gene_inference_{species}_visible{visible_count}_conditions.parquet", index=False)
    scores, attribution = gene_scores(condition_results, panels, eligible, PROBE_FOLDS)
    array = np.asarray(matrix)
    stats = pd.DataFrame({"model_index": np.arange(array.shape[1]),
        "expression_mean": array.mean(0), "expression_sd": array.std(0),
        "detection_fraction": (array > 0).mean(0)})
    ranking = genes.merge(scores, on="model_index", how="left").merge(stats, on="model_index", how="left")
    ranking["species"] = species; ranking["visible_genes"] = visible_count
    if ranking.loc[ranking.model_index.isin(eligible), "information_score"].isna().any():
        missing = ranking.loc[ranking.model_index.isin(eligible) & ranking.information_score.isna(), "gene"]
        raise RuntimeError(f"{species} visible={visible_count}: {len(missing)} eligible genes lack attribution")
    ranking = ranking.sort_values("information_rank", na_position="last")
    ranking.to_parquet(RESULTS / f"gene_inference_{species}_visible{visible_count}_ranking.parquet", index=False)
    attribution["species"] = species; attribution["visible_genes"] = visible_count
    attribution.to_parquet(RESULTS / f"gene_inference_{species}_visible{visible_count}_attribution.parquet", index=False)


def worker(species: str, device_name: str) -> None:
    matrix, genes, selected = load_balanced_cohort(species)
    print(f"[{species}] balanced discovery: {len(selected)} samples, "
          f"{selected.gse_candidates_str.nunique()} studies", flush=True)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    for visible_count in VISIBLE_COUNTS:
        run_size(species, matrix, genes, model, device, visible_count)


def summarize(species_names: list[str]) -> None:
    stability = []
    performance_rows = []
    for species in species_names:
        rankings = {size: pd.read_parquet(
            RESULTS / f"gene_inference_{species}_visible{size}_ranking.parquet") for size in VISIBLE_COUNTS}
        merged = rankings[500][["model_index", "gene", "information_score"]].merge(
            rankings[1000][["model_index", "information_score"]], on="model_index",
            suffixes=("_visible500", "_visible1000")).dropna()
        merged["consensus_score"] = (merged.information_score_visible500.rank(pct=True)
                                      + merged.information_score_visible1000.rank(pct=True)) / 2
        merged["gene_inference_rank"] = merged.consensus_score.rank(ascending=False, method="min").astype("Int64")
        merged = merged.sort_values("gene_inference_rank")
        merged.to_parquet(RESULTS / f"gene_inference_{species}_consensus_ranking.parquet", index=False)
        rho = spearmanr(merged.information_score_visible500, merged.information_score_visible1000).statistic
        top500 = set(merged.nlargest(1000, "information_score_visible500").model_index)
        top1000 = set(merged.nlargest(1000, "information_score_visible1000").model_index)
        stability.append({"species": species, "eligible_genes": len(merged),
            "visible_size_score_spearman": rho, "top1000_overlap": len(top500 & top1000),
            "top1000_jaccard": len(top500 & top1000) / len(top500 | top1000)})
        for size in VISIBLE_COUNTS:
            conditions = pd.read_parquet(
                RESULTS / f"gene_inference_{species}_visible{size}_conditions.parquet")
            performance_rows.append({"species": species, "visible_genes": size,
                "conditions": len(conditions), "samples": int(conditions.samples.iloc[0]),
                "pearson_mean": conditions.pearson.mean(), "pearson_sd": conditions.pearson.std(ddof=1),
                "spearman_mean": conditions.spearman.mean(), "spearman_sd": conditions.spearman.std(ddof=1),
                "mse_mean": conditions.mse.mean(), "mse_sd": conditions.mse.std(ddof=1)})
    summary = pd.DataFrame(stability)
    summary.to_parquet(RESULTS / "gene_inference_ranking_stability.parquet", index=False)
    summary.to_csv(RESULTS / "gene_inference_ranking_stability.csv", index=False)
    performance = pd.DataFrame(performance_rows).sort_values(["species", "visible_genes"])
    performance.to_parquet(RESULTS / "gene_inference_condition_performance_summary.parquet", index=False)
    performance.to_csv(RESULTS / "gene_inference_condition_performance_summary.csv", index=False)
    comparisons = []
    for row in summary.itertuples(index=False):
        expected = 1000 * 1000 / row.eligible_genes
        comparisons.append({"comparison": f"{row.species}: visible 500 vs 1000",
            "comparison_type": "visible_size", "genes_compared": row.eligible_genes,
            "spearman": row.visible_size_score_spearman, "top_n": 1000,
            "top_n_overlap": row.top1000_overlap, "expected_overlap": expected,
            "overlap_enrichment": row.top1000_overlap / expected,
            "top_n_jaccard": row.top1000_jaccard})
    if {"human", "mouse"}.issubset(species_names):
        for size in VISIBLE_COUNTS:
            human = pd.read_parquet(RESULTS / f"gene_inference_human_visible{size}_ranking.parquet")
            mouse = pd.read_parquet(RESULTS / f"gene_inference_mouse_visible{size}_ranking.parquet")
            merged = human[["model_index", "information_score"]].merge(
                mouse[["model_index", "information_score"]], on="model_index",
                suffixes=("_human", "_mouse")).dropna()
            human_top = set(merged.nlargest(1000, "information_score_human").model_index)
            mouse_top = set(merged.nlargest(1000, "information_score_mouse").model_index)
            overlap = len(human_top & mouse_top); expected = 1000 * 1000 / len(merged)
            comparisons.append({"comparison": f"human vs mouse: visible {size}",
                "comparison_type": "cross_species", "genes_compared": len(merged),
                "spearman": spearmanr(merged.information_score_human,
                                      merged.information_score_mouse).statistic,
                "top_n": 1000, "top_n_overlap": overlap, "expected_overlap": expected,
                "overlap_enrichment": overlap / expected,
                "top_n_jaccard": overlap / (2000 - overlap)})
        human = pd.read_parquet(RESULTS / "gene_inference_human_consensus_ranking.parquet")
        mouse = pd.read_parquet(RESULTS / "gene_inference_mouse_consensus_ranking.parquet")
        merged = human[["model_index", "consensus_score", "gene_inference_rank"]].merge(
            mouse[["model_index", "consensus_score", "gene_inference_rank"]], on="model_index",
            suffixes=("_human", "_mouse"))
        human_top = set(merged.nsmallest(1000, "gene_inference_rank_human").model_index)
        mouse_top = set(merged.nsmallest(1000, "gene_inference_rank_mouse").model_index)
        overlap = len(human_top & mouse_top); expected = 1000 * 1000 / len(merged)
        comparisons.append({"comparison": "human vs mouse: size consensus",
            "comparison_type": "cross_species_consensus", "genes_compared": len(merged),
            "spearman": spearmanr(merged.consensus_score_human,
                                  merged.consensus_score_mouse).statistic,
            "top_n": 1000, "top_n_overlap": overlap, "expected_overlap": expected,
            "overlap_enrichment": overlap / expected,
            "top_n_jaccard": overlap / (2000 - overlap)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_parquet(RESULTS / "gene_inference_comparison_summary.parquet", index=False)
    comparison.to_csv(RESULTS / "gene_inference_comparison_summary.csv", index=False)
    provenance = {"status": "exploratory_full-vocabulary_ranking", "objective":
        "marginal visible-gene utility for predicting individual masked genes across biological samples",
        "visible_counts": list(VISIBLE_COUNTS), "probe_folds": PROBE_FOLDS,
        "panels_per_fold": PANELS_PER_FOLD, "samples_per_species": SAMPLES_PER_SPECIES,
        "sample_selection": "frozen deterministic one-sample-per-unseen-study",
        "species": species_names, "benchmark_seed": int(CONFIG["benchmark_seed"]),
        "top_n_selected_during_discovery": None,
        "existing_rankings_modified": False}
    (RESULTS / "gene_inference_ranking_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nRanking stability\n" + summary.to_string(index=False), flush=True)
    print("\nPer-size reconstruction performance\n" + performance.to_string(index=False), flush=True)
    print("\nRanking comparisons\n" + comparison.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", choices=["human", "mouse"], default=["human", "mouse"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--worker-species", choices=["human", "mouse"])
    parser.add_argument("--worker-device")
    args = parser.parse_args()
    if args.worker_species:
        worker(args.worker_species, args.worker_device or "cuda:0"); return
    processes = []
    for index, species in enumerate(args.species):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-species", species,
                   "--worker-device", args.devices[index % len(args.devices)]]
        processes.append((species, subprocess.Popen(command)))
    started = time.monotonic()
    while processes:
        time.sleep(10); active = []
        for species, process in processes:
            code = process.poll()
            if code is None: active.append((species, process))
            elif code: raise subprocess.CalledProcessError(code, process.args)
            else: print(f"[orchestrator] {species} complete", flush=True)
        processes = active
        if processes and int(time.monotonic() - started) % 60 < 10:
            print(f"[orchestrator heartbeat] elapsed={(time.monotonic()-started)/60:.1f}m "
                  f"running={','.join(x[0] for x in processes)}", flush=True)
    summarize(args.species)


if __name__ == "__main__":
    main()
