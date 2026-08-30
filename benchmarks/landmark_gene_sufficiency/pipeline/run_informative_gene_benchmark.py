#!/usr/bin/env python3
"""Run resumable human/mouse informative-gene attribution across available GPUs."""

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
from scipy.stats import hypergeom, pearsonr, spearmanr

from common import CONFIG, REPO_ROOT, RESULTS, WORK
from rank_informative_genes import build_design, gene_scores, seeded_rng

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct, score_masked_rows


def digest(values: pd.Series) -> str:
    """Hash an ordered identifier series for reproducibility checks."""
    text = "".join(f"{value}\n" for value in values.astype(str))
    return hashlib.sha256(text.encode()).hexdigest()


def select_definitive_samples(species: str, sample_count: int) -> pd.DataFrame:
    """Select one deterministic sample from each of ``sample_count`` studies."""
    samples = pd.read_parquet(WORK / f"{species}_information_discovery_samples.parquet")
    cohort = pd.read_parquet(RESULTS / "information_discovery_cohort.parquet")
    cohort = cohort.loc[cohort.species.eq(species), ["sample_id", "gse_candidates_str",
                                                     "split", "study_exposure", "mapping_status"]]
    candidates = samples.merge(cohort, on="sample_id", how="left", validate="one_to_one")
    if candidates.gse_candidates_str.isna().any():
        raise ValueError(f"{species}: discovery samples are missing GSE metadata")
    expected = candidates.split.eq("unseen") & candidates.study_exposure.eq("unseen_study") \
        & candidates.mapping_status.eq("mapped_single")
    if not expected.all():
        raise ValueError(f"{species}: definitive candidates violate the frozen discovery criteria")
    rng = seeded_rng(f"definitive-study-balanced|{species}")
    candidates = candidates.assign(_tie_break=rng.random(len(candidates)))
    representatives = (candidates.sort_values(["gse_candidates_str", "_tie_break"])
                        .drop_duplicates("gse_candidates_str"))
    if len(representatives) < sample_count:
        raise ValueError(f"{species}: only {len(representatives):,} eligible studies for {sample_count:,} samples")
    selected = representatives.iloc[rng.permutation(len(representatives))[:sample_count]].copy()
    selected = selected.drop(columns="_tie_break").sort_values("row_index").reset_index(drop=True)
    if selected.sample_id.duplicated().any() or selected.gse_candidates_str.duplicated().any():
        raise AssertionError("Definitive ranking selection must contain unique samples and studies")
    selected.to_parquet(RESULTS / f"definitive_{species}_ranking_samples.parquet", index=False)
    provenance = {
        "species": species,
        "selection": "deterministic one-sample-per-study",
        "source_pool": str(WORK / f"{species}_information_discovery_samples.parquet"),
        "criteria": {"split": "unseen", "study_exposure": "unseen_study",
                     "mapping_status": "mapped_single"},
        "samples": len(selected), "studies": selected.gse_candidates_str.nunique(),
        "ordered_sample_sha256": digest(selected.sample_id),
        "benchmark_seed": int(CONFIG["benchmark_seed"]),
    }
    (RESULTS / f"definitive_{species}_ranking_sample_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    return selected


def worker(species: str, device_name: str, mode: str) -> None:
    settings = CONFIG[f"information_density_{mode}"]
    if mode == "pilot":
        samples = int(settings["samples"])
        matrix_path = WORK / "human_discovery_log1p_tpm.npy"
        genes_path = WORK / "human_discovery_genes.parquet"
    else:
        samples = int(settings["ranking_samples_per_species"])
        matrix_path = WORK / f"{species}_information_discovery_log1p_tpm.npy"
        genes_path = WORK / f"{species}_information_discovery_genes.parquet"
    full_matrix = np.load(matrix_path, mmap_mode="r")
    if mode == "definitive":
        selected = select_definitive_samples(species, samples)
        matrix = np.asarray(full_matrix[selected.row_index.to_numpy(int)])
        print(f"[{species}] definitive cohort: {len(selected):,} samples from "
              f"{selected.gse_candidates_str.nunique():,} studies", flush=True)
    else:
        matrix = full_matrix[:samples]
    genes = pd.read_parquet(genes_path)
    if "model_index" not in genes and "native_index" in genes:
        genes = genes.rename(columns={"native_index": "model_index"})
    eligible = np.flatnonzero(genes.observed.to_numpy(bool))
    folds, count = int(settings["probe_folds"]), int(settings["panels_per_fold"])
    design, panels = build_design(eligible, folds, count, int(settings["visible_genes"]))
    fold_members = np.array_split(seeded_rng("probe-folds").permutation(eligible), folds)
    for condition in design.itertuples(index=False):
        overlap = np.intersect1d(fold_members[condition.fold],
                                 panels[(condition.fold, condition.replicate)])
        if overlap.size:
            raise AssertionError(f"Probe/visible leakage in fold {condition.fold}, "
                                 f"panel {condition.replicate}: {overlap[:5].tolist()}")
    cache = WORK / f"information_conditions_{mode}" / species
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    rows = []
    for condition in design.itertuples(index=False):
        path = cache / f"fold{condition.fold:02d}_panel{condition.replicate:03d}.parquet"
        probe = np.sort(fold_members[condition.fold])
        if path.is_file():
            result = pd.read_parquet(path).iloc[0].to_dict()
            print(f"[{species} {device}] reuse {path.name}", flush=True)
        else:
            visible = panels[(condition.fold, condition.replicate)]
            masked = mask_except_panel(matrix, visible, float(CONFIG["mask_token"]))
            prediction = reconstruct(model, masked, device, int(CONFIG["batch_size"]),
                f"{species} fold={condition.fold + 1}/{folds} panel={condition.replicate + 1}/{count}")
            pearson, spearman, mse = score_masked_rows(matrix, prediction, probe)
            result = {"fold": condition.fold, "replicate": condition.replicate,
                      "pearson": np.nanmean(pearson), "spearman": np.nanmean(spearman),
                      "mse": np.nanmean(mse), "samples": samples}
            pd.DataFrame([result]).to_parquet(path, index=False)
        result["probe_indices"] = tuple(map(int, probe)); rows.append(result)
    scores, attribution = gene_scores(pd.DataFrame(rows), panels, eligible, folds)
    array = np.asarray(matrix)
    stats = pd.DataFrame({"model_index": np.arange(array.shape[1]),
        "expression_mean": array.mean(axis=0), "expression_sd": array.std(axis=0),
        "detection_fraction": (array > 0).mean(axis=0)})
    ranking = genes.merge(scores, on="model_index", how="left").merge(stats, on="model_index", how="left")
    ranking["species"] = species
    ranking = ranking.sort_values("information_rank", na_position="last")
    ranking.to_parquet(RESULTS / f"{mode}_{species}_informative_gene_ranking.parquet", index=False)
    attribution["species"] = species
    attribution.to_parquet(RESULTS / f"{mode}_{species}_informative_gene_attribution.parquet", index=False)
    print(f"[{species}] ranking complete: {ranking.information_score.notna().sum():,} genes", flush=True)


def summarize(mode: str, species: list[str]) -> None:
    rankings = {name: pd.read_parquet(RESULTS / f"{mode}_{name}_informative_gene_ranking.parquet")
                for name in species}
    l1000 = pd.read_parquet(RESULTS / "l1000_model_mapping.parquet")
    l1000_indices = set(l1000.loc[l1000.jointly_evaluable, "model_index"].dropna().astype(int))
    n = len(l1000_indices)
    rows, property_rows = [], []
    for name, frame in rankings.items():
        eligible = frame.loc[frame.information_score.notna()].copy()
        top = eligible.nsmallest(n, "information_rank")
        overlap = set(top.model_index.astype(int)) & l1000_indices
        expected = n * len(l1000_indices) / len(eligible)
        pvalue = hypergeom.sf(len(overlap) - 1, len(eligible), len(l1000_indices), n)
        rows.append({"comparison": f"{name}_top{n}_vs_l1000", "genes_a": n,
                     "genes_b": len(l1000_indices), "overlap": len(overlap),
                     "jaccard": len(overlap) / len(set(top.model_index) | l1000_indices),
                     "expected_overlap": expected, "enrichment": len(overlap) / expected,
                     "hypergeometric_p": pvalue})
        for group, subset in (("top", top), ("background", eligible)):
            property_rows.append({"species": name, "group": group, "genes": len(subset),
                "expression_mean": subset.expression_mean.mean(), "expression_sd": subset.expression_sd.mean(),
                "detection_fraction": subset.detection_fraction.mean()})
    if {"human", "mouse"}.issubset(rankings):
        merged = rankings["human"][["model_index", "information_score"]].merge(
            rankings["mouse"][["model_index", "information_score"]], on="model_index",
            suffixes=("_human", "_mouse")).dropna()
        human_top = set(rankings["human"].nsmallest(n, "information_rank").model_index.astype(int))
        mouse_top = set(rankings["mouse"].nsmallest(n, "information_rank").model_index.astype(int))
        overlap = human_top & mouse_top
        rows.append({"comparison": f"human_top{n}_vs_mouse_top{n}", "genes_a": n, "genes_b": n,
            "overlap": len(overlap), "jaccard": len(overlap) / len(human_top | mouse_top),
            "expected_overlap": n * n / len(merged), "enrichment": len(overlap) / (n * n / len(merged)),
            "hypergeometric_p": hypergeom.sf(len(overlap) - 1, len(merged), n, n),
            "pearson_score_correlation": pearsonr(merged.information_score_human, merged.information_score_mouse).statistic,
            "spearman_rank_correlation": spearmanr(merged.information_score_human, merged.information_score_mouse).statistic})
    pd.DataFrame(rows).to_parquet(RESULTS / f"{mode}_informative_gene_overlap_summary.parquet", index=False)
    pd.DataFrame(property_rows).to_parquet(RESULTS / f"{mode}_informative_gene_property_summary.parquet", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "final", "definitive"], default="final")
    parser.add_argument("--species", nargs="+", choices=["human", "mouse"], default=["human", "mouse"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--worker-species", choices=["human", "mouse"])
    parser.add_argument("--worker-device")
    args = parser.parse_args()
    if args.worker_species:
        worker(args.worker_species, args.worker_device or "cuda:0", args.mode); return
    if args.mode in {"final", "definitive"}:
        required = [WORK / f"{s}_information_discovery_log1p_tpm.npy" for s in args.species]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError("Prepare final discovery matrices first with prepare_information_expression.py")
    processes = []
    for index, name in enumerate(args.species):
        device = args.devices[index % len(args.devices)]
        command = [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode,
                   "--worker-species", name, "--worker-device", device]
        processes.append((name, device, subprocess.Popen(command)))
    started = time.monotonic()
    while processes:
        time.sleep(min(args.heartbeat_seconds, 10))
        remaining = []
        for name, device, process in processes:
            code = process.poll()
            if code is None: remaining.append((name, device, process))
            elif code: raise subprocess.CalledProcessError(code, process.args)
            else: print(f"[orchestrator] {name} finished on {device}", flush=True)
        processes = remaining
        if processes and int(time.monotonic() - started) % args.heartbeat_seconds < 10:
            print(f"[orchestrator heartbeat] elapsed={(time.monotonic()-started)/60:.1f}m "
                  f"running={','.join(x[0] for x in processes)}", flush=True)
    summarize(args.mode, args.species)


if __name__ == "__main__": main()
