#!/usr/bin/env python3
"""Pilot model-based gene ranking with cross-fitted randomized panel attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

from common import CONFIG, REPO_ROOT, RESULTS, WORK

sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct, score_masked_rows


def seeded_rng(label: str) -> np.random.Generator:
    token = f"{CONFIG['benchmark_seed']}|{label}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def build_design(eligible: np.ndarray, folds: int, panels_per_fold: int,
                 visible_count: int) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    shuffled = seeded_rng("probe-folds").permutation(eligible)
    fold_members = np.array_split(shuffled, folds)
    records, panels = [], {}
    for fold, probe in enumerate(fold_members):
        candidates = np.setdiff1d(eligible, probe, assume_unique=True)
        if visible_count >= len(candidates): raise ValueError("visible_genes must be smaller than candidate pool")
        usage = np.zeros(len(candidates), dtype=int)
        for replicate in range(panels_per_fold):
            rng = seeded_rng(f"fold={fold}|panel={replicate}")
            # Prefer the least-used genes, randomizing ties. This preserves exact
            # panel size while guaranteeing broad and nearly balanced coverage.
            chosen_positions = np.lexsort((rng.random(len(candidates)), usage))[:visible_count]
            usage[chosen_positions] += 1
            visible = np.sort(candidates[chosen_positions])
            panels[(fold, replicate)] = visible
            records.append({"fold": fold, "replicate": replicate,
                            "probe_genes": len(probe), "candidate_genes": len(candidates),
                            "visible_genes": len(visible)})
    return pd.DataFrame(records), panels


def gene_scores(design_results: pd.DataFrame, panels: dict[tuple[int, int], np.ndarray],
                eligible: np.ndarray, folds: int) -> pd.DataFrame:
    rows = []
    all_eligible = set(eligible.tolist())
    for fold in range(folds):
        frame = design_results.loc[design_results.fold.eq(fold)].sort_values("replicate")
        probe = set(frame.probe_indices.iloc[0])
        candidates = np.asarray(sorted(all_eligible - probe), dtype=int)
        included = np.zeros((len(frame), len(candidates)), dtype=bool)
        lookup = {gene: col for col, gene in enumerate(candidates)}
        for row, replicate in enumerate(frame.replicate):
            for gene in panels[(fold, int(replicate))]: included[row, lookup[int(gene)]] = True
        for metric in ("pearson", "spearman", "mse"):
            values = frame[metric].to_numpy(float)
            z = (values - values.mean()) / (values.std(ddof=0) or 1.0)
            for col, gene in enumerate(candidates):
                yes, no = included[:, col], ~included[:, col]
                raw = values[yes].mean() - values[no].mean() if yes.any() and no.any() else np.nan
                zdelta = z[yes].mean() - z[no].mean() if yes.any() and no.any() else np.nan
                if metric == "mse": raw, zdelta = -raw, -zdelta
                rows.append({"model_index": gene, "fold": fold, "metric": metric,
                             "inclusion_count": int(yes.sum()), "exclusion_count": int(no.sum()),
                             "improvement": raw, "standardized_improvement": zdelta})
    long = pd.DataFrame(rows)
    scores = long.groupby(["model_index", "metric"], as_index=False).agg(
        improvement=("improvement", "mean"),
        standardized_improvement=("standardized_improvement", "mean"),
        folds=("fold", "nunique"), inclusion_count=("inclusion_count", "sum"),
        exclusion_count=("exclusion_count", "sum"))
    wide = scores.pivot(index="model_index", columns="metric")
    wide.columns = [f"{metric}_{field}" for field, metric in wide.columns]
    wide = wide.reset_index()
    zcols = [f"{metric}_standardized_improvement" for metric in ("pearson", "spearman", "mse")]
    wide["information_score"] = wide[zcols].mean(axis=1)
    wide["information_rank"] = wide.information_score.rank(ascending=False, method="min").astype("Int64")
    return wide, long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, help="Override pilot discovery-sample count")
    args = parser.parse_args()
    settings = CONFIG["information_density_pilot"]
    sample_count = args.samples or int(settings["samples"])
    matrix = np.load(WORK / "human_discovery_log1p_tpm.npy", mmap_mode="r")[:sample_count]
    genes = pd.read_parquet(WORK / "human_discovery_genes.parquet")
    if "model_index" not in genes and "native_index" in genes:
        genes = genes.rename(columns={"native_index": "model_index"})
    eligible = np.flatnonzero(genes.observed.to_numpy(bool))
    folds, panel_count = int(settings["probe_folds"]), int(settings["panels_per_fold"])
    manifest, panels = build_design(eligible, folds, panel_count, int(settings["visible_genes"]))
    device_name = str(CONFIG["device"])
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", num_genes=15165, device=str(device))
    results = []
    fold_members = np.array_split(seeded_rng("probe-folds").permutation(eligible), folds)
    for condition in manifest.itertuples(index=False):
        visible = panels[(condition.fold, condition.replicate)]
        probe = np.sort(fold_members[condition.fold])
        masked = mask_except_panel(matrix, visible, float(CONFIG["mask_token"]))
        prediction = reconstruct(model, masked, device, int(CONFIG["batch_size"]),
            f"ranking fold={condition.fold + 1}/{folds} panel={condition.replicate + 1}/{panel_count}")
        pearson, spearman, mse = score_masked_rows(matrix, prediction, probe)
        results.append({"fold": condition.fold, "replicate": condition.replicate,
                        "probe_indices": tuple(map(int, probe)), "pearson": np.nanmean(pearson),
                        "spearman": np.nanmean(spearman), "mse": np.nanmean(mse)})
    design_results = pd.DataFrame(results)
    serializable = design_results.drop(columns="probe_indices")
    serializable.to_parquet(RESULTS / "pilot_information_condition_results.parquet", index=False)
    scores, long = gene_scores(design_results, panels, eligible, folds)
    stats = pd.DataFrame({"model_index": np.arange(matrix.shape[1]),
                          "expression_mean": np.asarray(matrix).mean(axis=0),
                          "expression_sd": np.asarray(matrix).std(axis=0),
                          "detection_fraction": (np.asarray(matrix) > 0).mean(axis=0)})
    output = genes.merge(scores, on="model_index", how="left").merge(stats, on="model_index", how="left")
    output = output.sort_values("information_rank", na_position="last")
    RESULTS.mkdir(parents=True, exist_ok=True)
    output.to_parquet(RESULTS / "pilot_informative_gene_ranking.parquet", index=False)
    output.to_csv(RESULTS / "pilot_informative_gene_ranking.csv", index=False)
    long.to_parquet(RESULTS / "pilot_informative_gene_attribution.parquet", index=False)
    manifest.to_parquet(RESULTS / "pilot_information_panel_manifest.parquet", index=False)
    provenance = {"method": "cross-fitted randomized panel marginal attribution",
                  "selection_species": "human", "selection_role": "discovery",
                  "samples": sample_count, "probe_folds": folds,
                  "panels_per_fold": panel_count, "visible_genes": int(settings["visible_genes"]),
                  "evaluation_cohorts_used_for_selection": False,
                  "score_direction": "larger is more informative; MSE sign inverted"}
    (RESULTS / "pilot_informative_gene_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output[["information_rank", "gene", "information_score", "expression_mean",
                  "expression_sd"]].head(25).to_string(index=False))


if __name__ == "__main__": main()
