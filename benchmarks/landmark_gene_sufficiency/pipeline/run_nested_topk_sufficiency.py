#!/usr/bin/env python3
"""Exploratory nested Top-k curves for profile and gene-inference rankings."""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import CONFIG, REPO_ROOT, RESULTS, WORK, sha256
from run_external_per_gene_evaluation import columnwise_metrics

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct, score_masked_rows


SIZES = (1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
RANDOM_REPLICATES = 5
DATASETS = {"tcga_human": "human", "osdr_mouse_filtered": "mouse", "gtex_human": "human"}
DATASET_SAMPLES = 1000
CACHE = WORK / "nested_topk_conditions"
PREFIX = "nested_topk_sufficiency"


def build_panels(species: str) -> pd.DataFrame:
    profile_path = RESULTS / f"final_{species}_informative_gene_ranking.parquet"
    inference_path = RESULTS / f"gene_inference_{species}_consensus_ranking.parquet"
    profile = pd.read_parquet(profile_path).sort_values(["information_rank", "model_index"])
    inference = pd.read_parquet(inference_path).sort_values(["gene_inference_rank", "model_index"])
    genes = profile.set_index("model_index").gene.astype(str)
    universe = np.sort(profile.model_index.to_numpy(int))
    rows = []
    for size in SIZES:
        for panel_type, ranking, rank_column in (("profile_top", profile, "information_rank"),
                                                  ("gene_inference_top", inference, "gene_inference_rank")):
            selected = ranking.nsmallest(size, rank_column)
            for row in selected.itertuples(index=False):
                rows.append({"panel_id": f"{panel_type}_{size}", "panel_type": panel_type,
                    "visible_genes": size, "replicate": pd.NA, "model_index": int(row.model_index),
                    "gene": str(row.gene), "seed": pd.NA})
    seed = int(CONFIG["benchmark_seed"])
    for replicate in range(RANDOM_REPLICATES):
        panel_seed = seed + 40000 + replicate
        permutation = np.random.default_rng(panel_seed).permutation(universe)
        for size in SIZES:
            for index in np.sort(permutation[:size]):
                rows.append({"panel_id": f"random_{size}_r{replicate:02d}",
                    "panel_type": "random", "visible_genes": size, "replicate": replicate,
                    "model_index": int(index), "gene": str(genes.loc[index]), "seed": panel_seed})
    panels = pd.DataFrame(rows)
    counts = panels.groupby("panel_id").size()
    expected = {f"{kind}_{size}": size for size in SIZES
                for kind in ("profile_top", "gene_inference_top")}
    expected.update({f"random_{size}_r{rep:02d}": size
                     for rep in range(RANDOM_REPLICATES) for size in SIZES})
    if counts.to_dict() != expected or panels.duplicated(["panel_id", "model_index"]).any():
        raise AssertionError("Invalid nested-panel construction")
    for kind in ("profile_top", "gene_inference_top"):
        previous = set()
        for size in SIZES:
            current = set(panels.loc[panels.panel_id.eq(f"{kind}_{size}"), "model_index"])
            if not previous.issubset(current): raise AssertionError(f"{kind} panels are not nested")
            previous = current
    return panels


def run_dataset(dataset: str, device: torch.device, model: torch.nn.Module,
                shard_index: int = 0, shard_count: int = 1,
                batch_size: int | None = None) -> None:
    species = DATASETS[dataset]
    panels = build_panels(species)
    if shard_index == 0:
        panels.to_parquet(RESULTS / f"{PREFIX}_{dataset}_panel_manifest.parquet", index=False)
    if dataset == "gtex_human":
        complete_matrix = np.load(WORK / "gtex_model_log1p_tpm.npy", mmap_mode="r")
        complete_samples = pd.read_parquet(RESULTS / "gtex_model_samples.parquet")
        rng = np.random.default_rng(int(CONFIG["benchmark_seed"]) + 53000)
        selected_rows = np.sort(rng.choice(len(complete_matrix), size=DATASET_SAMPLES, replace=False))
        matrix = np.asarray(complete_matrix[selected_rows])
        samples = complete_samples.iloc[selected_rows].copy()
        samples.insert(0, "benchmark_row", np.arange(len(samples)))
        samples["selection_seed"] = int(CONFIG["benchmark_seed"]) + 53000
        if shard_index == 0:
            samples.to_parquet(RESULTS / f"{PREFIX}_{dataset}_samples.parquet", index=False)
        genes = pd.read_parquet(RESULTS / "gtex_model_tpm_mapping.parquet")
        eligible = set(genes.loc[genes.observed_in_gtex, "model_index"].astype(int))
    else:
        matrix = np.load(WORK / f"external_{dataset}_log1p_tpm.npy", mmap_mode="r")
        genes = pd.read_parquet(WORK / f"external_{dataset}_genes.parquet")
        eligible = set(genes.loc[genes.observed, "model_index"].astype(int))
    manifest = panels[["panel_id", "panel_type", "visible_genes", "replicate"]].drop_duplicates()
    cache = CACHE / dataset; cache.mkdir(parents=True, exist_ok=True)
    manifest = manifest.sort_values("panel_id").reset_index(drop=True)
    manifest = manifest.loc[manifest.index % shard_count == shard_index]
    for panel in manifest.itertuples(index=False):
        output = cache / f"{panel.panel_id}.parquet"
        if output.is_file():
            print(f"[{dataset}] reuse {output.name}", flush=True); continue
        visible = np.sort(panels.loc[panels.panel_id.eq(panel.panel_id), "model_index"].to_numpy(int))
        score_indices = np.asarray(sorted(eligible - set(visible)), dtype=int)
        prediction = reconstruct(model, mask_except_panel(matrix, visible, float(CONFIG["mask_token"])),
            device, batch_size or int(CONFIG["batch_size"]),
            f"nested Top-k {dataset} {panel.panel_id}")
        sample_pearson, sample_spearman, sample_mse = score_masked_rows(matrix, prediction, score_indices)
        gene = columnwise_metrics(matrix, prediction, score_indices)
        result = {"dataset": dataset, "species": species, "panel_id": panel.panel_id,
            "panel_type": panel.panel_type, "visible_genes": panel.visible_genes,
            "effective_visible_genes": len(eligible & set(visible)), "replicate": panel.replicate,
            "samples": len(matrix), "masked_observed_genes": len(score_indices),
            "sample_pearson": np.nanmean(sample_pearson),
            "sample_spearman": np.nanmean(sample_spearman), "sample_mse": np.nanmean(sample_mse),
            "gene_pearson": gene.pearson.mean(), "gene_spearman": gene.spearman.mean(),
            "gene_mse": gene.mse.mean()}
        pd.DataFrame([result]).to_parquet(output, index=False)
        print(f"[{dataset}] completed {panel.panel_id}", flush=True)


def plateau_k(frame: pd.DataFrame, metric: str) -> int:
    """First k reaching 90% of the observed Top-k improvement from 1,500 to 5,000."""
    ordered = frame.sort_values("visible_genes")
    values = ordered[metric].to_numpy(float)
    oriented = -values if metric.endswith("mse") else values
    start, end = oriented[0], oriented[-1]
    if end <= start: return int(ordered.visible_genes.iloc[-1])
    threshold = start + 0.9 * (end - start)
    indices = np.flatnonzero(oriented >= threshold)
    return int(ordered.visible_genes.iloc[indices[0]]) if len(indices) else int(ordered.visible_genes.iloc[-1])


def aggregate(datasets: list[str]) -> None:
    files = sorted(path for dataset in datasets for path in (CACHE / dataset).glob("*.parquet"))
    if len(files) != len(datasets) * len(SIZES) * (2 + RANDOM_REPLICATES):
        raise RuntimeError(f"Incomplete cache: found {len(files)} condition files")
    conditions = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    conditions.to_parquet(RESULTS / f"{PREFIX}_conditions.parquet", index=False)
    rows = []
    objectives = {"profile": ("profile_top", "sample"),
                  "gene_inference": ("gene_inference_top", "gene")}
    for dataset in datasets:
        source = conditions.loc[conditions.dataset.eq(dataset)]
        for objective, (top_type, metric_prefix) in objectives.items():
            for size in SIZES:
                subset = source.loc[source.visible_genes.eq(size)]
                for method, frame in (("Top-k", subset.loc[subset.panel_type.eq(top_type)]),
                                      ("Random-k", subset.loc[subset.panel_type.eq("random")])):
                    row = {"dataset": dataset, "objective": objective, "method": method,
                           "visible_genes": size, "replicates": len(frame)}
                    for metric in ("pearson", "spearman", "mse"):
                        values = frame[f"{metric_prefix}_{metric}"]
                        row[f"{metric}_mean"] = values.mean()
                        row[f"{metric}_sd"] = values.std(ddof=1)
                    rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_parquet(RESULTS / f"{PREFIX}_summary.parquet", index=False)
    summary.to_csv(RESULTS / f"{PREFIX}_summary.csv", index=False)
    plateaus = []
    for (dataset, objective), frame in summary.loc[summary.method.eq("Top-k")].groupby(["dataset", "objective"]):
        for metric in ("pearson_mean", "spearman_mean", "mse_mean"):
            plateaus.append({"dataset": dataset, "objective": objective,
                "metric": metric.removesuffix("_mean"), "plateau_k_90pct": plateau_k(frame, metric),
                "definition": "first k reaching 90% of observed 1500-to-5000 Top-k improvement"})
    plateau = pd.DataFrame(plateaus)
    plateau.to_parquet(RESULTS / f"{PREFIX}_plateau.parquet", index=False)
    plateau.to_csv(RESULTS / f"{PREFIX}_plateau.csv", index=False)
    figures = RESULTS / "figures"; figures.mkdir(exist_ok=True)
    for dataset in datasets:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
        for row_index, objective in enumerate(("profile", "gene_inference")):
            frame = summary[(summary.dataset.eq(dataset)) & (summary.objective.eq(objective))]
            for ax, metric in zip(axes[row_index], ("pearson", "spearman", "mse")):
                for method, color in (("Top-k", "#E45756"), ("Random-k", "#A0A0A0")):
                    curve = frame.loc[frame.method.eq(method)].sort_values("visible_genes")
                    ax.errorbar(curve.visible_genes, curve[f"{metric}_mean"],
                        yerr=curve[f"{metric}_sd"].fillna(0), marker="o", capsize=3,
                        color=color, label=method)
                ax.set(title=metric.capitalize(), xlabel="Visible genes")
                if ax is axes[row_index, 0]: ax.set_ylabel(objective.replace("_", " ").title())
        axes[0, 0].legend(frameon=False); fig.suptitle(f"Nested Top-k sufficiency: {dataset}")
        fig.tight_layout(); fig.savefig(figures / f"{PREFIX}_{dataset}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    provenance = {"status": "exploratory", "datasets": datasets, "visible_sizes": list(SIZES),
        "random_replicates": RANDOM_REPLICATES, "random_panels_nested_by_replicate": True,
        "dataset_samples": {"tcga_human": 1000, "osdr_mouse_filtered": 934,
            "gtex_human": DATASET_SAMPLES},
        "gtex_sampling": "deterministic sample without replacement; benchmark_seed + 53000",
        "profile_metric_scope": "masked genes within each sample",
        "gene_inference_metric_scope": "each masked gene across external samples",
        "rankings": {species: {
            "profile": {"path": str(RESULTS / f"final_{species}_informative_gene_ranking.parquet"),
                "sha256": sha256(RESULTS / f"final_{species}_informative_gene_ranking.parquet")},
            "gene_inference": {"path": str(RESULTS / f"gene_inference_{species}_consensus_ranking.parquet"),
                "sha256": sha256(RESULTS / f"gene_inference_{species}_consensus_ranking.parquet")}}
            for species in set(DATASETS[x] for x in datasets)},
        "reranking": False, "fine_tuning": False,
        "plateau_definition": "first k reaching 90% of observed 1500-to-5000 Top-k improvement"}
    (RESULTS / f"{PREFIX}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nPlateau estimates\n" + plateau.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    started = time.monotonic()
    for dataset in args.datasets:
        run_dataset(dataset, device, model, args.shard_index, args.shard_count, args.batch_size)
    # Preserve the completed datasets when a single new dataset is appended.
    if not args.no_aggregate:
        aggregate(list(DATASETS))
    print(f"Completed in {(time.monotonic()-started)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
