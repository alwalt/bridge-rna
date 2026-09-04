#!/usr/bin/env python3
"""Integrated Gradients and enrichment for fixed Task 3C BridgeRNA modes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "task3d_mode_ig"
WORK = HERE / "work" / "task3d_mode_ig"
FIGURES = OUT / "figures"
GPROFILER = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

sys.path.insert(0, str(ROOT / "benchmarks/tcga_downstream/pipeline"))
sys.path.insert(0, str(ROOT))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_design() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[int, np.ndarray]]:
    clusters = pd.read_csv(RESULTS / "task3c_cluster_assignments.csv").sort_values("contrast_id")
    members = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv")
    manifest = pd.read_csv(RESULTS / "sample_manifest.csv")
    members = members.merge(manifest[["sample_id"]].reset_index(names="sample_index"),
                            on="sample_id", validate="many_to_one")
    archive = np.load(RESULTS / "task3b_bridgerna_response_vectors.npz", allow_pickle=True)
    response = pd.DataFrame(archive["delta_z"], index=archive["contrast_id"].astype(str))
    response = response.loc[clusters.contrast_id].to_numpy(np.float32)
    directions = {}
    for mode in (1, 2):
        rows = clusters.geometry_cluster.eq(mode).to_numpy()
        directions[mode] = unit(unit(response[rows]).mean(axis=0)).astype(np.float32)
    return clusters, members, np.load(HERE / "work/bridgerna_log1p_tpm_inputs.npy", mmap_mode="r"), directions


def score(model: torch.nn.Module, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return (model._encode_hidden(x).mean(1) * direction).sum(1)


def integrated_gradients(model: torch.nn.Module, values: np.ndarray, direction: np.ndarray,
                         device: torch.device, steps: int, path_batch: int) -> tuple[np.ndarray, float, float]:
    baseline = torch.zeros((1, len(values)), device=device)
    observed = torch.from_numpy(np.array(values, copy=True)).to(device).unsqueeze(0)
    target = torch.from_numpy(direction).to(device).unsqueeze(0)
    total = torch.zeros_like(baseline)
    alphas = (np.arange(steps, dtype=np.float32) + .5) / steps
    for start in range(0, steps, path_batch):
        alpha = torch.from_numpy(alphas[start:start + path_batch]).to(device).view(-1, 1)
        x = (baseline + alpha * (observed - baseline)).requires_grad_(True)
        value = score(model, x, target.expand(len(x), -1))
        total += torch.autograd.grad(value.sum(), x)[0].detach().sum(0, keepdim=True)
    attribution = ((observed - baseline) * total / steps)[0]
    with torch.no_grad():
        endpoint = score(model, observed, target)[0]
        origin = score(model, baseline, target)[0]
    return attribution.cpu().numpy().astype(np.float32), float(endpoint - origin), float(attribution.sum())


def worker(mode: int, device_name: str, steps: int, path_batch: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    clusters, members, expression, directions = load_design()
    selected = clusters[clusters.geometry_cluster.eq(mode)].sort_values("contrast_id")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = load_frozen_encoder(device)
    direction = directions[mode]
    effects, completeness = [], []
    started = time.monotonic()
    for number, contrast in enumerate(selected.contrast_id, 1):
        group = members[members.contrast_id.eq(contrast)]
        role_values = {}
        for condition in ("FLT", "GC"):
            indices = group.loc[group.condition.eq(condition), "sample_index"].to_numpy(int)
            profile = np.asarray(expression[indices]).mean(axis=0).astype(np.float32)
            attr, score_delta, attr_sum = integrated_gradients(model, profile, direction, device, steps, path_batch)
            role_values[condition] = attr
            completeness.append({"mode": mode, "contrast_id": contrast, "condition": condition,
                                 "samples": len(indices), "score_endpoint_minus_zero": score_delta,
                                 "attribution_sum": attr_sum, "completeness_delta": attr_sum - score_delta,
                                 "relative_completeness_error": abs(attr_sum - score_delta) / max(abs(score_delta), 1e-12)})
        effects.append(role_values["FLT"] - role_values["GC"])
        elapsed = time.monotonic() - started
        eta = elapsed / number * (len(selected) - number)
        log(f"Mode {mode}: contrast {number}/{len(selected)} complete; elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")
    np.save(WORK / f"mode_{mode}_contrast_ig.npy", np.stack(effects))
    selected[["contrast_id", "OSD", "mission", "geometry_cluster"]].to_csv(
        WORK / f"mode_{mode}_contrast_order.csv", index=False)
    pd.DataFrame(completeness).to_csv(OUT / f"mode_{mode}_ig_completeness.csv", index=False)
    log(f"Mode {mode} worker complete on {device}")


def mouse_symbol_map(genes: list[str]) -> dict[str, str]:
    table = pd.read_csv(ROOT / "data/ensembl/orthologs_one2one.txt", sep="\t", low_memory=False)
    table["human"] = table["Human gene name"].astype(str).str.upper()
    table["mouse"] = table["Gene name"].astype(str)
    pairs = table.dropna(subset=["human", "mouse"])[["human", "mouse"]].drop_duplicates()
    unique = pairs.groupby("human").filter(lambda frame: frame.mouse.nunique() == 1).drop_duplicates("human")
    mapping = unique.set_index("human").mouse.to_dict()
    return {gene: mapping.get(gene, "") for gene in genes}


def combine_and_rank(directions: dict[int, np.ndarray]) -> dict[str, list[str]]:
    genes = load_canonical_genes(ROOT / "data/ensembl/canonical_genes.csv")
    if len(genes) != 15165 or len(set(genes)) != 15165:
        raise AssertionError("The IG/enrichment universe must be exactly 15,165 unique genes")
    mouse = mouse_symbol_map(genes)
    np.savez(OUT / "mode_response_directions.npz", mode_1=directions[1], mode_2=directions[2])
    ranking_frames, queries = [], {}
    effect_archive = {}
    for mode in (1, 2):
        effects = np.load(WORK / f"mode_{mode}_contrast_ig.npy")
        order_meta = pd.read_csv(WORK / f"mode_{mode}_contrast_order.csv")
        effect_archive[f"mode_{mode}"] = effects
        mean = effects.mean(axis=0); absolute = np.abs(effects)
        sd = effects.std(axis=0, ddof=1); sem = sd / np.sqrt(len(effects))
        nonzero = np.sign(mean) != 0
        consistency = np.where(nonzero, (np.sign(effects) == np.sign(mean)).mean(axis=0), 0)
        order = np.lexsort((-consistency, -np.abs(mean)))
        frame = pd.DataFrame({"mode": mode, "gene_symbol_human": genes,
                              "gene_symbol_mouse": [mouse[x] for x in genes],
                              "mean_ig_change": mean, "absolute_mean_ig_change": np.abs(mean),
                              "mean_absolute_contrast_ig": absolute.mean(axis=0), "sd_ig_change": sd,
                              "sem_ig_change": sem, "direction_consistency": consistency})
        frame["rank"] = np.empty(len(genes), dtype=int); frame.loc[order, "rank"] = np.arange(1, len(genes) + 1)
        frame = frame.sort_values("rank")
        ranking_frames.append(frame)
        queries[f"mode_{mode}_top100"] = frame.head(100).gene_symbol_human.tolist()
        queries[f"mode_{mode}_top250"] = frame.head(250).gene_symbol_human.tolist()
        # Per-contrast scores remain compact in NPZ; its exact row order is saved alongside it.
        order_meta.to_csv(OUT / f"mode_{mode}_contrast_order.csv", index=False)
    np.savez_compressed(OUT / "all_contrast_ig_changes.npz", gene_symbol=np.asarray(genes), **effect_archive)
    rankings = pd.concat(ranking_frames, ignore_index=True)
    rankings.to_parquet(OUT / "mode_ig_gene_rankings.parquet", index=False)
    rankings.to_csv(OUT / "mode_ig_gene_rankings.csv.gz", index=False, compression="gzip")
    top1, top2 = set(queries["mode_1_top100"]), set(queries["mode_2_top100"])
    queries["shared_top100"] = sorted(top1 & top2)
    queries["mode_1_specific_top100"] = [x for x in queries["mode_1_top100"] if x not in top2]
    queries["mode_2_specific_top100"] = [x for x in queries["mode_2_top100"] if x not in top1]
    membership = pd.DataFrame({"gene_symbol_human": genes,
                               "mode_1_top100": [x in top1 for x in genes],
                               "mode_2_top100": [x in top2 for x in genes]})
    membership["relationship"] = np.select(
        [membership.mode_1_top100 & membership.mode_2_top100, membership.mode_1_top100,
         membership.mode_2_top100], ["shared", "mode_1_specific", "mode_2_specific"], default="neither")
    membership.to_csv(OUT / "mode_top100_overlap_genes.csv", index=False)
    pd.DataFrame([{"top_n": n,
                   "mode_1_genes": n, "mode_2_genes": n,
                   "shared_genes": len(set(queries[f'mode_1_top{n}']) & set(queries[f'mode_2_top{n}'])),
                   "jaccard": len(set(queries[f'mode_1_top{n}']) & set(queries[f'mode_2_top{n}'])) /
                              len(set(queries[f'mode_1_top{n}']) | set(queries[f'mode_2_top{n}']))}
                  for n in (100, 250)]).to_csv(OUT / "mode_gene_overlap_summary.csv", index=False)
    plot_rankings(rankings)
    return queries


def plot_rankings(rankings: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), layout="constrained")
    for ax, mode in zip(axes, (1, 2)):
        frame = rankings.query("mode == @mode").head(30).sort_values("absolute_mean_ig_change")
        ax.barh(frame.gene_symbol_human, frame.mean_ig_change,
                color=np.where(frame.mean_ig_change >= 0, "#D55E00", "#0072B2"))
        ax.set(title=f"Mode {mode}: consensus Integrated Gradients", xlabel="Mean FLT − GC attribution")
    fig.savefig(FIGURES / "mode_consensus_top30_genes.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURES / "mode_consensus_top30_genes.pdf", bbox_inches="tight")
    plt.close(fig)


def enrich(queries: dict[str, list[str]]) -> None:
    background = load_canonical_genes(ROOT / "data/ensembl/canonical_genes.csv")
    payload = {"organism": "hsapiens", "query": queries, "sources": ["GO:BP", "KEGG", "REAC"],
               "user_threshold": .05, "domain_scope": "custom", "background": background,
               "no_evidences": False}
    response = requests.post(GPROFILER, json=payload, timeout=600)
    response.raise_for_status(); raw = response.json()
    (OUT / "mode_ig_enrichment_raw.json").write_text(json.dumps(raw, indent=2) + "\n")
    result = pd.DataFrame(raw.get("result", []))
    columns = ["query", "source", "native", "name", "p_value", "significant", "term_size",
               "query_size", "intersection_size", "effective_domain_size", "precision", "recall", "intersection"]
    if not result.empty:
        result = result[[x for x in columns if x in result]].sort_values(["query", "source", "p_value"])
    result.to_parquet(OUT / "mode_ig_enrichment.parquet", index=False)
    result.to_csv(OUT / "mode_ig_enrichment.csv", index=False)
    summary = []
    for query, genes in queries.items():
        for source in ("GO:BP", "KEGG", "REAC"):
            z = result[(result["query"] == query) & (result["source"] == source)] if not result.empty else result
            summary.append({"gene_set": query, "source": source, "query_genes": len(genes),
                            "background_genes": len(background), "significant_terms": len(z)})
    pd.DataFrame(summary).to_csv(OUT / "mode_ig_enrichment_summary.csv", index=False)
    if not result.empty:
        top = result.groupby(["query", "source"], group_keys=False).head(5).copy()
        top["label"] = top["query"].str.replace("_", " ") + " | " + top.source + " | " + top.name.str.slice(0, 62)
        top = top.sort_values("p_value", ascending=False)
        fig, ax = plt.subplots(figsize=(13, max(7, .23 * len(top))), layout="constrained")
        ax.scatter(-np.log10(top.p_value.clip(lower=np.finfo(float).tiny)), np.arange(len(top)),
                   s=25 + 15 * top.intersection_size,
                   c=top.source.map({"GO:BP": "#4C78A8", "KEGG": "#F58518", "REAC": "#54A24B"}))
        ax.set(yticks=np.arange(len(top)), yticklabels=top.label,
               xlabel="−log10 adjusted p-value", title="Task 3D mode-specific IG enrichment")
        fig.savefig(FIGURES / "mode_ig_enrichment_top_terms.png", dpi=400, bbox_inches="tight")
        fig.savefig(FIGURES / "mode_ig_enrichment_top_terms.pdf", bbox_inches="tight")
        plt.close(fig)


def orchestrate(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    _, _, _, directions = load_design()
    commands, processes = [], []
    for mode, device in zip((1, 2), args.devices):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-mode", str(mode),
                   "--device", device, "--ig-steps", str(args.ig_steps), "--path-batch", str(args.path_batch)]
        commands.append(command); processes.append(subprocess.Popen(command))
        log(f"Started Mode {mode} worker on {device}, PID={processes[-1].pid}")
    failed = []
    for mode, process in zip((1, 2), processes):
        code = process.wait()
        if code: failed.append((mode, code))
    if failed:
        raise RuntimeError(f"IG worker failure(s): {failed}")
    queries = combine_and_rank(directions)
    log("IG ranking complete; starting GO/KEGG/Reactome enrichment")
    enrich(queries)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": "model/r7hnr92k/best_model.pt",
                  "encoder_frozen": True, "target": "dot product of mean-pooled 512-D embedding and fixed unit mode direction",
                  "mode_direction": "unit-normalized mean of unit-normalized Task 3B FLT-minus-GC vectors in each fixed Task 3C cluster",
                  "ig_input": "15,165-gene natural log1p(TPM) model input",
                  "ig_baseline": "all-zero log1p(TPM) profile", "ig_steps": args.ig_steps,
                  "integration_rule": "midpoint Riemann", "contrast_attribution": "IG(mean FLT profile) minus IG(mean GC profile)",
                  "enrichment": "g:Profiler GO:BP, KEGG, Reactome with custom exact 15,165-gene background",
                  "multiple_testing": "g:SCS", "no_edgeR_GSEA_batch_correction_or_deletion": True,
                  "worker_commands": commands}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    log("Task 3D mode IG and enrichment complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--worker-mode", type=int, choices=[1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--path-batch", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_mode:
        worker(parsed.worker_mode, parsed.device, parsed.ig_steps, parsed.path_batch)
    else:
        orchestrate(parsed)
