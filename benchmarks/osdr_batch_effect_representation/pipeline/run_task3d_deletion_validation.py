#!/usr/bin/env python3
"""Two-GPU deletion validation for fixed Task 3D IG rankings."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
IG_DIR = RESULTS / "task3d_mode_ig"
OUT = IG_DIR / "deletion_validation"
WORK = HERE / "work" / "task3d_deletion_validation"
FIGURES = OUT / "figures"
SIZES = [25, 50, 100, 250, 500, 1000]

sys.path.insert(0, str(ROOT / "benchmarks/tcga_downstream/pipeline"))
sys.path.insert(0, str(ROOT))
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes


def encode(model: torch.nn.Module, matrix: np.ndarray, mask: np.ndarray,
           device: torch.device, batch_size: int) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            x = torch.from_numpy(np.array(matrix[start:start + batch_size], copy=True)).to(device)
            if len(mask):
                x[:, mask] = -10.0
            output.append(model._encode_hidden(x).mean(1).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def contrast_responses(embeddings: np.ndarray, contrasts: pd.DataFrame,
                       members: pd.DataFrame) -> np.ndarray:
    responses = []
    for contrast in contrasts.contrast_id:
        group = members[members.contrast_id.eq(contrast)]
        flt = group.loc[group.condition.eq("FLT"), "sample_index"].to_numpy(int)
        gc = group.loc[group.condition.eq("GC"), "sample_index"].to_numpy(int)
        responses.append(embeddings[flt].mean(0) - embeddings[gc].mean(0))
    return np.stack(responses)


def worker(mode: int, device_name: str, random_replicates: int, seed: int,
           batch_size: int, events: mp.Queue) -> None:
    try:
        genes = load_canonical_genes(ROOT / "data/ensembl/canonical_genes.csv")
        rankings = pd.read_parquet(IG_DIR / "mode_ig_gene_rankings.parquet").query("mode == @mode").sort_values("rank")
        gene_index = {gene: i for i, gene in enumerate(genes)}
        ig_order = np.asarray([gene_index[x] for x in rankings.gene_symbol_human], dtype=int)
        expression = np.load(HERE / "work/bridgerna_log1p_tpm_inputs.npy", mmap_mode="r")
        manifest = pd.read_csv(RESULTS / "sample_manifest.csv")
        contrasts = pd.read_csv(RESULTS / "task3c_cluster_assignments.csv").sort_values("contrast_id").reset_index(drop=True)
        members = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv").merge(
            manifest[["sample_id"]].reset_index(names="sample_index"), on="sample_id", validate="many_to_one")
        direction = np.load(IG_DIR / "mode_response_directions.npz")[f"mode_{mode}"]
        original_archive = np.load(RESULTS / "task3b_bridgerna_response_vectors.npz", allow_pickle=True)
        original = pd.DataFrame(original_archive["delta_z"], index=original_archive["contrast_id"].astype(str))
        original = original.loc[contrasts.contrast_id].to_numpy(np.float32)
        assigned = contrasts.geometry_cluster.eq(mode).to_numpy()
        original_signal = original @ direction
        conditions = []
        for size in SIZES:
            conditions.append(("ig_ranked", 0, size, ig_order[:size]))
            for replicate in range(random_replicates):
                rng = np.random.default_rng(np.random.SeedSequence([seed, mode, size, replicate]))
                conditions.append(("random", replicate, size,
                                   np.sort(rng.choice(len(genes), size, replace=False))))
        device = torch.device(device_name)
        model = load_frozen_encoder(device)
        condition_rows, result_rows, panel_rows, response_vectors = [], [], [], []
        started = time.monotonic()
        for number, (panel_type, replicate, size, panel) in enumerate(conditions, 1):
            embeddings = encode(model, expression, panel, device, batch_size)
            response = contrast_responses(embeddings, contrasts, members)
            response_vectors.append(response)
            masked_signal = response @ direction
            condition_rows.append({"condition_index": number - 1, "mode": mode,
                                   "panel_type": panel_type, "replicate": replicate,
                                   "genes_masked": size})
            panel_rows.extend({"mode": mode, "panel_type": panel_type, "replicate": replicate,
                               "genes_masked": size, "model_index": int(index), "gene": genes[index]}
                              for index in panel)
            for index, contrast in contrasts[assigned].iterrows():
                result_rows.append({"mode": mode, "contrast_id": contrast.contrast_id,
                                    "OSD": contrast.OSD, "panel_type": panel_type,
                                    "replicate": replicate, "genes_masked": size,
                                    "original_signal": original_signal[index],
                                    "masked_signal": masked_signal[index],
                                    "fraction_signal_remaining": masked_signal[index] / original_signal[index],
                                    "absolute_signal_change": abs(masked_signal[index] - original_signal[index])})
            elapsed = time.monotonic() - started
            events.put(("done", mode, panel_type, replicate, size, number, len(conditions), elapsed))
        np.savez_compressed(WORK / f"mode_{mode}_masked_response_vectors.npz",
                            response_vectors=np.stack(response_vectors),
                            contrast_id=contrasts.contrast_id.to_numpy())
        pd.DataFrame(condition_rows).to_csv(WORK / f"mode_{mode}_condition_order.csv", index=False)
        pd.DataFrame(result_rows).to_parquet(WORK / f"mode_{mode}_results.parquet", index=False)
        pd.DataFrame(panel_rows).to_parquet(WORK / f"mode_{mode}_panels.parquet", index=False)
        events.put(("finished", mode))
    except Exception as error:
        events.put(("error", mode, repr(error)))
        raise


def aggregate(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_rows = []
    for keys, frame in results.groupby(["mode", "panel_type", "genes_masked", "replicate"]):
        mode, panel_type, size, replicate = keys
        replicate_rows.append({"mode": mode, "panel_type": panel_type, "genes_masked": size,
                               "replicate": replicate, "contrasts": len(frame),
                               "aggregate_fraction_remaining": frame.masked_signal.mean() / frame.original_signal.mean(),
                               "mean_contrast_fraction_remaining": frame.fraction_signal_remaining.mean(),
                               "median_contrast_fraction_remaining": frame.fraction_signal_remaining.median(),
                               "mean_absolute_signal_change": frame.absolute_signal_change.mean()})
    per_replicate = pd.DataFrame(replicate_rows)
    summary = per_replicate.groupby(["mode", "panel_type", "genes_masked"], as_index=False).agg(
        fraction_remaining_mean=("aggregate_fraction_remaining", "mean"),
        fraction_remaining_sd=("aggregate_fraction_remaining", "std"),
        mean_contrast_fraction_remaining=("mean_contrast_fraction_remaining", "mean"),
        mean_absolute_signal_change=("mean_absolute_signal_change", "mean"),
        replicates=("replicate", "nunique"), contrasts=("contrasts", "max"))
    return per_replicate, summary


def compare_panels(per_replicate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, size), frame in per_replicate.groupby(["mode", "genes_masked"]):
        ig = float(frame.loc[frame.panel_type.eq("ig_ranked"), "aggregate_fraction_remaining"].iloc[0])
        random = frame.loc[frame.panel_type.eq("random"), "aggregate_fraction_remaining"].to_numpy()
        ig_loss = 1 - ig; random_loss = 1 - random.mean()
        rows.append({"mode": mode, "genes_masked": size,
                     "ig_fraction_remaining": ig,
                     "ig_fraction_lost": ig_loss,
                     "random_fraction_remaining_mean": random.mean(),
                     "random_fraction_remaining_sd": random.std(ddof=1),
                     "random_fraction_lost_mean": random_loss,
                     "additional_fraction_lost_by_ig": ig_loss - random_loss,
                     "ig_to_random_loss_ratio": ig_loss / random_loss if random_loss > 0 else np.nan,
                     "random_replicates": len(random),
                     "empirical_one_sided_p": (np.count_nonzero(random <= ig) + 1) / (len(random) + 1)})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    colors = {"ig_ranked": "#D55E00", "random": "#777777"}
    for ax, mode in zip(axes, (1, 2)):
        for panel_type in ("ig_ranked", "random"):
            frame = summary.query("mode == @mode and panel_type == @panel_type").sort_values("genes_masked")
            label = "IG-ranked" if panel_type == "ig_ranked" else "Random (10 panels)"
            ax.plot(frame.genes_masked, frame.fraction_remaining_mean, marker="o",
                    color=colors[panel_type], label=label)
            if panel_type == "random":
                ax.fill_between(frame.genes_masked,
                                frame.fraction_remaining_mean - frame.fraction_remaining_sd,
                                frame.fraction_remaining_mean + frame.fraction_remaining_sd,
                                color=colors[panel_type], alpha=.18)
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set(title=f"Mode {mode}", xlabel="Genes masked",
               ylabel="Fraction of original mode signal remaining")
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
    fig.suptitle("Task 3D deletion validation")
    fig.savefig(FIGURES / "mode_ig_vs_random_deletion.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIGURES / "mode_ig_vs_random_deletion.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--random-replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This configured validation requires two visible CUDA GPUs")
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    conditions_per_mode = len(SIZES) * (1 + args.random_replicates)
    print(f"[estimate] modes=2 conditions_per_mode={conditions_per_mode} total={2*conditions_per_mode} "
          f"samples_per_condition=112 GPUs=2 approximate_runtime=20-35m", flush=True)
    context = mp.get_context("spawn"); events = context.Queue()
    processes = [context.Process(target=worker, args=(mode, device, args.random_replicates,
                                                      args.seed, args.batch_size, events))
                 for mode, device in zip((1, 2), args.devices)]
    for process in processes:
        process.start()
    completed = finished = 0; total = 2 * conditions_per_mode; started = time.monotonic()
    while finished < 2:
        try:
            event = events.get(timeout=args.heartbeat_seconds)
        except queue.Empty:
            elapsed = time.monotonic() - started
            rate = completed / elapsed if completed else 0
            eta = (total - completed) / rate if rate else float("nan")
            print(f"[heartbeat] completed={completed}/{total} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
            continue
        if event[0] == "done":
            completed += 1; elapsed = time.monotonic() - started
            eta = elapsed / completed * (total - completed)
            print(f"[progress] completed={completed}/{total} mode={event[1]} panel={event[2]} "
                  f"size={event[4]} replicate={event[3]} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
        elif event[0] == "finished":
            finished += 1; print(f"[worker] Mode {event[1]} finished ({finished}/2)", flush=True)
        else:
            raise RuntimeError(f"Mode {event[1]} failed: {event[2]}")
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise RuntimeError(f"Worker exit codes: {[process.exitcode for process in processes]}")
    results = pd.concat([pd.read_parquet(WORK / f"mode_{mode}_results.parquet") for mode in (1, 2)], ignore_index=True)
    panels = pd.concat([pd.read_parquet(WORK / f"mode_{mode}_panels.parquet") for mode in (1, 2)], ignore_index=True)
    per_replicate, summary = aggregate(results)
    comparison = compare_panels(per_replicate)
    results.to_parquet(OUT / "deletion_per_contrast.parquet", index=False)
    panels.to_parquet(OUT / "deletion_panels.parquet", index=False)
    per_replicate.to_csv(OUT / "deletion_per_replicate.csv", index=False)
    summary.to_csv(OUT / "deletion_summary.csv", index=False)
    comparison.to_csv(OUT / "deletion_comparison.csv", index=False)
    for mode in (1, 2):
        source = np.load(WORK / f"mode_{mode}_masked_response_vectors.npz", allow_pickle=True)
        np.savez_compressed(OUT / f"mode_{mode}_masked_response_vectors.npz", **{k: source[k] for k in source.files})
        pd.read_csv(WORK / f"mode_{mode}_condition_order.csv").to_csv(OUT / f"mode_{mode}_condition_order.csv", index=False)
    plot(summary)
    provenance = {"sizes": SIZES, "random_replicates": args.random_replicates,
                  "random_seed": args.seed, "mask_value": -10.0,
                  "panels": ["fixed Task 3D IG ranking", "deterministic size-matched random"],
                  "representations_recomputed": "full 512-D frozen BridgeRNA embeddings for all 112 samples per panel",
                  "response": "strict Task 3B within-contrast mean(FLT) minus mean(GC)",
                  "outcome": "mean masked mode projection divided by mean original mode projection across the seven assigned contrasts",
                  "excluded": ["edgeR", "DE-ranked deletion", "GSEA", "additional enrichment"],
                  "devices": args.devices}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("[complete] Task 3D deletion validation complete", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
