#!/usr/bin/env python3
"""L1000-style gene-recall evaluation of two frozen panels on local GTEx."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from analyze_gtex_tissue_patterns import GTEX_H5, build_hgnc_mapping, norm_gene
from common import REPO_ROOT, RESULTS, WORK, sha256

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from fm_embed.model import load_expression_performer
from fm_embed.reconstruction import mask_except_panel, reconstruct
from fm_embed.species import load_exon_length_map


MODEL_GENES = RESULTS / "gene_inference_human_consensus_ranking.parquet"
PANELS = RESULTS / "frozen_gene_inference_validation_panels.parquet"
LENGTHS = REPO_ROOT / "data/gencode/gencode_v49_gene_exon_lengths.csv"
MATRIX = WORK / "gtex_model_log1p_tpm.npy"
MAPPING = RESULTS / "gtex_model_tpm_mapping.parquet"
SAMPLES = RESULTS / "gtex_model_samples.parquet"
PREFIX = "gtex_l1000_style_gene_inference"
PANEL_IDS = ("conserved_gene_inference_top1000", "l1000_all_mapped_922")


def prepare_gtex() -> None:
    """Map raw GTEx counts and convert them to model-ordered natural log1p(TPM)."""
    ranking = pd.read_parquet(MODEL_GENES).sort_values("model_index")
    model_genes = ranking.gene.astype(str).tolist()
    length_map = {norm_gene(key): value for key, value in load_exon_length_map(LENGTHS).items()}
    with h5py.File(GTEX_H5, "r") as handle:
        source_genes = [norm_gene(x.decode() if isinstance(x, bytes) else x)
                        for x in handle["meta/genes"][:]]
        crosswalk = build_hgnc_mapping(source_genes)
        approved_to_source = {crosswalk[source]: index for index, source in enumerate(source_genes)
                              if source in crosswalk}
        source_rows = np.asarray([approved_to_source.get(norm_gene(gene), -1)
                                  for gene in model_genes], dtype=int)
        observed = source_rows >= 0
        source_approved = [crosswalk.get(gene, gene) for gene in source_genes]
        source_lengths = np.asarray([length_map.get(gene, np.nan) for gene in source_approved], dtype=float)
        valid_source = np.isfinite(source_lengths) & (source_lengths > 0)
        # A mapped model gene must also have a valid length to enter TPM.
        observed &= np.asarray([source_lengths[row] > 0 and np.isfinite(source_lengths[row])
                                if row >= 0 else False for row in source_rows])
        sample_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["meta/sampid"][:]]
        temp = MATRIX.with_suffix(".tmp.npy")
        output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                           shape=(len(sample_ids), len(model_genes)))
        output[:] = 0
        expression = handle["data/expression"]
        valid_lengths_kb = source_lengths[valid_source] / 1000.0
        selected_rows = source_rows[observed]
        selected_lengths_kb = source_lengths[selected_rows] / 1000.0
        for start in range(0, len(sample_ids), 64):
            stop = min(start + 64, len(sample_ids))
            counts = np.asarray(expression[start:stop], dtype=np.float32)
            total_rate = (counts[:, valid_source] / valid_lengths_kb).sum(axis=1, keepdims=True)
            selected_rate = counts[:, selected_rows] / selected_lengths_kb
            tpm = np.divide(selected_rate, total_rate, out=np.zeros_like(selected_rate),
                            where=total_rate > 0) * 1e6
            output[start:stop, observed] = np.log1p(tpm)
            if stop == len(sample_ids) or stop % 1024 == 0:
                print(f"[GTEx preparation] {stop:,}/{len(sample_ids):,}", flush=True)
        output.flush(); del output
    temp.replace(MATRIX)
    pd.DataFrame({"model_index": ranking.model_index.astype(int), "gene": model_genes,
        "gtex_source_index": source_rows, "observed_in_gtex": observed}).to_parquet(MAPPING, index=False)
    pd.DataFrame({"row_index": np.arange(len(sample_ids)), "sample_id": sample_ids}).to_parquet(
        SAMPLES, index=False)


def correlation_inputs(truth: np.ndarray, prediction: np.ndarray,
                       indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured = np.asarray(truth[:, indices], dtype=np.float32)
    inferred = np.asarray(prediction[:, indices], dtype=np.float32)
    measured -= measured.mean(axis=0, keepdims=True)
    inferred -= inferred.mean(axis=0, keepdims=True)
    measured_norm = np.linalg.norm(measured, axis=0)
    inferred_norm = np.linalg.norm(inferred, axis=0)
    measured_variable = measured_norm > 0
    # Measured-constant genes have undefined Pearson and are not evaluable.
    measured = measured[:, measured_variable]
    inferred = inferred[:, measured_variable]
    indices = indices[measured_variable]
    measured_norm = measured_norm[measured_variable]
    inferred_norm = inferred_norm[measured_variable]
    measured /= measured_norm
    inferred = np.divide(inferred, inferred_norm, out=np.zeros_like(inferred),
                         where=inferred_norm > 0)
    return measured, inferred, indices


def correlation_blocks(predicted: torch.Tensor, measured: torch.Tensor, block_size: int):
    genes = predicted.shape[1]
    for start in range(0, genes, block_size):
        stop = min(start + block_size, genes)
        correlations = predicted[:, start:stop].T @ measured
        correlations.clamp_(-1.0, 1.0)
        local = torch.arange(stop - start, device=correlations.device)
        correlations[local, torch.arange(start, stop, device=correlations.device)] = 2.0
        yield start, stop, correlations


def l1000_recall(measured: np.ndarray, inferred: np.ndarray, device: torch.device,
                 block_size: int, bins: int = 20000) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Compute matched Pearson and the exact pooled-null 95th-percentile cutoff."""
    measured_tensor = torch.from_numpy(np.ascontiguousarray(measured)).to(device)
    inferred_tensor = torch.from_numpy(np.ascontiguousarray(inferred)).to(device)
    matched = np.sum(measured * inferred, axis=0, dtype=np.float64).astype(np.float32)
    histogram = torch.zeros(bins, dtype=torch.int64, device=device)
    started = time.monotonic()
    for start, stop, correlations in correlation_blocks(inferred_tensor, measured_tensor, block_size):
        histogram += torch.histc(correlations, bins=bins, min=-1.0, max=1.0).to(torch.int64)
        print(f"[null pass 1] inferred genes={stop:,}/{inferred.shape[1]:,} "
              f"elapsed={(time.monotonic()-started)/60:.1f}m", flush=True)
    counts = histogram.cpu().numpy(); total = int(counts.sum())
    expected = inferred.shape[1] * (inferred.shape[1] - 1)
    if total != expected: raise AssertionError(f"Null count {total:,} != {expected:,}")
    target = int(np.ceil(0.95 * total)) - 1
    cumulative = np.cumsum(counts)
    threshold_bin = int(np.searchsorted(cumulative, target + 1))
    lower = -1.0 + 2.0 * threshold_bin / bins
    upper = -1.0 + 2.0 * (threshold_bin + 1) / bins
    in_bin = []
    for start, stop, correlations in correlation_blocks(inferred_tensor, measured_tensor, block_size):
        selected = correlations[(correlations >= lower) &
                                (correlations < upper if threshold_bin < bins - 1 else correlations <= upper)]
        if selected.numel(): in_bin.append(selected.cpu().numpy())
        print(f"[null pass 2] inferred genes={stop:,}/{inferred.shape[1]:,}", flush=True)
    values = np.concatenate(in_bin)
    before = int(cumulative[threshold_bin - 1]) if threshold_bin else 0
    within_rank = target - before
    threshold = float(np.partition(values, within_rank)[within_rank])
    # Histogram interpolation provides per-gene recall estimates; the exact
    # threshold above determines the reported well-inferred classification.
    bin_width = 2.0 / bins
    positions = np.clip(((matched + 1.0) / bin_width).astype(int), 0, bins - 1)
    before_bins = np.r_[0, cumulative[:-1]][positions]
    fractions = np.clip((matched - (-1.0 + positions * bin_width)) / bin_width, 0, 1)
    recall = (before_bins + fractions * counts[positions]) / total
    del measured_tensor, inferred_tensor, histogram
    return matched, threshold, recall, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=int(4))
    parser.add_argument("--correlation-block-size", type=int, default=256)
    parser.add_argument("--rebuild-gtex", action="store_true")
    args = parser.parse_args()
    if args.rebuild_gtex or not all(path.is_file() for path in (MATRIX, MAPPING, SAMPLES)):
        prepare_gtex()
    matrix = np.load(MATRIX, mmap_mode="r")
    mapping = pd.read_parquet(MAPPING).sort_values("model_index")
    panels = pd.read_parquet(PANELS)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model, device = load_expression_performer(REPO_ROOT / "model/r7hnr92k/best_model.pt",
        REPO_ROOT / "model/r7hnr92k/config.json", 15165, str(device))
    summaries, gene_tables = [], []
    cache = WORK / "gtex_l1000_style_predictions"; cache.mkdir(parents=True, exist_ok=True)
    for panel_id in PANEL_IDS:
        requested = np.sort(panels.loc[panels.panel_id.eq(panel_id), "model_index"].to_numpy(int))
        observed = set(mapping.loc[mapping.observed_in_gtex, "model_index"].astype(int))
        visible = np.asarray(sorted(set(requested) & observed), dtype=int)
        hidden = np.asarray(sorted(observed - set(visible)), dtype=int)
        prediction_path = cache / f"{panel_id}.npy"
        if prediction_path.is_file():
            prediction = np.load(prediction_path, mmap_mode="r")
            print(f"[{panel_id}] reuse prediction cache", flush=True)
        else:
            prediction = reconstruct(model,
                mask_except_panel(matrix, visible, float(-10.0)), device, args.batch_size,
                f"GTEx L1000-style {panel_id}")
            np.save(prediction_path, prediction)
        measured_norm, inferred_norm, evaluated = correlation_inputs(matrix, prediction, hidden)
        matched, threshold, recall, null_size = l1000_recall(
            measured_norm, inferred_norm, device, args.correlation_block_size)
        well = matched >= threshold
        genes = mapping.set_index("model_index").loc[evaluated, "gene"].astype(str).to_numpy()
        gene_table = pd.DataFrame({"panel_id": panel_id, "model_index": evaluated,
            "gene": genes, "matched_pearson": matched, "gene_recall_approx": recall,
            "null_95th_percentile": threshold, "well_inferred": well})
        gene_tables.append(gene_table)
        summaries.append({"panel_id": panel_id, "requested_visible_genes": len(requested),
            "visible_genes_mapped_used": len(visible), "hidden_genes_evaluated": len(evaluated),
            "well_inferred_genes": int(well.sum()), "well_inferred_percent": well.mean(),
            "median_matched_pearson": float(np.nanmedian(matched)),
            "null_95th_percentile": threshold, "null_correlations": null_size,
            "gtex_samples": len(matrix)})
    per_gene = pd.concat(gene_tables, ignore_index=True)
    summary = pd.DataFrame(summaries)
    per_gene.to_parquet(RESULTS / f"{PREFIX}_per_gene.parquet", index=False)
    summary.to_parquet(RESULTS / f"{PREFIX}_summary.parquet", index=False)
    summary.to_csv(RESULTS / f"{PREFIX}_summary.csv", index=False)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Subramanian et al. 2017 pooled non-matching-gene Pearson null",
        "well_inferred_criterion": "matched Pearson >= exact 95th percentile of all non-matched correlations",
        "paper_reference": "Cell 2017; doi:10.1016/j.cell.2017.10.049",
        "paper_gtex_samples": 8555, "local_gtex_samples": len(matrix),
        "sample_set_note": "Historical paper 8,555-sample matrix unavailable; all 9,662 local GTEx samples used",
        "preprocessing": "raw counts -> GENCODE v49 exon-length TPM -> natural log1p",
        "model_checkpoint": str(REPO_ROOT / "model/r7hnr92k/best_model.pt"),
        "model_checkpoint_sha256": sha256(REPO_ROOT / "model/r7hnr92k/best_model.pt"),
        "frozen_panels": str(PANELS), "frozen_panels_sha256": sha256(PANELS),
        "gtex_h5": str(GTEX_H5), "gtex_h5_sha256": sha256(GTEX_H5),
        "random_panels": 0, "fine_tuning": False, "reranking": False}
    (RESULTS / f"{PREFIX}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nGTEx L1000-style summary\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
