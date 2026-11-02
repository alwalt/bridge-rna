#!/usr/bin/env python3
"""Validate and produce sample-specific zero-reference Hallmark IG outputs."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special import expit

from common import CONFIG, REPO, RESULTS, WORK, read_gmt, read_vocab, resolve, save_json

sys.path.insert(0, str(REPO / "src"))
from fm_embed.model import load_expression_performer

OUT = RESULTS / "ig_validation"


def pathway_scores(model, inputs, weights, biases, mean, scale):
    hidden = model._encode_hidden(inputs).mean(1)
    standardized = (hidden - mean) / scale
    return standardized @ weights.T + biases


def gradients_on_nested_path(model, observed, weights, biases, mean, scale,
                             device, max_steps, path_batch):
    """Return d(score_k)/dx at endpoints j/max_steps, including zero."""
    baseline = torch.zeros_like(torch.as_tensor(observed, dtype=torch.float32, device=device))
    endpoint = torch.as_tensor(observed, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    biases_t = torch.as_tensor(biases, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    chunks = []
    for start in range(0, max_steps + 1, path_batch):
        stop = min(start + path_batch, max_steps + 1)
        alpha = torch.arange(start, stop, device=device, dtype=torch.float32) / max_steps
        inputs = (baseline[None] + alpha[:, None] * (endpoint - baseline)[None]).requires_grad_(True)
        scores = pathway_scores(model, inputs, weights_t, biases_t, mean_t, scale_t)
        pathway_gradients = []
        for pathway in range(scores.shape[1]):
            gradient = torch.autograd.grad(
                scores[:, pathway].sum(), inputs,
                retain_graph=pathway + 1 < scores.shape[1],
            )[0]
            pathway_gradients.append(gradient.detach().cpu())
        chunks.append(torch.stack(pathway_gradients, dim=1))
        del inputs, scores, pathway_gradients
    return baseline, endpoint, torch.cat(chunks, dim=0).numpy()


def attribution_from_nested_gradients(observed, gradients, steps, max_steps):
    if max_steps % steps:
        raise ValueError(f"{steps} does not divide reference grid {max_steps}")
    selected = gradients[::max_steps // steps]
    if len(selected) != steps + 1:
        raise AssertionError((steps, len(selected)))
    trapezoid_mean = (selected[0] / 2 + selected[1:-1].sum(axis=0) + selected[-1] / 2) / steps
    return observed[None, :] * trapezoid_mean


def top_overlap(a, b, k):
    left = set(np.argsort(-np.abs(a))[:k])
    right = set(np.argsort(-np.abs(b))[:k])
    return len(left & right) / k


def positive_top_overlap(a, b, k):
    left = set(np.argsort(-a)[:k])
    right = set(np.argsort(-b)[:k])
    return len(left & right) / k


def select_samples(meta):
    rows = []
    for contrast, (group_a, group_b) in CONFIG["tcga_contrasts"].items():
        for group in (group_a, group_b):
            candidates = meta.index[meta.cohort.eq(group)].to_numpy()
            if not len(candidates):
                raise AssertionError(f"No samples for {group}")
            rows.append({"contrast": contrast, "cohort": group, "row": int(candidates[0])})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = CONFIG["ig_validation"]
    step_counts = settings["steps"]
    max_steps = settings["reference_steps"]
    if max(step_counts) != max_steps or any(max_steps % step for step in step_counts):
        raise AssertionError("IG steps must be nested divisors of the reference grid")

    meta = pd.read_parquet(RESULTS / "tcga_sample_manifest.parquet")
    expression = np.load(resolve(CONFIG["tcga_bridge_expression"]), mmap_mode="r")
    probe = np.load(WORK / "frozen_gtex_hallmark_probe.npz")
    decoder = np.load(WORK / "hallmark_linear_decoder.npz")
    genes = np.asarray(read_vocab())
    gene_sets = read_gmt()
    hallmarks = probe["hallmarks"].astype(str).tolist()
    transfer = pd.read_csv(RESULTS / "gtex_tcga_probe_transfer_summary.csv")
    selected_pathways = transfer.nlargest(4, "pearson").hallmark.tolist()
    pathway_indices = np.asarray([hallmarks.index(name) for name in selected_pathways])
    samples = select_samples(meta)

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, device = load_expression_performer(
        resolve(CONFIG["checkpoint"]), resolve(CONFIG["model_config"]), 15165, device_name
    )
    model.requires_grad_(False)
    weights = probe["weight"][pathway_indices]
    biases = probe["bias"][pathway_indices]
    convergence_rows, attribution_rows, metric_rows = [], [], []
    attributions_by_step = {}
    started = time.time()

    for sample_number, selected in samples.iterrows():
        row = int(selected.row)
        sample = meta.iloc[row]
        observed = np.array(expression[row], dtype=np.float32, copy=True)
        baseline_np = np.zeros_like(observed)
        with torch.no_grad():
            endpoints = pathway_scores(
                model,
                torch.as_tensor(np.stack([observed, baseline_np]), device=device),
                torch.as_tensor(weights, device=device),
                torch.as_tensor(biases, device=device),
                torch.as_tensor(probe["mean"], device=device),
                torch.as_tensor(probe["scale"], device=device),
            ).cpu().numpy()
        score_difference = endpoints[0] - endpoints[1]
        cache = WORK / f"ig_nested_gradients_{sample.sample_id}_{max_steps}.npy"
        if cache.exists():
            gradients = np.load(cache)
            if gradients.shape != (max_steps + 1, len(selected_pathways), len(genes)):
                raise AssertionError((cache, gradients.shape))
        else:
            _, _, gradients = gradients_on_nested_path(
                model, observed, weights, biases, probe["mean"], probe["scale"],
                device, max_steps, settings["path_batch"],
            )
            np.save(cache, gradients.astype(np.float32))
        attrs = {
            step: attribution_from_nested_gradients(observed, gradients, step, max_steps)
            for step in step_counts
        }
        reference = attrs[max_steps]
        for pathway_position, hallmark in enumerate(selected_pathways):
            for step in step_counts:
                values = attrs[step][pathway_position]
                residual = float(values.sum() - score_difference[pathway_position])
                relative = abs(residual) / max(abs(float(score_difference[pathway_position])), 1e-8)
                rho = stats.spearmanr(values, reference[pathway_position]).statistic
                convergence_rows.append({
                    "sample_id": sample.sample_id, "patient_id": sample.patient_id,
                    "cohort": sample.cohort, "contrast_label": selected.contrast,
                    "hallmark": hallmark, "steps": step,
                    "score_x_minus_score_zero": float(score_difference[pathway_position]),
                    "attribution_sum": float(values.sum()),
                    "completeness_residual": residual,
                    "relative_completeness_error": relative,
                    "signed_rank_spearman_vs_reference": rho,
                    "absolute_rank_spearman_vs_reference": stats.spearmanr(np.abs(values), np.abs(reference[pathway_position])).statistic,
                    "top10_overlap_vs_reference": top_overlap(values, reference[pathway_position], 10),
                    "top25_overlap_vs_reference": top_overlap(values, reference[pathway_position], 25),
                    "top50_overlap_vs_reference": top_overlap(values, reference[pathway_position], 50),
                })
            for step in step_counts:
                attributions_by_step[(sample.sample_id, hallmark, step)] = attrs[step][pathway_position]
        elapsed = time.time() - started
        rate = (sample_number + 1) / elapsed
        eta = (len(samples) - sample_number - 1) / max(rate, 1e-12)
        print(f"[IG validation] {sample_number + 1}/{len(samples)} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(OUT / "step_convergence_and_completeness.csv", index=False)
    tolerance = settings["relative_completeness_tolerance"]
    rho_threshold = settings["rank_spearman_threshold"]
    overlap_threshold = settings["top50_overlap_threshold"]
    candidates = []
    for step in step_counts:
        block = convergence[convergence.steps.eq(step)]
        passes = (
            block.relative_completeness_error.le(tolerance).all()
            and block.absolute_rank_spearman_vs_reference.ge(rho_threshold).all()
            and block.top50_overlap_vs_reference.ge(overlap_threshold).all()
        )
        candidates.append({
            "steps": step, "all_profiles_pass": bool(passes),
            "max_relative_completeness_error": block.relative_completeness_error.max(),
            "min_absolute_rank_spearman_vs_reference": block.absolute_rank_spearman_vs_reference.min(),
            "min_top50_overlap_vs_reference": block.top50_overlap_vs_reference.min(),
        })
    selection = pd.DataFrame(candidates)
    selection.to_csv(OUT / "step_selection_summary.csv", index=False)
    passing = selection.loc[selection.all_profiles_pass, "steps"]
    validated_steps = int(passing.min()) if len(passing) else max_steps

    # Reuse the already calculated nested-grid attribution at the selected step.
    de_tables = {
        contrast: pd.read_parquet(RESULTS / f"de_{contrast}_bridge_vocab.parquet").set_index("gene")
        for contrast in CONFIG["tcga_contrasts"]
    }
    gsea = pd.read_csv(RESULTS / "conventional_gsea_results.csv")
    for _, selected in samples.iterrows():
        sample = meta.iloc[int(selected.row)]
        observed = np.array(expression[int(selected.row)], dtype=np.float32, copy=True)
        with torch.inference_mode():
            hidden = model._encode_hidden(torch.as_tensor(observed[None], device=device))[0].float().cpu().numpy()
        decoder_probability = expit(
            ((hidden - decoder["mean"]) / decoder["scale"]) @ decoder["weight"].T + decoder["bias"]
        )
        de = de_tables[selected.contrast]
        for hallmark in selected_pathways:
            # Select the validated attribution from the saved convergence grid by recalculating
            # the nested slice only if the accepted step is below the 64-step reference.
            reference_values = attributions_by_step[(sample.sample_id, hallmark, validated_steps)]
            pathway_index = hallmarks.index(hallmark)
            decoder_score = decoder_probability[:, pathway_index]
            absolute = np.abs(reference_values)
            lead_row = gsea[
                gsea.contrast.eq(selected.contrast)
                & gsea.universe.eq("bridge_15165")
                & gsea.hallmark.eq(hallmark)
            ]
            leading = set(str(lead_row.iloc[0].leading_edge).split(";")) if len(lead_row) else set()
            member = np.isin(genes, list(gene_sets[hallmark]))
            high = set(np.argsort(-absolute)[:50])
            odds, pvalue = stats.fisher_exact([
                [member[list(high)].sum(), 50 - member[list(high)].sum()],
                [member.sum() - member[list(high)].sum(), len(member) - member.sum() - (50 - member[list(high)].sum())],
            ], alternative="greater")
            metric_rows.append({
                "sample_id": sample.sample_id, "contrast_label": selected.contrast,
                "cohort": sample.cohort, "hallmark": hallmark,
                "validated_steps": validated_steps,
                "decoder_abs_ig_spearman": stats.spearmanr(decoder_score, absolute).statistic,
                "top10_overlap": positive_top_overlap(decoder_score, absolute, 10) * 10,
                "top25_overlap": positive_top_overlap(decoder_score, absolute, 25) * 25,
                "top50_overlap": positive_top_overlap(decoder_score, absolute, 50) * 50,
                "canonical_high_ig_odds_ratio": odds,
                "canonical_high_ig_p": pvalue,
                "leading_edge_top50_overlap": len({genes[index] for index in high} & leading),
            })
            for gene_index, gene in enumerate(genes):
                de_row = de.loc[gene] if gene in de.index else None
                attribution_rows.append({
                    "sample_id": sample.sample_id, "patient_id": sample.patient_id,
                    "cohort": sample.cohort, "contrast_label": selected.contrast,
                    "hallmark": hallmark, "gene": gene,
                    "de_log_expression_difference": de_row.log_expression_difference if de_row is not None else np.nan,
                    "de_statistic": de_row.statistic if de_row is not None else np.nan,
                    "de_fdr": de_row.fdr if de_row is not None else np.nan,
                    "gsea_leading_edge": gene in leading,
                    "decoder_score": decoder_score[gene_index],
                    "ig_signed": reference_values[gene_index],
                    "ig_absolute": absolute[gene_index],
                    "canonical_member": member[gene_index],
                })

    metrics = pd.DataFrame(metric_rows)
    gene_level = pd.DataFrame(attribution_rows)
    metrics.to_csv(OUT / "decoder_ig_comparison_metrics.csv", index=False)
    gene_level.to_parquet(OUT / "ig_de_gene_level_comparison.parquet", index=False)
    save_json(OUT / "provenance.json", {
        "target": "frozen GTEx-trained sample-level Bridge Hallmark probe output",
        "input": "one TCGA sample's 15,165-gene natural log1p(TPM) vector",
        "baseline": "all-zero natural log1p(TPM), equivalent to zero TPM for every input gene",
        "interpretation": "sample-specific input contribution relative to zero; not pairwise cancer attribution",
        "integration_rule": "nested composite-trapezoid integration on a straight zero-to-sample path",
        "tested_steps": step_counts, "reference_steps": max_steps,
        "validated_steps": validated_steps,
        "acceptance": {
            "relative_completeness_error_max": tolerance,
            "absolute_rank_spearman_vs_64_min": rho_threshold,
            "top50_overlap_vs_64_min": overlap_threshold,
            "must_pass_for_every_tested_sample_pathway": True,
        },
        "selected_pathways": selected_pathways,
        "selected_samples": samples.to_dict("records"),
        "gradient_cache": "ignored work/ig_nested_gradients_<sample>_<reference_steps>.npy",
        "bridge_frozen": True,
        "decoder_interpretation": "Hallmark-like information in a contextual gene representation",
        "ig_interpretation": "input-gene contribution to this sample's frozen Hallmark prediction",
    })
    make_plots(convergence, metrics, gene_level)
    print(json.dumps({"validated_steps": validated_steps, "profiles": len(metrics)}, indent=2))


def make_plots(convergence, metrics, gene_level):
    import matplotlib.pyplot as plt

    summary = convergence.groupby("steps", as_index=False).agg(
        median_relative_error=("relative_completeness_error", "median"),
        max_relative_error=("relative_completeness_error", "max"),
        median_rank_spearman=("absolute_rank_spearman_vs_reference", "median"),
        min_rank_spearman=("absolute_rank_spearman_vs_reference", "min"),
        median_top50_overlap=("top50_overlap_vs_reference", "median"),
        min_top50_overlap=("top50_overlap_vs_reference", "min"),
    )
    summary.to_csv(OUT / "step_convergence_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(summary.steps, summary.median_relative_error, marker="o", label="median")
    axes[0].plot(summary.steps, summary.max_relative_error, marker="o", label="maximum")
    axes[0].set(yscale="log", xlabel="IG steps", ylabel="Relative completeness error")
    axes[1].plot(summary.steps, summary.median_rank_spearman, marker="o", label="median")
    axes[1].plot(summary.steps, summary.min_rank_spearman, marker="o", label="minimum")
    axes[1].set(xlabel="IG steps", ylabel="Spearman of |IG| ranks vs reference steps", ylim=(0, 1.01))
    axes[2].plot(summary.steps, summary.median_top50_overlap, marker="o", label="median")
    axes[2].plot(summary.steps, summary.min_top50_overlap, marker="o", label="minimum")
    axes[2].set(xlabel="IG steps", ylabel="Top-50 overlap vs reference steps", ylim=(0, 1.01))
    for axis in axes: axis.legend()
    fig.tight_layout(); fig.savefig(OUT / "ig_step_convergence.png", dpi=220); fig.savefig(OUT / "ig_step_convergence.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(metrics.decoder_abs_ig_spearman, metrics.canonical_high_ig_odds_ratio)
    ax.set(xlabel="Spearman(decoder score, |IG|)", ylabel="Canonical-member enrichment odds ratio",
           title="Contextual identity and sample-specific contribution are distinct")
    fig.tight_layout(); fig.savefig(OUT / "decoder_vs_validated_ig.png", dpi=220); fig.savefig(OUT / "decoder_vs_validated_ig.pdf"); plt.close(fig)

    first_hallmark = gene_level.hallmark.iloc[0]
    first = gene_level[gene_level.hallmark.eq(first_hallmark)].nlargest(50, "ig_absolute").sort_values("ig_absolute")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(first.gene, first.ig_signed, color=np.where(first.canonical_member, "tab:red", "tab:blue"))
    ax.set(xlabel="Signed validated IG", title=f"{first_hallmark}: top |IG| genes (red = canonical member)")
    fig.tight_layout(); fig.savefig(OUT / "selected_pathway_validated_ig.png", dpi=220); fig.savefig(OUT / "selected_pathway_validated_ig.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
