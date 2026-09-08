#!/usr/bin/env python3
"""Diagnose why PC1-2 removal affects RR3-39 more than RR3-40.

The controlled reference, samples, preprocessing, and frozen BridgeRNA encoder
are reused unchanged. Primary component biology uses zero-baseline Integrated
Gradients, matching the validated Task 4 attribution implementation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
OUT = HERE / "results/task4_confounding_profiler/rr3_functional_overlap"
WORK = HERE / "work/task4_confounding_profiler/rr3_functional_overlap"
FIG = OUT / "figures"
T3 = REPO / "benchmarks/osdr_batch_effect_representation"
R3, W3 = T3 / "results", T3 / "work"
CONTROL = HERE / "work/datasets/chen_2020_tcells"
BIO = HERE / "results/task4_confounding_profiler/independent_biological_replication"
FULLVOC = HERE / "results/task4_full_vs_bridge_vocab_expression"
ROBUST = HERE / "results/task4_response_robustness"
GMT_ROOT = REPO / "benchmarks/cross_species_exercise_response/results/per_study_ranked_gsea"
GMT = {"GO:BP": "GO_Biological_Process_2026.gmt", "KEGG": "KEGG_2026.gmt", "REAC": "Reactome_Pathways_2024.gmt"}
SEED = 44021
N_PERM = 1000
N_BOOT = 5000
N_GENE_PERM = 10000

sys.path[:0] = [str(REPO / "benchmarks/tcga_downstream/pipeline"), str(REPO)]
from run_attention_pooling import load_frozen_encoder
from src.fm_embed.vocab import load_canonical_genes

GENES = np.asarray(load_canonical_genes(REPO / "data/ensembl/canonical_genes.csv"))
assert len(GENES) == 15165 and len(set(GENES)) == 15165

SPECS = {
    "RR3-39": ("C01_OSD137_original_matched", "C01_OSD168_all_ERCC"),
    "RR3-40": ("C02_OSD137_original_matched", "C02_OSD168_all_ERCC"),
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def unit(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), 1e-12)


def condition(sample: str) -> str:
    return "FLT" if "_FLT_" in sample else "GC" if "_GC_" in sample else "other"


def controlled_basis() -> tuple[np.ndarray, np.ndarray]:
    manifest = pd.read_parquet(CONTROL / "manifest.parquet").reset_index(drop=True)
    embeddings = np.load(CONTROL / "bridgerna_embeddings.npy").astype(float)
    differences = []
    for _, group in manifest.groupby("pair_id", sort=True):
        poly = embeddings[group.index[group.library_prep.eq("polyA")]].mean(0)
        ribo = embeddings[group.index[group.library_prep.eq("ribo")]].mean(0)
        differences.append(ribo - poly)
    _, singular, vt = np.linalg.svd(np.stack(differences), full_matrices=False)
    return vt[:2], singular**2 / np.sum(singular**2)


def design() -> tuple[dict, np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(R3 / "sample_manifest.csv")
    index = dict(zip(manifest.sample_id, range(len(manifest))))
    design_table = pd.read_csv(R3 / "task3_osd168_technical_replication/technical_response_design.csv")
    wanted = {x for pair in SPECS.values() for x in pair}
    specs = {}
    for row in design_table[design_table.representation.isin(wanted)].itertuples():
        samples = str(row.samples).split(" | ")
        specs[row.representation] = {
            "samples": samples,
            "indices": [index[x] for x in samples],
            "conditions": [condition(x) for x in samples],
            "technical_condition": row.technical_condition,
            "source_contrast": row.source_contrast,
        }
    assert set(specs) == wanted
    x = np.load(W3 / "bridgerna_log1p_tpm_inputs.npy", mmap_mode="r")
    z = np.load(W3 / "bridgerna_embeddings.npy").astype(float)
    return specs, x, z, manifest


def response(spec: dict, z: np.ndarray) -> np.ndarray:
    f = [i for i, c in zip(spec["indices"], spec["conditions"]) if c == "FLT"]
    g = [i for i, c in zip(spec["indices"], spec["conditions"]) if c == "GC"]
    return z[f].mean(0) - z[g].mean(0)


def project(v: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parallel = (v @ basis.T) @ basis
    return parallel, v - parallel


def metadata_audit(specs: dict, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correspondence = pd.read_csv(R3 / "task3_osd168_technical_replication/biological_sample_correspondence.csv")
    tech = pd.read_csv(R3 / "task3_library_diagnostic/authoritative_technical_metadata.csv")
    source_tech = tech[tech.representation.eq("OSD-137 original RR-3")].iloc[0]
    remeasure_tech = tech[tech.representation.eq("OSD-168 RR-3 ERCC")].iloc[0]
    rows = []
    for timepoint, (original, remeasure) in SPECS.items():
        orig_samples = specs[original]["samples"]
        corr = correspondence[(correspondence.source_OSD == "OSD-137") & correspondence.source_sample.isin(orig_samples)]
        for row in corr.itertuples():
            source_manifest = manifest[manifest.sample_id.eq(row.source_sample)].iloc[0]
            target_manifest = manifest[manifest.sample_id.eq(row.OSD168_sample)].iloc[0]
            rows.append({
                "timepoint": timepoint, "animal_id": row.animal_id, "condition": row.group,
                "OSD137_sample": row.source_sample, "OSD168_sample": row.OSD168_sample,
                "exact_animal_match": row.exact_animal_match,
                "biological_material_status": row.biological_material_status,
                "identical_RNA_status": row.identical_RNA_status,
                "OSD137_preservation": source_manifest.preservation,
                "OSD137_library_selection": source_tech.library_selection,
                "OSD137_library_kit": source_tech.library_kit,
                "OSD137_layout": source_tech.layout, "OSD137_read_length": source_tech.read_length,
                "OSD137_instrument": source_tech.instrument, "OSD137_facility": source_tech.facility,
                "OSD137_ERCC": source_tech.ERCC,
                "OSD168_preservation": target_manifest.preservation,
                "OSD168_library_selection": remeasure_tech.library_selection,
                "OSD168_library_kit": remeasure_tech.library_kit,
                "OSD168_layout": remeasure_tech.layout, "OSD168_read_length": remeasure_tech.read_length,
                "OSD168_instrument": remeasure_tech.instrument, "OSD168_facility": remeasure_tech.facility,
                "OSD168_ERCC": remeasure_tech.ERCC, "OSD168_ERCC_mix": row.ERCC_condition,
            })
    sample_table = pd.DataFrame(rows).sort_values(["timepoint", "condition", "animal_id"])
    sample_table.to_csv(OUT / "rr3_sample_correspondence_and_protocol.csv", index=False)
    variables = []
    fields = ["preservation", "library_selection", "library_kit", "layout", "read_length", "instrument", "facility", "ERCC"]
    for field in fields:
        a = str(sample_table[f"OSD137_{field}"].dropna().iloc[0])
        b = str(sample_table[f"OSD168_{field}"].dropna().iloc[0])
        variables.append({"variable": field, "OSD137": a, "OSD168": b, "remeasurement_status": "held constant" if a == b else "changed/reported differently"})
    variable_table = pd.DataFrame(variables)
    variable_table["between_39d_and_40d_status"] = "held constant"
    extra = pd.DataFrame([
        {"variable": "flight duration", "OSD137": "39 day vs 40 day", "OSD168": "same source-animal strata",
         "remeasurement_status": "held constant within animal", "between_39d_and_40d_status": "different by design"},
        {"variable": "source animals", "OSD137": "F1/F2; G1/G2 vs F3/F4/F5; G3/G5", "OSD168": "matched F1/F2; G1/G2 vs F3/F4; G3/G5",
         "remeasurement_status": "exact animal IDs matched; same RNA material supported", "between_39d_and_40d_status": "different animals"},
        {"variable": "technical-comparison sample size", "OSD137": "39d 2/2; 40d matched 2/2", "OSD168": "39d 2/2; 40d 2/2",
         "remeasurement_status": "held constant", "between_39d_and_40d_status": "held constant"},
        {"variable": "biological edgeR sample size", "OSD137": "39d 2/2; 40d 3/2", "OSD168": "not used for conventional biological edgeR",
         "remeasurement_status": "not applicable", "between_39d_and_40d_status": "different; F5 has no OSD-168 counterpart"},
    ])
    variable_table = pd.concat([variable_table, extra], ignore_index=True)
    variable_table.to_csv(OUT / "protocol_variable_audit.csv", index=False)
    return sample_table, variable_table


def decomposition(specs: dict, z: np.ndarray, basis: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows, vectors = [], {}
    for timepoint, pair in SPECS.items():
        for measurement, name in zip(["OSD-137 original", "OSD-168 remeasurement"], pair):
            full = response(specs[name], z)
            parallel, orthogonal = project(full, basis)
            vectors[(timepoint, measurement, "full")] = full
            vectors[(timepoint, measurement, "parallel")] = parallel
            vectors[(timepoint, measurement, "orthogonal")] = orthogonal
            rows.append({
                "timepoint": timepoint, "measurement": measurement, "response_name": name,
                "n_FLT": specs[name]["conditions"].count("FLT"), "n_GC": specs[name]["conditions"].count("GC"),
                "total_norm": np.linalg.norm(full), "parallel_norm": np.linalg.norm(parallel),
                "orthogonal_norm": np.linalg.norm(orthogonal),
                "aligned_fraction": np.dot(parallel, parallel) / np.dot(full, full),
                "pythagorean_relative_error": abs(np.dot(full, full) - np.dot(parallel, parallel) - np.dot(orthogonal, orthogonal)) / np.dot(full, full),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "response_component_metrics.csv", index=False)
    pair_rows = []
    for timepoint in SPECS:
        for component in ["full", "parallel", "orthogonal"]:
            a = vectors[(timepoint, "OSD-137 original", component)]
            b = vectors[(timepoint, "OSD-168 remeasurement", component)]
            pair_rows.append({"timepoint": timepoint, "component": component, "cosine": cosine(a, b), "spearman": spearmanr(a, b).statistic})
    pd.DataFrame(pair_rows).to_csv(OUT / "component_replication_similarity.csv", index=False)
    return table, vectors


def bootstrap_stability(specs: dict, z: np.ndarray, basis: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows = []
    for timepoint, (a_name, b_name) in SPECS.items():
        a, b = specs[a_name], specs[b_name]
        # Corresponding original/remeasurement samples have identical list order by condition/animal.
        for rep in range(N_BOOT):
            response_vectors = []
            for spec in [a, b]:
                fi = np.asarray([i for i, c in zip(spec["indices"], spec["conditions"]) if c == "FLT"])
                gi = np.asarray([i for i, c in zip(spec["indices"], spec["conditions"]) if c == "GC"])
                # Shared random draws preserve animal correspondence across technical measurements.
                response_vectors.append((fi, gi))
            nf, ng = len(response_vectors[0][0]), len(response_vectors[0][1])
            fdraw, gdraw = rng.integers(nf, size=nf), rng.integers(ng, size=ng)
            built = []
            for fi, gi in response_vectors:
                v = z[fi[fdraw]].mean(0) - z[gi[gdraw]].mean(0)
                p, o = project(v, basis)
                built.append((v, p, o))
            rows.append({
                "timepoint": timepoint, "replicate": rep,
                "full_cosine": cosine(built[0][0], built[1][0]),
                "parallel_cosine": cosine(built[0][1], built[1][1]),
                "orthogonal_cosine": cosine(built[0][2], built[1][2]),
                "cosine_after_removal": cosine(built[0][2], built[1][2]),
                "loss_after_removal": cosine(built[0][2], built[1][2]) - cosine(built[0][0], built[1][0]),
                "original_aligned_fraction": np.dot(built[0][1], built[0][1]) / max(np.dot(built[0][0], built[0][0]), 1e-12),
                "remeasure_aligned_fraction": np.dot(built[1][1], built[1][1]) / max(np.dot(built[1][0], built[1][0]), 1e-12),
            })
    raw = pd.DataFrame(rows)
    raw.to_parquet(OUT / "paired_bootstrap_stability.parquet", index=False)
    metrics = ["full_cosine", "parallel_cosine", "orthogonal_cosine", "loss_after_removal", "original_aligned_fraction", "remeasure_aligned_fraction"]
    summary = raw.groupby("timepoint")[metrics].agg(["mean", "median", "std", lambda x: x.quantile(.025), lambda x: x.quantile(.975)])
    summary.columns = [f"{a}_{b if isinstance(b, str) else 'quantile'}" for a, b in summary.columns]
    # Pandas names lambda columns <lambda_0>/<lambda_1>; make names explicit.
    summary = raw.groupby("timepoint")[metrics].agg(
        **{f"{m}_{stat}": (m, fn) for m in metrics for stat, fn in [
            ("mean", "mean"), ("median", "median"), ("sd", "std"),
            ("ci_low", lambda x: x.quantile(.025)), ("ci_high", lambda x: x.quantile(.975)),
        ]}
    ).reset_index()
    summary.to_csv(OUT / "paired_bootstrap_stability_summary.csv", index=False)
    return raw, summary


def score(model, x, direction):
    return (model._encode_hidden(x).mean(1) * direction).sum(1)


def integrated_gradients(model, values, direction, device, steps=16, path_batch=4):
    baseline = torch.zeros((1, len(values)), device=device)
    observed = torch.as_tensor(np.array(values, dtype=np.float32, copy=True), device=device)[None]
    target = torch.as_tensor(direction.astype(np.float32), device=device)[None]
    total = torch.zeros_like(baseline)
    alphas = (np.arange(steps, dtype=np.float32) + .5) / steps
    for start in range(0, steps, path_batch):
        alpha = torch.as_tensor(alphas[start:start + path_batch], device=device)[:, None]
        path = (baseline + alpha * (observed - baseline)).requires_grad_(True)
        output = score(model, path, target.expand(len(path), -1))
        total += torch.autograd.grad(output.sum(), path)[0].detach().sum(0, keepdim=True)
    attribution = ((observed - baseline) * total / steps)[0]
    with torch.no_grad():
        delta = score(model, observed, target)[0] - score(model, baseline, target)[0]
    return attribution.cpu().numpy(), float(delta), float(attribution.sum().cpu())


def attribution_worker(timepoint, device_name, steps, path_batch):
    specs, x, z, _ = design()
    basis, _ = controlled_basis()
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = load_frozen_encoder(device)
    attrs, checks = {}, []
    start = time.monotonic()
    for measurement, name in zip(["OSD-137 original", "OSD-168 remeasurement"], SPECS[timepoint]):
        full = response(specs[name], z)
        parallel, orthogonal = project(full, basis)
        for component, vector in [("parallel", parallel), ("orthogonal", orthogonal)]:
            direction = unit(vector).astype(np.float32)
            condition_attrs = {}
            for group in ["FLT", "GC"]:
                indices = [i for i, c in zip(specs[name]["indices"], specs[name]["conditions"]) if c == group]
                profile = np.asarray(x[indices]).mean(0).astype(np.float32)
                attr, endpoint, total = integrated_gradients(model, profile, direction, device, steps, path_batch)
                condition_attrs[group] = attr
                checks.append({"timepoint": timepoint, "measurement": measurement, "component": component, "condition": group,
                               "n_samples": len(indices), "endpoint_delta": endpoint, "attribution_sum": total,
                               "completeness_error": total - endpoint})
            attrs[f"{measurement}|{component}"] = condition_attrs["FLT"] - condition_attrs["GC"]
            elapsed = (time.monotonic() - start) / 60
            print(f"[IG heartbeat] {timepoint} {measurement} {component} elapsed={elapsed:.1f}m", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / f"{timepoint}_component_ig.npz", **attrs)
    pd.DataFrame(checks).to_csv(WORK / f"{timepoint}_ig_completeness.csv", index=False)


def load_attributions() -> tuple[pd.DataFrame, dict]:
    rows, arrays = [], {}
    for timepoint in SPECS:
        data = np.load(WORK / f"{timepoint}_component_ig.npz")
        for key in data.files:
            measurement, component = key.split("|")
            values = data[key]
            arrays[(timepoint, measurement, component)] = values
            order = np.argsort(-np.abs(values)); rank = np.empty(len(values), int); rank[order] = np.arange(1, len(values) + 1)
            rows.append(pd.DataFrame({"timepoint": timepoint, "measurement": measurement, "component": component,
                                      "gene_symbol": GENES, "signed_attribution": values,
                                      "absolute_attribution": np.abs(values), "absolute_rank": rank}))
    detailed = pd.concat(rows, ignore_index=True)
    # Equal-weight consensus of normalized attribution magnitudes across the two measurements.
    for timepoint in SPECS:
        for component in ["parallel", "orthogonal"]:
            vals = []
            for measurement in ["OSD-137 original", "OSD-168 remeasurement"]:
                v = np.abs(arrays[(timepoint, measurement, component)])
                vals.append(v / max(v.sum(), 1e-12))
            consensus = np.mean(vals, axis=0)
            arrays[(timepoint, "consensus", component)] = consensus
            order = np.argsort(-consensus); rank = np.empty(len(consensus), int); rank[order] = np.arange(1, len(consensus) + 1)
            rows.append(pd.DataFrame({"timepoint": timepoint, "measurement": "consensus", "component": component,
                                      "gene_symbol": GENES, "signed_attribution": consensus,
                                      "absolute_attribution": consensus, "absolute_rank": rank}))
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_parquet(OUT / "component_gene_attribution_rankings.parquet", index=False)
    (all_rows[all_rows.measurement.eq("consensus")]
     .sort_values(["timepoint", "component", "absolute_rank"])
     .groupby(["timepoint", "component"], as_index=False, group_keys=False).head(25)
     .to_csv(OUT / "top25_component_genes.csv", index=False))
    pd.concat([pd.read_csv(WORK / f"{t}_ig_completeness.csv") for t in SPECS]).to_csv(OUT / "ig_completeness.csv", index=False)
    return all_rows, arrays


def pathway_family(term: str) -> str:
    t = term.upper()
    if "RRNA PROCESS" in t or "RIBOSOM" in t:
        return "ribosome/rRNA processing"
    if "SPLIC" in t or "RNA PROCESS" in t or "MRNA PROCESS" in t:
        return "RNA processing/splicing"
    if "CHROMATIN" in t or "NUCLEOSOME" in t or "HISTONE" in t:
        return "chromatin organization/remodeling"
    if "DNA REPAIR" in t or "DNA METABOL" in t or "DNA DAMAGE" in t:
        return "DNA repair/DNA-damage response"
    if any(x in t for x in ["LIPID", "FATTY ACID", "PEROXISOM", "CHOLESTEROL", "BILE", "SMALL MOLECULE", "CATABOL", "METABOL"]):
        return "hepatic lipid/fatty-acid/peroxisomal metabolism"
    return "other"


def enrichment(rankings: pd.DataFrame) -> pd.DataFrame:
    cache = OUT / "component_pathway_enrichment.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    target = rankings[rankings.measurement.eq("consensus")]
    total = target.groupby(["timepoint", "component"]).ngroups * len(GMT)
    done, started = 0, time.monotonic()
    for (timepoint, component), group in target.groupby(["timepoint", "component"], sort=False):
        ranking = group[["gene_symbol", "absolute_attribution"]].sort_values("absolute_attribution", ascending=False)
        for source, filename in GMT.items():
            result = gp.prerank(rnk=ranking, gene_sets=str(GMT_ROOT / filename), min_size=10, max_size=500,
                                permutation_num=N_PERM, threads=8, seed=SEED, outdir=None, verbose=False).res2d
            result = result.rename(columns={"Term": "pathway", "ES": "es", "NES": "nes", "NOM p-val": "nominal_p", "FDR q-val": "fdr", "Lead_genes": "leading_edge"})
            result["timepoint"], result["component"], result["source"] = timepoint, component, source
            rows.append(result[["timepoint", "component", "source", "pathway", "es", "nes", "nominal_p", "fdr", "leading_edge"]])
            done += 1; elapsed = time.monotonic() - started
            print(f"[GSEA heartbeat] {done}/{total} elapsed={elapsed/60:.1f}m eta={(elapsed/done*(total-done))/60:.1f}m", flush=True)
    answer = pd.concat(rows, ignore_index=True)
    answer["family"] = answer.pathway.map(pathway_family)
    answer.to_parquet(cache, index=False)
    answer.to_csv(OUT / "component_pathway_enrichment.csv.gz", index=False, compression="gzip")
    return answer


def family_summary(enrichment_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = ["RNA processing/splicing", "ribosome/rRNA processing", "chromatin organization/remodeling",
                "DNA repair/DNA-damage response", "hepatic lipid/fatty-acid/peroxisomal metabolism"]
    for (timepoint, component), group in enrichment_table.groupby(["timepoint", "component"]):
        for family in families:
            z = group[group.family.eq(family)].copy()
            significant = z[(z.fdr < .05) & (z.nes > 0)]
            best = z.sort_values(["fdr", "nes"], ascending=[True, False]).iloc[0] if len(z) else None
            rows.append({"timepoint": timepoint, "component": component, "family": family,
                         "significant_positive_terms": len(significant),
                         "best_pathway": best.pathway if best is not None else None,
                         "best_nes": best.nes if best is not None else np.nan,
                         "best_fdr": best.fdr if best is not None else np.nan,
                         "best_positive_nes": significant.nes.max() if len(significant) else np.nan,
                         "best_positive_fdr": significant.fdr.min() if len(significant) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "component_program_family_summary.csv", index=False)
    top = (enrichment_table.assign(abs_nes=enrichment_table.nes.abs())
           .sort_values(["timepoint", "component", "fdr", "abs_nes"], ascending=[True, True, True, False])
           .groupby(["timepoint", "component"], as_index=False, group_keys=False).head(10))
    top.to_csv(OUT / "top_component_pathways.csv", index=False)
    return result


def conventional_comparison(rankings: pd.DataFrame, enrichment_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge = pd.read_csv(FULLVOC / "bridge_vocab_edger.csv.gz")
    conventional_pathway = pd.read_parquet(FULLVOC / "bridge_vocab_pathway_enrichment.parquet")
    ids = {"RR3-39": "C01__OSD-137__RR3__39-day", "RR3-40": "C02__OSD-137__RR3__40-day"}
    gene_rows, pathway_rows = [], []
    for timepoint, cid in ids.items():
        de = edge[(edge.contrast_id.eq(cid)) & edge.tested].dropna(subset=["signed_statistic"]).copy()
        de["absolute_expression_statistic"] = de.signed_statistic.abs()
        for component in ["parallel", "orthogonal"]:
            ig = rankings[(rankings.timepoint.eq(timepoint)) & rankings.measurement.eq("consensus") & rankings.component.eq(component)]
            merged = de.merge(ig[["gene_symbol", "absolute_attribution", "absolute_rank"]], on="gene_symbol")
            n = len(merged); a = set(merged.nlargest(100, "absolute_expression_statistic").gene_symbol); b = set(merged.nsmallest(100, "absolute_rank").gene_symbol)
            overlap = len(a & b)
            gene_rows.append({"timepoint": timepoint, "component": component, "genes_compared": n,
                              "spearman_abs_expression_vs_abs_attribution": spearmanr(merged.absolute_expression_statistic, merged.absolute_attribution).statistic,
                              "top100_overlap": overlap, "top100_expected": 10000 / n,
                              "top100_hypergeom_p": hypergeom.sf(overlap - 1, n, 100, 100)})
        conv = conventional_pathway[(conventional_pathway.contrast_id.eq(cid))]
        for family in ["RNA processing / splicing"]:
            z = conv[conv.family.eq(family)].sort_values(["fdr", "nes"], ascending=[True, False])
            best = z.iloc[0]
            pathway_rows.append({"timepoint": timepoint, "evidence": "conventional expression", "family": family,
                                 "best_pathway": best.pathway, "nes": best.nes, "fdr": best.fdr})
        for component in ["parallel", "orthogonal"]:
            z = enrichment_table[(enrichment_table.timepoint.eq(timepoint)) & enrichment_table.component.eq(component) &
                                 enrichment_table.family.isin(["RNA processing/splicing", "ribosome/rRNA processing"])].sort_values(["fdr", "nes"], ascending=[True, False])
            best = z.iloc[0]
            pathway_rows.append({"timepoint": timepoint, "evidence": f"Bridge {component} component", "family": "RNA/ribosome processing",
                                 "best_pathway": best.pathway, "nes": best.nes, "fdr": best.fdr})
    genes = pd.DataFrame(gene_rows); pathways = pd.DataFrame(pathway_rows)
    genes.to_csv(OUT / "conventional_gene_rank_comparison.csv", index=False)
    pathways.to_csv(OUT / "conventional_vs_component_pathway_comparison.csv", index=False)
    return genes, pathways


def gene_set_permutation(rankings: pd.DataFrame) -> pd.DataFrame:
    """Competitive rank test: is each family unusually concentrated in parallel vs orthogonal?"""
    sets = {family: set() for family in ["RNA processing/splicing", "ribosome/rRNA processing", "chromatin organization/remodeling",
                                        "DNA repair/DNA-damage response", "hepatic lipid/fatty-acid/peroxisomal metabolism"]}
    for filename in GMT.values():
        for term, genes in gp.parser.read_gmt(path=str(GMT_ROOT / filename)).items():
            family = pathway_family(term)
            if family in sets:
                sets[family] |= set(genes) & set(GENES)
    rng = np.random.default_rng(SEED)
    rows = []
    for timepoint in SPECS:
        p = rankings[(rankings.timepoint.eq(timepoint)) & rankings.measurement.eq("consensus") & rankings.component.eq("parallel")].set_index("gene_symbol").absolute_attribution
        o = rankings[(rankings.timepoint.eq(timepoint)) & rankings.measurement.eq("consensus") & rankings.component.eq("orthogonal")].set_index("gene_symbol").absolute_attribution
        # Percentiles make scales comparable across the two component targets.
        diff = p.rank(pct=True) - o.rank(pct=True)
        values = diff.reindex(GENES).to_numpy()
        for family, genes in sets.items():
            idx = np.flatnonzero(np.isin(GENES, list(genes)))
            observed = float(np.mean(values[idx]))
            null = np.empty(N_GENE_PERM)
            for i in range(N_GENE_PERM):
                null[i] = values[rng.choice(len(values), len(idx), replace=False)].mean()
            rows.append({"timepoint": timepoint, "family": family, "genes": len(idx),
                         "mean_parallel_minus_orthogonal_rank_percentile": observed,
                         "empirical_two_sided_p": (1 + np.sum(np.abs(null) >= abs(observed))) / (N_GENE_PERM + 1),
                         "null_mean": null.mean(), "null_sd": null.std(ddof=1),
                         "family_rank_by_parallel_preference": np.nan})
    result = pd.DataFrame(rows)
    result["family_rank_by_parallel_preference"] = result.groupby("timepoint").mean_parallel_minus_orthogonal_rank_percentile.rank(ascending=False, method="min").astype(int)
    result.to_csv(OUT / "component_family_competitive_permutation.csv", index=False)
    return result


def random_subspace_control(component_similarity: pd.DataFrame) -> pd.DataFrame:
    null = pd.read_parquet(ROBUST / "random_subspace_metrics.parquet")
    null = null[null.removed_components.eq(2)].copy()
    rows = []
    for timepoint, col in [("RR3-39", "rr3_39_cosine"), ("RR3-40", "rr3_40_cosine")]:
        original = component_similarity[(component_similarity.timepoint.eq(timepoint)) & component_similarity.component.eq("full")].cosine.iloc[0]
        corrected = component_similarity[(component_similarity.timepoint.eq(timepoint)) & component_similarity.component.eq("orthogonal")].cosine.iloc[0]
        observed_change = corrected - original
        random_change = null[col] - original
        rows.append({"timepoint": timepoint, "original_cosine": original, "after_PC1_2_cosine": corrected,
                     "observed_change": observed_change, "random_change_mean": random_change.mean(),
                     "random_change_sd": random_change.std(ddof=1), "random_change_low": random_change.quantile(.025),
                     "random_change_high": random_change.quantile(.975),
                     "empirical_p_random_change_at_most_observed": (1 + np.sum(random_change <= observed_change)) / (len(random_change) + 1),
                     "observed_percentile": np.mean(random_change <= observed_change)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "random_subspace_loss_control.csv", index=False)
    return result


def figures(decomp, similarity, family, bootstrap, random_control):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, timepoint in zip(axes, SPECS):
        z = similarity[similarity.timepoint.eq(timepoint)].set_index("component")
        ax.bar(["Full", "PC1–2", "Orthogonal"], z.loc[["full", "parallel", "orthogonal"], "cosine"], color=["#4C78A8", "#E45756", "#72B7B2"])
        ax.axhline(0, color="black", lw=.8); ax.set(ylim=(-1, 1), ylabel="Original vs remeasurement cosine", title=timepoint)
        for i, value in enumerate(z.loc[["full", "parallel", "orthogonal"], "cosine"]): ax.text(i, value + (.04 if value >= 0 else -.08), f"{value:.3f}", ha="center")
    fig.suptitle("RR3 response replication by controlled-reference component")
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"component_replication_similarity.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    pivot = family.pivot_table(index="family", columns=["timepoint", "component"], values="best_nes").reindex(columns=pd.MultiIndex.from_product([list(SPECS), ["parallel", "orthogonal"]]))
    fdr = family.pivot_table(index="family", columns=["timepoint", "component"], values="best_fdr").reindex(index=pivot.index, columns=pivot.columns)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained"); im = ax.imshow(pivot, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    ax.set(yticks=range(len(pivot)), yticklabels=pivot.index, xticks=range(len(pivot.columns)), xticklabels=[f"{a}\n{b}" for a, b in pivot.columns], title="Significant component-associated programs")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                marker = "*" if fdr.iloc[i, j] < .05 else ""
                ax.text(j, i, f"{v:.2f}{marker}", ha="center", va="center", color="white" if abs(v) > 1.35 else "black")
    fig.colorbar(im, ax=ax, label="Best family NES (* FDR < 0.05)")
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"component_programs.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    raw = pd.read_parquet(OUT / "paired_bootstrap_stability.parquet")
    for ax, timepoint in zip(axes, SPECS):
        z = raw[raw.timepoint.eq(timepoint)]
        ax.hist(z.loss_after_removal, bins=45, color="#9C755F", alpha=.8)
        observed = random_control[random_control.timepoint.eq(timepoint)].observed_change.iloc[0]
        ax.axvline(observed, color="black", lw=2, label=f"Observed {observed:+.3f}")
        ax.set(xlabel="Orthogonal cosine − full cosine", ylabel="Paired bootstrap replicates", title=timepoint); ax.legend()
    fig.suptitle("Sample-level stability of PC1–2 removal effect")
    for ext in ["png", "pdf"]: fig.savefig(FIG / f"bootstrap_removal_effect.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def compact_summary(decomp, similarity, family, bootstrap, random_control, conventional_paths):
    biological = pd.read_csv(BIO / "independent_contrast_summary.csv")
    ids = {"RR3-39": "C01__OSD-137__RR3__39-day", "RR3-40": "C02__OSD-137__RR3__40-day"}
    rows = []
    for timepoint, cid in ids.items():
        bio = biological[biological.contrast_id.eq(cid)].iloc[0]
        d = decomp[(decomp.timepoint.eq(timepoint)) & decomp.measurement.eq("OSD-137 original")].iloc[0]
        s = similarity[similarity.timepoint.eq(timepoint)].set_index("component")
        conv = conventional_paths[(conventional_paths.timepoint.eq(timepoint)) & conventional_paths.evidence.eq("conventional expression")].iloc[0]
        b = bootstrap[bootstrap.timepoint.eq(timepoint)].iloc[0]
        rc = random_control[random_control.timepoint.eq(timepoint)].iloc[0]
        def top(component):
            z = family[(family.timepoint.eq(timepoint)) & family.component.eq(component)].copy()
            z["abs_nes"] = z.best_nes.abs(); z = z.sort_values(["best_fdr", "abs_nes"], ascending=[True, False]).head(3)
            return "; ".join(f"{r.family} (NES {r.best_nes:+.2f}, FDR {r.best_fdr:.3g}{'' if r.best_fdr < .05 else ', NS'})" for r in z.itertuples())
        rows.append({"Metric": timepoint, "FLT_GC_n_biological": f"{bio.n_FLT}/{bio.n_GC}", "FLT_GC_n_technical_matched": f"{int(d.n_FLT)}/{int(d.n_GC)}",
                     "conventional_RNA_processing_NES": conv.nes, "conventional_RNA_processing_FDR": conv.fdr,
                     "original_replication_cosine": s.loc["full", "cosine"], "cosine_after_PC1_2_removal": s.loc["orthogonal", "cosine"],
                     "change_after_removal": s.loc["orthogonal", "cosine"] - s.loc["full", "cosine"],
                     "original_total_response_norm": d.total_norm, "original_PC1_2_aligned_fraction": d.aligned_fraction,
                     "parallel_replication_cosine": s.loc["parallel", "cosine"], "orthogonal_replication_cosine": s.loc["orthogonal", "cosine"],
                     "top_parallel_programs": top("parallel"), "top_orthogonal_programs": top("orthogonal"),
                     "bootstrap_change_median": b.loss_after_removal_median, "bootstrap_change_95CI": f"{b.loss_after_removal_ci_low:.3f} to {b.loss_after_removal_ci_high:.3f}",
                     "random_subspace_p": rc.empirical_p_random_change_at_most_observed})
    answer = pd.DataFrame(rows)
    answer.to_csv(OUT / "compact_rr3_39_vs_40_summary.csv", index=False)
    return answer


def main(args):
    OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    specs, x, z, manifest = design(); basis, evr = controlled_basis()
    samples, variables = metadata_audit(specs, manifest)
    decomp, vectors = decomposition(specs, z, basis)
    bootstrap_raw, bootstrap = bootstrap_stability(specs, z, basis)
    if not args.reuse_attributions:
        jobs = []
        for timepoint, device in zip(SPECS, args.devices):
            command = [sys.executable, __file__, "--worker", "--timepoint", timepoint, "--device", device,
                       "--ig-steps", str(args.ig_steps), "--path-batch", str(args.path_batch)]
            print("[launch]", " ".join(command), flush=True); jobs.append(subprocess.Popen(command))
        for job in jobs:
            if job.wait() != 0: raise RuntimeError(f"Attribution worker failed: {job.args}")
    rankings, arrays = load_attributions()
    enriched = enrichment(rankings); families = family_summary(enriched)
    genes, conventional_paths = conventional_comparison(rankings, enriched)
    competitive = gene_set_permutation(rankings)
    similarity = pd.read_csv(OUT / "component_replication_similarity.csv")
    random_control = random_subspace_control(similarity)
    figures(decomp, similarity, families, bootstrap, random_control)
    compact = compact_summary(decomp, similarity, families, bootstrap, random_control, conventional_paths)
    # Classification is data-driven but conservative; write evidence needed for manual audit.
    r39 = compact.set_index("Metric").loc["RR3-39"]; r40 = compact.set_index("Metric").loc["RR3-40"]
    rna = competitive[competitive.family.eq("RNA processing/splicing")].set_index("timepoint")
    stable39 = bootstrap.set_index("timepoint").loc["RR3-39"]
    if stable39.loss_after_removal_ci_high >= 0:
        decision = "D. UNSTABLE / UNDERPOWERED"
    elif (rna.loc["RR3-39", "empirical_two_sided_p"] < .05 and rna.loc["RR3-39", "mean_parallel_minus_orthogonal_rank_percentile"] > 0 and
          rna.loc["RR3-39", "mean_parallel_minus_orthogonal_rank_percentile"] > rna.loc["RR3-40", "mean_parallel_minus_orthogonal_rank_percentile"]):
        decision = "A. FUNCTIONAL OVERLAP SUPPORTED"
    elif r39.parallel_replication_cosine > r39.orthogonal_replication_cosine:
        decision = "B. BROAD BIOLOGICAL OVERLAP"
    else:
        decision = "C. NO FUNCTIONAL EXPLANATION"
    result = {
        "decision": decision, "controlled_reference": "uncentered T-cell PolyA/Ribo PC1-2",
        "PC1_variance_fraction": float(evr[0]), "PC1_2_variance_fraction": float(evr[:2].sum()),
        "RR3_39_change_after_removal": float(r39.change_after_removal), "RR3_40_change_after_removal": float(r40.change_after_removal),
        "RR3_39_random_subspace_p": float(r39.random_subspace_p), "RR3_40_random_subspace_p": float(r40.random_subspace_p),
        "RR3_39_RNA_parallel_preference": float(rna.loc["RR3-39", "mean_parallel_minus_orthogonal_rank_percentile"]),
        "RR3_39_RNA_competitive_p": float(rna.loc["RR3-39", "empirical_two_sided_p"]),
        "RR3_40_RNA_parallel_preference": float(rna.loc["RR3-40", "mean_parallel_minus_orthogonal_rank_percentile"]),
        "RR3_40_RNA_competitive_p": float(rna.loc["RR3-40", "empirical_two_sided_p"]),
        "limitations": ["RR3-39 has 2 FLT/2 GC; matched RR3-40 technical comparison has 2/2.",
                        "Component IG identifies input influence on a latent target; it does not label dimensions as technical-only.",
                        "Matched random 2D controls test geometric loss; functional specificity uses gene-set competitive permutations because random-subspace IG would require new gradients per subspace.",
                        "OSD metadata supports same RR3 RNA material and exact animal matches, but does not independently prove aliquot identity for every library."],
    }
    (OUT / "decision_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "bridge_frozen": True, "embeddings_recomputed": False,
                  "samples_redefined": False, "controlled_reference_redefined": False, "ig_baseline": "all-zero log1p(TPM)",
                  "ig_steps": args.ig_steps, "gsea_permutations": N_PERM, "gsea_seed": SEED,
                  "paired_bootstrap_replicates": N_BOOT, "gene_set_competitive_permutations": N_GENE_PERM,
                  "random_subspace_source": str((ROBUST / "random_subspace_metrics.parquet").relative_to(REPO))}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nCompact summary\n", compact.to_string(index=False), flush=True)
    print("\nDecision\n", json.dumps(result, indent=2), flush=True)
    print("[complete]", OUT, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timepoint", choices=list(SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--path-batch", type=int, default=4)
    parser.add_argument("--reuse-attributions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        attribution_worker(arguments.timepoint, arguments.device, arguments.ig_steps, arguments.path_batch)
    else:
        main(arguments)
