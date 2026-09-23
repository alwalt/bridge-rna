#!/usr/bin/env python3
"""Run frozen ExpressionPerformer inference/IG for one locked de Weerd cohort."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parents[1]
PREP = HERE / "work/prepared"
CACHE = HERE / "work/bridge"
OUT = HERE / "results/bridge"
ROOT = Path("/home/walt/bridge-rna")
sys.path.insert(0, str(ROOT))
from src.fm_embed.model import load_expression_performer

CHECKPOINT = ROOT / "model/r7hnr92k/best_model.pt"
CHECKPOINT_CONFIG = ROOT / "model/r7hnr92k/config.json"
CANONICAL = ROOT / "data/ensembl/canonical_genes.csv"


def say(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def load_model(device_name):
    genes = pd.read_csv(CANONICAL).gene_symbol.astype(str).tolist()
    model, device = load_expression_performer(CHECKPOINT, CHECKPOINT_CONFIG, len(genes), device_name)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device, genes


def encode(model, values, device, batch_size=2):
    rows = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                         enabled=device.type == "cuda"):
        for start in range(0, len(values), batch_size):
            tensor = torch.as_tensor(np.array(values[start:start + batch_size], copy=True),
                                     dtype=torch.float32, device=device)
            rows.append(model._encode_hidden(tensor).mean(1).float().cpu().numpy())
    return np.concatenate(rows)


def arm_effect(values: np.ndarray, manifest: pd.DataFrame, donor_weighted: bool) -> np.ndarray:
    frame = pd.DataFrame({"donor": manifest.donor, "role": manifest.role})
    if donor_weighted:
        donor_values, roles = [], []
        for (_, role), group in frame.groupby(["donor", "role"], sort=True):
            donor_values.append(values[group.index].mean(axis=0))
            roles.append(role)
        donor_values = np.stack(donor_values)
        roles = np.asarray(roles)
        return donor_values[roles == "case"].mean(0) - donor_values[roles == "control"].mean(0)
    case = manifest.role.eq("case").to_numpy()
    return values[case].mean(0) - values[~case].mean(0)


def leave_donor_out_directions(embeddings, manifest, donor_weighted):
    overall = unit(arm_effect(embeddings, manifest, donor_weighted)).astype(np.float32)
    directions = []
    for donor in manifest.donor:
        kept = manifest.donor.ne(donor).to_numpy()
        subset = manifest.loc[kept].reset_index(drop=True)
        if subset.role.nunique() < 2:
            directions.append(overall)
        else:
            directions.append(unit(arm_effect(embeddings[kept], subset, donor_weighted)).astype(np.float32))
    return overall, np.stack(directions)


def score(model, values, direction):
    return (model._encode_hidden(values).mean(1) * direction).sum(1)


def integrated_gradients(model, values, direction, device, steps=16, path_batch=4):
    baseline = torch.zeros((1, len(values)), device=device)
    observed = torch.as_tensor(np.array(values, copy=True), device=device).view(1, -1)
    target = torch.as_tensor(direction, device=device).view(1, -1)
    total = torch.zeros_like(baseline)
    alphas = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    for start in range(0, steps, path_batch):
        alpha = torch.as_tensor(alphas[start:start + path_batch], device=device).view(-1, 1)
        path = (baseline + alpha * (observed - baseline)).requires_grad_(True)
        gradient = torch.autograd.grad(score(model, path, target.expand(len(path), -1)).sum(), path)[0]
        total += gradient.detach().sum(0, keepdim=True)
    attribution = ((observed - baseline) * total / steps)[0].detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        delta = float(score(model, observed, target) - score(model, baseline, target))
    return attribution, delta, float(attribution.sum() - delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", required=True, choices=["GSE138614", "GSE101794", "GSE72509"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ig-steps", type=int, default=16)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(HERE / "results/manifests/locked_samples.parquet")
    manifest = manifest[manifest.accession.eq(args.accession)].reset_index(drop=True)
    rows = pd.read_csv(PREP / f"{args.accession}_matrix_rows.csv")
    manifest = manifest.merge(rows, on="sample_id", validate="one_to_one").sort_values("matrix_row").reset_index(drop=True)
    expression = np.load(PREP / f"{args.accession}_log1p_tpm.npy", mmap_mode="r")
    model, device, genes = load_model(args.device)

    embedding_path = CACHE / f"{args.accession}_embeddings.npy"
    if embedding_path.exists():
        embeddings = np.load(embedding_path)
    else:
        embeddings = encode(model, expression, device)
        np.save(embedding_path, embeddings)
    donor_weighted = args.accession == "GSE138614"
    direction, sample_directions = leave_donor_out_directions(embeddings, manifest, donor_weighted)
    np.save(CACHE / f"{args.accession}_condition_direction.npy", direction)
    np.save(CACHE / f"{args.accession}_sample_directions.npy", sample_directions)

    attribution_path = CACHE / f"{args.accession}_sample_ig.npy"
    completeness_rows = []
    if attribution_path.exists():
        attributions = np.load(attribution_path)
    else:
        partial_path = CACHE / f"{args.accession}_sample_ig.partial.npy"
        if partial_path.exists():
            partial = np.lib.format.open_memmap(partial_path, mode="r+")
            if partial.shape != (len(manifest), len(genes)):
                raise RuntimeError(f"invalid partial attribution shape: {partial.shape}")
        else:
            partial = np.lib.format.open_memmap(
                partial_path, mode="w+", dtype=np.float32, shape=(len(manifest), len(genes)))
            partial[:] = np.nan
            partial.flush()
        start_time = time.monotonic()
        for index, row in manifest.iterrows():
            if np.isfinite(partial[index, 0]):
                say(f"{args.accession} IG {index + 1}/{len(manifest)} restored from partial cache")
                continue
            attribution, delta, error = integrated_gradients(
                model, expression[row.matrix_row], sample_directions[index], device, args.ig_steps)
            partial[index] = attribution
            partial.flush()
            completeness_rows.append({"accession": args.accession, "sample_id": row.sample_id,
                                      "score_difference": delta, "attribution_sum": float(attribution.sum()),
                                      "completeness_delta": error})
            elapsed = time.monotonic() - start_time
            say(f"{args.accession} IG {index + 1}/{len(manifest)} elapsed={elapsed / 60:.1f}m "
                f"eta={(elapsed / (index + 1)) * (len(manifest) - index - 1) / 60:.1f}m")
        attributions = np.asarray(partial).copy()
        if not np.isfinite(attributions).all():
            raise RuntimeError("partial attribution cache remains incomplete")
        np.save(attribution_path, attributions)
        pd.DataFrame(completeness_rows).to_csv(OUT / f"{args.accession}_ig_completeness.csv", index=False)

    signed = arm_effect(attributions, manifest, donor_weighted)
    signed = signed / max(float(np.abs(signed).sum()), 1e-12)
    ranking = pd.DataFrame({"accession": args.accession, "gene": genes,
                            "signed_bridge_score": signed, "absolute_bridge_score": np.abs(signed)})
    ranking = ranking.sort_values(["absolute_bridge_score", "gene"], ascending=[False, True]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_parquet(OUT / f"{args.accession}_gene_ranking.parquet", index=False)
    embeddings_out = pd.DataFrame(embeddings, columns=[f"z{i}" for i in range(embeddings.shape[1])])
    embeddings_out.insert(0, "sample_id", manifest.sample_id)
    embeddings_out.insert(0, "accession", args.accession)
    embeddings_out.to_parquet(OUT / f"{args.accession}_sample_embeddings.parquet", index=False)
    provenance = {
        "accession": args.accession, "checkpoint": str(CHECKPOINT), "checkpoint_config": str(CHECKPOINT_CONFIG),
        "encoder_frozen": True, "input": "natural log1p(TPM), canonical 15,165 genes",
        "condition_direction": "case-minus-control pooled embedding; donor-weighted for GSE138614",
        "attribution_direction": "leave-one-whole-donor-out",
        "attribution": "signed Integrated Gradients; all-zero baseline; midpoint Riemann",
        "ig_steps": args.ig_steps, "score_normalization": "L1 absolute sum",
    }
    (OUT / f"{args.accession}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    say(f"{args.accession} complete")


if __name__ == "__main__":
    main()
