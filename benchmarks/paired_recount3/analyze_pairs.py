#!/usr/bin/env python3
"""Analyze expression agreement and paired embedding retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def row_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", left_centered, right_centered)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(right_centered, axis=1)
    return np.divide(numerator, denominator, out=np.full(len(left), np.nan), where=denominator > 0)


def row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = np.empty(len(left), dtype=np.float64)
    for i in range(len(left)):
        values[i] = row_pearson(
            rankdata(left[i])[None, :], rankdata(right[i])[None, :]
        )[0]
    return values


def normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm > 0)


def load_archs4_embeddings(samples: pd.DataFrame, embedding_dir: Path) -> np.ndarray:
    locations = pd.read_parquet(embedding_dir / "sample_locations.parquet")
    index = samples[["gsm"]].merge(
        locations[["geo_accession", "global_index"]], left_on="gsm",
        right_on="geo_accession", how="left", validate="one_to_one",
    )
    if index["global_index"].isna().any():
        raise ValueError("Some paired GSMs lack precomputed ARCHS4 embeddings")
    spec = json.loads((embedding_dir / "embedding_manifest.json").read_text())
    dtype = np.dtype(spec.get("embedding_dtype", "float16"))
    matrix = np.memmap(
        embedding_dir / f"sample_embeddings.{dtype.name}.mmap", mode="r", dtype=dtype,
        shape=(int(spec["total_samples"]), int(spec["embedding_dim"])),
    )
    return np.asarray(matrix[index["global_index"].astype(int).to_numpy()], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-dir", type=Path, default=HERE / "outputs/paired_expression")
    parser.add_argument("--recount3-embeddings", type=Path, default=HERE / "outputs/recount3_embeddings.npy")
    parser.add_argument("--archs4-embedding-dir", type=Path, default=REPO_ROOT / "embeddings/archs4")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/results")
    parser.add_argument("--skip-spearman", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = pd.read_parquet(args.expression_dir / "samples.parquet")
    arch_expression = np.load(args.expression_dir / "archs4_log1p_tpm.npy", mmap_mode="r")
    recount_expression = np.load(args.expression_dir / "recount3_log1p_tpm.npy", mmap_mode="r")
    recount_embeddings = np.load(args.recount3_embeddings)
    arch_embeddings = load_archs4_embeddings(samples, args.archs4_embedding_dir)
    expected = len(samples)
    if any(matrix.shape[0] != expected for matrix in [arch_expression, recount_expression,
                                                       recount_embeddings, arch_embeddings]):
        raise ValueError("Sample counts differ among paired artifacts")

    metrics = samples.copy()
    metrics["expression_pearson"] = row_pearson(arch_expression, recount_expression)
    if not args.skip_spearman:
        metrics["expression_spearman"] = row_spearman(arch_expression, recount_expression)
    arch_unit, recount_unit = normalize(arch_embeddings), normalize(recount_embeddings)
    similarity = recount_unit @ arch_unit.T
    metrics["embedding_cosine"] = np.diag(similarity)
    order = np.argsort(-similarity, axis=1)
    metrics["paired_retrieval_rank"] = np.argmax(
        order == np.arange(expected)[:, None], axis=1
    ) + 1
    metrics["recall_at_1"] = metrics["paired_retrieval_rank"].le(1)
    metrics["recall_at_5"] = metrics["paired_retrieval_rank"].le(5)
    metrics["recall_at_10"] = metrics["paired_retrieval_rank"].le(10)
    gse = metrics["gse"].astype(str).to_numpy()
    metrics["top1_same_gse"] = gse[order[:, 0]] == gse
    metrics["top5_same_gse"] = np.asarray([
        np.any(gse[neighbors] == gse[index])
        for index, neighbors in enumerate(order[:, :5])
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.output_dir / "paired_metrics.parquet", index=False)
    summary = metrics.groupby("cohort").agg(
        samples=("gsm", "size"), median_expression_pearson=("expression_pearson", "median"),
        median_embedding_cosine=("embedding_cosine", "median"),
        median_retrieval_rank=("paired_retrieval_rank", "median"),
        recall_at_1=("recall_at_1", "mean"), recall_at_5=("recall_at_5", "mean"),
        recall_at_10=("recall_at_10", "mean"),
        top1_same_gse=("top1_same_gse", "mean"), top5_same_gse=("top5_same_gse", "mean"),
    ).reset_index()
    if "expression_spearman" in metrics:
        summary = summary.merge(
            metrics.groupby("cohort")["expression_spearman"].median().rename(
                "median_expression_spearman"
            ), on="cohort",
        )
    overall = pd.DataFrame([{
        "cohort": "overall", "samples": len(metrics),
        "median_expression_pearson": metrics["expression_pearson"].median(),
        "median_embedding_cosine": metrics["embedding_cosine"].median(),
        "median_retrieval_rank": metrics["paired_retrieval_rank"].median(),
        "recall_at_1": metrics["recall_at_1"].mean(),
        "recall_at_5": metrics["recall_at_5"].mean(),
        "recall_at_10": metrics["recall_at_10"].mean(),
        "top1_same_gse": metrics["top1_same_gse"].mean(),
        "top5_same_gse": metrics["top5_same_gse"].mean(),
        "median_expression_spearman": (
            metrics["expression_spearman"].median()
            if "expression_spearman" in metrics else np.nan
        ),
    }])
    summary = pd.concat([overall, summary], ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    labels = list(metrics["cohort"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label in labels:
        subset = metrics[metrics["cohort"].eq(label)]
        axes[0].hist(subset["expression_pearson"].dropna(), bins=40, alpha=.55, label=label)
        axes[1].hist(subset["embedding_cosine"].dropna(), bins=40, alpha=.55, label=label)
    axes[0].set(xlabel="Paired expression Pearson r", ylabel="Samples")
    axes[1].set(xlabel="Paired embedding cosine similarity", ylabel="Samples")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "paired_agreement.png", dpi=300)
    fig.savefig(args.output_dir / "paired_agreement.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for label in labels:
        subset = metrics[metrics["cohort"].eq(label)]
        ax.scatter(subset["expression_pearson"], subset["embedding_cosine"],
                   s=14, alpha=.55, label=label)
    ax.set(xlabel="Paired expression Pearson r", ylabel="Paired embedding cosine similarity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "expression_vs_embedding.png", dpi=300)
    fig.savefig(args.output_dir / "expression_vs_embedding.pdf")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
