#!/usr/bin/env python3
"""Audit and freeze the Zhong benchmark exclusively from saved outputs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
RESULTS = BENCH / "results"
FIGURES = BENCH / "figures"
REFERENCE = BENCH / "references" / "gene-embedding-benchmarks"
SHARED = Path("/home/walt/bridge-rna")

REP_LABELS = {"E0_static": "E0", "E1_null_L12": "E1", "E2_GTEx200_L12": "E2"}
TASK_LABELS = {"SL": "Synthetic lethal", "NG": "Negative genetic", "TF": "TF-target"}
COLORS = {"E0": "#7f7f7f", "E1": "#2878B5", "E2": "#D95319"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_and_assert() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single = pd.read_csv(RESULTS / "single_gene_zhong_matched_summary.csv")
    pair_folds = pd.read_csv(RESULTS / "pair_outer_fold_results_all.csv")
    pair_summary = pd.read_csv(RESULTS / "pair_representation_summary.csv")
    leakage = pd.read_csv(RESULTS / "pair_leakage_qc.csv")
    assert len(single) == 6
    assert single.groupby("Task")["N_terms"].first().to_dict() == {"GO": 56, "OMIM": 103}
    assert len(pair_folds) == 81 and len(pair_summary) == 27
    assert (pair_folds.groupby(["interaction_type", "representation", "joining_operation"]).size() == 3).all()
    assert not pair_folds[["AUROC", "AUPRC"]].isna().any().any()
    assert (leakage["train_test_gene_overlap"] == 0).all()
    return single, pair_folds, pair_summary, leakage


def make_tables(single: pd.DataFrame, pair_summary: pd.DataFrame, leakage: pd.DataFrame) -> None:
    single_table = pd.DataFrame(
        {
            "Biological_scale": "Single gene",
            "Task": single["Task"],
            "Representation": single["Representation"],
            "Pair_operation": "—",
            "AUROC": single["Mean_AUROC"],
            "AUPRC": single["Mean_AUPRC"],
            "Aggregation": single["N_terms"].map(lambda n: f"unweighted mean across {n} terms"),
        }
    )
    pair_table = pair_summary.rename(
        columns={
            "interaction_type": "Task",
            "representation": "Representation",
            "joining_operation": "Pair_operation",
            "AUROC_mean": "AUROC",
            "AUPRC_mean": "AUPRC",
        }
    )
    pair_table["Task"] = pair_table["Task"].map(TASK_LABELS)
    pair_table["Representation"] = pair_table["Representation"].map(REP_LABELS)
    pair_table["Biological_scale"] = "Gene pair"
    pair_table["Aggregation"] = "mean across 3 outer folds"
    pair_table = pair_table[
        ["Biological_scale", "Task", "Representation", "Pair_operation", "AUROC", "AUPRC", "Aggregation"]
    ]
    pd.concat([single_table, pair_table], ignore_index=True).to_csv(
        RESULTS / "final_benchmark_summary.csv", index=False
    )

    best = pair_summary.loc[pair_summary.groupby("interaction_type")["AUROC_mean"].idxmax()].copy()
    best["Task"] = best["interaction_type"].map(TASK_LABELS)
    best["Representation"] = best["representation"].map(REP_LABELS)
    best.rename(
        columns={"joining_operation": "Pair_operation", "AUROC_mean": "AUROC", "AUPRC_mean": "AUPRC"}
    )[["Task", "Representation", "Pair_operation", "AUROC", "AUPRC", "AUROC_sd", "AUPRC_sd"]].to_csv(
        RESULTS / "final_pair_best_conditions.csv", index=False
    )

    leak_unique = leakage.drop_duplicates(
        ["task", "outer_fold", "train_unique_genes", "test_unique_genes", "n_train_pairs", "n_test_pairs"]
    ).sort_values(["task", "outer_fold"])
    assert len(leak_unique) == 9 and (leak_unique["train_test_gene_overlap"] == 0).all()
    leak_unique.to_csv(RESULTS / "final_pair_leakage_qc.csv", index=False)


def figure_ablation(single: pd.DataFrame) -> None:
    paired = {
        task: pd.read_csv(RESULTS / f"E0_E1_E2_{task}_paired.csv") for task in ["GO", "OMIM"]
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, task in enumerate(["GO", "OMIM"]):
        data = paired[task]
        for col, metric in enumerate(["AUROC", "AUPRC"]):
            ax = axes[row, col]
            cols = [f"E0_static_{metric}", f"E1_null_L12_{metric}", f"E2_GTEx200_L12_{metric}"]
            x = np.arange(3)
            for values in data[cols].to_numpy(float):
                ax.plot(x, values, color="#777777", alpha=0.13, linewidth=0.7)
            means = data[cols].mean().to_numpy()
            ax.plot(x, means, color="#111111", marker="o", linewidth=2.2, label="Mean")
            for xi, value, rep in zip(x, means, ["E0", "E1", "E2"]):
                ax.scatter(xi, value, color=COLORS[rep], s=55, zorder=3)
            ax.set_title(f"{task} {metric}")
            ax.set_ylabel(metric)
            ax.set_xticks(x, ["E0 static", "E1 null L12", "E2 GTEx-200 L12"])
            ax.grid(axis="y", alpha=0.22)
            if row == 0 and col == 0:
                ax.legend(frameon=False)
    fig.suptitle("Bridge representation ablation: individual terms and unweighted means", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "final_representation_ablation")


def figure_pairs(pair_folds: pd.DataFrame) -> None:
    df = pair_folds.copy()
    df["Representation"] = df["representation"].map(REP_LABELS)
    operations = ["sum", "product", "concat"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, task in zip(axes, ["SL", "NG", "TF"]):
        sub = df[df["interaction_type"] == task]
        x = np.arange(3)
        offsets = {"E0": -0.22, "E1": 0.0, "E2": 0.22}
        for rep in ["E0", "E1", "E2"]:
            means = []
            for i, op in enumerate(operations):
                values = sub[(sub["Representation"] == rep) & (sub["joining_operation"] == op)]["AUROC"]
                means.append(values.mean())
                ax.scatter(np.full(len(values), x[i] + offsets[rep]), values, color=COLORS[rep], alpha=0.5, s=22)
            ax.plot(x + offsets[rep], means, color=COLORS[rep], marker="o", linewidth=2, label=rep)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x, operations)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Pair operation")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Outer-fold AUROC")
    axes[-1].legend(title="Representation", frameon=False)
    fig.suptitle("Gene-pair benchmark: all conditions and outer-fold observations", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "final_gene_pair_benchmark")


def figure_scale(single: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    pair_best_rep = pair_summary.loc[
        pair_summary.groupby(["interaction_type", "representation"])["AUROC_mean"].idxmax()
    ].copy()
    pair_best_rep["Representation"] = pair_best_rep["representation"].map(REP_LABELS)
    pair_best_rep["Task"] = pair_best_rep["interaction_type"].map(TASK_LABELS)
    figure_data = pd.concat(
        [
            single.assign(
                Biological_scale="Single gene",
                AUROC=single["Mean_AUROC"],
                Operation="—",
            )[["Biological_scale", "Task", "Representation", "Operation", "AUROC"]],
            pair_best_rep.assign(
                Biological_scale="Gene pair",
                Operation=pair_best_rep["joining_operation"],
                AUROC=pair_best_rep["AUROC_mean"],
            )[["Biological_scale", "Task", "Representation", "Operation", "AUROC"]],
        ],
        ignore_index=True,
    )
    figure_data.to_csv(RESULTS / "final_biological_scale_figure_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, scale, tasks in [
        (axes[0], "Single gene", ["GO", "OMIM"]),
        (axes[1], "Gene pair", ["Synthetic lethal", "Negative genetic", "TF-target"]),
    ]:
        sub = figure_data[figure_data["Biological_scale"] == scale]
        x = np.arange(len(tasks))
        for rep, offset in zip(["E0", "E1", "E2"], [-0.2, 0.0, 0.2]):
            values = [sub[(sub["Task"] == task) & (sub["Representation"] == rep)]["AUROC"].iloc[0] for task in tasks]
            ax.scatter(x + offset, values, s=65, color=COLORS[rep], label=rep)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xticks(x, tasks, rotation=15, ha="right")
        ax.set_title(f"{scale} aggregation")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(title="Representation", frameon=False)
    fig.suptitle("Biological-scale summary (panels use distinct aggregation units)", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "final_biological_scale_summary")


def freeze_provenance(single: pd.DataFrame, pair_summary: pd.DataFrame, leakage: pd.DataFrame) -> None:
    config = SHARED / "model/r7hnr92k/config.json"
    checkpoint = SHARED / "model/r7hnr92k/best_model.pt"
    vocab = SHARED / "data/ensembl/canonical_genes.csv"
    split_relpaths = [
        "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold1_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold2_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold3_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/go_folds_splits/go_holdout_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold1_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold2_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold3_dict_all.pkl",
        "data/data_splits/gene_level_benchmark/omim_folds_splits/omim_holdout_dict_all.pkl",
        "data/data_splits/gene_pair_benchmark/sl_nested_cv_splits.pkl",
        "data/data_splits/gene_pair_benchmark/ng_nested_cv_splits.pkl",
        "data/data_splits/gene_pair_benchmark/tf_nested_cv_splits.pkl",
    ]
    mapping = pd.read_csv(BENCH / "bridge_gene_mapping.tsv", sep="\t")
    coverage = pd.read_csv(BENCH / "coverage/coverage_summary.csv")
    pair_coverage = pd.read_csv(BENCH / "coverage/pair_coverage.csv")
    gtex_manifest = BENCH / "representations/E2_GTEx200_L12/gtex200_samples.tsv"
    frozen_manifest = RESULTS / "gtex200_samples.tsv"
    shutil.copyfile(gtex_manifest, frozen_manifest)
    e0 = json.loads((BENCH / "representations/E0_static/provenance.json").read_text())
    e1 = json.loads((BENCH / "representations/E1_null_L12/provenance.json").read_text())
    e2 = json.loads((BENCH / "representations/E2_GTEx200_L12/provenance.json").read_text())
    provenance = {
        "status": "frozen_complete",
        "scope": "audit and consolidation from saved results only; no inference, SVM, CV, split, or control reruns",
        "bridge": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "model_config_path": str(config),
            "model_config_sha256": sha256(config),
            "model_config": json.loads(config.read_text()),
            "vocabulary_path": str(vocab),
            "vocabulary_sha256": sha256(vocab),
            "vocabulary_size": 15165,
            "embedding_dimension": 512,
            "gene_ordering": "canonical token_id ascending; row_index = token_id - 1",
            "expression_preprocessing": "natural log1p(TPM)",
            "mask_value": -10.0,
            "species_embedding_used": False,
        },
        "representations": {"E0": e0, "E1": e1, "E2": e2},
        "gtex200_manifest": {
            "path": str(frozen_manifest.relative_to(ROOT)),
            "sha256": sha256(frozen_manifest),
            "rows": 200,
            "tissue_counts": pd.read_csv(frozen_manifest, sep="\t")["context"].value_counts().sort_index().to_dict(),
            "aggregation": "unweighted arithmetic mean of final layer-12 hidden states across 200 samples",
            "vector_normalization": "none",
        },
        "zhong": {
            "upstream_repository": "https://github.com/ylaboratory/gene-embedding-benchmarks",
            "upstream_git_commit": "d1320026a2a4ee033d49517f91e2d1c2ccc8df1e",
            "split_files": [
                {"path": rel, "sha256": sha256(REFERENCE / rel)} for rel in split_relpaths
            ],
            "database_resources": {
                "GO": {
                    "description": "Upstream GO resource at the frozen commit",
                    "path": "data/obo/go.obo",
                    "sha256": sha256(REFERENCE / "data/obo/go.obo"),
                },
                "OMIM_DOID": {
                    "description": "Upstream disease ontology and OMIM-derived fixed splits at the frozen commit",
                    "doid_path": "data/obo/doid.obo",
                    "doid_sha256": sha256(REFERENCE / "data/obo/doid.obo"),
                    "omim_gmt_path": "data/gmt/omim.20231030.prop.gmt",
                    "omim_gmt_sha256": sha256(REFERENCE / "data/gmt/omim.20231030.prop.gmt"),
                },
                "pair_sources": "Upstream SL, NG, and TF fixed nested-CV split files at the frozen commit",
                "ncbi_gene_info": {
                    "url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
                    "sha256": sha256(BENCH / "references/Homo_sapiens.gene_info.gz"),
                    "note": "Per-row NCBI Modification_date is retained in the frozen source snapshot.",
                },
            },
        },
        "gene_mapping": {
            "bridge_vocabulary_size": 15165,
            "mapped_unique_entrez": int((mapping["map_status"] == "mapped").sum()),
            "unmapped": int((mapping["map_status"] == "unmapped").sum()),
            "ambiguous_excluded": int(mapping["map_status"].str.startswith("ambiguous").sum()),
            "duplicate_entrez_excluded": int(mapping["is_duplicate_entrez_mapping"].sum()),
            "benchmark_union_genes": int(coverage.loc[coverage.metric == "benchmark_genes_union", "value"].iloc[0]),
            "overlapping_genes": int(coverage.loc[coverage.metric == "overlapping_genes", "value"].iloc[0]),
            "mapping_file": str((BENCH / "bridge_gene_mapping.tsv").relative_to(ROOT)),
            "mapping_file_sha256": sha256(BENCH / "bridge_gene_mapping.tsv"),
            "single_gene_terms": {"GO": 56, "OMIM": 103},
            "pair_coverage": pair_coverage.to_dict(orient="records"),
        },
        "pair_split_rigor": {
            "all_outer_folds_gene_disjoint": bool((leakage["train_test_gene_overlap"] == 0).all()),
            "unique_fold_records": pd.read_csv(RESULTS / "final_pair_leakage_qc.csv").to_dict(orient="records"),
        },
        "final_results": {
            "single_gene": single.to_dict(orient="records"),
            "pair_all_conditions": pair_summary.to_dict(orient="records"),
        },
        "limitations": [
            "Bridge uses its own available vocabulary and a benchmark adaptation; literature values are not a formal head-to-head rerun.",
            "Single-gene metrics are unweighted across biological terms; pair metrics are means across three outer folds.",
            "No permutation or Gaussian controls were run for the frozen final benchmark.",
        ],
    }
    (RESULTS / "final_benchmark_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    single, pair_folds, pair_summary, leakage = load_and_assert()
    make_tables(single, pair_summary, leakage)
    figure_ablation(single)
    figure_pairs(pair_folds)
    figure_scale(single, pair_summary)
    freeze_provenance(single, pair_summary, leakage)
    print("Final saved-output audit complete: 6 single-gene rows, 27 pair conditions, zero pair gene overlap.")


if __name__ == "__main__":
    main()
