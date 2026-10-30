#!/usr/bin/env python3
"""Build the public cached-mode gene embedding benchmark notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "gene_embedding_benchmarks.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Gene embedding benchmarks: GO, OMIM, and gene pairing

This is the public, cached-first report for the validated Zhong-style gene-biology benchmark. It evaluates frozen BridgeRNA representations on GO Biological Process, OMIM disease association, and three gene-pair tasks. It also includes a static-only single-gene extension for BulkFormer-50M and BulkFormer-147M.

**Cached mode performs no model inference, SVM fitting, cross-validation, split generation, or controls.** It rebuilds all displayed summaries from compact per-term and per-fold artifacts, checks them against frozen expected tables, audits the pair splits, and regenerates the figures."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

HERE = Path.cwd()
if HERE.name != "gene_embedding_benchmarks":
    HERE = Path("notebooks/gene_embedding_benchmarks")
CACHE = HERE / "cached"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

with (CACHE / "manifest.json").open() as handle:
    manifest = json.load(handle)
assert manifest["mode"].startswith("cached validated artifacts")
for item in manifest["files"]:
    path = CACHE / item["file"]
    assert path.stat().st_size == item["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
print(f"Cached mode: {len(manifest['files'])} validated artifacts")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Representations and evaluation design

- **Bridge E0:** static checkpoint gene embedding, before expression encoding or transformer processing.
- **Bridge E1:** final layer-12 state under uniform `log1p(TPM)=0`; a null context, not a biological average.
- **Bridge E2:** final layer-12 state averaged across a fixed tissue-balanced panel of 200 GTEx samples.
- **BulkFormer static:** checkpoint tensor `gene_emb_onehot_layer.weight`; no BulkFormer forward inference or expression context.

For every GO or OMIM term, the RBF-SVM hyperparameter was selected using only three supplied training folds. The selected classifier was refit on all training genes and evaluated once on the untouched holdout. The headline statistic is the unweighted mean of one final held-out AUROC per term. Pair results are means across three supplied, strictly gene-disjoint outer folds."""
        ),
        nbf.v4.new_markdown_cell("## Single-gene GO and OMIM results"),
        nbf.v4.new_code_cell(
            """def summarize_terms(frame, task, model, embedding_dim):
    return {
        "Task": task, "Model": model, "N_terms": len(frame), "Embedding_dim": embedding_dim,
        "Mean_AUROC": frame.AUROC.mean(), "Median_AUROC": frame.AUROC.median(),
        "SD_AUROC": frame.AUROC.std(ddof=1), "Mean_AUPRC": frame.AUPRC.mean(),
        "Median_AUPRC": frame.AUPRC.median(), "SD_AUPRC": frame.AUPRC.std(ddof=1),
    }

single_specs = [
    ("GO", "Bridge E0", 512, "bridge_E0_GO.csv"),
    ("GO", "Bridge E1", 512, "bridge_E1_GO.csv"),
    ("GO", "Bridge E2", 512, "bridge_E2_GO.csv"),
    ("GO", "BulkFormer-50M static", 256, "bulkformer_50M_GO.csv"),
    ("GO", "BulkFormer-147M static", 640, "bulkformer_147M_GO.csv"),
    ("OMIM", "Bridge E0", 512, "bridge_E0_OMIM.csv"),
    ("OMIM", "Bridge E1", 512, "bridge_E1_OMIM.csv"),
    ("OMIM", "Bridge E2", 512, "bridge_E2_OMIM.csv"),
    ("OMIM", "BulkFormer-50M static", 256, "bulkformer_50M_OMIM.csv"),
    ("OMIM", "BulkFormer-147M static", 640, "bulkformer_147M_OMIM.csv"),
]
rows = []
for task, model, dim, filename in single_specs:
    frame = pd.read_csv(CACHE / filename)
    expected_n = 56 if task == "GO" else 103
    assert len(frame) == expected_n and frame.term.nunique() == expected_n
    assert np.isfinite(frame[["AUROC", "AUPRC"]]).all().all()
    rows.append(summarize_terms(frame, task, model, dim))
single_summary = pd.DataFrame(rows)

# Compare rebuilt summaries with the frozen research summaries.
bridge_expected = pd.read_csv(CACHE / "single_gene_expected_summary.csv")
bridge_expected["Model"] = "Bridge " + bridge_expected.Representation
bulk_expected = pd.read_csv(CACHE / "bulkformer_expected_summary.csv")
bulk_expected["Model"] = bulk_expected.Model + " static"
for expected in (bridge_expected, bulk_expected):
    merged = single_summary.merge(expected, on=["Task", "Model", "N_terms"], suffixes=("_rebuilt", "_expected"))
    assert len(merged) == len(expected)
    for metric in ["Mean_AUROC", "Median_AUROC", "SD_AUROC", "Mean_AUPRC", "Median_AUPRC", "SD_AUPRC"]:
        assert np.allclose(merged[f"{metric}_rebuilt"], merged[f"{metric}_expected"], atol=1e-12)

display(single_summary.round(4))"""
        ),
        nbf.v4.new_code_cell(
            """model_order = ["Bridge E0", "BulkFormer-50M static", "BulkFormer-147M static", "Bridge E1", "Bridge E2"]
colors = ["#8A8F98", "#7A9E9F", "#4F7C7D", "#2878B5", "#D95319"]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
for ax, task, subtitle in zip(
    axes, ["GO", "OMIM"],
    ["Mean across 56 GO Biological Process terms", "Mean across 103 OMIM disease terms"],
):
    sub = single_summary[single_summary.Task == task].set_index("Model").loc[model_order]
    bars = ax.bar(np.arange(len(sub)), sub.Mean_AUROC, color=colors, width=0.68)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in sub.Mean_AUROC], padding=4, fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(sub)), ["Bridge\\nE0", "BulkFormer\\n50M static", "BulkFormer\\n147M static", "Bridge\\nE1", "Bridge\\nE2"])
    ax.set_title(task, fontsize=17, fontweight="bold", pad=22)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", color="#555555", fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
axes[0].set_ylabel("Mean held-out AUROC")
fig.suptitle("Single-gene biology from frozen gene representations", fontsize=19, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "single_gene_go_omim.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURES / "single_gene_go_omim.pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """Bridge E1 and E2 are substantially stronger than Bridge E0, making transformer processing the dominant transition. Both BulkFormer static tables are more informative than Bridge E0, showing that a weak static result is not inevitable for bulk-transcriptome models. Bridge E1/E2 have higher mean AUROC than the tested BulkFormer static references, but this is not a formal architecture ranking because model vocabularies differ and BulkFormer was not evaluated contextually."""
        ),
        nbf.v4.new_markdown_cell(
            """## Published Zhong bulk RNA-seq comparison

The Zhong metadata classifies exactly three published methods as `gene expression (bulk)`: **GENE2VEC**, **TCGA-EMBEDDING**, and **FROGS-ARCHS4**. Single-cell expression, amino-acid, literature, PPI, and mixed-input methods are excluded. The cached files are the authors' common-intersection results used by their paper figures, pinned to the repository revision recorded below."""
        ),
        nbf.v4.new_code_cell(
            """ZHONG_CACHE = CACHE / "zhong"
with (ZHONG_CACHE / "provenance.json").open() as handle:
    zhong_provenance = json.load(handle)
assert zhong_provenance["commit"] == "d1320026a2a4ee033d49517f91e2d1c2ccc8df1e"

zhong_meta = pd.read_csv(ZHONG_CACHE / "embed_meta.csv")
bulk_meta = zhong_meta[zhong_meta.Category.eq("gene expression (bulk)")].copy()
eligible_bulk = ["GENE2VEC", "TCGA-EMBEDDING", "FROGS-ARCHS4"]
assert set(bulk_meta.Embeding) == set(eligible_bulk)
bulk_meta = bulk_meta.set_index("Embeding").loc[eligible_bulk].reset_index()
bulk_meta.columns = ["Model", "Method", "Training data", "Original dimensions"]
display(bulk_meta)
display(pd.DataFrame([{
    "Repository": zhong_provenance["repository"],
    "Commit": zhong_provenance["commit"],
    "Download date": zhong_provenance["download_date"],
    "Downloaded files": len(zhong_provenance["files"]),
}]))"""
        ),
        nbf.v4.new_code_cell(
            """zhong_single_rows = []
for task, filename in [("GO", "single_gene_go_AUROC.tsv"), ("OMIM", "single_gene_omim_AUROC.tsv")]:
    frame = pd.read_csv(ZHONG_CACHE / filename, sep="\\t", index_col=0)
    expected_n = 56 if task == "GO" else 103
    assert frame.shape[1] == expected_n and set(eligible_bulk).issubset(frame.index)
    for model in eligible_bulk:
        values = frame.loc[model].astype(float)
        zhong_single_rows.append({
            "Task": task, "Model": model, "N_terms": len(values),
            "Mean_AUROC": values.mean(), "Median_AUROC": values.median(),
            "SD_AUROC": values.std(ddof=1),
        })
zhong_single = pd.DataFrame(zhong_single_rows)
display(zhong_single.round(4))

single_comparability = pd.DataFrame([
    ["Bridge E0/E1/E2", "Local adaptation", "Bridge mapped vocabulary", "Same Zhong term split files and mean held-out AUROC; test genes differ after vocabulary filtering", "Protocol-matched; not a head-to-head ranking"],
    ["BulkFormer-50M/147M static", "Local adaptation", "Each model's mapped vocabulary", "Same Zhong term split files and mean held-out AUROC; test genes differ after vocabulary filtering", "Protocol-matched; not a head-to-head ranking"],
    ["GENE2VEC / TCGA-EMBEDDING / FROGS-ARCHS4", "Published Zhong", "Zhong common-intersection genes", "Original published 56 GO and 103 OMIM per-term held-out AUROCs", "Directly comparable to one another"],
], columns=["Models", "Result source", "Gene universe", "Evaluation relationship", "Comparison status"])
display(single_comparability)"""
        ),
        nbf.v4.new_code_cell(
            """local_order = ["Bridge E0", "BulkFormer-50M static", "BulkFormer-147M static", "Bridge E1", "Bridge E2"]
published_order = eligible_bulk
fig, axes = plt.subplots(1, 2, figsize=(16, 6.3), sharey=True)
for ax, task in zip(axes, ["GO", "OMIM"]):
    local = single_summary[single_summary.Task.eq(task)].set_index("Model").loc[local_order]
    pub = zhong_single[zhong_single.Task.eq(task)].set_index("Model").loc[published_order]
    values = np.r_[local.Mean_AUROC.to_numpy(), pub.Mean_AUROC.to_numpy()]
    positions = np.r_[np.arange(5), np.arange(6, 9)]
    labels = ["Bridge E0", "BulkFormer 50M", "BulkFormer 147M", "Bridge E1", "Bridge E2", "Gene2vec", "TCGA embedding", "FRoGS-ARCHS4"]
    palette = ["#8A8F98", "#7A9E9F", "#4F7C7D", "#2878B5", "#D95319", "#6A4C93", "#9C6ADE", "#C7A4E0"]
    ax.axvspan(-0.55, 4.55, color="#EAF2F8", alpha=0.65, zorder=0)
    ax.axvspan(5.45, 8.55, color="#F3ECF8", alpha=0.75, zorder=0)
    bars = ax.bar(positions, values, color=palette, width=0.72, zorder=2)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9, fontweight="bold")
    ax.set_xticks(positions, labels, rotation=27, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.65, 8.65)
    ax.set_title(task, fontsize=16, fontweight="bold", pad=35)
    ax.text(2, 1.015, "Local adaptations\\nmodel-specific universes", ha="center", va="bottom", fontsize=10, color="#355C7D", transform=ax.get_xaxis_transform())
    ax.text(7, 1.015, "Published Zhong\\ncommon-intersection universe", ha="center", va="bottom", fontsize=10, color="#6A4C93", transform=ax.get_xaxis_transform())
    ax.axvline(5, color="#777777", linewidth=1, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
axes[0].set_ylabel("Mean held-out AUROC")
fig.suptitle("Bulk RNA-seq-derived gene embeddings", fontsize=19, fontweight="bold", y=1.03)
fig.text(0.5, -0.015, "All models are shown together for context; shaded source groups use different evaluated gene universes and do not form a strict ranking.", ha="center", fontsize=9.5, color="#555555")
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(FIGURES / "bulk_rnaseq_single_gene_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURES / "bulk_rnaseq_single_gene_comparison.pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## Zhong literature context"),
        nbf.v4.new_code_cell(
            """literature = pd.read_csv(CACHE / "zhong_literature_reference.csv")
display(literature.round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """The literature values above were reported by Zhong et al. and were not recomputed here. Bridge and BulkFormer use their own available vocabularies in this adaptation, so the table provides context rather than a formal head-to-head ranking."""
        ),
        nbf.v4.new_markdown_cell("## Gene-pair results"),
        nbf.v4.new_code_cell(
            """pair_folds = pd.read_csv(CACHE / "pair_outer_fold_results.csv")
assert len(pair_folds) == 81
pair_summary = (
    pair_folds.groupby(["interaction_type", "representation", "joining_operation"], as_index=False)
    .agg(outer_folds=("outer_fold", "nunique"), AUROC_mean=("AUROC", "mean"),
         AUROC_sd=("AUROC", "std"), AUPRC_mean=("AUPRC", "mean"), AUPRC_sd=("AUPRC", "std"),
         **{"Precision@10_mean": ("Precision@10", "mean"), "Precision@10_sd": ("Precision@10", "std")})
)
assert (pair_summary.outer_folds == 3).all() and len(pair_summary) == 27
pair_expected = pd.read_csv(CACHE / "pair_expected_summary.csv")
keys = ["interaction_type", "representation", "joining_operation"]
check = pair_summary.merge(pair_expected, on=keys, suffixes=("_rebuilt", "_expected"))
assert len(check) == 27
for metric in ["AUROC_mean", "AUROC_sd", "AUPRC_mean", "AUPRC_sd", "Precision@10_mean", "Precision@10_sd"]:
    assert np.allclose(check[f"{metric}_rebuilt"], check[f"{metric}_expected"], atol=1e-12)

task_labels = {"NG": "Negative genetic", "SL": "Synthetic lethal", "TF": "TF-target"}
rep_labels = {"E0_static": "E0", "E1_null_L12": "E1", "E2_GTEx200_L12": "E2"}
best = pair_summary.loc[pair_summary.groupby("interaction_type").AUROC_mean.idxmax()].copy()
best["Task"] = best.interaction_type.map(task_labels)
best["Representation"] = best.representation.map(rep_labels)
best = best.rename(columns={"joining_operation": "Pair_operation", "AUROC_mean": "AUROC", "AUPRC_mean": "AUPRC", "Precision@10_mean": "Precision@10"})
best = best[["Task", "Representation", "Pair_operation", "AUROC", "AUPRC", "Precision@10", "AUROC_sd", "AUPRC_sd"]]
best_expected = pd.read_csv(CACHE / "pair_expected_best.csv")
best_check = best.merge(best_expected, on=["Task", "Representation", "Pair_operation"], suffixes=("_rebuilt", "_expected"))
assert len(best_check) == 3
assert np.allclose(best_check.AUROC_rebuilt, best_check.AUROC_expected, atol=1e-12)

leakage = pd.read_csv(CACHE / "pair_leakage_qc.csv")
assert len(leakage) == 9 and (leakage.train_test_gene_overlap == 0).all()
display(best.round(3))
display(leakage)"""
        ),
        nbf.v4.new_code_cell(
            """plot_order = ["Synthetic lethal", "TF-target", "Negative genetic"]
plot_best = best.set_index("Task").loc[plot_order]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
bars = ax.bar(np.arange(3), plot_best.AUROC, color=["#4472C4", "#5B9BD5", "#9DC3E6"], width=0.62)
ax.bar_label(bars, labels=[f"{value:.3f}" for value in plot_best.AUROC], padding=5, fontsize=13, fontweight="bold")
ax.set_xticks(np.arange(3), ["Synthetic lethal", "TF → target", "Negative genetic"])
ax.set_ylim(0, 1)
ax.set_ylabel("Mean outer-fold AUROC")
ax.set_title("Gene–gene relationship prediction", fontsize=18, fontweight="bold", pad=16)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(False)
fig.text(0.5, -0.01, "Bars show the best observed representation/operation condition for each task.", ha="center", fontsize=10, color="#555555")
fig.tight_layout()
fig.savefig(FIGURES / "gene_pair_best.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURES / "gene_pair_best.pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """Synthetic-lethal and TF-target relationships are meaningfully predictable from Bridge representations, whereas negative genetic interaction prediction is weaker. E2 is not universally superior: the best synthetic-lethal and TF-target AUROCs use E1, while negative genetic interaction uses E2. BulkFormer pair tasks were not run."""
        ),
        nbf.v4.new_markdown_cell(
            """### Published Zhong bulk RNA-seq pair results

Zhong reports the mean AUROC, mean AUPRC, and mean precision@10 across three outer folds for each pair operation. The cached fold-level AUROC tables below independently reproduce every published mean. Bridge used the same nested split definitions but filtered them to its mapped genes, whereas these published results use Zhong's common-intersection universe; the two sections are therefore protocol-matched literature context, not one combined ranking."""
        ),
        nbf.v4.new_code_cell(
            """published_pair_rows = []
task_names = {"ng": "Negative genetic", "sl": "Synthetic lethal", "tf": "TF-target"}
for task_code, task_name in task_names.items():
    for operation in ["concat", "product", "sum"]:
        stem = f"paired_gene_{task_code}_{operation}_outerloop"
        folds = pd.read_csv(ZHONG_CACHE / f"{stem}_AUC.tsv", sep="\\t", index_col=0)
        avgs = pd.read_csv(ZHONG_CACHE / f"{stem}_avgs.tsv", sep="\\t", index_col=0)
        fold_columns = ["outer_AUC_fold_0", "outer_AUC_fold_1", "outer_AUC_fold_2"]
        assert np.allclose(folds[fold_columns].mean(axis=1), folds.outer_AUC_average, atol=1e-12)
        assert np.allclose(folds.outer_AUC_average, avgs.outer_AUC, atol=1e-12)
        for model in eligible_bulk:
            published_pair_rows.append({
                "Task": task_name, "Model": model, "Pair_operation": operation,
                "AUROC": avgs.loc[model, "outer_AUC"],
                "AUROC_SD": folds.loc[model, fold_columns].std(ddof=1),
                "AUPRC": avgs.loc[model, "outer_AUPRC"],
                "Precision@10": avgs.loc[model, "outer_PR@10"],
                "Outer_folds": 3,
            })
zhong_pair = pd.DataFrame(published_pair_rows)
assert len(zhong_pair) == 27
display(zhong_pair.round(4))

zhong_pair_best = zhong_pair.loc[zhong_pair.groupby(["Task", "Model"]).AUROC.idxmax()].copy()
zhong_pair_best["Result source"] = "Published Zhong common-intersection"
bridge_pair_best = best[["Task", "Representation", "Pair_operation", "AUROC", "AUPRC", "Precision@10", "AUROC_sd"]].copy()
bridge_pair_best["Model"] = "Bridge " + bridge_pair_best.pop("Representation")
bridge_pair_best = bridge_pair_best.rename(columns={"AUROC_sd": "AUROC_SD"})
bridge_pair_best["Outer_folds"] = 3
bridge_pair_best["Result source"] = "Local Bridge adaptation"
pair_comparison = pd.concat([
    bridge_pair_best[["Task", "Model", "Pair_operation", "AUROC", "AUROC_SD", "AUPRC", "Precision@10", "Outer_folds", "Result source"]],
    zhong_pair_best[["Task", "Model", "Pair_operation", "AUROC", "AUROC_SD", "AUPRC", "Precision@10", "Outer_folds", "Result source"]],
], ignore_index=True)
display(pair_comparison.sort_values(["Task", "Result source", "Model"]).round(4))"""
        ),
        nbf.v4.new_code_cell(
            """task_order = ["Synthetic lethal", "TF-target", "Negative genetic"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharey=True)
for col, task in enumerate(task_order):
    bridge_row = bridge_pair_best[bridge_pair_best.Task.eq(task)].iloc[0]
    ax = axes[0, col]
    bar = ax.bar([0], [bridge_row.AUROC], color="#2878B5", width=0.55)
    ax.bar_label(bar, labels=[f"{bridge_row.AUROC:.3f}"], padding=4, fontweight="bold")
    ax.set_xticks([0], [f"{bridge_row.Model}\\n{bridge_row.Pair_operation}"])
    ax.set_title(task, fontsize=13, fontweight="bold")
    pub = zhong_pair_best[zhong_pair_best.Task.eq(task)].set_index("Model").loc[eligible_bulk]
    ax = axes[1, col]
    bars = ax.bar(np.arange(3), pub.AUROC, color=["#6A4C93", "#9C6ADE", "#C7A4E0"], width=0.65)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in pub.AUROC], padding=4, fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(3), [f"Gene2vec\\n{pub.loc['GENE2VEC', 'Pair_operation']}", f"TCGA embedding\\n{pub.loc['TCGA-EMBEDDING', 'Pair_operation']}", f"FRoGS-ARCHS4\\n{pub.loc['FROGS-ARCHS4', 'Pair_operation']}"])
for ax in axes.flat:
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
axes[0, 0].set_ylabel("Local Bridge adaptation\\nMean outer-fold AUROC")
axes[1, 0].set_ylabel("Published Zhong intersection\\nMean outer-fold AUROC")
fig.suptitle("Gene-pair tasks: best observed operation within each evaluation", fontsize=18, fontweight="bold")
fig.text(0.5, 0.01, "Rows are separated because the evaluated gene universes differ; operation labels appear below each bar.", ha="center", fontsize=9.5, color="#555555")
fig.tight_layout(rect=[0, 0.035, 1, 0.96])
fig.savefig(FIGURES / "bulk_rnaseq_pair_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(FIGURES / "bulk_rnaseq_pair_comparison.pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate": None, "ModDate": None})
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Conclusions

1. Bridge transformer-derived states encode substantially more GO and OMIM information than its static lookup table.
2. Real GTEx context provides a smaller, task-dependent improvement beyond the null-context transformer state.
3. The eligible Zhong bulk-expression references are Gene2vec, TCGA-Embedding, and FRoGS-ARCHS4; their published common-intersection results provide protocol-matched context, not an identical-universe ranking against local runs.
4. BulkFormer static tables contain appreciable single-gene biology, but their scaling pattern is not monotonic across GO and OMIM.
5. Bridge supports generalization to unseen genes for selected pair tasks, but pair performance is task-dependent and does not imply universal relational understanding.
6. These probes establish decodability—not causal or mechanistic understanding, batch correction, or universal model superiority.

Major limitations are different evaluated gene universes across the local adaptations and Zhong's published intersection benchmark, no contextual or pair evaluation for BulkFormer, three outer folds for pair tasks, unavailable single-gene AUPRC in the published Zhong TSVs, and reliance on Zhong's annotation, negative-sampling, and split design."""
        ),
        nbf.v4.new_markdown_cell(
            """## Full recomputation

This public notebook intentionally uses cached artifacts. The complete expensive workflow is documented in the adjacent `README.md`. In brief, full recomputation requires the frozen checkpoints and vocabularies, Zhong's supplied split files, Bridge E0/E1 extraction, the GTEx-200 E2 inference pass, per-term GO/OMIM SVM fitting, nested gene-pair SVM fitting, and final aggregation. The BulkFormer extension reads only static checkpoint tensors. Do not launch the full workflow merely to execute this report."""
        ),
    ]
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
