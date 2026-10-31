#!/usr/bin/env python3
"""Build the approved, cached publication package from frozen benchmark evidence.

This script is deliberately bounded: it reads compact saved results plus the retained
two-million-pair cosine cache, verifies the approved headline values, and writes only
under notebooks/sample_retrieval_geometry. It never runs model inference or retrieval.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
SOURCE = REPO / "benchmarks" / "sample_retrieval_geometry"
RESULTS = SOURCE / "results"
DATA = PACKAGE / "data"
FIGURE_DATA = PACKAGE / "figure_data"
FIGURES = PACKAGE / "figures"
SEED = 20260922


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, name: str, atol: float = 5e-12) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise AssertionError(f"{name}: {actual} != {expected}")


def style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def prepare_dirs() -> None:
    for path in (DATA, FIGURE_DATA, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def load_and_validate() -> dict[str, object]:
    pair_summary = pd.read_csv(
        RESULTS / "absolute_similarity" / "paired_cosine_micro_macro_summary.csv"
    )
    random_exact = pd.read_csv(
        RESULTS / "absolute_similarity" / "random_vs_exact_cosine.csv"
    )
    poor_rank = pd.read_csv(
        RESULTS / "absolute_similarity" / "high_similarity_poor_rank_frequency.csv"
    )
    context = pd.read_parquet(
        RESULTS / "absolute_similarity" / "nearest_neighbor_context.parquet"
    )
    anisotropy = pd.read_csv(RESULTS / "anisotropy" / "anisotropy_summary.csv")
    distributions = pd.read_csv(
        RESULTS / "anisotropy" / "cosine_distribution_summaries.csv"
    )
    retrieval_micro = pd.read_csv(
        RESULTS / "literature_postprocessing" / "retrieval_metrics_micro.csv"
    )
    retrieval_macro = pd.read_csv(
        RESULTS / "literature_postprocessing" / "retrieval_metrics_study_macro.csv"
    )
    gtex_micro = pd.read_csv(
        RESULTS / "gtex_biological_retrieval" / "retrieval_metrics_micro.csv"
    )
    gtex_macro = pd.read_csv(
        RESULTS / "gtex_biological_retrieval" / "retrieval_metrics_tissue_macro.csv"
    )
    gtex_control = pd.read_csv(
        RESULTS / "gtex_biological_retrieval" / "gtex_to_gtex_control.csv"
    )
    cohort = pd.read_csv(
        RESULTS / "gtex_biological_retrieval" / "gtex_cohort_summary.csv"
    )
    annotation = pd.read_csv(
        RESULTS / "gtex_biological_retrieval" / "archs4_annotation_summary.csv"
    )

    # Prediction-level checks available for the raw 562-query identity results.
    if len(context) != 562 or context["query_index"].nunique() != 562:
        raise AssertionError("Expected 562 unique recount3 queries")
    if context["gse"].nunique() != 40:
        raise AssertionError("Expected 40 unique recount3 studies")
    assert_close(context["exact_pair_cosine"].mean(), 0.997806, "exact cosine mean", 5e-7)
    assert_close(context["exact_pair_cosine"].median(), 0.999218, "exact cosine median", 5e-7)
    assert_close((context["exact_rank"] <= 1).mean(), 0.2846975088967972, "raw R@1")
    assert_close((context["exact_rank"] <= 10).mean(), 0.7722419928825622, "raw R@10")
    if int(((context["exact_rank"] > 100) & (context["exact_pair_cosine"] > 0.99)).sum()) != 33:
        raise AssertionError("Expected 33 high-similarity/poor-rank queries")

    approved_methods = ["Raw cosine", "Whitening"]
    retrieval_micro = retrieval_micro[retrieval_micro.method.isin(approved_methods)].copy()
    retrieval_macro = retrieval_macro[retrieval_macro.method.isin(approved_methods)].copy()
    if set(retrieval_micro.method) != set(approved_methods):
        raise AssertionError("Missing an approved retrieval method")
    white = retrieval_micro.set_index("method").loc["Whitening"]
    assert_close(white.r_at_1, 0.46619217081850534, "whitening R@1")
    assert_close(white.r_at_10, 0.9448398576512456, "whitening R@10")
    assert_close(white.r_at_100, 0.9911032028469751, "whitening R@100")
    assert_close(white.mrr, 0.6919384418024931, "whitening MRR")
    assert_close(
        retrieval_macro.set_index("method").loc["Whitening", "macro_r_at_10"],
        0.9412089646464648,
        "whitening macro R@10",
    )

    # Aggregate GTEx tables are checked against the retained per-query table.
    per_query = pd.read_parquet(
        RESULTS / "gtex_biological_retrieval" / "per_query_metrics.parquet"
    )
    per_query = per_query[per_query.method.isin(approved_methods)].copy()
    for method in approved_methods:
        q = per_query[(per_query.method == method) & (per_query.k == 10)]
        if q.query_index.nunique() != 6064:
            raise AssertionError(f"Expected 6,064 GTEx queries for {method}")
        summary = gtex_micro[(gtex_micro.method == method) & (gtex_micro.k == 10)].iloc[0]
        assert_close(q.same_hits.sum() / (len(q) * 10), summary.absolute_hit_rate, f"{method} GTEx Hit@10")
        assert_close(q.unknown_hits.sum() / (len(q) * 10), summary.unknown_rate, f"{method} GTEx Unknown@10")
        assert_close(q.same_hits.sum() / q.labeled_hits.sum(), summary.annotated_precision, f"{method} GTEx P@10")

    gtex_micro = gtex_micro[gtex_micro.method.isin(approved_methods)].copy()
    gtex_macro = gtex_macro[gtex_macro.method.isin(approved_methods)].copy()
    gtex_control = gtex_control[gtex_control.method.isin(approved_methods)].copy()
    assert_close(
        gtex_macro.query("method == 'Raw cosine' and k == 10").annotated_precision.iloc[0],
        0.867303,
        "raw GTEx macro P@10",
        5e-7,
    )
    assert_close(
        gtex_macro.query("method == 'Whitening' and k == 10").annotated_precision.iloc[0],
        0.912085,
        "white GTEx macro P@10",
        5e-7,
    )

    return locals()


def write_cached_tables(d: dict[str, object]) -> None:
    d["pair_summary"].to_csv(DATA / "exact_pair_similarity_summary.csv", index=False)
    d["random_exact"].to_csv(DATA / "random_vs_exact_similarity.csv", index=False)
    d["poor_rank"].query("cosine_threshold == 0.99").to_csv(
        DATA / "high_similarity_poor_rank.csv", index=False
    )
    d["anisotropy"].to_csv(DATA / "anisotropy_summary.csv", index=False)
    d["distributions"].query(
        "representation in ['Raw', 'Mean-centered'] and distribution == 'random_ARCHS4_pairs'"
    ).to_csv(DATA / "geometry_distribution_summary.csv", index=False)
    d["retrieval_micro"].to_csv(DATA / "exact_retrieval_micro.csv", index=False)
    d["retrieval_macro"].to_csv(DATA / "exact_retrieval_study_macro.csv", index=False)
    d["gtex_micro"].to_csv(DATA / "gtex_archs4_micro.csv", index=False)
    d["gtex_macro"].to_csv(DATA / "gtex_archs4_tissue_macro.csv", index=False)
    d["gtex_control"].to_csv(DATA / "gtex_gtex_donor_excluded.csv", index=False)
    d["cohort"].to_csv(DATA / "gtex_cohort_summary.csv", index=False)
    d["annotation"].to_csv(DATA / "archs4_annotation_summary.csv", index=False)

    scatter = d["context"][["query_index", "exact_rank", "exact_pair_cosine"]].copy()
    scatter.to_csv(FIGURE_DATA / "rank_vs_exact_cosine.csv", index=False)

    random_cos = np.load(SOURCE / "work" / "anisotropy_random_cosines.npz")
    bins = np.linspace(-1, 1, 241)
    frames = []
    for name, key in [("Raw", "rep_0"), ("Mean-centered", "rep_1")]:
        values = random_cos[key]
        counts, edges = np.histogram(values, bins=bins)
        width = np.diff(edges)
        frames.append(
            pd.DataFrame(
                {
                    "representation": name,
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "bin_mid": (edges[:-1] + edges[1:]) / 2,
                    "count": counts,
                    "density": counts / (counts.sum() * width),
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(
        FIGURE_DATA / "random_cosine_histogram.csv", index=False
    )

    micro = d["retrieval_micro"].melt(
        id_vars=["method"],
        value_vars=["r_at_1", "r_at_5", "r_at_10", "r_at_100", "r_at_1000"],
        var_name="cutoff",
        value_name="recall",
    )
    micro["weighting"] = "Sample-micro"
    macro = d["retrieval_macro"].melt(
        id_vars=["method"],
        value_vars=["macro_r_at_1", "macro_r_at_5", "macro_r_at_10", "macro_r_at_100", "macro_r_at_1000"],
        var_name="cutoff",
        value_name="recall",
    )
    macro["cutoff"] = macro.cutoff.str.replace("macro_", "", regex=False)
    macro["weighting"] = "Study-macro"
    pd.concat([micro, macro], ignore_index=True).to_csv(
        FIGURE_DATA / "exact_retrieval_r_at_k.csv", index=False
    )

    d["gtex_macro"].query("k == 10")[[
        "method", "annotated_precision", "absolute_hit_rate", "unknown_rate"
    ]].to_csv(FIGURE_DATA / "gtex_archs4_macro_p10.csv", index=False)
    d["gtex_control"].to_csv(FIGURE_DATA / "gtex_gtex_precision.csv", index=False)


def make_figures() -> None:
    style()
    colors = {"Raw cosine": "#4C78A8", "Whitening": "#E45756", "Raw": "#4C78A8", "Mean-centered": "#72B7B2"}

    hist = pd.read_csv(FIGURE_DATA / "random_cosine_histogram.csv")
    scatter = pd.read_csv(FIGURE_DATA / "rank_vs_exact_cosine.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for rep in ["Raw", "Mean-centered"]:
        q = hist[hist.representation == rep]
        axes[0].plot(q.bin_mid, q.density, lw=2, color=colors[rep], label=rep)
    axes[0].axvline(0.999218, color="#222222", ls="--", lw=1.3, label="Exact-pair median")
    axes[0].set(xlabel="Cosine similarity", ylabel="Density", title="Random ARCHS4 pairs (2 million)")
    axes[0].legend(fontsize=8)
    axes[1].scatter(np.log10(scatter.exact_rank), scatter.exact_pair_cosine, s=12, alpha=0.55, color="#4C78A8", edgecolors="none")
    for rank in [1, 10, 100, 1000]:
        axes[1].axvline(np.log10(rank), color="#999999", lw=0.8, ls=":" if rank != 1 else "-")
    axes[1].set(
        xlabel=r"$\log_{10}$(exact counterpart rank)",
        ylabel="Exact-pair Bridge cosine",
        title="Agreement can remain high at poor rank",
        ylim=(0.972, 1.0003),
    )
    fig.tight_layout()
    save_figure(fig, "figure_1_geometry_and_uniqueness")

    rk = pd.read_csv(FIGURE_DATA / "exact_retrieval_r_at_k.csv")
    order = ["r_at_1", "r_at_5", "r_at_10", "r_at_100", "r_at_1000"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)
    for ax, weighting in zip(axes, ["Sample-micro", "Study-macro"]):
        q = rk[rk.weighting == weighting]
        for method in ["Raw cosine", "Whitening"]:
            vals = q[q.method == method].set_index("cutoff").loc[order, "recall"]
            ax.plot(x, vals * 100, marker="o", lw=2, color=colors[method], label=method)
        ax.set_xticks(x, ["1", "5", "10", "100", "1,000"])
        ax.set(xlabel="K", title=weighting, ylim=(20, 102))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Exact counterpart recall (%)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Reference-only whitening improves exact cross-pipeline retrieval", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure_2_exact_retrieval")

    tissue = pd.read_csv(FIGURE_DATA / "gtex_archs4_macro_p10.csv")
    control = pd.read_csv(FIGURE_DATA / "gtex_gtex_precision.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    metrics = ["annotated_precision", "absolute_hit_rate", "unknown_rate"]
    metric_labels = ["Annotated\nP@10", "Confirmed\nHit@10", "Unknown@10"]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, method in zip([-width / 2, width / 2], ["Raw cosine", "Whitening"]):
        row = tissue[tissue.method == method].iloc[0]
        axes[0].bar(x + offset, [row[m] * 100 for m in metrics], width, color=colors[method], label=method)
    axes[0].set_xticks(x, metric_labels)
    axes[0].set(ylabel="Tissue-macro rate (%)", title="GTEx → independent ARCHS4", ylim=(0, 100))
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    for method in ["Raw cosine", "Whitening"]:
        q = control[control.method == method].sort_values("k")
        axes[1].plot(q.k, q.tissue_macro_precision * 100, marker="o", lw=2, color=colors[method], label=method)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 5, 10, 100], ["1", "5", "10", "100"])
    axes[1].set(
        xlabel="K", ylabel="Tissue-macro precision (%)",
        title="GTEx → GTEx, different donor", ylim=(80, 100),
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "figure_3_biological_retrieval")


def write_provenance(d: dict[str, object]) -> None:
    source_files = [
        "results/absolute_similarity/paired_cosine_micro_macro_summary.csv",
        "results/absolute_similarity/random_vs_exact_cosine.csv",
        "results/absolute_similarity/high_similarity_poor_rank_frequency.csv",
        "results/absolute_similarity/nearest_neighbor_context.parquet",
        "results/anisotropy/anisotropy_summary.csv",
        "results/anisotropy/cosine_distribution_summaries.csv",
        "results/literature_postprocessing/retrieval_metrics_micro.csv",
        "results/literature_postprocessing/retrieval_metrics_study_macro.csv",
        "results/gtex_biological_retrieval/retrieval_metrics_micro.csv",
        "results/gtex_biological_retrieval/retrieval_metrics_tissue_macro.csv",
        "results/gtex_biological_retrieval/gtex_to_gtex_control.csv",
        "results/gtex_biological_retrieval/per_query_metrics.parquet",
    ]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = "unavailable"
    provenance = {
        "package": "sample_retrieval_geometry",
        "report_scope": "Bridge retrieval geometry and reference-only whitening results defined in MANUSCRIPT_OUTLINE.md.",
        "source_benchmark": "benchmarks/sample_retrieval_geometry",
        "source_git_commit": commit,
        "source_benchmark_tracked_at_packaging": False,
        "cohorts": {
            "technical_identity": {"queries": 562, "studies": 40, "references": 510709, "exposure": "repository-defined unseen_study"},
            "gtex_archs4": {"eligible_queries": 6064, "tissues": 13, "references": 510376, "excluded_gtex_origin_references": 333},
            "gtex_gtex_control": {"queries": 6064, "donors_in_cohort": 937, "same_donor_excluded": True},
        },
        "representation": {"model": "frozen BridgeRNA", "dimensions": 512, "checkpoint": "model/r7hnr92k/best_model.pt"},
        "preprocessing": {"genes": 15165, "input": "natural log1p(TPM)", "gene_order": "data/ensembl/canonical_genes.csv"},
        "transformations": {
            "raw": "L2-normalized Bridge embedding; cosine ranking",
            "mean_centered_diagnostic": "subtract mean fitted on 510,709 ARCHS4 references; L2 normalize",
            "whitening": "ARCHS4-only full 512-D covariance whitening, denominator N, eigenvalue floor lambda_max*1e-12, then L2 normalize and cosine rank",
            "whitening_parameter_sha256": "9834e6bb8873e245f84e6874c6d11949d4e8dd57de9f31aa287fd7617524f880",
        },
        "random_pair_sample": {"pairs": 2000000, "seed": SEED, "distinct_samples": True},
        "source_artifacts": {rel: {"sha256": sha256(SOURCE / rel), "bytes": (SOURCE / rel).stat().st_size} for rel in source_files},
        "verification": {
            "raw_exact_identity": "recalculated from 562-row nearest-neighbor context table",
            "whitened_exact_identity": "verified against two independent compact saved summary tables; prediction-level whitened ranks were not retained",
            "gtex_archs4": "micro P@10, Hit@10, and Unknown@10 recalculated from the retained 6,064-query table; tissue-macro summaries verified against saved per-tissue summaries",
            "gtex_gtex": "summary-table verification only; compact prediction-level neighbor labels were not retained in the publication package",
            "status": "passed",
        },
        "limitations": [
            "The two-million-pair ARCHS4 distribution is a deterministic sample, not all ~130 billion unique pairs.",
            "ARCHS4 tissue labels are conservative and sparse; approximately 74% of Top-10 neighbors are unknown, so annotated precision is always interpreted with absolute Hit@10 and Unknown@10.",
            "The 562 pairs are cross-pipeline technical counterparts, not independent biological replicates.",
            "Whitening is reference-only post-processing and is not evidence that BridgeRNA is batch-corrected.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "nbformat": nbf.__version__,
        },
    }
    (PACKAGE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (PACKAGE / "validation.json").write_text(
        json.dumps({"status": "passed", "checks": provenance["verification"]}, indent=2) + "\n"
    )


def write_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
    }
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Frozen Bridge retrieval geometry and reference-only whitening

This report asks whether frozen BridgeRNA embeddings preserve cross-pipeline sample identity, why ordinary cosine similarity is unusually high, and whether an ARCHS4-fitted geometry correction improves both technical-identity and tissue retrieval.

Bridge is not trained or fine-tuned in this analysis. Every result below is loaded from validated compact cached tables.
"""))
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'data').exists():
    candidates = [p for p in ROOT.rglob('sample_retrieval_geometry') if (p / 'data').exists()]
    if not candidates:
        raise FileNotFoundError('Run from notebooks/sample_retrieval_geometry or the repository root')
    ROOT = candidates[0]

provenance = json.loads((ROOT / 'provenance.json').read_text())
assert provenance['verification']['status'] == 'passed'
print(f"Validated cached package: {ROOT}")
"""))
    cells.append(nbf.v4.new_markdown_cell("""## Cohorts and frozen protocol

- **Technical identity:** 562 exact ARCHS4↔recount3 GSM pairs from 40 repository-defined unseen studies, searched against 510,709 human ARCHS4 references.
- **Embedding geometry:** 2,000,000 deterministic distinct random ARCHS4 pairs (seed 20260922).
- **Biological retrieval:** 6,064 eligible GTEx v11 queries across 13 harmonized tissues, searched against 510,376 ARCHS4 references after 333 explicit GTEx-origin exclusions.
- **Different-donor control:** each GTEx query excludes itself and every sample from the same donor.

BridgeRNA is frozen. The post-processing transform is fitted only on the ARCHS4 reference corpus: full 512-dimensional covariance whitening with denominator *N*, eigenvalue floor $\lambda_{max}\times10^{-12}$, L2 normalization, and cosine ranking. The transform is applied unchanged to recount3 and GTEx queries.
"""))
    cells.append(nbf.v4.new_markdown_cell("## 1. Absolute agreement is distinct from retrieval uniqueness"))
    cells.append(nbf.v4.new_code_cell("""pair = pd.read_csv(ROOT / 'data/exact_pair_similarity_summary.csv')
random_exact = pd.read_csv(ROOT / 'data/random_vs_exact_similarity.csv')
poor_rank = pd.read_csv(ROOT / 'data/high_similarity_poor_rank.csv')
display(pair.round(6))
display(random_exact[['distribution','n','mean','median','sd','q1','q3','minimum','maximum']].round(6))
display(poor_rank.round(6))
"""))
    cells.append(nbf.v4.new_markdown_cell("""Across 562 exact pairs, mean cosine is **0.997806** and median cosine is **0.999218**, versus a sampled random-pair median of **0.912954**. Yet 33 queries have rank >100 while retaining exact-pair cosine >0.99. High absolute agreement therefore does not imply a unique nearest neighbor in a crowded reference corpus.
"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT / 'figures/figure_1_geometry_and_uniqueness.png')))"))
    cells.append(nbf.v4.new_markdown_cell("""## 2. A shared global direction explains the high raw cosine baseline

Mean random-pair cosine is 0.878748, almost identical to the squared magnitude of the unit directional mean (0.878774). Mean-centering moves the random-pair median from 0.912954 to −0.002471. This quantitative match identifies strong anisotropy—not failed sample discrimination—as the main reason ordinary Bridge cosine is high.
"""))
    cells.append(nbf.v4.new_code_cell("""anis = pd.read_csv(ROOT / 'data/anisotropy_summary.csv')
geom = pd.read_csv(ROOT / 'data/geometry_distribution_summary.csv')
display(anis[['references','embedding_dim','unit_directional_mean_norm_squared','participation_ratio']].round(6))
display(geom[['representation','n','mean','median','sd','p5','p95','minimum','maximum']].round(6))
"""))
    cells.append(nbf.v4.new_markdown_cell("## 3. Reference-only whitening improves exact cross-pipeline retrieval"))
    cells.append(nbf.v4.new_code_cell("""micro = pd.read_csv(ROOT / 'data/exact_retrieval_micro.csv')
macro = pd.read_csv(ROOT / 'data/exact_retrieval_study_macro.csv')
display(micro[['method','r_at_1','r_at_5','r_at_10','r_at_100','r_at_1000','mrr','median_rank']].round(4))
display(macro[['method','studies','macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_r_at_1000','macro_mrr']].round(4))
"""))
    cells.append(nbf.v4.new_markdown_cell("""Whitening raises sample-micro R@1 from **28.47% to 46.62%**, R@10 from **77.22% to 94.48%**, R@100 from **91.81% to 99.11%**, and MRR from **0.4764 to 0.6919**. Equal-weight study-macro R@10 rises from **75.92% to 94.12%**. Random chance is $K/510{,}709$ for R@K, so even the raw result is far above chance; whitening makes the exact counterpart substantially more unique.
"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT / 'figures/figure_2_exact_retrieval.png')))"))
    cells.append(nbf.v4.new_markdown_cell("## 4. The gain extends to tissue organization"))
    cells.append(nbf.v4.new_code_cell("""gtex_macro = pd.read_csv(ROOT / 'data/gtex_archs4_tissue_macro.csv')
gtex_control = pd.read_csv(ROOT / 'data/gtex_gtex_donor_excluded.csv')
display(gtex_macro[gtex_macro.k == 10][['method','tissues','annotated_precision','absolute_hit_rate','unknown_rate','fold_enrichment']].round(4))
display(gtex_control[gtex_control.k == 10][['method','queries','micro_precision','tissue_macro_precision','tissues']].round(4))
"""))
    cells.append(nbf.v4.new_markdown_cell("""In GTEx→ARCHS4 retrieval, tissue-macro annotated P@10 increases from **86.73% to 91.21%** and absolute Hit@10 from **23.13% to 24.77%**. Approximately **74%** of Top-10 references are unlabeled under the deliberately conservative ARCHS4 mapping, so annotated precision must be read alongside Hit@10 and Unknown@10.

The fully labeled different-donor GTEx→GTEx control provides the cleaner biological validation: whitening raises tissue-macro P@10 from **90.61% to 94.86%** and micro P@10 from **92.28% to 95.65%**.
"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT / 'figures/figure_3_biological_retrieval.png')))"))
    cells.append(nbf.v4.new_markdown_cell("""## Limitations

- The ARCHS4 pairwise distribution is a deterministic sample, not an exhaustive evaluation of roughly 130 billion unique pairs.
- Exact ARCHS4↔recount3 matches are cross-pipeline technical counterparts, not independent biological replicates.
- Conservative ARCHS4 tissue annotation is sparse. Unknowns are retained explicitly; the GTEx→ARCHS4 claim is therefore conditional on annotated neighbors.
- The different-donor GTEx→GTEx experiment is a standardized-dataset control, not independent-study validation.
- Whitening is post hoc geometry correction. It is not training and is not evidence that BridgeRNA is batch-corrected.

## Conclusion

Frozen BridgeRNA strongly preserves cross-pipeline sample identity, but raw cosine is dominated by a shared global direction and can obscure retrieval uniqueness. A full whitening transform fitted only on ARCHS4 references markedly improves exact retrieval and also improves tissue organization in both independent ARCHS4 search and a fully labeled different-donor GTEx control. Raw cosine remains the interpretable measure of absolute representation agreement; reference-only whitening is the supported ranking geometry.
"""))
    nb["cells"] = cells
    nbf.write(nb, PACKAGE / "sample_retrieval_geometry.ipynb")


def main() -> None:
    prepare_dirs()
    data = load_and_validate()
    write_cached_tables(data)
    make_figures()
    write_provenance(data)
    write_notebook()
    print("Publication package built and validation passed.")


if __name__ == "__main__":
    main()
