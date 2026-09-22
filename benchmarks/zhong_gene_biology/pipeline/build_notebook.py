#!/usr/bin/env python3
"""Build the human-readable Zhong benchmark report from saved results."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
OUT = BENCH / "zhong_gene_biology.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# BridgeRNA gene-biology benchmark

This report adapts the fixed splits from Zhong et al. (2025) to the frozen BridgeRNA model. It compares the static token embedding (E0), the layer-12 representation under a null input (E1), and the layer-12 representation averaged over a frozen 200-sample GTEx panel (E2). All values below are loaded from pipeline outputs; the notebook does not recompute embeddings or probes."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

BENCH = Path.cwd()
if BENCH.name != "zhong_gene_biology":
    BENCH = Path("benchmarks/zhong_gene_biology")
RESULTS = BENCH / "results"
COVERAGE = BENCH / 'coverage'"""
        ),
        nbf.v4.new_markdown_cell(
            """## Cohort, mapping, and representations

BridgeRNA is evaluated in its canonical 15,165-token order. Human Entrez mappings are retained only when unambiguous and one-to-one. E2 uses a fixed 200-sample GTEx manifest and natural `log1p(TPM)` input; the backbone remains frozen."""
        ),
        nbf.v4.new_code_cell(
            """coverage = pd.read_csv(COVERAGE / "coverage_summary.csv")
display(coverage)
with (COVERAGE / "representation_qc_e0_e1_e2.json").open() as handle:
    representation_qc = json.load(handle)
representation_qc"""
        ),
        nbf.v4.new_markdown_cell(
            """## Single-gene GO and OMIM prediction

RBF SVM hyperparameters are chosen only from the supplied training folds, then evaluated once on the supplied holdout. Terms that lose either class after mapping are excluded explicitly."""
        ),
        nbf.v4.new_code_cell(
            """single = pd.read_csv(RESULTS / "representation_summary.csv")
display(single.round(3))
display(pd.read_csv(RESULTS / "comparison_summary_AUROC.csv").round(3))
display(pd.read_csv(RESULTS / "comparison_summary_AUPRC.csv").round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """### Zhong-matched aggregation across terms

Zhong et al. report **mean AUROC across biological terms** for their headline single-gene comparison. The table below therefore uses the unweighted arithmetic mean of the final held-out AUROC for 56 GO Biological Process terms or 103 OMIM terms. Each term contributes exactly one value. The median is retained as a distribution-robust reference, and sample SD (`ddof=1`) describes between-term variation. CV-fold scores, individual predictions, and gene-count-weighted or pooled metrics are not included in this aggregation."""
        ),
        nbf.v4.new_code_cell(
            """zhong_matched = pd.read_csv(RESULTS / "single_gene_zhong_matched_summary.csv")
display(zhong_matched[[
    "Task", "Representation", "N_terms", "Mean_AUROC", "Median_AUROC", "SD_AUROC",
    "Mean_AUPRC", "Median_AUPRC", "SD_AUPRC"
]].round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """### Literature-reference comparison

The non-Bridge values below are literature reference values explicitly reported by Zhong et al. They were not regenerated here. Bridge uses its own available vocabulary and an adapted evaluation, so this is not a formal head-to-head rerun and should not be used by itself to rank models or claim superiority. AUPRC is supplementary for Bridge and is not compared with Zhong without a corresponding reported statistic."""
        ),
        nbf.v4.new_code_cell(
            """zhong_literature = pd.read_csv(RESULTS / "zhong_literature_comparison.csv")
display(zhong_literature.round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """The contextual layer-12 representations outperform the static lookup embedding on both tasks. E2 is modestly stronger than E1, so real expression context adds signal beyond the architecture/null-input baseline, but the dominant gain is E0 to E1. These are probe results on fixed annotation splits, not evidence of causal biology or batch correction."""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(BENCH / "figures/final_representation_ablation.png")))"""
        ),
        nbf.v4.new_markdown_cell("## Gene-pair prediction"),
        nbf.v4.new_code_cell(
            """pair_path = RESULTS / "pair_representation_summary.csv"
if pair_path.exists():
    pair = pd.read_csv(pair_path)
    display(pair.round(3))
    display(pd.read_csv(RESULTS / "final_pair_best_conditions.csv").round(3))
    display(pd.read_csv(RESULTS / "final_pair_leakage_qc.csv"))
    display(Image(filename=str(BENCH / "figures/final_gene_pair_benchmark.png")))
else:
    print("Pair benchmark is still running; rerun this notebook after the checkpointed pipeline completes.")"""
        ),
        nbf.v4.new_markdown_cell("## Biological-scale summary"),
        nbf.v4.new_code_cell(
            """display(pd.read_csv(RESULTS / "final_benchmark_summary.csv").round(4))
display(Image(filename=str(BENCH / "figures/final_biological_scale_summary.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Limitations

The benchmark inherits Zhong et al.'s annotation sources, negative sampling, and split construction. Gene overlap between train and test pairs is reported separately and can permit gene-level exposure even though pair assignments are held out. GTEx-200 is a fixed bulk-tissue panel rather than a study-independent biological validation cohort. Control distributions should be interpreted only after all declared seeds finish."""
        ),
    ]
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
