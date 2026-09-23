#!/usr/bin/env python3
"""Complete paired and delta-versus-variance diagnostics from saved metrics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluate_results import bootstrap_delta, sign_flip_p


BENCH = Path(__file__).resolve().parent.parent
CONFIG = json.loads((BENCH / "config.json").read_text())
RESULTS = BENCH / "results"
WORK = Path(CONFIG["shared_work_root"])


def main() -> None:
    metrics = pd.read_csv(RESULTS / "analysis/per_gene_metrics.csv")
    wide = metrics.pivot(index="gene_id", columns="representation", values="pcc")
    pairs = [
        ("bridge_contextual", "bulkformer_contextual"),
        ("bridge_contextual", "pca_context"),
        ("bridge_contextual", "target_expression"),
        ("bridge_plus_pca", "bridge_contextual"),
        ("bridge_plus_pca", "pca_context"),
        ("bulkformer_contextual", "target_expression"),
    ]
    rows = []
    for left, right in pairs:
        delta = (wide[left] - wide[right]).dropna().to_numpy()
        low, high = bootstrap_delta(delta)
        rows.append({"comparison": f"{left}_minus_{right}", "n_genes": len(delta),
            "mean_delta_pcc": delta.mean(), "median_delta_pcc": np.median(delta),
            "ci95_mean_low": low, "ci95_mean_high": high,
            "fraction_left_greater": np.mean(delta > 0),
            "fraction_right_greater": np.mean(delta < 0),
            "paired_sign_flip_p": sign_flip_p(delta)})
    pd.DataFrame(rows).to_csv(RESULTS / "analysis/paired_comparisons.csv", index=False)

    overlap = pd.read_csv(RESULTS / "gene_overlap.csv")
    overlap = overlap[overlap.all_method_common].set_index("gene_id")
    expression = np.load(WORK / "expression_released_log2_tpm_plus_1.float32.npy", mmap_mode="r")
    columns = overlap.expression_column_index.astype(int).to_numpy()
    variance = pd.Series(np.var(expression[:, columns], axis=0, ddof=1), index=overlap.index)
    diagnostic_rows = []
    for left, right in (("bridge_contextual", "pca_context"),
                        ("bridge_contextual", "target_expression"),
                        ("bridge_contextual", "bulkformer_contextual")):
        delta = (wide[left] - wide[right]).dropna()
        common = delta.index.intersection(variance.index)
        result = spearmanr(delta.loc[common], variance.loc[common])
        diagnostic_rows.append({"delta": f"{left}_minus_{right}", "n_genes": len(common),
            "expression_variance_definition": "full-cohort descriptive variance; not used in model fitting",
            "spearman_rho": result.statistic, "two_sided_p": result.pvalue})
    pd.DataFrame(diagnostic_rows).to_csv(RESULTS / "analysis/delta_vs_expression_variance.csv", index=False)


if __name__ == "__main__":
    main()
