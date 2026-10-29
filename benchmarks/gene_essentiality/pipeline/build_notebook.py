#!/usr/bin/env python3
"""Build the human-readable benchmark notebook from saved results."""

from pathlib import Path

import nbformat as nbf


BENCH = Path(__file__).resolve().parent.parent
notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Bridge contextual gene-essentiality benchmark\n\n"
        "This notebook is a view over saved, out-of-fold results. Computation lives "
        "in `pipeline/`; no metric is hard-coded here."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "from IPython.display import Image, Markdown, display\n"
        "ROOT = Path.cwd()\n"
        "if ROOT.name != 'gene_essentiality':\n"
        "    ROOT = ROOT / 'benchmarks' / 'gene_essentiality'\n"
        "RESULTS = ROOT / 'results'\n"
        "assert (RESULTS / 'primary_metrics.csv').exists()\n"
    ),
    nbf.v4.new_markdown_cell(
        "## Cohort and protocol\n\n"
        "The matched benchmark contains 1,108 cell lines and 14,415 dependency "
        "genes. Ten-fold cell-line cross-validation produces one out-of-fold "
        "prediction for every cell line. Mean PCC is the unweighted mean of "
        "per-gene correlations across cell lines. PCA is fitted on training lines only."
    ),
    nbf.v4.new_code_cell(
        "audit = pd.read_json(RESULTS / 'dataset_audit.json', typ='series')\n"
        "metrics = pd.read_csv(RESULTS / 'primary_metrics.csv')\n"
        "display(metrics.style.format({\n"
        "    'mean_pcc': '{:.5f}', 'median_pcc': '{:.5f}',\n"
        "    'mean_scc': '{:.5f}', 'median_scc': '{:.5f}'\n"
        "}))"
    ),
    nbf.v4.new_markdown_cell("## Paired gene-level diagnostics"),
    nbf.v4.new_code_cell(
        "paired = pd.read_csv(RESULTS / 'analysis' / 'paired_comparisons.csv')\n"
        "display(paired.style.format({\n"
        "    c: '{:.5f}' for c in paired.columns if c != 'comparison' and c != 'n_genes'\n"
        "}))"
    ),
    nbf.v4.new_code_cell(
        "display(Image(filename=str(RESULTS / 'figures' / 'bridge_vs_pca_pcc.png'), width=760))\n"
        "display(Image(filename=str(RESULTS / 'figures' / 'bridge_vs_target_expression_pcc.png'), width=760))"
    ),
    nbf.v4.new_markdown_cell(
        "## Conclusion\n\n"
        "Bridge contextual tokens contain weak predictive signal but do not beat "
        "target-gene expression. Bridge is slightly above PCA in mean PCC, while "
        "its median paired difference is negative and only 47.95% of genes favor "
        "Bridge. Bridge + PCA improves mean PCC slightly, indicating limited "
        "complementarity. The published BulkFormer value of 0.186 is retained as "
        "external context because its exact evaluation procedure could not be recovered."
    ),
    nbf.v4.new_code_cell(
        "variance = pd.read_csv(RESULTS / 'analysis' / 'variance_control_correlations.csv')\n"
        "display(variance)"
    ),
]

nbf.write(notebook, BENCH / "gene_essentiality.ipynb")
print(BENCH / "gene_essentiality.ipynb")
