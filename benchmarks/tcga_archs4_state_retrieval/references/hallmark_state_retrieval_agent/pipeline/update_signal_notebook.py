#!/usr/bin/env python3
"""Add/replace the generated signal-matching section and execute the notebook."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "hallmark_state_retrieval.ipynb"
TAG = "signal-matching-generated"


def main() -> None:
    notebook = nbformat.read(PATH, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if TAG not in cell.get("metadata", {}).get("tags", [])]
    md = {"tags": [TAG]}
    notebook.cells.extend([
        nbformat.v4.new_markdown_cell("""## Dominant-signal matching audit

Six frozen 50-D scoring rules are compared on all 33 TCGA centroids, with Bridge-512 retained as a global-transcriptome reference. Signal coverage requires matching sign and high within-sample rank; background percentiles use all 510,709 ARCHS4 samples.""", metadata=md),
        nbformat.v4.new_code_cell("""signal_macro = pd.read_csv(ROOT/'results/signal_matching_audit/macro_method_comparison.csv')
signal_macro[signal_macro['k'].eq(10)]""", metadata=md),
        nbformat.v4.new_code_cell("""from IPython.display import Image, display
display(Image(filename=str(ROOT/'results/signal_matching_audit/figures/method_comparison.png')))
display(Image(filename=str(ROOT/'results/signal_matching_audit/figures/per_cancer_signal_coverage.png')))""", metadata=md),
        nbformat.v4.new_markdown_cell("""Sparse matching improves its target metric but sacrifices context. Top-10 overlap raises signed signal coverage from 0.791 to 0.888 and containment to 0.854, while exact-disease P@10 falls from 0.482 to 0.388. Top-5 overlap is hub-prone and query-weighting saturates for unrelated samples. Signal matching is best retained as an interpretable secondary view or reranker, not the sole retrieval metric.""", metadata=md),
    ])
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, PATH)


if __name__ == "__main__":
    main()
