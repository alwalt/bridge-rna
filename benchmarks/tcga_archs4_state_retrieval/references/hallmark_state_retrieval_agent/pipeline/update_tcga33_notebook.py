#!/usr/bin/env python3
"""Add/replace the generated all-33 TCGA section and execute the benchmark notebook."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "hallmark_state_retrieval.ipynb"
TAG = "tcga33-generated"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.get("metadata", {}).get("tags", [])]
    metadata = {"tags": [TAG]}
    notebook.cells.extend(
        [
            nbformat.v4.new_markdown_cell(
                """## All-33 TCGA frozen-output evaluation

This section loads the saved exact-retrieval results for 9,942 individual patients and 33 cancer-type centroids. The primary table reports confirmed exact-disease or identifiable same-tissue-cancer precision; missing or ambiguous ARCHS4 metadata remain unconfirmed and are audited separately through coverage and conditional precision.""",
                metadata=metadata,
            ),
            nbformat.v4.new_code_cell(
                """tcga33 = pd.read_csv(ROOT/'results/tcga33_retrieval/tcga33_cancer_comparison.csv')
tcga33""",
                metadata=metadata,
            ),
            nbformat.v4.new_code_cell(
                """macro33 = pd.read_csv(ROOT/'results/tcga33_retrieval/macro_hypothesis_metrics.csv')
macro33[macro33['k'].eq(50)].pivot(index='criterion', columns='method', values='macro_precision')""",
                metadata=metadata,
            ),
            nbformat.v4.new_code_cell(
                """from IPython.display import Image, display
display(Image(filename=str(ROOT/'results/tcga33_retrieval/figures/per_cancer_method_difference.png')))
display(Image(filename=str(ROOT/'results/tcga33_retrieval/figures/per_cancer_criterion_heatmap.png')))""",
                metadata=metadata,
            ),
            nbformat.v4.new_markdown_cell(
                """No cancer crossed the descriptive +0.10 P@50 threshold for clear Hallmark improvement; OV, PCPG, and THCA crossed −0.10. UVM improved at P@10 but converged by P@100. Centroid inspection found biologically plausible functional reranking for UVM and CHOL, plus three LUAD→LUSC same-lung hits, but the individual-patient macro results favor Bridge-512 for every criterion. OSDR remains a separate cross-species/tissue stress test.""",
                metadata=metadata,
            ),
        ]
    )
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
