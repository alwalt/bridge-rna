#!/usr/bin/env python3
"""Add the all-eligible OSDR atlas section and execute the benchmark notebook."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "hallmark_state_retrieval.ipynb"
TAG = "osdr-atlas-generated"


def main() -> None:
    notebook = nbformat.read(PATH, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.get("metadata", {}).get("tags", [])]
    metadata = {"tags": [TAG]}
    notebook.cells.extend([
        nbformat.v4.new_markdown_cell(
            """## All-eligible OSDR reference-atlas analysis

The frozen cohort contains 1,471 samples from 64 studies, 45 tissues, and 88 exact FLT/GC strata. Individual retrieval is primary; centroids summarize each condition. Differential retrieval uses reciprocal-rank enrichment and 1,000 within-block label permutations.""",
            metadata=metadata),
        nbformat.v4.new_code_cell(
            """osdr_consistency = pd.read_csv(ROOT/'results/osdr_atlas/replicate_consistency_complete.csv')
osdr_consistency[osdr_consistency['k'].eq(10)].groupby(['method','condition'])[['mean_neighbor_jaccard','mean_gse_jaccard','metadata_category_jaccard','same_tissue_fraction','signal_coverage_top10']].mean()""",
            metadata=metadata),
        nbformat.v4.new_code_cell(
            """osdr_summary = json.loads((ROOT/'results/osdr_atlas/scientific_summary.json').read_text())
osdr_summary""",
            metadata=metadata),
        nbformat.v4.new_code_cell(
            """from IPython.display import Image, display
display(Image(filename=str(ROOT/'results/osdr_atlas/figures/replicate_consistency.png')))
display(Image(filename=str(ROOT/'results/osdr_atlas/figures/flt_gc_category_shifts.png')))""",
            metadata=metadata),
        nbformat.v4.new_markdown_cell(
            """Retrieval is replicate-consistent, but no universal FLT state emerges. Bridge preserves tissue context; Hallmark retrieves substantially different program analogs and is more hub-prone. Strict replicate-consistent FLT enrichment occurs in 24/88 groups. The evidence supports a qualified, hypothesis-generating reference-atlas search tool—not an autonomous disease-state classifier.""",
            metadata=metadata),
    ])
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, PATH)


if __name__ == "__main__":
    main()
