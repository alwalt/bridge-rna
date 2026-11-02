# TCGA–ARCHS4 state retrieval report source

The notebook generator is maintained here; the original retrieval analysis and large caches remain in the source agent worktree.

## Scientific notebook source

The report is saved under `notebooks/tcga_archs4_state_retrieval/`. Its generation and cache-preparation scripts are maintained in this research directory:

- `pipeline/make_notebook.py` (from the repository root: `python benchmarks/tcga_archs4_state_retrieval/pipeline/make_notebook.py`).

See [the report README](../../notebooks/tcga_archs4_state_retrieval/README.md) for dependencies, cached execution, and regeneration details. These scripts use saved evidence; do not launch the full research pipeline to execute the report.
