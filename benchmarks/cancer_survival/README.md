# Cancer survival prediction from frozen transcriptomic representations

This directory contains an executed scientific notebook built from validated cached aggregate evidence. The original research pipeline and primary research notebook remain under [`research/tcga_downstream/`](research_archive/tcga_downstream/).

## Contents

- [`cancer_survival.ipynb`](cancer_survival.ipynb): executed report.
- [`data/`](data/): compact result tables, QA, validation, and provenance.
- [`figures/`](figures/): regenerated PNG and PDF figures.

No raw TCGA data, model checkpoints, embeddings, H5 files, or large prediction tables are duplicated here.

## Lightweight cached rebuild

From the repository root:

```bash
python3 research/tcga_downstream/pipeline/build_oof_survival_notebook.py
jupyter nbconvert --to notebook --execute benchmarks/cancer_survival/tcga_downstream.ipynb \
  --output cancer_survival.ipynb --output-dir benchmarks/cancer_survival \
  --ExecutePreprocessor.timeout=300
```

Tested dependencies include Python 3.10, NumPy, pandas, Matplotlib, lifelines 0.30.0, nbformat, nbconvert, Jupyter, SciPy, and scikit-learn. Exact versions are recorded in [`data/provenance.json`](data/provenance.json). A minimal environment can be prepared with:

```bash
python3 -m pip install numpy pandas matplotlib lifelines==0.30.0 nbformat nbconvert jupyter scipy scikit-learn
```

## Full research pipeline

The cached report does not require model inference. Full downstream reproduction from existing frozen representations uses:

```bash
python3 research/tcga_downstream/pipeline/run_survival_5fold_oof.py
python3 research/tcga_downstream/pipeline/run_survival_head_factorial.py
python3 research/tcga_downstream/pipeline/finalize_qa.py
```

These commands require the shared TCGA expression assets, frozen Bridge embeddings, configured checkpoint, and benchmark work caches described in the research [README](research_archive/tcga_downstream/README.md). They can be computationally expensive and were not rerun to build this report. Long-running execution writes logs under `research/tcga_downstream/results/`; monitor the relevant log with `tail -f`.

## Validation

The builder checks prediction uniqueness, identical patients/folds/outcomes across all six representation × head combinations, complete 9,668-patient coverage, and recalculates fold-aware C-indices from saved OOF predictions using the established concordance definition. Results must match saved summaries to numerical tolerance before the notebook is generated.

The scientific scope is limited to Bridge 45.6M, PCA-128, and full raw expression under complete five-fold OOF survival evaluation. Literature values are contextual and not directly paired.

## Directory layout

- `data/`: cached evidence, provenance, validation records, and dependency files.
- `figures/`: report figures and associated plot tables.
- `research_archive/`: local, Git-ignored historical research copies; not required to execute this notebook.
- `cancer_survival.ipynb`: scientific report.

Run the notebook from this directory with its documented Python dependencies. Historical source paths in provenance describe the original runs and are retained unchanged. Research scripts now live under the local `research/` directory; archive copies preserve their original contents.

## Regenerate the within-cancer forest plots

Run `.venv/bin/python benchmarks/cancer_survival/pipeline/plot_survival_forest.py`
from the repository root. This reads saved per-cancer metrics only, sorts all
32 cancer types by decreasing Bridge C-index, and writes combined and Bridge-only
PNG/PDF/SVG figures. A small x-axis margin below zero keeps interval endpoints
visible. Point estimates, bootstrap intervals, and event counts are unchanged.

The Bridge-only forest plot uses two columns of 16 cancers: decreasing Bridge
C-index from top to bottom in the left panel, then the right panel.

Survival plot display labels use **BRIDGE**; underlying saved representation keys
are preserved. Figures use the manuscript convention: 183-mm width, Arial
7–8.5-pt typography, restrained colors, embedded-font PDF, editable SVG, and
600-dpi PNG. Sorting subtitles, rank annotations, and the redundant single-model
legend are omitted. Axis labels, cancer labels, intervals, and events/N remain.
