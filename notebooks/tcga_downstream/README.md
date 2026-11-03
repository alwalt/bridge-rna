# Complete five-fold OOF TCGA survival evaluation

This directory contains an executed scientific notebook built from validated cached aggregate evidence. The original research pipeline and primary research notebook remain under [`benchmarks/tcga_downstream/`](../../benchmarks/tcga_downstream/).

## Contents

- [`tcga_downstream.ipynb`](tcga_downstream.ipynb): executed report.
- [`cached_evidence/`](cached_evidence/): compact result tables, QA, validation, and provenance.
- [`figures/`](figures/): regenerated PNG and PDF figures.

No raw TCGA data, model checkpoints, embeddings, H5 files, or large prediction tables are duplicated here.

## Lightweight cached rebuild

From the repository root:

```bash
python3 benchmarks/tcga_downstream/pipeline/build_oof_survival_notebook.py
jupyter nbconvert --to notebook --execute notebooks/tcga_downstream/tcga_downstream.ipynb \
  --output tcga_downstream.ipynb --output-dir notebooks/tcga_downstream \
  --ExecutePreprocessor.timeout=300
```

Tested dependencies include Python 3.10, NumPy, pandas, Matplotlib, lifelines 0.30.0, nbformat, nbconvert, Jupyter, SciPy, and scikit-learn. Exact versions are recorded in [`cached_evidence/provenance.json`](cached_evidence/provenance.json). A minimal environment can be prepared with:

```bash
python3 -m pip install numpy pandas matplotlib lifelines==0.30.0 nbformat nbconvert jupyter scipy scikit-learn
```

## Full research pipeline

The cached report does not require model inference. Full downstream reproduction from existing frozen representations uses:

```bash
python3 benchmarks/tcga_downstream/pipeline/run_survival_5fold_oof.py
python3 benchmarks/tcga_downstream/pipeline/run_survival_head_factorial.py
python3 benchmarks/tcga_downstream/pipeline/finalize_qa.py
```

These commands require the shared TCGA expression assets, frozen Bridge embeddings, configured checkpoint, and benchmark work caches described in the research [README](../../benchmarks/tcga_downstream/README.md). They can be computationally expensive and were not rerun to build this report. Long-running execution writes logs under `benchmarks/tcga_downstream/results/`; monitor the relevant log with `tail -f`.

## Validation

The builder checks prediction uniqueness, identical patients/folds/outcomes across all six representation × head combinations, complete 9,668-patient coverage, and recalculates fold-aware C-indices from saved OOF predictions using the established concordance definition. Results must match saved summaries to numerical tolerance before the notebook is generated.

The scientific scope is limited to Bridge 45.6M, PCA-128, and full raw expression under complete five-fold OOF survival evaluation. Literature values are contextual and not directly paired.
