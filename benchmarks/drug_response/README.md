# Frozen drug-response representation benchmark

This directory contains an executed scientific report built from validated cached evidence in `research/drug_response/`. The notebook runs without model inference, network access, raw expression matrices, checkpoints, embeddings, or prediction arrays.

## Lightweight cached report

Tested with Python 3.11, NumPy 1.26, pandas 2.3, SciPy 1.17, scikit-learn 1.8, Matplotlib 3.10, nbformat 5.11, and Jupyter nbconvert.

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib nbformat jupyter
cd benchmarks/drug_response
jupyter nbconvert --to notebook --execute --inplace drug_response.ipynb \
  --ExecutePreprocessor.timeout=1200
```

Cached execution should complete in seconds on CPU. The notebook reads only `data/` and `figures/`.

## Rebuild the cached package

From the repository root, after the full benchmark outputs are available:

```bash
python research/drug_response/pipeline/build_scientific_notebook.py
```

This validates saved out-of-fold predictions against per-screen metrics, copies compact evidence, and regenerates the unexecuted notebook. Execute it with the command above.

## Full benchmark

The source release is [Zenodo record 10.5281/zenodo.15744294](https://doi.org/10.5281/zenodo.15744294). Full execution additionally requires the Bridge checkpoint, both BulkFormer checkpoints and vocabularies, the canonical Bridge gene vocabulary, a configured PyTorch environment, and sufficient local storage for the released expression/response tables and cached arrays. Paths are configured in `research/drug_response/config.json` and may need local adjustment.

```bash
python research/drug_response/pipeline/audit_data.py
python research/drug_response/pipeline/prepare_inputs.py
python research/drug_response/pipeline/make_splits.py
python research/drug_response/pipeline/extract_embeddings.py --model bridge_45.6m --device cuda:0
python research/drug_response/pipeline/extract_embeddings.py --model bulkformer_50m --device cuda:0
python research/drug_response/pipeline/extract_embeddings.py --model bulkformer_147m --device cuda:0
python research/drug_response/pipeline/evaluate.py
python research/drug_response/pipeline/make_figures.py
```

Neural representation extraction is the expensive stage and should use an available GPU; runtime depends on hardware. The saved run processes 700 cell lines. The downstream PCA/ridge evaluation is CPU-light (minutes on the validation machine). Pipeline commands emit progress heartbeats; redirect output to a log and monitor with `tail -f <log>` for long inference runs.

## Evidence and interpretation

- `data/tables/`: compact numerical source tables.
- `data/provenance.json`: source hashes, cohort, protocol, dependencies, validation outcomes, and limitations.
- `figures/`: PNG and PDF figures paired with CSV source tables in cached evidence.
- Scientific review: [`research_archive/drug_response/PUBLICATION_REVIEW.md`](research_archive/drug_response/PUBLICATION_REVIEW.md)
- Manuscript outline: [`research_archive/drug_response/MANUSCRIPT_OUTLINE.md`](research_archive/drug_response/MANUSCRIPT_OUTLINE.md)

The research pipeline under `research/drug_response/pipeline/` remains the computational source of truth.

## Directory layout

- `data/`: cached evidence, provenance, validation records, and dependency files.
- `figures/`: report figures and associated plot tables.
- `research_archive/`: local, Git-ignored historical research copies; not required to execute this notebook.
- `drug_response.ipynb`: scientific report.

Run the notebook from this directory with its documented Python dependencies. Historical source paths in provenance describe the original runs and are retained unchanged. Research scripts now live under the local `research/` directory; archive copies preserve their original contents.

## Saved observed-versus-predicted IC50 plot

`figures/bridge_oof_ic50_regression.{png,pdf,svg}` shows all 212,299 measured cell-line–screen responses against saved held-out BRIDGE predictions. No target standardization, averaging, screen selection, or recalibration is applied. The all-point correlation includes between-screen differences and is distinct from mean per-screen PCC. A compressed full-precision plotted-value CSV and provenance JSON accompany the figure. PDF/SVG point layers are rasterized at 600 dpi, with vector axes/text.

Regenerate from the saved points with:

```bash
.venv/bin/python benchmarks/drug_response/pipeline/plot_ic50_predictions.py
```

To re-extract points from original research caches, pass `--research-root /path/to/original/drug_response`. That directory must contain the saved prediction array, cell-line splits, and observation manifest. All 355 screen correlations are checked before export. The notebook displays the saved plot without requiring original worktree caches.

The notebook pairs this scatter with `figures/bridge_per_screen_pcc_histogram.{png,pdf,svg}`, showing all 355 within-screen correlations (mean 0.448). Regenerate with `.venv/bin/python benchmarks/drug_response/pipeline/plot_screen_correlations.py`.
