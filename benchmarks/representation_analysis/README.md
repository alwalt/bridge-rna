# Representation analysis

This benchmark examines how frozen Bridge gene identity vectors develop functional neighborhoods and context dependence at L0, L1, L6, and L12. It compares the resulting cohort-level geometry with Pearson/Spearman coexpression and tests incremental shared-annotation prediction after specified expression controls.

## Execute the report

From the repository root, create an environment and run the notebook:

```bash
python3 -m venv .venv-representation-report
. .venv-representation-report/bin/activate
python -m pip install -r benchmarks/representation_analysis/data/requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
  benchmarks/representation_analysis/representation_analysis.ipynb \
  --ExecutePreprocessor.timeout=300
```

The notebook can also run with its kernel working directory set to this benchmark folder or `benchmarks/`. It uses only `data/` and `figures/`; it requires no network, checkpoint, embeddings, source expression matrices, research archives, or model fitting. Cells regenerate seven PNG/PDF figures (plus SVG for the context UMAP) and unrounded figure-data CSVs under `data/figure_data/`. Allow approximately one minute on CPU, depending on plotting and notebook startup.

## Evidence and interpretation

- The balanced cohort includes 200 GTEx samples across 10 tissues and 200 TCGA samples across 10 tumor contexts, with 20 samples per context. The exact frozen sample manifest is in `data/sample_manifest.csv`.
- L0 is sample-invariant; contextual summaries use sample SD over 200 samples. Context UMAPs use a separate organ-corresponding cohort: 500 genes averaged across 20 samples per tissue/cancer group (400 samples total). The original quantitative cohort remains in `data/sample_manifest.csv`; visualization samples and organ/color pairs are in `data/context_visualization_sample_manifest.csv` and `data/context_visualization_palette.json`. Saved gene-centered PCA50/UMAP layouts are descriptive, not comparable effect sizes or patient/cell-type matches.
- GTEx has greater between-context differentiation; TCGA has lower within-context same-gene directional consistency. Neither quantity is absolute embedding variance or a paired tumor effect.
- Cohort-level coexpression comparisons differ from per-sample neighborhoods. Their confidence intervals resample genes; conditional prediction intervals resample anchors.
- Conditional metrics were recalculated from 535,206 saved OOF prediction rows. Partner sampling includes repeats: 722 directed GO rows and 198 directed KEGG rows per cohort. Rows are retained as originally sampled. The split groups anchors, not both genes, so it does not establish fully gene-disjoint generalization.
- `data/validation.json` records numerical checks. `data/provenance.json` records source hashes, dependencies, preprocessing, seeds, checkpoint identity, and limitations. Original prediction rows are not duplicated; their source path and hash are recorded.
- The static baseline has weak functional organization under the tested geometry. L12 adds annotation-predictive signal beyond specified expression/correlation controls, without establishing expression-independent biology or causal mechanisms.

## Regenerate from saved research evidence

The local source generator is `research/static_gene_embeddings/pipeline/build_representation_analysis_report.py`. It validates saved predictions/aggregates, refreshes the compact evidence, and generates notebook source:

```bash
python research/static_gene_embeddings/pipeline/build_representation_analysis_report.py
jupyter nbconvert --to notebook --execute --inplace \
  benchmarks/representation_analysis/representation_analysis.ipynb \
  --ExecutePreprocessor.timeout=300
```

The generator remains in the ignored local research tree. A fresh clone can execute the report using its supplied compact evidence without that generator. The historical research scripts and notebook are preserved in the local ignored `research_archive/static_gene_embeddings/`; they are not runtime dependencies. No archive files or historical results are modified by the report.

## Full research reproduction

Full recomputation is a separate, expensive workflow. Before running it:

1. Obtain the frozen Bridge `model/r7hnr92k/best_model.pt` and model configuration, canonical 15,165-gene vocabulary, GENCODE v49 human gene lengths, and GO Biological Process/KEGG annotation snapshots identified in the source provenance.
2. Restore the canonical ARCHS4 expression/training manifests, GTEx expression/sample metadata, TCGA expression/cohort metadata, and the frozen balanced cohort selection. Preserve original sample order and natural `log1p(TPM)` preprocessing.
3. Inspect the scripts' input constants. Original research scripts retain historical `benchmarks/...` paths from before the local directory rename; remap these to the corresponding `research/...` assets before execution. Do not point them at this report's compact `data/` files as substitutes for expression arrays or embeddings. Original source provenance is retained unchanged in `data/*_source_provenance.json`.
4. Prepare the broader research environment: PyTorch with appropriate CUDA support, h5py, SciPy, scikit-learn, statsmodels, UMAP-learn, NumPy, pandas, Matplotlib, and supporting packages used by the historical scripts. The minimal report requirements are not a complete training/inference environment.

Once inputs and historical path mappings have been restored, the source entrypoints are:

```bash
python research/static_gene_embeddings/pipeline/run_static_embeddings.py
python research/static_gene_embeddings/pipeline/characterize_static_geometry.py --exploratory
python research/static_gene_embeddings/pipeline/contextual_depth_followup.py
python research/static_gene_embeddings/pipeline/matched_gtex_tcga.py --device cuda:0
python research/static_gene_embeddings/pipeline/contextual_gene_modules.py --device cuda:0
python research/static_gene_embeddings/pipeline/contextual_module_expression_controls.py
python research/static_gene_embeddings/pipeline/coexpression_baseline.py --device cuda:0
python research/static_gene_embeddings/pipeline/l12_conditional_function.py
python research/static_gene_embeddings/pipeline/context_colored_gene_projections.py --device cuda:0
python research/static_gene_embeddings/pipeline/build_paper_summary.py
```

These commands were not run to prepare this report. Recorded source timings include approximately 70 minutes for balanced contextual analysis, 18 minutes for cohort modules, 14 minutes for linear expression controls, and 4.4 minutes for conditional modeling; current hardware and cache availability will change these estimates. Full extraction requires substantial shared source storage, GPU inference, and several large intermediate arrays. Precise peak storage/GPU memory was not established in the retained provenance. Avoid launching these commands merely to execute the report.

For the matched contextual run, monitor its existing flushed log with:

```bash
tail -f research/static_gene_embeddings/results/matched_gtex_tcga/run.log
```

## Directory layout

- `representation_analysis.ipynb`: executed scientific report.
- `data/`: compact numerical evidence, exact cohort, source provenance, validation, dependencies, and figure data.
- `figures/`: regenerated PNG/PDF figures.
- `research_archive/`: local, Git-ignored historical research copy.
- `PUBLICATION_REVIEW.md` and `MANUSCRIPT_OUTLINE.md`: local, Git-ignored writing references when present.
