# TCGA downstream: complete five-fold OOF survival

This benchmark's manuscript-facing scope is the complete five-fold out-of-fold evaluation of censoring-aware overall-survival prediction using:

- frozen Bridge 45.6M embeddings;
- training-fold-only PCA-128;
- full raw expression.

Every eligible patient receives exactly one held-out prediction per representation and head. Bridge remains frozen; only downstream Cox heads are trained.

## Primary scientific question

Among patients with the same TCGA cancer type, how well can each transcriptomic representation rank overall-survival risk?

The benchmark evaluates both a bias-free linear Cox head and Cox MLP `[512,256]` for every representation, preserving the PanCanAtlas time-to-event endpoint and censoring.

## Cohort

- 9,668 eligible patients.
- 32 cancer types.
- 2,774 overall-survival events.
- One deterministic eligible specimen per patient.
- Positive `OS.time` and valid binary `OS` required.
- Five mutually exclusive cancer × event-balanced OOF folds.

## Representations

- **Bridge 45.6M:** frozen 512-dimensional mean-pooled L12 representation from the ordered 15,165-gene natural `log1p(TPM)` input.
- **PCA-128:** scaling and PCA fitted separately inside each outer training fold.
- **Full raw expression:** 25,150 `log1p(CPM)` features with training-fold-only standardization.

## Primary results

| Representation | Linear Cox C-index | Cox MLP C-index |
| --- | ---: | ---: |
| Bridge 45.6M | 0.7589 | 0.7580 |
| PCA-128 | 0.7615 | 0.7638 |
| Full raw expression | 0.7533 | 0.7590 |

All six combinations contain exactly 9,668 predictions and 3,068,664 fold-valid comparable pairs. Bridge is nearly head-insensitive. No representation or head contrast remains significant after Benjamini–Hochberg correction across seven paired cancer-level tests.

Median within-cancer C-indices are 0.6239 for Bridge, 0.6360 for PCA-128, and 0.6008 for full expression. Bridge exceeds each comparator in 15/32 cancers but shows no systematic cancer-level advantage.

## Primary files

### Reports

- [Scientific report](SCIENTIFIC_REPORT.md)
- [Publication review](PUBLICATION_REVIEW.md)
- [Manuscript outline](MANUSCRIPT_OUTLINE.md)
- [Executed notebook](tcga_downstream_benchmark.ipynb)

### Cohort and predictions

- [Fold manifest](results/survival_5fold_oof_fold_manifest.csv)
- [OOF predictions](results/survival_5fold_oof_predictions.csv)
- [Matched-head predictions](results/survival_head_factorial_predictions.csv)
- [Per-cancer metrics](results/survival_5fold_oof_per_cancer.csv)
- [Representation comparison](results/survival_5fold_oof_representation_comparison.csv)

### Summary tables

- [Complete OOF metrics](results/survival_5fold_oof_metrics.csv)
- [Matched-head metrics](results/survival_head_factorial_metrics.csv)
- [Matched-head statistical summary](results/survival_head_factorial_statistical_summary.csv)
- [Training QC](results/survival_head_factorial_training_qc.csv)
- [Layerwise summary](results/bridge_layerwise_survival_summary.csv)
- [Final QA](results/final_qa.json)

### Figures

- [Bridge alphabetical forest plot](results/figures/bridge_5fold_oof_per_cancer_survival_forest.png)
- [PCA-128 alphabetical forest plot](results/figures/pca128_5fold_oof_per_cancer_survival_forest.png)
- [Full-expression alphabetical forest plot](results/figures/raw_expression_5fold_oof_per_cancer_survival_forest.png)
- [Combined alphabetical forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest.png)
- [Combined Bridge-ranked forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest_bridge_sorted.png)
- [Three-panel comparison](results/figures/bridge_pca_raw_5fold_oof_survival_three_panel.png)
- [Delta-C-index comparison](results/figures/bridge_vs_conventional_5fold_oof_delta_cindex.png)
- [Matched-head overall comparison](results/figures/survival_head_factorial_overall.png)
- [Matched-head per-cancer comparison](results/figures/survival_head_factorial_per_cancer_two_panel.png)
- [Layerwise depth curve](results/figures/bridge_layerwise_survival_cindex.png)

## Reproduction

The expensive foundation-model inference is already cached. Reproduce downstream OOF survival from cached representations with:

```bash
python3 benchmarks/tcga_downstream/pipeline/run_survival_5fold_oof.py
python3 benchmarks/tcga_downstream/pipeline/run_survival_head_factorial.py
python3 benchmarks/tcga_downstream/pipeline/finalize_qa.py
```

The scripts enforce complete coverage, patient uniqueness, fold exclusivity, representation matching, training-only learned preprocessing, frozen Bridge status, and fold-aware concordance.

## Interpretation boundary

The benchmark supports preservation of prognostic information in a compact frozen representation. It does not demonstrate systematic superiority over PCA-128 or full expression, clinical utility, external validation, or incremental value beyond clinical covariates.
