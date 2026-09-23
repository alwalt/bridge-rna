#!/usr/bin/env python3
"""Insert/update the BulkFormer-parity section in the primary notebook."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tcga_downstream_benchmark.ipynb"
TAG = "bulkformer-parity-generated"
COX_TAG = "cox-per-cancer-generated"
SURVIVAL_UNCERTAINTY_TAG = "survival-uncertainty-generated"
SURVIVAL_OOF_TAG = "survival-oof-generated"


def markdown(text: str):
    cell = nbformat.v4.new_markdown_cell(text)
    cell.metadata["tags"] = [TAG]
    return cell


def code(text: str):
    cell = nbformat.v4.new_code_cell(text)
    cell.metadata["tags"] = [TAG]
    return cell


notebook = nbformat.read(NOTEBOOK, as_version=4)
notebook.cells = [cell for cell in notebook.cells
                  if not ({TAG, COX_TAG, SURVIVAL_UNCERTAINTY_TAG, SURVIVAL_OOF_TAG} &
                          set(cell.metadata.get("tags", [])))]

for cell in notebook.cells:
    if cell.cell_type != "markdown":
        continue
    if cell.source.startswith("## 2. Five-cohort"):
        cell.source = cell.source.replace("## 2.", "## A. Common local benchmark\n\n### A1.", 1)
    elif cell.source.startswith("## 3. Pan-cancer"):
        cell.source = cell.source.replace("## 3.", "### A2.", 1)
    elif cell.source.startswith("## 3.1 Mean"):
        cell.source = cell.source.replace("## 3.1", "### A3", 1)
    elif cell.source.startswith("## 4. Published"):
        cell.source = cell.source.replace("## 4.", "## C.", 1)
    elif cell.source.startswith("### 4.1 Complete"):
        cell.source = cell.source.replace("### 4.1", "### C1", 1)

insert_at = next(i for i, cell in enumerate(notebook.cells)
                 if cell.cell_type == "markdown" and cell.source.startswith("## C. Published"))
section = [
    markdown("""## B. BulkFormer-parity benchmark

The final Cell Systems release reports TCGA classification weighted F1 = 0.907, but the authors do not release downstream training code or fold assignments. Their released `TCGA_survival.h5ad` contains 10,429 sample rows for 9,679 patients, including 750 repeated-patient rows and 709 normal-tissue aliquots. The literature value is therefore not patient-level paired with the results below.

The conservative local reconstruction uses all 33 TCGA labels, one specimen per patient (primary tumors plus LAML peripheral blood), native model vocabularies, frozen encoders, training-fold-only PCA-128, random forests, and ten fixed patient-level folds. All representations share the same 9,942 patients and folds. Existing caches cover 9,816 patients; only 126 LAML patients required new inference. BulkFormer uses mean pooling, matching the final release notebook example; the inaccessible final downstream implementation leaves pooling and several classifier details ambiguous."""),
    code("""parity_classification = pd.read_csv('results/bulkformer_parity_pan_cancer_classification_summary.csv')
parity_classification[['representation', 'method', 'folds', 'macro_f1_mean',
                       'weighted_f1_mean', 'balanced_accuracy_mean']].round(4)"""),
    code("""from IPython.display import Image
display(Image(filename=str(FIGURES / 'bulkformer_parity_pan_cancer_classification.png')))"""),
    markdown("""### B1. Pan-cancer classification conclusion

On identical patient-level folds, full raw expression is strongest. BulkFormer-147M is the strongest frozen FM, Bridge 45.6M is next, and BulkFormer-50M is lower. The authors' 0.907 is shown only as a literature reference because their released cohort includes repeated patients and normal aliquots and their exact folds/code are unavailable."""),
    markdown("""### B2. BulkFormer-style alive/dead prognosis

This binary endpoint discards follow-up time and censoring structure. It is reported separately and is not interchangeable with the Cox/C-index benchmark in section A2. The final release retains prognosis modeling and releases `OS` plus `OS.time`, but does not publish final prognosis metrics in the repository README; AUROC/AUPRC = 0.747/0.549 are historical preprint values only."""),
    code("""parity_prognosis = pd.read_csv('results/bulkformer_parity_alive_dead_prognosis_summary.csv')
parity_prognosis[['representation', 'method', 'folds', 'auroc_mean', 'auprc_mean']].round(4)"""),
    code("""display(Image(filename=str(FIGURES / 'bulkformer_parity_alive_dead_prognosis.png')))"""),
    markdown("""### B3. Per-cancer alive/dead performance

The heatmap below calculates AUROC separately within each cancer from pooled out-of-fold predictions, so every patient contributes exactly once per representation. `NA` means that a cancer has only one observed outcome class and AUROC is undefined. Small-event cancers should be interpreted descriptively; this remains a binary alive/dead analysis rather than censoring-aware survival."""),
    code("""per_cancer_prognosis = pd.read_csv('results/bulkformer_parity_alive_dead_prognosis_per_cancer.csv')
display(per_cancer_prognosis.sort_values(['cancer_label', 'representation'])
        [['cancer_label', 'representation', 'patients', 'events', 'alive', 'auroc', 'auprc']]
        .style.format({'auroc': '{:.3f}', 'auprc': '{:.3f}'}).hide(axis='index'))"""),
    code("""display(Image(filename=str(FIGURES / 'bulkformer_parity_alive_dead_prognosis_per_cancer.png')))"""),
    markdown("""### B3.1 Direct visual comparison with Cox survival

For visibility, the two per-cancer plots are aligned below. The left panel is the BulkFormer-style binary alive/dead endpoint (AUROC); the right panel is the censoring-aware Cox endpoint (C-index). The metrics are different and should not be numerically equated. LAML is `NA` for Cox because the original primary-tumor Cox cohort contains 32 solid-cancer projects."""),
    code("""display(Image(filename=str(FIGURES / 'per_cancer_alive_dead_vs_cox_comparison.png')))"""),
    markdown("""### B4. Prognosis conclusion

Raw expression is strongest for alive/dead prediction. Bridge 45.6M and BulkFormer-147M are effectively tied in mean AUROC/AUPRC on the common patient-level folds; BulkFormer-50M is lower. This does not alter the stronger scientific status of the existing time-to-event Cox analysis."""),
]
notebook.cells[insert_at:insert_at] = section

cox_cells = [
    markdown("""### A2.2 Legacy split-mean Cox heatmap

This heatmap reports censoring-aware C-index within each cancer on held-out patients, averaged across the same five seeded splits used by the primary Cox benchmark. `NA` indicates that a held-out cancer subset had no comparable pairs or no event. Small cohorts and low-event cancers are descriptive and can have high split-to-split variance."""),
    code("""cox_per_cancer = pd.read_csv('results/cox_mlp_per_cancer_summary.csv')
display(cox_per_cancer.sort_values(['cancer_label', 'representation'])
        [['cancer_label', 'representation', 'splits', 'test_patients_mean',
          'events_mean', 'c_index_mean', 'c_index_sd']]
        .style.format({'test_patients_mean': '{:.1f}', 'events_mean': '{:.1f}',
                       'c_index_mean': '{:.3f}', 'c_index_sd': '{:.3f}'}).hide(axis='index'))"""),
    code("""from IPython.display import Image
display(Image(filename=str(FIGURES / 'cox_mlp_per_cancer_c_index.png')))"""),
    markdown("""### A2.3 Legacy Bridge split-SD bar chart

The bar chart isolates Bridge 45.6M and shows the mean held-out within-cancer C-index with SD across the five fixed splits. The dashed line marks chance concordance at 0.5. Large error bars primarily reflect small held-out event counts, so the cancer-specific values should be interpreted descriptively."""),
    code("""display(Image(filename=str(FIGURES / 'bridge_cox_per_cancer_c_index_bar.png')))"""),
]

uncertainty_cells = [
    markdown("""### A2.1 Per-cancer survival discrimination

These forest plots calculate C-index **within each cancer** from the existing held-out Cox risk predictions. Censoring and observed survival times are retained. The original protocol consists of five repeated stratified 80/20 splits rather than exhaustive K-fold OOF prediction: concordance pairs are therefore formed only within each seed's held-out set and then aggregated by comparable-pair count. Some patients appear in multiple held-out splits and some in none.

Confidence intervals are percentile intervals from 1,000 patient-level bootstrap resamples within cancer (fixed seed 20260922). A patient's survival outcome and all held-out appearances stay together. Event counts are shown because sparse-event estimates can look extreme while remaining highly uncertain. Flags identify fewer than 10 events as `very_low_event_support`, 10–19 as `low_event_support`, CI width above 0.25 as `wide_ci`, and fewer than 900 valid bootstrap replicates as `unreliable_bootstrap`."""),
    code("""per_cancer_survival = pd.read_csv('results/per_cancer_survival_metrics.csv')
bridge_survival = per_cancer_survival.query("representation == 'Bridge 45.6M'")
display(bridge_survival[['cancer', 'n_patients', 'n_events', 'n_censored',
                         'censoring_fraction', 'comparable_pairs', 'c_index',
                         'ci_low', 'ci_high', 'bootstrap_valid', 'stability_flag']]
        .sort_values('cancer').style.format({
            'censoring_fraction': '{:.1%}', 'c_index': '{:.3f}',
            'ci_low': '{:.3f}', 'ci_high': '{:.3f}'}).hide(axis='index'))"""),
    code("""from IPython.display import Image
display(Image(filename=str(FIGURES / 'bridge_per_cancer_survival_forest.png')))"""),
    markdown("""The alphabetical Bridge-only forest plot is the manuscript-facing default. The companion plot below uses the same cancers and x-axis limits for the conventional PCA-128 and full-expression controls. Both controls use the exact same held-out patients as Bridge; PCA and all learned preprocessing were fitted on training patients only."""),
    code("""display(Image(filename=str(FIGURES / 'conventional_per_cancer_survival_forest.png')))"""),
    code("""survival_stats = pd.read_csv('results/per_cancer_survival_statistical_summary.csv')
display(survival_stats.style.format({'value': '{:.4g}'}).hide(axis='index'))"""),
    markdown("""Across the 32 cancers, Bridge's median within-cancer C-index is 0.598 and its patient-count-weighted mean is 0.603. Bridge is above 0.5 in 26 cancers; 15 cancer-specific intervals exclude 0.5. Bridge exceeds PCA-128 and full raw expression in 13 cancers each. The paired median differences are −0.008 versus PCA and −0.011 versus raw expression; neither the paired sign-flip tests nor Wilcoxon tests indicate a systematic difference (all p > 0.35). A nonsignificant result is not evidence of equivalence.

The strongest well-supported Bridge estimates are KIRP, ACC, LGG, UCEC, CESC, LIHC, LUAD, and KIRC. Estimates near or below chance include PRAD, READ, PCPG, THCA, THYM, SKCM, and DLBC, but PRAD, PCPG, THCA, THYM, and DLBC have fewer than 10 events, while READ and SKCM have fewer than 20; these should not be read as stable failures. Conversely, TGCT's apparently high C-index is based on only two events and is not reliable. The pooled pan-cancer C-index remains a separate quantity that can exploit between-cancer prognosis differences; these within-cancer plots do not replace it."""),
]
for cell in uncertainty_cells:
    cell.metadata["tags"] = [SURVIVAL_UNCERTAINTY_TAG]
uncertainty_at = next(i for i, cell in enumerate(notebook.cells)
                      if cell.cell_type == "markdown" and cell.source.startswith("### A3"))
notebook.cells[uncertainty_at:uncertainty_at] = uncertainty_cells

oof_cells = [
    markdown("""## Complete 5-fold out-of-fold survival evaluation

The authoritative primary-tumor manifest contains 9,816 patients. The unchanged OS eligibility rule retains 9,668 patients and 2,774 events; 21 patients lack a valid binary event, 34 have a valid event but missing time, and 93 have nonpositive time. The original five repeated 80/20 tests measure split-to-split stability, but overlapping test sets are expected: only 6,467 of 9,668 eligible patients appeared at least once. This fixed true five-fold evaluation partitions all 9,668 patients into mutually exclusive test folds, balanced deterministically within cancer × event strata. Every patient is trained against in four folds and receives exactly one prediction from the remaining fold.

Bridge remained frozen and its cached embeddings were reused; only the established `[512, 256]` downstream Cox heads were fitted. Full expression uses the same head and training-fold standardization. PCA-128 and its scaler are fitted independently inside each training fold, followed by the existing penalized Cox model. Overall and cancer-specific concordance is **fold-aware**: concordant and comparable-pair counts are summed across held-out folds, but risk scores from separately fitted heads are never compared across folds. Patient-level bootstrap resampling retains each patient's fixed fold membership (1,000 draws, seed 20260923)."""),
    code("""oof_metrics = pd.read_csv('results/survival_5fold_oof_metrics.csv')
oof_metrics.style.format({'oof_survival_c_index': '{:.3f}', 'comparable_pairs': '{:,.0f}'}).hide(axis='index')"""),
    code("""oof_per_cancer = pd.read_csv('results/survival_5fold_oof_per_cancer.csv')
display(oof_per_cancer.query("representation == 'Bridge 45.6M'")
        [['cancer', 'n_patients', 'n_events', 'n_censored', 'censoring_fraction',
          'comparable_pairs', 'c_index', 'ci_low', 'ci_high', 'bootstrap_valid', 'stability_flag']]
        .sort_values('cancer').style.format({'censoring_fraction': '{:.1%}',
            'c_index': '{:.3f}', 'ci_low': '{:.3f}', 'ci_high': '{:.3f}'}).hide(axis='index'))"""),
    markdown("""### Individual manuscript-facing forest plots

The three plots use identical patients, folds, endpoint, bootstrap definition, alphabetical cancer order, dimensions, support annotations, and x-axis limits."""),
    code("""display(Image(filename=str(FIGURES / 'bridge_5fold_oof_per_cancer_survival_forest.png')))"""),
    code("""display(Image(filename=str(FIGURES / 'pca128_5fold_oof_per_cancer_survival_forest.png')))"""),
    code("""display(Image(filename=str(FIGURES / 'raw_expression_5fold_oof_per_cancer_survival_forest.png')))"""),
    markdown("""### Direct representation comparisons"""),
    code("""display(Image(filename=str(FIGURES / 'bridge_pca_raw_5fold_oof_survival_forest.png')))"""),
    code("""display(Image(filename=str(FIGURES / 'bridge_pca_raw_5fold_oof_survival_three_panel.png')))"""),
    code("""display(Image(filename=str(FIGURES / 'bridge_vs_conventional_5fold_oof_delta_cindex.png')))"""),
    code("""oof_comparison = pd.read_csv('results/survival_5fold_oof_representation_comparison.csv')
display(oof_comparison.style.format({column: '{:.3f}' for column in oof_comparison.columns
                                     if 'index' in column.lower() or 'CI' in column or '−' in column})
        .hide(axis='index'))"""),
    markdown("""### Comparison with repeated 80/20 tests"""),
    code("""repeated_comparison = pd.read_csv('results/survival_5fold_oof_comparison_to_repeated_splits.csv')
bridge_repeat_comparison = repeated_comparison.query("representation == 'Bridge 45.6M'")
display(bridge_repeat_comparison[['cancer', 'n_patients_repeated', 'n_patients_oof',
    'n_events_repeated', 'n_events_oof', 'c_index_repeated', 'c_index_oof',
    'ci_width_repeated', 'ci_width_oof', 'ci_width_change']].sort_values('cancer')
    .style.format({column: '{:.3f}' for column in ['c_index_repeated', 'c_index_oof',
        'ci_width_repeated', 'ci_width_oof', 'ci_width_change']}).hide(axis='index'))"""),
    code("""oof_stats = pd.read_csv('results/survival_5fold_oof_statistical_summary.csv')
display(oof_stats.style.format({'value': '{:.4g}'}).hide(axis='index'))"""),
    markdown("""Complete OOF coverage raises Bridge's median within-cancer C-index from 0.598 to 0.624 and narrows the median CI width from 0.179 to 0.148; 28 of 32 Bridge intervals narrow. Bridge is above chance by point estimate in 31 cancers and has a CI wholly above 0.5 in 17. KIRP, ACC, LGG, UCEC, CESC, LIHC, LUAD, and KIRC remain supported; MESO, PAAD, BLCA, BRCA, COAD, SARC, HNSC, and STAD also exclude chance.

Bridge exceeds PCA and raw expression in 15 cancers each. Median differences are small (−0.002 versus PCA and −0.003 versus raw) and neither paired sign-flip nor Wilcoxon tests indicates a systematic difference (all p > 0.73); nonsignificance is not evidence of equivalence. KICH, PCPG, PRAD, SKCM, and THYM narrow relative to the repeated-split analysis but remain uncertain. TGCT remains especially uninterpretable—only four events spread across five folds—and its interval widens. Complete coverage improves sampling uncertainty but cannot create event information absent from TCGA."""),
]
for cell in oof_cells:
    cell.metadata["tags"] = [SURVIVAL_OOF_TAG]
oof_at = next(i for i, cell in enumerate(notebook.cells)
              if cell.cell_type == "markdown" and cell.source.startswith("### A3"))
notebook.cells[oof_at:oof_at] = oof_cells

for cell in cox_cells:
    cell.metadata["tags"] = [COX_TAG]
cox_at = next(i for i, cell in enumerate(notebook.cells)
              if cell.cell_type == "markdown" and cell.source.startswith("### A3"))
notebook.cells[cox_at:cox_at] = cox_cells
nbformat.write(notebook, NOTEBOOK)
