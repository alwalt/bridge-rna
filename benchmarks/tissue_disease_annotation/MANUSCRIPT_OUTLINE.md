# Frozen BridgeRNA representations support tissue and cancer annotation

Source: [Publication review](PUBLICATION_REVIEW.md)

## Supplementary Methods

- **Objective:** Test how linearly accessible tissue and cancer identities are from frozen BridgeRNA sample representations, without assuming that Bridge should outperform transcriptomic baselines.
- **Cohorts:** Evaluate 8,883 GTEx profiles from 549 donors across 51 detailed tissue classes; evaluate 9,942 unique TCGA patients across 33 cancer types, using one specimen per patient. As a secondary analysis, evaluate the same GTEx profiles across 30 broad tissue classes.
- **Representations:** Compare frozen 512-dimensional Bridge embeddings with fold-local PCA-512 and raw 15,165-gene natural `log1p(TPM)` expression. Of 15,165 canonical genes, 15,105 were observed and 60 unavailable genes were zero-filled before `log1p`.
- **Readout and evaluation:** Use the same linear classifier, samples, and held-out folds for all representations. Apply donor-grouped five-fold evaluation for GTEx and patient-level stratified five-fold evaluation for TCGA; report macro F1 as the primary metric, with accuracy and weighted F1 as secondary metrics.
- **Leakage controls:** Keep each GTEx donor within one fold, remove repeated GTEx biospecimens, retain one TCGA row per patient, and fit PCA and all scalers only on each training fold. Keep Bridge frozen and exclude annotation labels from representation construction.
- **Quality control:** Confirm identical held-out samples across Bridge, PCA, and raw expression, numerical reproduction of saved metrics from predictions, and convergence of all 45 logistic-regression fits. Exact summaries and paired comparisons are available in [metrics_summary.csv](results/metrics_summary.csv) and [paired_macro_f1_differences.csv](results/paired_macro_f1_differences.csv).

## Supplementary Results

- **GTEx detailed tissues:** Bridge achieved macro F1 of **0.877 ± 0.023** across 51 classes, compared with **0.925 ± 0.028** for PCA and **0.924 ± 0.022** for raw expression.
- **TCGA pan-cancer:** Bridge achieved macro F1 of **0.910 ± 0.005** across 33 cancer types, compared with **0.938 ± 0.009** for PCA and **0.942 ± 0.010** for raw expression.
- **GTEx broad tissues:** Bridge achieved macro F1 of **0.926 ± 0.013** across 30 classes, compared with **0.962 ± 0.024** for PCA and **0.960 ± 0.024** for raw expression.
- **Paired comparisons:** Bridge was below both baselines in every paired fold. Mean Bridge-minus-PCA / Bridge-minus-raw macro-F1 differences were **−0.0476 / −0.0468** for detailed GTEx, **−0.0278 / −0.0325** for TCGA, and **−0.0359 / −0.0331** for broad GTEx; no post-hoc significance test was introduced for five folds.
- **Compact representation:** The 512-dimensional Bridge embedding represents an approximately **29.6-fold** dimensionality reduction relative to 15,165-gene expression and achieved descriptive Bridge/PCA macro-F1 ratios of **94.9%** for detailed GTEx and **97.0%** for TCGA. These ratios are descriptive and do not establish information retention or representation superiority.
- **Class-level behavior:** Better-powered errors were concentrated among adjacent annotations, including gastroesophageal junction versus esophagus muscularis, transverse versus sigmoid colon, sun-exposed versus non-sun-exposed skin, and related brain regions. In TCGA, READ→COAD occurred in **106/166 (63.9%)**, COAD→READ in **46/458 (10.0%)**, and ESCA→STAD in **41/184 (22.3%)**.
- **Rare classes:** Cervix–Ectocervix contained six samples and had Bridge F1=0, Cervix–Endocervix contained five samples, Fallopian Tube contained seven samples, and CHOL contained 36 patients. These estimates are unstable and should not be treated as evidence that Bridge lacks the corresponding biological information.

## Supplementary Discussion

- Frozen Bridge representations retain substantial tissue and cancer identity that is accessible through a fixed linear readout, supporting their use as compact biological sample representations.
- PCA and raw expression remain more linearly separable on these straightforward annotation tasks, indicating that Bridge preserves much, but not all, of the tissue- and disease-identifying information available in the original transcriptome.
- The dimensionality reduction and descriptive performance ratios are useful tradeoff measures, but dimensionality alone does not establish superiority and the paired deficits should be reported directly.
- Biologically adjacent confusions are descriptively plausible but should not be interpreted mechanistically; very rare GTEx tissues and CHOL provide especially noisy class-level estimates.
- ARCHS4 pretraining may contain public GTEx and/or TCGA profiles. Because pretraining used label-free masked-expression reconstruction rather than tissue or cancer labels, this benchmark demonstrates biological linear separability of frozen representations rather than guaranteed sample-unseen external generalization.
- Fold variability reflects one fixed five-fold partition rather than independent model-training replicates. Fold-level PCA cumulative explained variance was not retained, and GTEx-versus-TCGA normal/tumor classification remains out of scope because cohort effects would be confounded with disease state.

## Suggested main-text reference

Frozen 512-dimensional BridgeRNA representations supported strong linear annotation of 51 detailed GTEx tissues (macro F1 **0.877 ± 0.023**) and 33 TCGA cancer types (**0.910 ± 0.005**), although PCA and raw expression remained more linearly separable in every paired fold. The supplementary analysis therefore demonstrates substantial label-free biological separability under approximately 29.6-fold dimensional compression, while possible ARCHS4 exposure precludes interpreting the result as guaranteed sample-unseen external generalization.
