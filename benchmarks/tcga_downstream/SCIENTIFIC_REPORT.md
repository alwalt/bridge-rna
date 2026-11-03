# TCGA survival evaluation of frozen Bridge representations

## 1. Scientific objective

This analysis tests whether a frozen Bridge representation retains information relevant to censoring-aware patient prognosis and within-cancer survival-risk ranking. Bridge is compared with conventional PCA-128 and full raw expression on identical patients and folds. The objective is downstream representation evaluation rather than universal superiority or clinical validation.

## 2. Representations

### Bridge

Bridge has approximately 45.6 million parameters and uses an ordered 15,165-gene vocabulary. Natural `log1p(TPM)` inputs are encoded by the frozen checkpoint recorded in benchmark provenance. The primary L12 sample representation is 512-dimensional and mean pooled. No Bridge parameter is updated during Cox-head training.

### Conventional controls

PCA-128 is fitted separately within every outer training fold after training-fold-only scaling, then applied to the held-out fold. Full raw expression contains 25,150 TCGA features represented as `log1p(CPM)` and standardized using training patients only.

## 3. TCGA survival cohort

The cohort contains 9,668 eligible patients across 32 cancers and 2,774 overall-survival events. Outcomes come from the PanCanAtlas `OS` event indicator and positive `OS.time` in days. One deterministic eligible specimen is retained per patient, so technical aliquots are not counted as independent observations.

The primary-tumor manifest contains 9,816 unique patients. The survival cohort excludes 148 patients: 21 without a valid binary event indicator, 34 with a valid event but missing time, and 93 with nonpositive time.

## 4. Complete five-fold OOF design

Patients are assigned to five mutually exclusive test folds balanced deterministically within cancer × event strata, with a sparse-stratum fallback. Every patient appears in the training set for four folds, in the test set exactly once, and receives one held-out prediction per representation and head.

Bridge remains frozen. Only downstream Cox heads are trained. PCA and all learned preprocessing are fitted using training patients only. Patient IDs, folds, survival outcomes, and eligibility match across Bridge, PCA-128, and full expression.

Risk scores from independently fitted Cox heads are not assumed to share a common scale. Concordant and comparable-pair counts are calculated inside each held-out fold and summed. The same fold-aware definition is used within cancer.

## 5. Cox heads and metrics

Every representation is evaluated with:

- a bias-free linear Cox head;
- a nonlinear Cox MLP with hidden layers `[512,256]`.

Both heads optimize Cox partial likelihood and retain censored observations. Harrell's C-index measures ranking discrimination among comparable patient pairs: 0.5 represents chance ordering, and larger values indicate that patients with earlier observed events generally receive higher predicted risk.

Cancer-specific pointwise 95% confidence intervals use 1,000 patient-level bootstrap draws within cancer while retaining fold membership. Event/N annotations and stability flags distinguish low estimated performance from high uncertainty caused by sparse events.

## 6. Complete-cohort results

All six representation × head combinations contain exactly 9,668 held-out predictions and 3,068,664 fold-valid comparable pairs.

| Representation | Linear Cox C-index | Cox MLP C-index |
| --- | ---: | ---: |
| Bridge 45.6M | 0.7589 | 0.7580 |
| PCA-128 | 0.7615 | 0.7638 |
| Full raw expression | 0.7533 | 0.7590 |

Bridge is nearly head-insensitive. With the linear head fixed, Bridge is 0.0026 below PCA-128 and 0.0056 above full expression overall. With the MLP fixed, Bridge is 0.0058 below PCA-128 and 0.0011 below full expression.

No representation or head contrast remains significant after Benjamini–Hochberg correction across seven paired cancer-level tests. These results support similar broad prognostic information across the three representations; nonsignificance is not proof of equivalence.

## 7. Within-cancer survival discrimination

The within-cancer analysis asks whether a representation can rank survival risk among patients who have the same cancer type.

| Summary | Bridge | PCA-128 | Full expression |
| --- | ---: | ---: | ---: |
| Median cancer C-index | 0.6239 | 0.6360 | 0.6008 |
| Patient-weighted mean | 0.6247 | 0.645 | 0.632 |

Bridge exceeds PCA-128 in 15/32 cancers and full expression in 15/32. Median paired differences are −0.0024 versus PCA and −0.0030 versus full expression. Paired sign-flip/Wilcoxon p-values are 0.761/0.733 versus PCA and 0.995/0.963 versus full expression.

Bridge pointwise bootstrap intervals lie wholly above 0.5 in 17 cancers: ACC, BLCA, BRCA, CESC, COAD, HNSC, KIRC, KIRP, LGG, LIHC, LUAD, MESO, PAAD, SARC, STAD, UCEC, and UVM.

Strong estimates with useful event support include:

| Cancer | N | Events | Bridge C-index | 95% CI |
| --- | ---: | ---: | ---: | --- |
| KIRP | 287 | 44 | 0.858 | 0.785–0.914 |
| ACC | 79 | 28 | 0.821 | 0.718–0.905 |
| LGG | 509 | 125 | 0.775 | 0.719–0.824 |
| KIRC | 529 | 174 | 0.687 | 0.643–0.729 |
| UCEC | 539 | 90 | 0.686 | 0.623–0.750 |
| CESC | 291 | 71 | 0.662 | 0.590–0.734 |
| LIHC | 365 | 130 | 0.659 | 0.602–0.716 |
| LUAD | 503 | 182 | 0.654 | 0.607–0.701 |

CHOL, DLBC, KICH, PCPG, PRAD, TGCT, THCA, and THYM have sparse event support and wide intervals. Complete patient coverage cannot compensate for intrinsically limited outcomes.

## 8. Forest-plot interpretation

Primary figures:

- [Bridge forest plot](results/figures/bridge_5fold_oof_per_cancer_survival_forest.png)
- [PCA-128 forest plot](results/figures/pca128_5fold_oof_per_cancer_survival_forest.png)
- [Full-expression forest plot](results/figures/raw_expression_5fold_oof_per_cancer_survival_forest.png)
- [Combined alphabetical forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest.png)
- [Combined Bridge-ranked forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest_bridge_sorted.png)
- [Three-panel comparison](results/figures/bridge_pca_raw_5fold_oof_survival_three_panel.png)
- [Paired delta-C-index plot](results/figures/bridge_vs_conventional_5fold_oof_delta_cindex.png)

Each point is a fold-aware held-out C-index, each whisker is a patient-bootstrap 95% interval, the dashed line marks C = 0.5, and events/N records outcome support. Alphabetical ordering is manuscript-facing; Bridge-ranked ordering is exploratory.

## 9. Exploratory Bridge depth

The frozen-depth ablation evaluates L1–L12 on the same complete OOF cohort and Cox head. L12 has the highest overall OOF C-index (0.7580) and median within-cancer C-index (0.6239). L11 has a trivially higher patient-weighted within-cancer mean (0.6250 versus 0.6247), but no earlier layer differs systematically from L12 after multiplicity correction.

Cancer-specific best layers are post hoc maxima across 12 layers and are subject to winner's curse. They are hypotheses for independent validation, not evidence for selecting cancer-specific deployment layers on this cohort.

## 10. Literature context

Gélard et al. evaluated frozen 256-dimensional BulkRNABert embeddings with a `[512,256]` Cox MLP and reported pooled C-indices of 0.753 ± 0.009 for a model pretrained on GTEx/ENCODE and 0.765 ± 0.011 for a model pretrained on TCGA ([PMLR paper](https://proceedings.mlr.press/v259/gelard25a.html)). A subsequent multimodal TCGA study applied penalized linear CoxPH to BulkRNABert embeddings and reported approximately 0.750 without PCA and 0.753 after PCA-128 ([arXiv:2505.07683](https://arxiv.org/abs/2505.07683)). Bridge's complete OOF C-index of 0.7580–0.7589 lies in the same numerical range.

These are not direct model comparisons. The studies differ in cohort construction, sample handling, preprocessing, pretraining exposure, split design, and concordance aggregation. The final BulkFormer prognosis protocol is not used as a Cox comparator because its censoring-aware time-to-event implementation and directly paired predictions are unavailable.

Liu et al. applied scFoundation to bulk TCGA data across 25 cancers ([PubMed 41909608](https://pubmed.ncbi.nlm.nih.gov/41909608/)). Under five-fold Ridge–Cox, Embed-1024 achieved mean cancer-level C-index 0.680 ± 0.086, comparable to gene expression at 0.686 ± 0.089 (Wilcoxon P = 0.211). Under Lasso–Cox, the embedding achieved 0.670 ± 0.081 versus 0.603 ± 0.120 for gene expression (P = 2.17 × 10⁻⁵). Their embedding-only deep survival model achieved a mean C-index of 0.624 across 25 cancer-specific test sets, while a later multimodal model combining embeddings, expression, and clinical variables reached 0.724.

The relevant shared conclusion is not that one embedding universally outperforms raw expression. Frozen transcriptomic embeddings can preserve Cox-accessible prognostic information, while the apparent advantage over expression depends on the downstream head, regularization, cohort, and inclusion of clinical variables. This supports the matched-head Bridge analysis and motivates future clinical-plus-representation evaluation.

## 11. Medical interpretation

The evaluated path is:

`tumor RNA-seq → frozen Bridge → molecular representation → Cox risk`.

The results demonstrate retrospective prognostic information, not clinical utility. Clinical utility would require clinical-covariate adjustment, comparisons with stage and grade, combined clinical-plus-expression models, calibration, external independent cohorts, and prospective or temporal validation.

## 12. Conclusions

1. Frozen Bridge representations retain substantial censoring-aware prognostic information.
2. Bridge supports stable above-chance within-cancer discrimination in a supported subset of cancers.
3. Bridge does not systematically outperform PCA-128 or full expression across cancers.
4. Similar linear and nonlinear Cox performance indicates that the prognostic signal is readily accessible in the frozen representation.
5. The benchmark establishes information preservation in a 512-dimensional representation, while leaving clinical incremental value and external generalization unresolved.

## 13. Limitations

- Retrospective TCGA cohort without external clinical validation.
- No clinical covariates or calibration analysis.
- Heterogeneous and sparse event counts in several cancers.
- Pointwise cancer-specific intervals without multiplicity-adjusted discovery claims.
- Survival-head and regularization dependence across input dimensions.
- Pooled C-index may reflect between-cancer prognosis differences.
- Practical benefits of representation compression were not directly tested.
- Cross-study foundation-model values are literature context only and cannot be ranked directly against the local OOF results.
