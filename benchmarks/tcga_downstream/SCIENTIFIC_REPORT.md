# TCGA Downstream Evaluation of Frozen Bridge Representations

## 1. Scientific objective

This benchmark tests whether a frozen pretrained Bridge representation retains information relevant to cancer phenotype, patient prognosis, and survival-risk ranking within individual cancers. Bridge is compared with conventional full expression, training-fold-only PCA, and locally evaluated BulkFormer representations where methodologically applicable. The objective is downstream representation evaluation, not a claim of universal superiority.

## 2. Models

### Bridge

Bridge has approximately 45.6 million parameters and uses an ordered 15,165-gene vocabulary. The evaluated checkpoint/config is `model/r7hnr92k/config.json` (run `r7hnr92k`), configured for natural `log1p(TPM)`. The encoder is frozen and produces a 512-dimensional sample representation. Only downstream task heads are trained.

### BulkFormer

The local comparison evaluates frozen BulkFormer-50M and BulkFormer-147M checkpoints. Their sample representations contain 259 and 643 features, respectively. BulkFormer uses its native ordered 20,010-gene vocabulary, natural `log1p(TPM)`, −10 for missing genes, and mean gene-token pooling in the final-release-aligned local reconstruction. Bridge and BulkFormer retain their native vocabularies; neither is forced into the other's input space. The full-expression baseline uses 25,150 TCGA features as `log1p(CPM)`.

## 3. TCGA cohort

Expression comes from the repository TCGA matrix (`data/tcga/tcga_matrix.h5`), and survival labels come from the TCGA PanCanAtlas curated overall-survival table. Primary tumors are selected, sorted deterministically, and deduplicated to one expression aliquot per patient. The five-cohort classification task contains 3,227 patients and uses BLCA, BRCA, GBM+LGG (`GBMLGG`), LUAD, and UCEC.

The pan-cancer classification cohort contains 9,942 unique patients and 33 labels: ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM. It uses primary tumors plus LAML peripheral-blood primary cancer. The final Cox cohort contains 9,668 unique primary-tumor patients across 32 cancers and 2,774 deaths. Overall survival uses PanCanAtlas `OS.time` in days and `OS` as the event indicator; eligibility requires `OS` in {0,1} and positive follow-up time. Of 9,816 primary-tumor patients, 148 are excluded: 21 lack a valid binary event, 34 have missing time with a valid event, and 93 have nonpositive time.

## 4. Five-cohort cancer classification

All representations use identical patient-level repeated 80/20 splits. The primary frozen-representation classifier is an MLP with hidden layers `[256,128]`; metrics are macro F1, weighted F1, and balanced accuracy. PCA-128 controls are fitted on training patients only. Full raw expression is evaluated separately with the same downstream architecture.

| Representation / head | Macro F1 | Weighted F1 | Balanced accuracy |
| --- | --- | --- | --- |
| BulkFormer-147M | 0.9837 | 0.9861 | 0.9833 |
| BulkFormer-50M | 0.9751 | 0.9796 | 0.9750 |
| Full raw expression | 0.9886 | 0.9904 | 0.9894 |
| Bridge 45.6M | 0.9796 | 0.9830 | 0.9794 |
| PCA-128 expression + logistic regression | 0.9937 | 0.9948 | 0.9947 |

This task is near-saturated. Full raw expression and the PCA control are strongest, while Bridge and both BulkFormer representations remain close in absolute performance. The result supports substantial cancer-phenotype information in all frozen representations without establishing an advantage over conventional expression.

## 5. 33-class pan-cancer classification

The final BulkFormer release reports weighted F1 = 0.907, but its released object has 10,429 rows for 9,679 patients, including repeated patients and normal aliquots, and its exact downstream code and folds are unavailable. That number is retained as a literature-only reference. The local paired reconstruction uses 10 fixed patient-level folds, one eligible specimen per patient, native model preprocessing, training-fold-only PCA-128, and random forests.

| Representation | Macro F1 | Weighted F1 | Balanced accuracy |
| --- | --- | --- | --- |
| Bridge 45.6M | 0.8575 | 0.8884 | 0.8588 |
| BulkFormer-50M | 0.8453 | 0.8799 | 0.8387 |
| BulkFormer-147M | 0.8608 | 0.8966 | 0.8551 |
| PCA-128 expression | 0.8891 | 0.9167 | 0.8903 |
| Full raw expression | 0.8968 | 0.9304 | 0.8952 |

Conventional expression is particularly strong for cancer-type classification. Bridge performs in the same broad range as the BulkFormer representations, below full expression and its PCA control. This is not a failure: it shows that the learned 512-dimensional Bridge representation retains much of the phenotype signal present in the transcriptome, while simple conventional representations remain highly effective for tissue-of-origin-like classification.

## 6. BulkFormer-style prognosis parity

The parity reconstruction predicts the binary alive/dead label with 10-fold patient-level random forests. This endpoint discards follow-up duration and does not model censoring; it is therefore distinct from time-to-event survival and is not the primary clinical survival analysis.

| Representation | AUROC | AUPRC |
| --- | --- | --- |
| Bridge 45.6M | 0.7492 | 0.5273 |
| BulkFormer-50M | 0.7337 | 0.5091 |
| BulkFormer-147M | 0.7474 | 0.5266 |
| PCA-128 expression | 0.7611 | 0.5477 |
| Full raw expression | 0.7611 | 0.5523 |

Historical preprint values of AUROC 0.747 and AUPRC 0.549 are literature-only and not paired to local folds. The final release does not provide a sufficiently complete prognosis implementation to treat those values as a direct local comparator.

## 7. Cox time-to-event survival

The primary survival path is `expression → frozen representation → [512,256] Cox risk head`. The head is optimized with Cox partial likelihood, retains censored patients, and is evaluated with C-index. A C-index of 0.5 corresponds to chance ordering; higher values mean that patients with earlier observed events tend to receive higher predicted risk, among comparable patient pairs.

| Representation / head | Pooled C-index | Weighted within-cancer C-index | Macro within-cancer C-index |
| --- | --- | --- | --- |
| BulkFormer-147M | 0.7620 | 0.6320 | 0.6242 |
| BulkFormer-50M | 0.7588 | 0.6168 | 0.6119 |
| Full raw expression | 0.7610 | 0.6145 | 0.6119 |
| Bridge 45.6M | 0.7597 | 0.5995 | 0.5963 |
| PCA-128 expression | 0.7647 | 0.6262 | 0.6211 |

The pooled pan-cancer C-index may partly exploit systematic prognosis differences between cancer types. Weighted and macro within-cancer summaries more directly ask whether risk is ranked correctly among patients with the same cancer.

## 8. Repeated 80/20 survival analysis

The original design uses five fixed repeated random 80/20 train/test splits. It is retained because repeated fitting measures split-to-split stability. Because the test sets overlap, 6,467 of 9,668 eligible patients (66.9%) appear in at least one held-out set; 3,201 do not. This is expected for independent repeated holdouts and does not make the design erroneous. It motivated a complementary true five-fold OOF analysis for complete manuscript-facing coverage.

## 9. Complete 5-fold OOF survival evaluation

The complete analysis assigns every eligible patient to exactly one of five mutually exclusive test folds using deterministic within-cancer × event balancing, with a sparse-stratum fallback. Fold sizes range from 1,932 to 1,936; cancer counts and cancer-specific event counts differ by at most one between folds. All 9,668 patients receive exactly one held-out risk prediction per representation. Bridge remains frozen; only downstream Cox heads are refitted. PCA/scaling is fitted independently within each training fold, and full-expression standardization uses training patients only.

Cox risk scales from independently trained heads are not assumed interchangeable. Concordant and comparable-pair counts are computed inside each held-out fold and then summed, so no pair crosses fitted heads.

| Representation | Fold-aware OOF C-index | Comparable pairs |
| --- | --- | --- |
| Bridge 45.6M | 0.7580 | 3068664.0000 |
| PCA-128 | 0.7621 | 3068664.0000 |
| Full raw expression | 0.7590 | 3068664.0000 |

## 10. Within-cancer survival discrimination

The question is: *among patients with the same cancer, can the transcriptomic representation correctly rank survival risk?* Confidence intervals use 1,000 patient-level bootstrap draws within cancer while retaining fixed fold membership and fold-aware concordance.

| Cancer | N | Events | Bridge C-index | 95% CI low | 95% CI high | Comparable pairs | Stability flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACC | 79.000 | 28.000 | 0.821 | 0.718 | 0.905 | 290.000 | stable |
| BLCA | 406.000 | 178.000 | 0.627 | 0.579 | 0.671 | 8505.000 | stable |
| BRCA | 1079.000 | 151.000 | 0.613 | 0.560 | 0.670 | 12859.000 | stable |
| CESC | 291.000 | 71.000 | 0.662 | 0.590 | 0.734 | 2180.000 | stable |
| CHOL | 36.000 | 18.000 | 0.507 | 0.302 | 0.723 | 69.000 | low_event_support;wide_ci |
| COAD | 439.000 | 98.000 | 0.594 | 0.525 | 0.663 | 5102.000 | stable |
| DLBC | 47.000 | 9.000 | 0.523 | 0.204 | 0.892 | 44.000 | very_low_event_support;wide_ci |
| ESCA | 184.000 | 77.000 | 0.536 | 0.449 | 0.623 | 1480.000 | stable |
| GBM | 154.000 | 122.000 | 0.521 | 0.450 | 0.592 | 1772.000 | stable |
| HNSC | 501.000 | 217.000 | 0.619 | 0.577 | 0.661 | 13057.000 | stable |
| KICH | 65.000 | 9.000 | 0.663 | 0.345 | 0.921 | 83.000 | very_low_event_support;wide_ci |
| KIRC | 529.000 | 174.000 | 0.687 | 0.643 | 0.729 | 11222.000 | stable |
| KIRP | 287.000 | 44.000 | 0.858 | 0.785 | 0.914 | 1285.000 | stable |
| LGG | 509.000 | 125.000 | 0.775 | 0.719 | 0.824 | 5477.000 | stable |
| LIHC | 365.000 | 130.000 | 0.659 | 0.602 | 0.716 | 5473.000 | stable |
| LUAD | 503.000 | 182.000 | 0.654 | 0.607 | 0.701 | 9778.000 | stable |
| LUSC | 494.000 | 212.000 | 0.542 | 0.499 | 0.588 | 11151.000 | stable |
| MESO | 85.000 | 73.000 | 0.680 | 0.592 | 0.767 | 632.000 | stable |
| OV | 420.000 | 262.000 | 0.539 | 0.497 | 0.579 | 10527.000 | stable |
| PAAD | 177.000 | 93.000 | 0.645 | 0.562 | 0.712 | 1834.000 | stable |
| PCPG | 179.000 | 6.000 | 0.636 | 0.405 | 0.816 | 121.000 | very_low_event_support;wide_ci |
| PRAD | 497.000 | 10.000 | 0.509 | 0.177 | 0.810 | 397.000 | low_event_support;wide_ci |
| READ | 159.000 | 25.000 | 0.622 | 0.456 | 0.774 | 378.000 | wide_ci |
| SARC | 259.000 | 98.000 | 0.589 | 0.513 | 0.659 | 2942.000 | stable |
| SKCM | 102.000 | 29.000 | 0.452 | 0.306 | 0.603 | 290.000 | wide_ci |
| STAD | 390.000 | 156.000 | 0.566 | 0.514 | 0.615 | 7328.000 | stable |
| TGCT | 134.000 | 4.000 | 0.672 | 0.204 | 0.895 | 67.000 | very_low_event_support;wide_ci |
| THCA | 504.000 | 16.000 | 0.555 | 0.389 | 0.716 | 766.000 | low_event_support;wide_ci |
| THYM | 119.000 | 9.000 | 0.626 | 0.456 | 0.820 | 107.000 | very_low_event_support;wide_ci |
| UCEC | 539.000 | 90.000 | 0.686 | 0.623 | 0.750 | 5679.000 | stable |
| UCS | 56.000 | 35.000 | 0.569 | 0.414 | 0.709 | 209.000 | wide_ci |
| UVM | 80.000 | 23.000 | 0.755 | 0.596 | 0.874 | 192.000 | wide_ci |

Bridge confidence intervals lie wholly above 0.5 for 17 cancers: ACC, BLCA, BRCA, CESC, COAD, HNSC, KIRC, KIRP, LGG, LIHC, LUAD, MESO, PAAD, SARC, STAD, UCEC, UVM. The previously highlighted KIRP, ACC, LGG, UCEC, CESC, LIHC, LUAD, and KIRC remain supported. Cancers with fewer than 20 events (CHOL, DLBC, KICH, PCPG, PRAD, TGCT, THCA, THYM) require particular caution; complete coverage cannot compensate for intrinsically sparse outcomes.

## 11. Bridge versus PCA and raw expression

Median within-cancer C-indices are 0.624 for Bridge, 0.636 for PCA-128, and 0.601 for full raw expression. Patient-count-weighted means are 0.625, 0.645, and 0.632, respectively. Bridge exceeds PCA in 15/32 cancers and raw expression in 15/32. Median paired differences are -0.002 versus PCA and -0.003 versus raw expression.

The paired sign-flip/Wilcoxon p-values are 0.761/0.733 against PCA and 0.995/0.963 against raw expression. There is no evidence of a systematic cancer-level difference; nonsignificance is not proof of equivalence. Bridge retains prognostic information comparable to conventional transcriptomic representations, but this benchmark does not demonstrate systematic survival-prediction superiority over PCA or full expression. Bridge compresses the 15,165-gene input into 512 learned dimensions, but storage, compute, transfer, and clinical advantages of that compression were not tested.

## 12. Forest-plot interpretation

Primary figures:

- [Bridge forest plot](results/figures/bridge_5fold_oof_per_cancer_survival_forest.png)
- [PCA-128 forest plot](results/figures/pca128_5fold_oof_per_cancer_survival_forest.png)
- [Full-expression forest plot](results/figures/raw_expression_5fold_oof_per_cancer_survival_forest.png)
- [Combined forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest.png)
- [Three-panel comparison](results/figures/bridge_pca_raw_5fold_oof_survival_three_panel.png)

Each point is the fold-aware held-out C-index, each whisker is a patient-bootstrap 95% interval, the dashed line marks C = 0.5, and `events / N` records outcome support. Sparse-event cancers have few comparable pairs and consequently wide or unstable intervals. Complete OOF coverage narrows Bridge intervals in 28/32 cancers; the median width falls from 0.179 to 0.148.

## 13. Medical interpretation

The evaluated path is `tumor RNA-seq → Bridge → molecular representation → Cox risk`. It demonstrates retrospective prognostic information, not clinical utility. Clinical utility would require clinical-covariate adjustment, comparisons with stage and grade, combined clinical-plus-expression models, calibration, external independent cohorts, and prospective or temporal validation. A logical future analysis is `Clinical` versus `Clinical + PCA` versus `Clinical + Raw` versus `Clinical + Bridge`; it is outside the present benchmark.

## 14. Major scientific conclusions

1. Bridge preserves cancer-phenotype information: five-cohort classification is near-saturated, and 33-class performance remains strong.
2. Bridge preserves prognostic information: its complete fold-aware OOF C-index is 0.758.
3. Within-cancer information is stable for a substantial subset of cancers, with 17 Bridge intervals wholly above chance, but not uniformly across TCGA.
4. Bridge does not systematically outperform PCA.
5. Bridge does not systematically outperform full expression.
6. Bridge lies in the same general performance range as locally evaluated BulkFormer representations; task-specific ordering varies.
7. The benchmark establishes that a frozen 512-dimensional Bridge representation retains substantial phenotype and prognosis signal under patient-held-out evaluation.
8. It leaves unresolved incremental value beyond clinical covariates, external generalization, calibration, clinical utility, and the practical value of representation compression.

## 15. Limitations

- TCGA is retrospective and lacks external clinical validation here.
- Event counts are heterogeneous and intrinsically sparse in several cancers.
- Pretraining exposure may differ across models; possible TCGA exposure is not uniform or always recoverable.
- Published BulkFormer protocols and local paired reconstructions differ, and exact final-release folds/code are unavailable.
- Clinical covariates are absent from the current representation comparison.
- Results depend on the declared downstream survival head and fixed hyperparameters.
- Pooled metrics can reflect cancer-type composition and between-cancer prognosis differences.
- Alive/dead parity results are not censoring-aware survival results.

## Reproducibility and final QA

The final lightweight QA passed: all required outputs are present; all 9,668 patients occur exactly once per representation; Bridge, PCA, and raw expression have identical patients, folds, and outcomes; fold exclusivity holds; learned preprocessing is training-fold-only; Bridge is frozen; all notebook cells execute; and figures regenerate from saved result tables. The benchmark is closed for the current manuscript unless a reproducibility defect is discovered.
