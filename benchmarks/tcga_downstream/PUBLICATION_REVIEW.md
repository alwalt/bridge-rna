# Publication review: complete five-fold OOF TCGA survival

**Review status:** Approved analysis scope; documentation narrowed to complete OOF results

**Benchmark:** `benchmarks/tcga_downstream/`

**Source Git revision:** `2015b886028479996cbfe7364fcd510739e13403`
**Review date:** 2026-11-02

## Benchmark objective

The manuscript-facing analysis asks whether a frozen 512-dimensional Bridge representation retains information that supports censoring-aware survival-risk ranking within TCGA cancers. Bridge is compared with PCA-128 and full raw expression on identical patients, folds, outcomes, and downstream head families. The objective is representation evaluation, not clinical validation or a prespecified superiority claim.

## Dataset and cohort

- 9,668 eligible patients across 32 TCGA cancers.
- 2,774 overall-survival events.
- PanCanAtlas `OS` event indicator and positive `OS.time` in days.
- One deterministic eligible specimen per patient; technical aliquots are not independent observations.
- The survival cohort excludes 148 primary-tumor-manifest patients: 21 without a valid event indicator, 34 with a valid event but missing time, and 93 with nonpositive time.
- TCGA is a single retrospective resource, not external biological replication.

## Methods and evaluation

- **Bridge 45.6M:** frozen checkpoint `model/r7hnr92k/config.json`; ordered 15,165-gene vocabulary; natural `log1p(TPM)`; 512-dimensional mean-pooled L12 representation.
- **PCA-128:** `StandardScaler + PCA-128` fitted separately inside each outer training fold.
- **Full expression:** 25,150 TCGA features represented as `log1p(CPM)` with training-fold-only standardization.
- Five mutually exclusive OOF folds are deterministically balanced within cancer × event strata, with a documented sparse-stratum fallback. Every eligible patient appears in exactly one test fold.
- Both a bias-free linear Cox head and Cox MLP `[512,256]` are applied to all three representations. Bridge remains frozen; only downstream Cox heads are trained.
- Concordant and comparable-pair counts are computed within held-out folds and summed, avoiding invalid cross-fold comparison of independently fitted Cox risk scales.
- Cancer-specific pointwise 95% intervals use 1,000 patient-level bootstrap draws while retaining fold membership. Sparse-event cancers remain visible with event/N annotations and stability flags.
- Paired cancer-level sign-flip and Wilcoxon tests assess systematic representation differences; Benjamini–Hochberg correction covers seven representation/head contrasts.
- The L1–L12 depth analysis is exploratory. L12 remains the prespecified primary representation.

## Validated results

### Complete OOF coverage

All six representation × head combinations contain exactly 9,668 unique held-out predictions and 3,068,664 fold-valid comparable pairs. Patient IDs, folds, outcomes, and eligibility match across representations. Saved QA confirms fold exclusivity, training-fold-only learned preprocessing, frozen Bridge status, complete notebook execution, and convergence below the accepted training ceiling.

| Representation | Linear Cox C-index | Cox MLP C-index |
| --- | ---: | ---: |
| Bridge 45.6M | 0.7589 | 0.7580 |
| PCA-128 | 0.7615 | 0.7638 |
| Full raw expression | 0.7533 | 0.7590 |

Sources: [results/survival_head_factorial_metrics.csv](results/survival_head_factorial_metrics.csv), [results/survival_head_factorial_predictions.csv](results/survival_head_factorial_predictions.csv), and [results/final_qa.json](results/final_qa.json).

Bridge is almost unchanged across head families. No representation or head contrast remains significant after Benjamini–Hochberg correction across the seven paired cancer-level tests. The matched-head analysis therefore supports the same neutral representation conclusion under linear and nonlinear Cox modeling.

### Within-cancer discrimination

- Median C-index: Bridge 0.6239, PCA-128 0.6360, full expression 0.6008.
- Patient-count-weighted mean: Bridge 0.6247, PCA-128 0.645, full expression 0.632.
- Bridge exceeds PCA-128 in 15/32 cancers and full expression in 15/32.
- Median paired difference: Bridge − PCA −0.0024; Bridge − full expression −0.0030.
- Sign-flip/Wilcoxon p-values: 0.761/0.733 versus PCA and 0.995/0.963 versus full expression. Nonsignificance is not evidence of equivalence.

Bridge pointwise bootstrap intervals lie wholly above 0.5 in 17 cancers: ACC, BLCA, BRCA, CESC, COAD, HNSC, KIRC, KIRP, LGG, LIHC, LUAD, MESO, PAAD, SARC, STAD, UCEC, and UVM. Supported estimates with useful event counts include KIRP (0.858; 44 events), ACC (0.821; 28), LGG (0.775; 125), KIRC (0.687; 174), UCEC (0.686; 90), CESC (0.662; 71), LIHC (0.659; 130), and LUAD (0.654; 182).

CHOL, DLBC, KICH, PCPG, PRAD, TGCT, THCA, and THYM remain difficult to interpret because of sparse events. Complete OOF coverage cannot solve intrinsically limited outcome support.

### Exploratory Bridge depth

L12 has the highest overall OOF C-index (0.7580) and median within-cancer C-index (0.6239). L11 has a negligibly higher patient-weighted within-cancer mean (0.6250 versus 0.6247), but no earlier layer differs systematically from L12 after multiplicity correction. Cancer-specific best layers are post hoc hypotheses and should not determine deployment without independent validation.

### Literature-only context

- BulkRNABert studies report pooled Cox C-indices of approximately 0.750–0.765 for frozen bulk-RNA embeddings, placing Bridge's 0.7580–0.7589 in the same numerical range ([Gélard et al.](https://proceedings.mlr.press/v259/gelard25a.html); [multimodal survival study](https://arxiv.org/abs/2505.07683)). Differences in cohorts, preprocessing, pretraining exposure, splits, and aggregation preclude direct ranking.
- A scFoundation study using bulk TCGA inputs reported Embed-1024 versus gene-expression mean cancer-level C-indices of 0.680 ± 0.086 versus 0.686 ± 0.089 under Ridge–Cox (P = 0.211), but 0.670 ± 0.081 versus 0.603 ± 0.120 under Lasso–Cox (P = 2.17 × 10⁻⁵) ([Liu et al.](https://pubmed.ncbi.nlm.nih.gov/41909608/)). This supports the importance of matched downstream heads and regularization.
- The final BulkFormer prognosis protocol is not used as a Cox comparator because a recoverable censoring-aware time-to-event implementation and paired predictions are unavailable.

## Positive and negative findings

### Positive

- A frozen 512-dimensional Bridge representation retains substantial censoring-aware prognostic information.
- Bridge supports stable above-chance within-cancer discrimination in a supported subset of cancers.
- Similar Bridge performance under linear and nonlinear Cox heads indicates that the signal is readily accessible in the frozen representation.
- Complete OOF coverage, matched folds, fold-aware concordance, saved predictions, event support, and patient-bootstrap uncertainty make the result technically auditable.
- Bridge's pooled discrimination lies in the range reported for other frozen transcriptomic embeddings, providing supportive context without constituting a direct comparison.

### Null or adverse

- Bridge does not systematically outperform PCA-128 or full expression across cancers.
- Several cancers remain uninformative because the complete cohort contains too few survival events.
- The analysis establishes retrospective prognostic information, not clinical utility or incremental value beyond clinical covariates.

## Limitations

1. Single retrospective TCGA resource without external or temporal validation.
2. No adjustment for stage, grade, age, treatment, molecular subtype, or other clinical covariates.
3. Heterogeneous and sometimes sparse event support.
4. Pointwise cancer-specific intervals are not multiplicity-adjusted discoveries.
5. Pooled concordance can exploit between-cancer prognosis differences; within-cancer results carry the primary biological interpretation.
6. Identical optimizer settings do not guarantee identical effective regularization across 128, 512, and 25,150 input dimensions.
7. Practical storage, compute, transfer, calibration, and clinical workflow benefits of compression were not tested.
8. Cross-study embedding values are literature-only and cannot be ranked directly against the locally paired OOF results.

## Publication recommendation

**Include the complete five-fold OOF survival evaluation in the main manuscript.**

Primary claim:

> A frozen 512-dimensional Bridge representation preserves substantial censoring-aware prognostic information in TCGA. Complete patient-level OOF evaluation shows stable within-cancer discrimination in a supported subset of cancers, while paired analyses do not demonstrate systematic superiority over PCA-128 or full expression.

Permitted literature context:

> Bridge's complete OOF C-index of 0.7580–0.7589 lies within the approximately 0.750–0.765 range reported for other frozen bulk-RNA embeddings, while an independent single-cell foundation-model study similarly found that embedding-versus-expression conclusions depend on Cox regularization. These cross-study comparisons are contextual rather than directly paired.

Primary table:

- Complete fold-aware OOF C-index for Bridge, PCA-128, and full expression under matched linear and nonlinear Cox heads.
- Median and patient-weighted within-cancer C-index and paired cancer-level comparisons.

Primary figure:

- [Combined alphabetical forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest.png).

Supplementary material:

- Individual Bridge, PCA-128, and full-expression forest plots.
- Three-panel and ranked forest plots.
- Delta-C-index and head-controlled sensitivity figures.
- Event-support, bootstrap-validity, and stability tables.
- Exploratory L1–L12 depth analysis, without cancer-specific layer-selection claims.

Exclude from claims:

- systematic predictive superiority over PCA-128 or full expression;
- clinical utility or external validation;
- favorable interpretation of sparse-event extremes;
- cancer-specific layer selection using the same outcomes.

## Evidence inspected

- [README.md](README.md)
- [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md)
- [tcga_downstream_benchmark.ipynb](tcga_downstream_benchmark.ipynb)
- [results/survival_5fold_oof_predictions.csv](results/survival_5fold_oof_predictions.csv)
- [results/survival_5fold_oof_metrics.csv](results/survival_5fold_oof_metrics.csv)
- [results/survival_5fold_oof_per_cancer.csv](results/survival_5fold_oof_per_cancer.csv)
- [results/survival_head_factorial_metrics.csv](results/survival_head_factorial_metrics.csv)
- [results/survival_head_factorial_statistical_summary.csv](results/survival_head_factorial_statistical_summary.csv)
- [results/bridge_layerwise_survival_summary.csv](results/bridge_layerwise_survival_summary.csv)
- [results/final_qa.json](results/final_qa.json)
- Final OOF forest and sensitivity figures under [results/figures/](results/figures/)
- Gélard et al., *BulkRNABert: Cancer prognosis from bulk RNA-seq based language models* ([PMLR 259](https://proceedings.mlr.press/v259/gelard25a.html))
- *Multimodal Survival Modeling in the Age of Foundation Models* ([arXiv:2505.07683](https://arxiv.org/abs/2505.07683))
- Liu et al., *Leveraging single-cell foundation models for accurate survival outcome prediction* ([PubMed 41909608](https://pubmed.ncbi.nlm.nih.gov/41909608/))
