# Publication review: frozen drug-response prediction benchmark

## Review record

- **Reviewed benchmark:** `benchmarks/drug_response/`
- **Source Git revision:** `26af449797c8497664505d4949d7c948108f9d1e`
- **Review date:** 2026-10-31
- **Working-tree context:** the review was performed with uncommitted changes limited to the figure-generation script, two regenerated PDFs, and the new Bridge-only regression PNG/PDF. The numerical result tables and prediction arrays were not modified during review.
- **Evidence inspected:** the executed primary notebook; README and scientific report; frozen configuration; dataset, protocol, asset, and pretraining-exposure audits; cohort and observation manifests; split definitions; input and evaluation provenance; primary, per-screen, paired-comparison, fold-diagnostic, variance-control, and published-reference tables; saved out-of-fold predictions and embedding shapes; pipeline source; and final figures.
- **Review mode:** saved evidence plus bounded, read-only in-memory checks. No notebook execution, fitting, inference, download, split regeneration, or pipeline run was performed.

## Benchmark objective

The scientific question is whether a **frozen Bridge-45.6M representation of baseline cell-line expression contains information that predicts measured drug sensitivity in cell lines excluded from downstream readout training**.

The testable hypothesis is that, under the same held-out-cell-line split and linear readout, Bridge produces higher within-screen out-of-fold Pearson correlation than alternative frozen transcriptomic representations and conventional expression controls. The intended contribution is evidence of transferable pharmacogenomic information in Bridge, not a new drug-response system or a claim of therapeutic efficacy.

The benchmark evaluates **known drug screen → unseen cell line** generalization. It does not evaluate unseen compounds, post-treatment expression, causal mechanisms, biomarker validity, clinical response, or drug repurposing.

## Datasets and cohort

| Item | Saved evidence | Review finding |
|---|---:|---|
| Baseline expression profiles | 700 | 700 unique DepMap cell-line models; all appear in the response manifest |
| Broad lineages | 27 | No missing broad-lineage labels; rare labels are collapsed for stratification, yielding 23 split strata |
| Response observations | 212,299 | Unique by cell line and dataset-version-specific screen; no exact duplicates |
| Dataset-version-specific screens | 355 | 134–697 observations per screen; median 658 |
| GDSC drug IDs / PubChem compounds | 295 / 255 | Sixty drug IDs occur in both GDSC1 and GDSC2; version-specific screens are correctly kept distinct |
| Expression genes | 19,098 | Released processed matrix; no missing or infinite values, but 1,782,260 values are negative |
| Bridge vocabulary overlap | 15,130 / 15,165 | Correct canonical Bridge vocabulary size and order are used; 35 genes are absent and zero-filled |
| BulkFormer overlap | 18,916 / 20,007 unique symbols | The preprocessing provenance also reports 18,919 / 20,010 vocabulary rows; the three-count difference is explained by duplicate symbols rather than a cohort discrepancy |
| Cross-validation | 5 folds of 140 cell lines | Deterministic, lineage-stratified, complete-cell-line holdout; train/test cell-line overlap is zero in every fold |

The 700 cell lines are distinct experimental cancer cell-line models spanning 27 broad lineages, not 700 samples from one tumor and not 700 biological replicates of a common condition. Each cell line has one released baseline expression profile reused across its drug screens. The 212,299 response rows and 355 screens therefore must not be interpreted as independent biological replicates: screens share many of the same cell lines, compounds can recur across GDSC versions, and responses within a cell line are correlated.

Exact Bridge pretraining exposure remains unresolved. The release contains DepMap identifiers but no GEO/SRA sample or study accessions, so held-out downstream cell lines cannot be claimed to be unseen during ARCHS4 pretraining.

## Methods and evaluation

### Preprocessing and frozen representations

The raw and PCA controls use the 19,098 released expression features unchanged. The released matrix is not literal natural `log1p(TPM)` because it contains negative values and its exact transformation is undocumented. For Bridge and BulkFormer inference, negative values are clipped to zero and remaining values are treated as `log2(TPM+1)` before multiplication by `ln(2)`. This is explicitly recorded as an approximation and is not a recoverable conversion to canonical natural `log1p(TPM)` if the source was batch-corrected or otherwise processed.

Bridge uses the frozen standard 512-dimensional sample representation from the recorded checkpoint. The local BulkFormer controls use frozen mean-pooled final contextual token representations. Saved matrices have shapes 700×512 for Bridge, 700×259 for BulkFormer-50M, and 700×643 for BulkFormer-147M; all values are finite.

### Splits, PCA, and readout

Five deterministic folds hold out complete cell lines. The saved split manifest contains exactly 140 test cell lines per fold, no train/test overlap, and every stratum appears in all five folds. This protects the downstream readout from response leakage across folds. It does not provide study-level or pretraining-level isolation because source-study identifiers are unavailable.

PCA is handled correctly with respect to the test folds: raw features are standardized using training cells only, PCA is fitted separately inside each training fold, and held-out cells are transformed by that fitted model. A post-review sensitivity extension reports all independently fitted dimensions from 8 through 512; PCA-8 is the best observed setting and is explicitly exploratory rather than prospectively selected. The PCA **scores** are supplied to ridge regression; PCA loadings are not regression inputs.

Every representation uses a standardized ridge readout with a separate output for each dataset-version-specific screen. The configured penalty scales with feature count (`alpha = 1000 × dimensions / 128`). This makes the comparison reproducible, but it is a fixed heuristic rather than a nested-CV-selected penalty and may affect representations differently.

Missing training responses are initialized to the training-screen mean and included as pseudo-targets in the ridge objective. With `missing_target_em_iterations = 1`, the code fits once and only then updates missing values, without a second fit; the effective saved protocol is therefore **one-pass train-mean completion**, not an iterative EM fit. Depending on screen and fold, 0.18%–81.43% of the 560 training rows are imputed. This rule is shared across representations and preserves the controlled comparison, but sensitivity to observed-only per-screen fitting is untested and is especially important for sparse screens.

### Metrics and uncertainty

Predictions are assembled out of fold. PCC, SCC, and RMSE are computed separately within each of the 355 screens and summarized as unweighted means across screens; median PCC is also retained. Read-only verification found exactly one finite out-of-fold prediction for every observed response and no predictions outside observed response positions, for all seven representations.

The paired intervals use 10,000 bootstrap resamples of the 355 screens. They quantify variation across screens under an exchangeability assumption, not variation across random splits, cell-line cohorts, studies, or model seeds. Because screens share cell lines and some compounds recur across versions, these intervals may be too narrow if interpreted as independent-replicate confidence intervals. Only one split seed and one representation extraction per model are available; no across-seed mean or SD is reported.

## Validated results

### Primary saved results

All summary values below reproduce from [`results/per_drug_metrics.csv`](results/per_drug_metrics.csv) to numerical precision and match [`results/primary_results.csv`](results/primary_results.csv). Each result contains 355 evaluable screens and 212,299 matched out-of-fold observations.

| Representation | Mean PCC | Median PCC | Mean SCC | Mean per-screen RMSE | Replication / uncertainty | Status |
|---|---:|---:|---:|---:|---|---|
| Bridge-45.6M | **0.448** | **0.461** | 0.424 | **1.235** | 355 screens; one fixed 5-fold partition | Validated from saved metrics and predictions |
| BulkFormer-50M | 0.438 | 0.450 | 0.414 | 1.238 | 355 screens; one fixed 5-fold partition | Validated locally |
| BulkFormer-147M | 0.436 | 0.452 | 0.416 | 1.244 | 355 screens; one fixed 5-fold partition | Validated locally |
| Raw expression | 0.196 | 0.156 | **0.440** | 2.257 | 355 screens; one fixed 5-fold partition | Validated; highly readout-sensitive control |
| PCA-8 | 0.313 | 0.325 | 0.382 | 1.387 | 355 screens; one fixed 5-fold partition | Best observed PCA dimension in exploratory full sweep |
| PCA-16 | 0.214 | 0.184 | 0.406 | 1.761 | 355 screens; one fixed 5-fold partition | Exploratory sensitivity control |
| PCA-32 | 0.213 | 0.167 | 0.425 | 1.891 | 355 screens; one fixed 5-fold partition | Exploratory sensitivity control |
| PCA-64 | 0.161 | 0.121 | 0.437 | 2.224 | 355 screens; one fixed 5-fold partition | Exploratory sensitivity control |
| PCA-128 | 0.088 | 0.065 | 0.424 | 3.499 | 355 screens; one fixed 5-fold partition | Validated; highly readout-sensitive control |
| PCA-256 | 0.041 | 0.028 | 0.395 | 4.090 | 355 screens; one fixed 5-fold partition | Exploratory sensitivity control |
| PCA-512 | 0.031 | 0.021 | 0.348 | 4.002 | 355 screens; one fixed 5-fold partition | Validated; highly readout-sensitive control |
| PCA-128 + Bridge | 0.284 | 0.273 | 0.432 | 1.587 | 355 screens; one fixed 5-fold partition | Validated; no gain over Bridge alone |

### Paired effects

| Contrast | Mean ΔPCC | Median ΔPCC | Screen-bootstrap 95% interval | Fraction left higher | Verification status |
|---|---:|---:|---:|---:|---|
| Bridge − BulkFormer-50M | +0.0102 | +0.0088 | +0.0083 to +0.0122 | 71.8% | Validated from paired saved screen metrics |
| Bridge − BulkFormer-147M | +0.0120 | +0.0117 | +0.0092 to +0.0147 | 74.6% | Validated from paired saved screen metrics |
| Bridge − PCA-8 | +0.1357 | +0.1137 | +0.1250 to +0.1468 | 97.7% | Validated; PCA-8 selected as best observed dimension in exploratory sweep |
| Bridge − PCA-512 | +0.4176 | +0.4291 | +0.4042 to +0.4308 | 100.0% | Validated, but baseline/readout sensitivity limits interpretation |
| PCA-128 + Bridge − PCA-128 | +0.1958 | +0.1960 | +0.1840 to +0.2075 | 99.4% | Validated evidence that Bridge adds signal beyond PCA-128 |

The local Bridge advantage over BulkFormer is small in magnitude but consistent across screens. The paired screen bootstrap excludes zero, but it is not a substitute for repeated-split uncertainty and does not fully account for dependence among screens.

### Discrepant or unverified claims

| Claim | Assessment |
|---|---|
| The published BulkFormer mean PCC of 0.373 is directly exceeded by Bridge | **Not supported.** The published value is verified as literature context, but final folds, screen grouping, pooling, head, and aggregation are not public and are not protocol-equivalent to the local evaluation. |
| The input is exact natural `log1p(TPM)` | **Not supported.** Frozen-model inference uses a documented approximation from an undocumented processed matrix containing negative values. |
| Held-out cell lines were unseen during Bridge pretraining | **Unverified.** Required GEO/SRA sample and study identifiers are absent. |
| The readout uses iterative EM completion | **Not supported by the saved configuration and implementation.** One configured iteration produces a single fit on train-mean-completed targets; the subsequent update is not refitted. |
| The pooled Bridge regression line demonstrates calibration | **Not supported.** The figure separately z-scores measured and predicted values within each screen, so its slope (0.462) is an association summary closely related to pooled within-screen correlation, not calibration in IC50 units. |
| Bridge is a state-of-the-art pharmacogenomic predictor or generalizes to unseen drugs | **Not tested.** Every screen has training responses and its own output head. |

## Positive and negative findings

### Positive findings

1. Frozen Bridge representations contain clear information about differential drug sensitivity across held-out cell lines: mean per-screen PCC is 0.448 over all 355 screens.
2. Under an identical local split, endpoint, completion rule, and readout heuristic, Bridge modestly exceeds both locally evaluated BulkFormer checkpoints by mean ΔPCC of 0.010–0.012 and is higher on 72%–75% of screens.
3. Bridge adds substantial information beyond PCA-128 under the same readout, supporting representation transfer beyond a low-rank expression summary.
4. The benchmark has strong auditability: exact cohort and split manifests, full per-screen outputs, cached out-of-fold predictions, frozen configuration, checksummed source assets, and explicit preprocessing and exposure limitations are preserved.

### Negative and null findings

1. PCA-128 + Bridge performs materially worse than Bridge alone (0.284 versus 0.448 mean PCC). The experiment therefore does not show useful complementarity from appending PCA features.
2. Raw expression has the highest mean SCC (0.440) despite poor PCC and RMSE, showing that the ranking, linear association, and calibration/error conclusions differ materially by metric.
3. PCA is strongly component-count sensitive: the best observed PCA-8 reaches mean PCC 0.313, while PCA-512 reaches 0.031. Bridge remains higher than every tested PCA dimension, but the sweep is exploratory and is not nested-CV model selection.
4. No evidence addresses unseen-drug generalization, mechanistic biomarkers, causal response programs, clinical efficacy, or batch correction.

## Limitations and unresolved issues

1. **Approximate, potentially distorted FM inputs.** The source expression transformation is undocumented and includes negative values. Clipping can erase information and may affect Bridge and BulkFormer differently because their vocabularies and architectures differ.
2. **Unresolved pretraining exposure.** Downstream split isolation is verified, but exact sample/study novelty relative to ARCHS4 cannot be established.
3. **Single split realization.** Five folds come from one deterministic partition and one seed. The small Bridge–BulkFormer effect lacks repeated-split or nested resampling uncertainty.
4. **Dependent screen bootstrap.** The 355 screens are not independent experiments: they reuse cell lines, and some compounds occur in both GDSC versions. Reported intervals should be labeled descriptive screen-bootstrap intervals.
5. **Mean-completed target objective.** Sparse screens can contain up to 81.4% imputed training pseudo-targets. An observed-only per-screen ridge sensitivity analysis is needed.
6. **Fixed dimension-scaled regularization.** The penalty was not tuned within training folds. Different representation dimensions (259, 512, 643, 128, and 19,098) receive different absolute penalties, and poor raw/PCA results may reflect this heuristic.
7. **No unseen-drug task.** Separate screen heads only interpolate to held-out cell lines for known screens.
8. **Limited biological interpretation.** No lineage-stratified performance, mechanism-level validation, or external cohort is provided.
9. **Published-protocol noncomparability.** The final BulkFormer score of 0.373 cannot be reconstructed and must remain contextual rather than comparative.
10. **Pooled regression figure is not a calibration plot.** Separate within-screen standardization removes original IC50 scale and forces unit variance in both axes; it should not be used to claim calibrated predictions.

## Publication recommendation

**Recommendation: supplementary material in its current form.**

The benchmark is scientifically useful as a transparent, controlled demonstration that frozen Bridge features transfer to a pharmacogenomic phenotype. It is stronger than an exploratory clustering result because it uses complete-cell-line holdout, out-of-fold prediction, identical local evaluation across representations, and auditable per-screen outputs.

It should not currently support a headline main-manuscript superiority claim. The Bridge–BulkFormer advantage is modest, preprocessing is necessarily approximate, exact pretraining exposure is unresolved, uncertainty comes from one partition, and the readout's mean-completion and regularization choices have not been stress-tested. These issues weaken comparative attribution to the representation itself.

Prerequisites for promotion to a main-manuscript result are:

1. repeat the held-out-cell-line comparison across multiple deterministic split seeds and report mean, SD, and paired effects;
2. add observed-only per-screen ridge fitting and training-only penalty selection or a justified sensitivity grid;
3. show that the Bridge–BulkFormer ordering is robust to those readout choices;
4. obtain raw counts/TPM or authoritative preprocessing provenance, or demonstrate robustness across defensible input interpretations;
5. use dependence-aware uncertainty, such as resampling cell lines and grouping repeated compounds across GDSC versions, with the estimand stated explicitly;
6. retain the local comparison framing and avoid numerical comparison to the published 0.373 unless its protocol becomes reproducible.

## Proposed publication scope and approval status

**Status: pending explicit user approval. Saving this review does not approve publication packaging.**

### Proposed supplementary scope

- **Cohort/protocol:** 700 DepMap cell lines, 212,299 GDSC1/GDSC2 observations, 355 version-specific screens, five deterministic held-out-cell-line folds, frozen representations, and the exact saved one-pass mean-completed ridge rule.
- **Primary claim:** “Frozen Bridge representations predict IC50 variation in held-out cell lines and show a modest advantage over locally evaluated frozen BulkFormer representations under one transparent common protocol.”
- **Required qualifiers:** known screens only; approximate FM preprocessing; unresolved ARCHS4 exposure; single split partition; screen-bootstrap dependence; no direct comparison with published BulkFormer PCC 0.373.
- **Primary table:** Bridge, BulkFormer-50M, and BulkFormer-147M mean/median PCC, mean SCC, mean per-screen RMSE, paired mean ΔPCC, descriptive screen-bootstrap intervals, and fraction of screens favoring Bridge.
- **Control table:** raw expression, PCA-128, PCA-512, and PCA-128 + Bridge, explicitly labeled readout-sensitive controls.
- **Recommended figures:** mean per-screen PCC summary and paired per-screen PCC distributions. The pooled z-score scatter may be included only as an illustrative association plot, not as evidence of IC50 calibration.
- **Negative results to retain:** PCA concatenation does not improve over Bridge alone; raw expression has the highest mean SCC; no unseen-drug or causal conclusion is supported.

If this scope is approved, publication packaging should preserve the exact saved values and limitations above rather than strengthening the claims.
