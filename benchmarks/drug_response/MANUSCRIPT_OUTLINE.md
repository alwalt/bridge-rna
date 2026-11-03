# Frozen representations for held-out-cell-line drug-response prediction

Source: [Publication review](./PUBLICATION_REVIEW.md)

## Supplementary Methods

- Test whether frozen Bridge-45.6M representations of baseline expression predict measured IC50 variation for known drug screens in cell lines excluded from downstream readout training; unseen-drug generalization was not evaluated.
- Analyze 700 unique DepMap cancer cell-line models spanning 27 broad lineages, with 212,299 GDSC1/GDSC2 response observations across 355 dataset-version-specific screens (295 drug IDs; 255 PubChem compounds).
- Use five deterministic, lineage-stratified folds of 140 held-out cell lines each, with no train/test cell-line overlap; fit preprocessing, PCA, and downstream readouts using training folds only.
- Compare frozen Bridge-45.6M (512 dimensions), locally evaluated frozen BulkFormer-50M (259 dimensions), BulkFormer-147M (643 dimensions), raw processed expression (19,098 genes), PCA-1–8 and PCA-16/32/64/128/256/512, and PCA-128 concatenated with Bridge.
- Apply a standardized ridge readout with one output per version-specific drug screen and a fixed dimension-scaled penalty (`alpha = 1000 × dimensions / 128`); use one-pass training-screen-mean completion for missing training responses.
- Assemble out-of-fold predictions and calculate PCC, SCC, and RMSE within each screen; report unweighted means across 355 screens, median PCC, and paired 10,000-resample screen-bootstrap intervals. These intervals describe variation across dependent screens rather than repeated splits or independent biological replicates.

## Supplementary Results

- Bridge achieved mean per-screen PCC 0.448, median PCC 0.461, mean SCC 0.424, and mean per-screen RMSE 1.235 across all 355 evaluable screens and 212,299 matched out-of-fold observations.
- Locally evaluated BulkFormer-50M and BulkFormer-147M achieved mean PCC 0.438 and 0.436. Bridge exceeded them by mean ΔPCC 0.0102 (95% screen-bootstrap interval 0.0083–0.0122; 71.8% of screens) and 0.0120 (0.0092–0.0147; 74.6% of screens), respectively.
- The complete exploratory PCA sensitivity curve had an interior optimum at PCA-4 (mean PCC 0.360); neighboring PCA-3 and PCA-5 achieved 0.359 and 0.357, whereas PCA-1 and PCA-512 achieved 0.085 and 0.031.
- Bridge exceeded PCA-4 by mean ΔPCC 0.0883 (median 0.0655; 95% screen-bootstrap interval 0.0810–0.0959) and was higher on 342 of 355 screens (96.3%). PCA-4 is the best observed setting from an exploratory post hoc sweep, not a prospectively selected hyperparameter.
- Raw expression achieved mean PCC 0.196 but the highest mean SCC, 0.440, indicating that ranking, linear association, and absolute-error conclusions differ. PCA-128 plus Bridge achieved mean PCC 0.284, below Bridge alone, and therefore did not establish useful complementarity.
- QC confirmed one finite out-of-fold prediction for every observed response and none outside observed-response positions for all 19 representations; exact per-screen metrics and plotting data accompany the report.

## Supplementary Discussion

- Frozen Bridge representations contain transferable baseline cellular-state information relevant to drug sensitivity in held-out cell lines under a lightweight common linear-readout protocol.
- The full PCA sweep reduces concern that the conventional control was weakened solely by high component counts; nevertheless, the PCA-4 comparison remains exploratory because dimensionality was selected after inspecting the benchmark.
- The Bridge advantage over locally evaluated BulkFormer is small and should be described as a controlled local comparison, not as direct improvement over the published BulkFormer mean PCC 0.373, whose folds, head, drug grouping, and aggregation cannot be reconstructed.
- Interpretation is limited by one split realization, dependence among screens, one-pass mean-completed targets, a fixed non-nested ridge penalty, approximate frozen-model preprocessing from an undocumented processed expression matrix, and unresolved Bridge pretraining exposure.
- The benchmark supports known-screen-to-unseen-cell-line representation utility, but not unseen-drug prediction, causal biomarkers, clinical efficacy, drug repurposing, or state-of-the-art pharmacogenomic prediction claims.

## Suggested main-text reference

In a supplementary held-out-cell-line analysis spanning 355 GDSC1/GDSC2 screens, frozen Bridge representations predicted IC50 variation with mean per-screen PCC 0.448 and exceeded the best observed exploratory PCA control (PCA-4, PCC 0.360) on 96.3% of screens. This result demonstrates downstream representation utility for known drug screens under one fixed readout protocol, while remaining limited by approximate preprocessing, a single split partition, dependent screens, and unresolved pretraining exposure.
