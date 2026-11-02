# Preprocessing report

## Released data

The released expression matrix is used without alteration for conventional
baselines. Its values are consistent with DepMap `log2(TPM + 1)` (range
0--17.0955, with exact zeros retained). The dependency matrix is used on its
released scale, with missing targets masked from training and evaluation.

For frozen-model inference only, expression is converted to Bridge's required
natural-log convention by multiplying by `ln(2)`, exactly implementing
`ln(TPM + 1) = log2(TPM + 1) * ln(2)`. No clipping, imputation, quantile
normalization, or fold-dependent transformation is applied before either
encoder.

## Gene mapping and universe

Gene identifiers are mapped explicitly through the released matrices and the
canonical model vocabularies. Duplicate and unmapped identifiers remain in
[gene_overlap.csv](results/gene_overlap.csv); they are never silently dropped.
The frozen matched universe is the 14,415 dependency genes that are present in
the expression matrix and both model vocabularies. All 1,108 cell lines are
retained. The BulkFormer-native sensitivity universe contains 17,525 genes.

## Leakage controls

The ten folds split cell lines, not individual cell-line/gene pairs. Each cell
line is held out exactly once. Validation lines are distinct from training and
test lines. PCA is refit within every fold on training cell lines only; its
mean, components, singular values, and explained variance are saved with each
fold's external prediction artifacts. All matched representations use the same
folds, masks, sampled training-pair budget, architecture, and seed schedule.
