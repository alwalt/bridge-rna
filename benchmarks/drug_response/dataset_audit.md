# Released drug-response dataset audit

The authoritative local inputs are the two final BulkFormer Zenodo files. Exact
machine-readable statistics are in `results/dataset_audit.json`, and every row is
preserved in `results/observation_manifest.parquet`.

## Dimensions and identifiers

| Item | Count |
|---|---:|
| Baseline cell-line expression profiles | 700 |
| Expression genes | 19,098 |
| Response observations | 212,299 |
| GDSC drug IDs | 295 |
| Unique PubChem CIDs | 255 |
| GDSC-version-specific screens | 355 |
| TCGA classification labels | 32 |

Cell lines use DepMap `ACH-*` model IDs, with COSMIC ID and human-readable cell
line names in the response table. Drugs have GDSC drug ID, PubChem CID, SMILES,
and name. Dataset version is `GDSC1` or `GDSC2`.

There are no missing response values and no exact duplicate rows. There are
65,546 rows participating in repeated `(cell line, drug ID)` keys across versions,
but zero repeated `(cell line, dataset version, drug ID)` keys. Version is therefore
part of the exact observation identity. Screens contain 134–697 observations
(median 658).

## Endpoints

The table includes fitted IC50, AUC, maximum assayed concentration, fit RMSE, and
Z-score. IC50 ranges from -9.800 to 12.355 and is already on a transformed scale;
the release does not document an additional local transformation. The benchmark
uses the values unchanged.

## Expression and gene overlap

The released matrix has one row per benchmark cell line and no NaN/Inf values.
It contains 1,782,260 negative entries (minimum -1.841), which rules out untreated
natural `log1p(TPM)` as a literal description. It is consistent with a processed
or batch-corrected baseline expression matrix, but the exact source transform is
unresolved.

- Bridge vocabulary: 15,130 / 15,165 symbols present.
- BulkFormer vocabulary: 18,916 / 20,007 unique symbols present (20,010 rows).

Raw/PCA controls preserve released values. FM inputs use an explicitly labeled
approximation: clip negative values to zero, assume the remaining values are
`log2(TPM+1)`, and multiply by `ln(2)`. Missing Bridge genes are zero; missing
BulkFormer genes use its native -10 sentinel. This limitation prevents claiming
exact preprocessing parity with either pretrained model.

## Baseline-state interpretation

Each DepMap ID has one expression profile reused across all drug screens. Thus the
expression is baseline cell-line state rather than post-treatment expression.
Nothing in this benchmark predicts a perturbational transcriptome.
