# Final-release BulkFormer gene-essentiality dataset audit

## Identity and dimensions

The authoritative inputs are the two checksum-verified files from Zenodo
record 15744294, reused in place from the shared checkout:

- expression: 1,108 cell lines x 18,757 Ensembl genes;
- CRISPR dependency: 1,108 cell lines x 17,910 Ensembl genes.

Both matrices have the same ordered ACH identifiers. Cell-line identifiers,
expression-gene identifiers, and dependency-gene identifiers are each unique.
There are 17,465 genes shared between expression and dependency.

The dependency matrix has 143,614 missing entries (0.7237048%): 837 genes and
907 cell lines have at least one missing entry. Missing values are retained and
masked in the downstream loss and metrics.

## Expression preprocessing audit

The released expression values are complete and have:

- range 0 to 17.0954895;
- mean 2.6816654 and SD 2.5482676;
- zero fraction 0.1710805;
- quartiles 0.0718464, 2.4443650, and 4.6247735.

The values are nonnegative, not centered or standardized, and their range and
numeric distribution have the fingerprint of DepMap's `log2(TPM + 1)` product.
The exact DepMap release metadata is absent, so that identification remains a
numeric inference rather than recovered provenance.

The released matrix is preserved unchanged for auditing and conventional
baselines. Frozen-model inference uses an explicit conversion
`released_value * ln(2)`, which converts `log2(TPM + 1)` to natural
`log1p(TPM)`, the documented input for both checkpoints. No second TPM
normalization is applied.

## Dependency audit

Finite dependency scores have range -5.6489091 to 4.4364781, mean -0.1414814,
and SD 0.4106874. Their distribution is consistent with continuous DepMap
CRISPR gene-effect values: negative values indicate stronger dependency. They
are neither binarized nor globally standardized by this benchmark.

## Gene overlap

| Universe | Dependency genes |
|---|---:|
| Released target matrix | 17,910 |
| BulkFormer representable | 17,525 |
| Bridge representable | 14,456 |
| Bridge-BulkFormer FM matched | 14,454 |
| All-method common (also present in expression) | 14,415 |

Thirty-nine FM-matched genes are absent from the released expression matrix.
The all-method common universe is frozen as the primary comparison set so the
target-expression and PCA controls are defined for every target. The 14,454
FM-only matched universe remains available as a sensitivity set. Mapping is by
clean Ensembl identifier to a unique current HGNC symbol and then to Bridge's
fixed canonical token order; ambiguous mappings are not silently rescued.

Exact row, gene, overlap, missingness, and fold tables are saved under
`results/`. Raw numerical audit values are in `results/dataset_audit.json`.

## Fixed folds

Ten cell-line folds were generated once with NumPy `RandomState(42)`. Every
cell line is test exactly once. Each fold contains 110 or 111 test cell lines,
100 validation cell lines, and 897 or 898 training cell lines. The three sets
are pairwise disjoint in every fold. No result was examined before this fold
file and the gene universes were frozen.
