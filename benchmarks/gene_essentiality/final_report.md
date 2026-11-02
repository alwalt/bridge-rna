# Bridge contextual gene essentiality benchmark

## Executive conclusion

Frozen Bridge contextual gene representations contain a weak but measurable
cell-line-specific dependency signal under this controlled benchmark. Bridge
achieves mean per-gene PCC **0.01456**, compared with **0.01307** for the
target-conditioned PCA baseline, **0.01864** for target-gene expression alone,
and **0.01775** for BulkFormer contextual tokens under the identical local
protocol. Bridge therefore does not outperform the simplest expression
baseline. Its small mean advantage over PCA is not systematic: the median
gene-wise difference is negative and Bridge wins for 47.95% of genes.

Adding PCA context to Bridge increases mean PCC to **0.01537**, a small paired
improvement of 0.00081 over Bridge alone. The result supports limited
complementarity, not a strong claim that Bridge captures dependency information
beyond conventional expression structure.

## 1. What task was evaluated?

The final released BulkFormer data define a dense matrix-prediction problem:
for every cancer cell line and target gene, predict the released CRISPR gene
dependency score from that gene's contextual representation in the cell line.
This implementation uses ten-fold cell-line cross-validation. Every cell line
is test data exactly once; a separate validation partition selects the epoch
within each fold. The primary metric is calculated by correlating out-of-fold
predictions with observations across cell lines separately for every gene,
then taking the unweighted mean over genes.

The exact historical code path that produced the final paper's 0.186 could not
be recovered. The full evidence and unresolved ambiguities are documented in
[bulkformer_protocol_audit.md](bulkformer_protocol_audit.md). Consequently,
this benchmark is a controlled local replication, and the paper's published
model numbers are external context rather than direct numerical comparators.

## 2. Data, cohort, and preprocessing

The released expression and dependency pickles were reused in place. Both have
1,108 cell lines in identical order. Expression contains 18,757 genes;
dependency contains 17,910 genes and 143,614 missing values (0.724%). The
released expression values are consistent with `log2(TPM + 1)`. Conventional
baselines use those values unchanged. Frozen-model input is multiplied by
`ln(2)` to obtain the natural `log1p(TPM)` scale required by Bridge.

Bridge overlaps 14,456 dependency genes and BulkFormer overlaps 17,525. The
prespecified all-method universe additionally requires expression availability,
leaving **14,415 genes** and **all 1,108 cell lines**. No test-dependent gene
filtering was performed. Mapping details and all exclusions are retained in
[gene_overlap.csv](results/gene_overlap.csv), with full QA in
[dataset_audit.md](dataset_audit.md) and [preprocessing_report.md](preprocessing_report.md).

## 3. Representations and prediction protocol

Bridge uses the normalized contextual output of its final encoder layer
(layer 12), one 512-dimensional token per target gene and cell line. It is not
the static embedding table or pooled sample embedding. The released Bridge
checkpoint is frozen.

BulkFormer uses the final normalized contextual trunk token, 640 dimensions,
excluding the three appended hand-engineered expression-summary covariates and
optional ESM2 features. This isolates contextual representation quality. A
separate native-universe sensitivity analysis evaluates its 17,525 overlapping
genes.

Every matched representation uses the same MLP: 256 and 128 hidden units,
GELU, dropout 0.1, MSE loss, AdamW (`lr=1e-4`, weight decay `1e-4`), batch size
8,192, up to 20 epochs, and patience 3. Each fold uses two million deterministic
training pairs per epoch and 500,000 fixed validation pairs. This is one
prespecified seed, not a multi-seed uncertainty estimate.

The target-expression baseline supplies only the released expression of target
gene g in cell line i. The PCA baseline fits 512 components on training cell
lines only. A cell-line/gene pair is represented by the elementwise product of
the cell line's PC scores and the target gene's PC loadings, plus its expression
value (513 dimensions). Thus each feature describes the target gene's
contribution to a conventional sample-level expression axis. Bridge + PCA
concatenates the 512-dimensional Bridge token and these 513 conventional
features.

## 4. Primary results

| Representation | Genes | Mean PCC | Median PCC | Mean SCC | Median SCC |
|---|---:|---:|---:|---:|---:|
| Target expression | 14,415 | 0.01864 | 0.01098 | 0.01149 | 0.00917 |
| PCA context | 14,415 | 0.01307 | 0.01091 | 0.00805 | 0.00765 |
| Bridge contextual | 14,415 | 0.01456 | 0.00861 | 0.01254 | 0.00741 |
| BulkFormer contextual | 14,415 | 0.01775 | 0.01037 | 0.01562 | 0.01010 |
| Bridge + PCA | 14,415 | 0.01537 | 0.00939 | 0.01294 | 0.00835 |
| BulkFormer native-universe sensitivity | 17,525 | 0.01419 | 0.00802 | 0.01248 | 0.00752 |

Raw values are in [primary_metrics.csv](results/primary_metrics.csv), with
gene-level results in [results/analysis/](results/analysis/). All values are
out-of-fold; no in-sample predictions enter the metrics.

## 5. Paired diagnostics

| Paired contrast (PCC) | Mean delta | Median delta | 95% bootstrap CI of mean | Fraction left wins |
|---|---:|---:|---:|---:|
| Bridge - target expression | -0.00407 | -0.00497 | [-0.00554, -0.00260] | 47.26% |
| Bridge - PCA | +0.00150 | -0.00303 | [0.00017, 0.00284] | 47.95% |
| Bridge - BulkFormer | -0.00319 | -0.00227 | [-0.00431, -0.00212] | 48.48% |
| Bridge + PCA - Bridge | +0.00081 | +0.00125 | [0.00025, 0.00137] | 51.47% |

The Bridge-versus-PCA result is distributionally uneven: a minority of larger
positive changes raises the mean even though most genes and the median favor
PCA. The Bridge-plus-PCA gain is small but more consistent. Paired sign-flip
tests and exact estimates are in
[paired_comparisons.csv](results/analysis/paired_comparisons.csv); scatterplots
with identity lines are in [results/figures/](results/figures/).

Bridge PCC has Spearman correlation 0.104 with training-fold mean expression,
0.001 with training-fold expression variance, and 0.180 with dependency
variance. Bridge-minus-PCA PCC has a weak negative correlation (-0.027) with
full-cohort expression variance in a descriptive, non-model-fitting analysis.
These statistics do not reduce Bridge to expression variance alone, but neither
do they establish substantial information beyond target expression.

## 6. Why 0.931 became 0.186 in the BulkFormer publication

The public record establishes that the repository, model, data, and benchmark
were comprehensively updated between the preprint and final release, and that
the final table reports mean PCC 0.186 instead of 0.931. It does not expose a
complete executable provenance chain sufficient to attribute the difference to
one verified methodological change. Likely contributors include changed data,
split/evaluation aggregation, and downstream implementation, but selecting one
would be speculation. This benchmark therefore preserves that question as
unresolved rather than inventing an explanation.

## 7. Answers to the primary questions

1. **Final BulkFormer task:** predict cell-line-specific CRISPR dependency for
   each gene from its contextual gene representation; the exact published
   aggregation/training implementation remains incompletely specified.
2. **Reason for 0.931 to 0.186:** the final release comprehensively changed the
   benchmark stack, but available artifacts do not identify a single exact
   causal change with reasonable confidence.
3. **Bridge-usable cell lines:** 1,108 of 1,108.
4. **Usable dependency genes:** 14,415 in the frozen matched comparison;
   Bridge alone overlaps 14,456.
5. **Bridge representation:** frozen, normalized final-layer contextual gene
   token, 512 dimensions, conditioned on each cell line's full expression.
6. **Bridge mean PCC:** 0.01456.
7. **Expression-only mean PCC:** 0.01864.
8. **PCA mean PCC:** 0.01307.
9. **Does Bridge win?** It exceeds PCA in mean PCC only, but loses to target
   expression and BulkFormer; most individual genes favor PCA over Bridge.
10. **Systematic difference:** no systematic Bridge advantage over the simple
    baselines; Bridge wins 47.95% versus PCA and 47.26% versus expression.
11. **Does combination help?** Bridge + PCA reaches 0.01537, a small mean gain
    of 0.00081 over Bridge, but remains below target expression.
12. **Direct-comparison limitations:** the published 0.186 procedure is not
    exactly reproducible from public artifacts; this run uses a declared local
    MLP, one seed, finite pair subsampling, model-specific vocabulary mapping,
    and no optional full-expression regularized baseline. Published table
    values must not be read as leader-board peers of these local results.

## 8. Limitations

The local protocol was frozen before test evaluation but is not claimed to be
the missing final-paper protocol. A single seed cannot quantify optimizer and
sampling instability. The common-gene universe favors comparability at the cost
of excluding genes absent from either model. Token caches use float16 to make
the 1,108-by-vocabulary contextual arrays tractable. Finally, association does
not establish causal biology: weak PCC may reflect limits of the representations,
the downstream head, measurement noise, or the intrinsically difficult
cross-cell-line dependency task.

## 9. Correlation aggregation diagnostic

The immutable saved out-of-fold predictions were rescored under the other
common correlation aggregations without changing any predictions, masks,
scales, genes, or folds. Results are documented in
[correlation_aggregation_diagnostic.md](correlation_aggregation_diagnostic.md).

| Representation | Mean per-gene PCC | Mean per-cell-line PCC | Global PCC | Mean fold-global PCC |
|---|---:|---:|---:|---:|
| Target expression | 0.01864 | 0.40942 | 0.40692 | 0.40717 |
| PCA context | 0.01307 | 0.47650 | 0.47475 | 0.47481 |
| Bridge contextual | 0.01456 | 0.90802 | 0.90696 | 0.90705 |
| BulkFormer contextual | 0.01775 | 0.90124 | 0.90027 | 0.90033 |
| Bridge + PCA | 0.01537 | 0.90679 | 0.90573 | 0.90584 |
| BulkFormer native-universe sensitivity | 0.01419 | 0.89463 | 0.89359 | 0.89369 |

For matched BulkFormer, the median per-cell-line PCC is 0.90534 and the mean
fold-global PCC is 0.90033 (SD 0.00253). Spearman results show the same sharp
axis dependence: BulkFormer's mean per-gene SCC is 0.01562, whereas its mean
per-cell-line, global, and mean fold-global SCC values are 0.60013, 0.59534,
and 0.59574. Bridge's corresponding SCC values are 0.01254, 0.63043, 0.62470,
and 0.62535.

No tested legitimate aggregation approaches 0.186. The nearest BulkFormer
Pearson summary is mean per-gene PCC 0.01775, an absolute difference of 0.16825;
the across-gene/global alternatives are about 0.90. Aggregation alone therefore
does not explain the published result, which remains literature-only.
