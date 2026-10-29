# Gene-essentiality benchmark summary

## Benchmark

- 1,108 cell lines
- 14,415 common matched genes
- Frozen Bridge and BulkFormer encoders
- 10-fold cell-line-disjoint cross-validation
- Identical downstream readout conditions
- Primary metric: mean per-gene PCC across held-out cell lines

## Final results

| Representation | Mean per-gene PCC | Mean SCC |
|---|---:|---:|
| Target expression | 0.01864 | 0.01149 |
| PCA context | 0.01307 | 0.00805 |
| Bridge contextual | 0.01456 | 0.01254 |
| BulkFormer contextual | 0.01775 | 0.01562 |
| Bridge + PCA | 0.01537 | 0.01294 |

## Interpretation

Bridge does not show a systematic advantage over PCA for context-specific
gene-essentiality prediction. It slightly exceeds PCA in mean PCC, but only
47.95% of genes favor Bridge and the median paired difference is negative.
Target-gene expression and locally evaluated BulkFormer outperform Bridge.
Bridge + PCA gives a small improvement, suggesting limited complementary
information.

Overall per-gene correlations are extremely weak. Under this evaluation,
baseline transcriptomic representations provide little ability to predict how
dependency on a particular gene varies across unseen cell lines.

## Row-versus-column correlation result

| Representation | Mean per-gene PCC | Mean per-cell-line PCC |
|---|---:|---:|
| Bridge contextual | 0.0146 | 0.908 |
| BulkFormer contextual | 0.0178 | 0.901 |

Per-gene PCC asks whether the model predicts variation in dependency for the
same gene across cell lines. Per-cell-line PCC instead asks whether it recovers
broad differences in essentiality between genes within one cell line. The
approximately 0.90 per-cell-line correlations therefore primarily reflect
stable between-gene essentiality structure and should not be interpreted as
strong prediction of context-specific genetic dependency.

This row-versus-column distinction is an ecological-fallacy concern for
transcriptomic benchmarks: strong aggregate or within-sample structure does not
imply accurate prediction of within-gene variation across cellular contexts.

## BulkFormer published result

BulkFormer reports mean PCC approximately 0.186 in the final publication, but
the exact final evaluation protocol could not be recovered from public
artifacts. The local aggregation diagnostic found BulkFormer PCC approximately
0.018 per gene, 0.90 per cell line, 0.90 globally, and 0.90 when global PCC was
computed within each fold and averaged. Correlation-axis choice alone therefore
does not explain 0.186.

The published value is literature-only and is not directly comparable to this
controlled local benchmark.

## Bottom line

> Under leakage-safe held-out-cell-line evaluation, Bridge does not reliably
> predict context-specific gene dependency beyond conventional expression
> representations. Its contextual embeddings perform similarly to PCA, while
> combining Bridge and PCA provides a small improvement. Both Bridge and
> BulkFormer strongly recover broad between-gene essentiality structure, but
> this does not translate into accurate prediction of how individual gene
> dependencies vary across unseen cellular contexts.
