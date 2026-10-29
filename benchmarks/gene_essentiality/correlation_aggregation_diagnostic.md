# Correlation aggregation diagnostic

This diagnostic uses the immutable saved out-of-fold predictions. It does not
rerun or alter inference, training, PCA, folds, preprocessing, gene filtering,
scaling, or missing-value handling. Correlations use every finite prediction /
dependency pair in the already frozen representation-specific universe.

## Pearson results

| Representation | Mean per-gene PCC | Mean per-cell-line PCC | Global PCC | Mean fold-global PCC |
|---|---:|---:|---:|---:|
| Target expression | 0.01864 | 0.40942 | 0.40692 | 0.40717 |
| PCA context | 0.01307 | 0.47650 | 0.47475 | 0.47481 |
| Bridge contextual | 0.01456 | 0.90802 | 0.90696 | 0.90705 |
| BulkFormer contextual | 0.01775 | 0.90124 | 0.90027 | 0.90033 |
| Bridge + PCA | 0.01537 | 0.90679 | 0.90573 | 0.90584 |
| BulkFormer native universe | 0.01419 | 0.89463 | 0.89359 | 0.89369 |

For matched BulkFormer, median per-gene PCC is 0.01037 and median
per-cell-line PCC is 0.90534. Fold-global PCC is 0.90033 +/- 0.00253 SD across
the ten folds. No legitimate Pearson aggregation is close to 0.186: among the
tested summaries, the nearest is mean per-gene PCC 0.01775, still 0.16825 away.

## Spearman check

Matched BulkFormer has mean/median per-gene SCC 0.01562/0.01010,
mean/median per-cell-line SCC 0.60013/0.60441, global SCC 0.59534, and mean
fold-global SCC 0.59574 (SD 0.00488). Bridge has corresponding values
0.01254/0.00741, 0.63043/0.63482, 0.62470, and 0.62535 (SD 0.00770).

The high across-gene and global correlations are not evidence of strong
cell-line-specific prediction. They are dominated by stable between-gene
differences that are removed when each gene is correlated across held-out cell
lines. Thus changing only the aggregation axis moves BulkFormer from about
0.018 to about 0.90, not to 0.186.

## Conclusion

**Correlation aggregation alone does not explain BulkFormer's published mean
PCC of 0.186.** The published result remains literature-only and is not made a
direct comparator to this local benchmark.

Exact full-precision summaries are in
[`results/analysis/correlation_aggregation_summary.csv`](results/analysis/correlation_aggregation_summary.csv),
per-cell-line values in
[`results/analysis/per_cell_line_correlations.csv`](results/analysis/per_cell_line_correlations.csv),
and per-fold values in
[`results/analysis/fold_global_correlations.csv`](results/analysis/fold_global_correlations.csv).
