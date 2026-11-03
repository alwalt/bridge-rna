# Complete OOF TCGA survival summary

The current manuscript scope is restricted to the complete five-fold OOF survival comparison of Bridge 45.6M, PCA-128, and full raw expression. Historical analyses are retained as reproducibility artifacts but are not part of the reported comparison.

## Cohort and design

- 9,668 patients across 32 cancers.
- 2,774 overall-survival events.
- Five mutually exclusive cancer × event-balanced folds.
- Exactly one held-out prediction per patient, representation, and head.
- Frozen Bridge encoder.
- Training-fold-only scaling and PCA.
- Fold-aware concordance aggregation.
- 1,000 patient-level bootstrap replicates for cancer-specific pointwise intervals.

## Results

| Representation | Linear Cox C-index | Cox MLP C-index |
| --- | ---: | ---: |
| Bridge 45.6M | 0.7589 | 0.7580 |
| PCA-128 | 0.7615 | 0.7638 |
| Full raw expression | 0.7533 | 0.7590 |

Bridge's median within-cancer C-index is 0.6239, compared with 0.6360 for PCA-128 and 0.6008 for full expression. Bridge exceeds each comparator in 15/32 cancers. Paired cancer-level tests do not show a systematic representation advantage after multiplicity correction.

## Conclusion

Frozen Bridge embeddings retain substantial censoring-aware prognostic information in 512 dimensions. Performance is in the same broad range as PCA-128 and full expression, without evidence of systematic superiority.

See the [scientific report](SCIENTIFIC_REPORT.md), [publication review](PUBLICATION_REVIEW.md), and [combined forest plot](results/figures/bridge_pca_raw_5fold_oof_survival_forest.png).
