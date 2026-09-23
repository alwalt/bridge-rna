# BulkFormer parity and TCGA benchmark summary

The TCGA downstream benchmark is closed for the current manuscript. Detailed methods, tables, interpretation, and limitations are in [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md).

## Completed analyses

- Five-cohort classification with frozen Bridge 45.6M, BulkFormer-50M, BulkFormer-147M, PCA controls, and full expression.
- Locally paired 33-class pan-cancer classification, separated from the final published BulkFormer literature value.
- BulkFormer-style alive/dead prognosis parity, kept separate from censoring-aware survival.
- Original repeated 80/20 Cox survival with pooled, weighted within-cancer, and macro within-cancer C-index.
- Complete true five-fold OOF Cox survival for all 9,668 eligible patients, including 1,000-draw patient-bootstrap intervals and matched Bridge/PCA/raw forest plots.

## Final complete OOF survival

| Representation | Fold-aware OOF C-index | Comparable pairs |
| --- | --- | --- |
| Bridge 45.6M | 0.7580 | 3068664.0000 |
| PCA-128 | 0.7621 | 3068664.0000 |
| Full raw expression | 0.7590 | 3068664.0000 |

Bridge's median within-cancer C-index is 0.624; PCA-128 is 0.636; full raw expression is 0.601. Bridge exceeds PCA and raw expression in 15 of 32 cancers each. Paired sign-flip and Wilcoxon analyses find no systematic difference. Seventeen Bridge cancer-specific intervals are wholly above chance.

## BulkFormer context

The final Cell Systems release reports TCGA classification weighted F1 = 0.907 as a literature value. It is not directly paired with the local evaluation because exact downstream code and folds are unavailable and the released object contains repeated-patient and normal-tissue rows. The local 33-class comparison uses one eligible specimen per patient and common folds while preserving native model vocabularies.

## Primary records

- [Scientific report](SCIENTIFIC_REPORT.md)
- [Executed notebook](tcga_downstream_benchmark.ipynb)
- [Gap audit](bulkformer_tcga_gap_audit.md)
- [OOF representation table](results/survival_5fold_oof_representation_comparison.csv)
- [OOF provenance](results/survival_5fold_oof_provenance.json)
- [Final QA](results/final_qa.json)
