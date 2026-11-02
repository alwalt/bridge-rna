# BridgeRNA gene-essentiality benchmark

This benchmark tests frozen contextual gene representations against CRISPR
dependency with a transparent, prespecified local protocol. The historical
audit could not recover the exact procedure behind BulkFormer's published
0.186, so published values are context only and are not treated as directly
comparable. See [bulkformer_protocol_audit.md](bulkformer_protocol_audit.md).

## Main result

All matched methods use the same 1,108 cell lines, 14,415 genes, ten cell-line
folds, and MLP protocol. Mean PCC is the unweighted mean of per-gene Pearson
correlations across out-of-fold cell-line predictions.

| Representation | Mean PCC | Mean SCC |
|---|---:|---:|
| Target expression | 0.01864 | 0.01149 |
| PCA context | 0.01307 | 0.00805 |
| Bridge contextual | 0.01456 | 0.01254 |
| BulkFormer contextual | 0.01775 | 0.01562 |
| Bridge + PCA | 0.01537 | 0.01294 |

Bridge is slightly above PCA by mean PCC, but below target expression and the
matched BulkFormer representation. The gene-wise Bridge-minus-PCA median is
negative, so the mean advantage is not systematic across genes. Combining
Bridge with PCA yields a small improvement over Bridge alone.

## Reproduce

The source of truth is in [pipeline/](pipeline/); large deterministic caches
and fold predictions are written to the external shared work root configured
in [config.json](config.json). Compact audits, metrics, and figures are in
[results/](results/).

```bash
/home/walt/bridge-rna/.venv/bin/python pipeline/prepare_dataset.py
/home/walt/bridge-rna/.venv/bin/python pipeline/cache_contextual_tokens.py bridge
/home/walt/bridge-rna/.venv/bin/python pipeline/cache_contextual_tokens.py bulkformer
/home/walt/bridge-rna/.venv/bin/python pipeline/cache_expression_features.py
/home/walt/bridge-rna/.venv/bin/python pipeline/train_contextual.py bridge matched
/home/walt/bridge-rna/.venv/bin/python pipeline/train_contextual.py bulkformer matched
/home/walt/bridge-rna/.venv/bin/python pipeline/train_contextual.py target_expression matched
/home/walt/bridge-rna/.venv/bin/python pipeline/train_pca.py pca
/home/walt/bridge-rna/.venv/bin/python pipeline/train_pca.py bridge_pca
/home/walt/bridge-rna/.venv/bin/python pipeline/evaluate_results.py
/home/walt/bridge-rna/.venv/bin/python pipeline/analyze_paired.py
/home/walt/bridge-rna/.venv/bin/python pipeline/build_figures.py
```

The narrative result is [final_report.md](final_report.md), and the executed
human-readable report is [gene_essentiality.ipynb](gene_essentiality.ipynb).
The post hoc, prediction-only aggregation check is in
[correlation_aggregation_diagnostic.md](correlation_aggregation_diagnostic.md).
