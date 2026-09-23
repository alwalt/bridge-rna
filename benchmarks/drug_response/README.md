# Drug-response prediction benchmark (#5)

This benchmark asks whether baseline cell-line transcriptomic representations
predict experimentally measured GDSC IC50. It does **not** predict post-treatment
expression.

The primary local protocol uses 355 GDSC-version-specific screens and five
deterministic held-out-cell-line folds. PCA is fit only on training cell lines.
Each representation uses the same multi-output ridge framework (independent
linear head per screen), train-only mean completion for missing targets, and a
dimension-scaled regularization rule. Metrics are computed per screen over pooled
out-of-fold predictions and averaged without weighting.

## Run

```bash
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/audit_data.py
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/prepare_inputs.py
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/make_splits.py
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/extract_embeddings.py --model bridge_45.6m --device cuda:0
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/extract_embeddings.py --model bulkformer_50m --device cuda:0
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/extract_embeddings.py --model bulkformer_147m --device cuda:0
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/evaluate.py
/home/walt/bridge-rna/.venv/bin/python benchmarks/drug_response/pipeline/make_figures.py
```

Execute the report:

```bash
cd benchmarks/drug_response
/home/walt/bridge-rna/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  drug_response.ipynb --ExecutePreprocessor.timeout=1200
```

## Interpretation

The final BulkFormer publication reports mean PCC 0.373, but its exact folds,
head, target grouping, and mean-PCC definition are not public. It is a literature
reference, not a directly comparable local score. The released expression matrix
also lacks an exact invertible TPM preprocessing description; see
`dataset_audit.md` before interpreting FM differences.
