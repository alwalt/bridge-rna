# Tissue and disease annotation

This benchmark asks how linearly accessible tissue and disease identities are
from frozen BridgeRNA sample representations. It compares frozen Bridge L12
mean-pooled embeddings, fold-local PCA-512, and canonical 15,165-gene
`log1p(TPM)` using the same multinomial logistic-regression readout and folds.

Primary tasks:

- GTEx v8 detailed tissue: 8,883 samples, 51 classes, donor-grouped 5-fold CV.
- TCGA pan-cancer: 9,942 unique patients, 33 classes, stratified 5-fold CV.

GTEx broad tissue annotation is a supplementary task using the exact same
samples, representations, donor groups, and folds. Cross-dataset normal-versus-
tumor classification is explicitly out of scope.

## Commands

Run stages from the repository root with the main checkout environment:

```bash
/home/walt/bridge-rna/.venv/bin/python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py prepare
/home/walt/bridge-rna/.venv/bin/python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py validate --device cuda:0
/home/walt/bridge-rna/.venv/bin/python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py embed --device cuda:0
/home/walt/bridge-rna/.venv/bin/python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py evaluate
/home/walt/bridge-rna/.venv/bin/python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py report
```

Continuous output is written to `results/run.log`.

```bash
tail -f benchmarks/tissue_disease_annotation/results/run.log
```

Large derived expression and embedding arrays live under `work/` and are not
versioned. Canonical datasets and existing TCGA caches are read in place from
`/home/walt/bridge-rna` and are never modified.
