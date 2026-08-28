# TCGA expression-imputation benchmark

Primary entry point: `tcga_imputation_benchmark.ipynb` (created after the
one-sample validation and full benchmark results are available).

This benchmark compares the frozen 45.6M ExpressionPerformer checkpoint with
the published BulkFormer-50M and BulkFormer-147M checkpoints on the same fixed
1,000 TCGA samples. It contains two analyses:

1. `shared_vocab` (primary): identical evaluable gene symbols and identical
   sample/seed masks across all models.
2. `native_vocab`: each model retains its complete native input vocabulary and
   masks only genes with measured TCGA ground truth.

All models receive their intended natural `log1p(TPM)` representation. Because
the checkpoints use different gene-length resources, MSE is evaluated in each
model's native log-TPM target space and must be interpreted with that provenance;
rank/correlation metrics are less sensitive to this scale difference.

## Gene annotation

The TCGA H5 is a recount2-era matrix whose symbols predate both current HGNC
and parts of the model vocabularies. Preparation therefore resolves the TCGA,
ExpressionPerformer, and BulkFormer labels through the pinned HGNC complete set
at `data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv`. Resolution uses, in
order, an approved symbol, a unique previous symbol, or a unique alias. Ambiguous
aliases are never guessed. If multiple model nodes resolve to one approved HGNC
symbol, that symbol is excluded from the primary shared-vocabulary benchmark.

The audit artifacts are `results/gene_annotation_summary.csv`,
`results/tcga_hgnc_crosswalk.csv`, and `results/shared_genes.csv`. The exact HGNC
file checksum and mapping-status counts are recorded in
`results/preparation_manifest.json`.

## Layout

```text
tcga_imputation/
├── README.md
├── config.json
├── tcga_imputation_benchmark.ipynb
├── pipeline/
├── results/
└── work/
```

`work/` contains frozen sample selection, vocabularies, and prepared matrices.
`results/` contains only benchmark tables and validation provenance.

## Run order

```bash
.venv/bin/pip install -r benchmarks/tcga_imputation/requirements.txt
.venv/bin/python benchmarks/tcga_imputation/pipeline/prepare_tcga.py
.venv/bin/python benchmarks/tcga_imputation/pipeline/run_imputation.py --validate-one
.venv/bin/python benchmarks/tcga_imputation/pipeline/run_imputation.py --run
```

The full run is intentionally unavailable until `--validate-one` writes a
successful `results/one_sample_validation.json` for all three checkpoints.

BulkFormer graph resources are the official `G_tcga.pt` and
`G_tcga_weight.pt` files from Zenodo record `15744294`; their expected MD5s are
recorded by the preparation manifest. The adapter preserves the official
notebook's transposed-sparse-adjacency convention when converting these files
to a native PyG edge index (`stored row 1 -> stored row 0`). This orientation
matters because the top-k correlation graph is directed. No fine-tuning occurs.
