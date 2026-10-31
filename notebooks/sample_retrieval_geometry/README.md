# Frozen Bridge retrieval geometry and reference-only whitening

This is the portable publication package for the `sample_retrieval_geometry` benchmark. It reports frozen BridgeRNA cross-pipeline agreement, anisotropy, reference-only whitening, and GTEx tissue retrieval.

## Lightweight cached execution

From the repository root:

```bash
python -m pip install -r notebooks/sample_retrieval_geometry/requirements.txt
jupyter nbconvert \
  --to notebook \
  --execute notebooks/sample_retrieval_geometry/sample_retrieval_geometry.ipynb \
  --output sample_retrieval_geometry.ipynb \
  --output-dir notebooks/sample_retrieval_geometry \
  --ExecutePreprocessor.timeout=300
```

The notebook reads only the compact tables and figure data committed beside it. It does not need network access, embeddings, expression matrices, checkpoints, or model inference.

To rebuild the publication cache from a local frozen research benchmark:

```bash
python notebooks/sample_retrieval_geometry/build_publication_package.py
```

That bounded command reads saved benchmark evidence and the retained two-million-pair cosine cache; it does not rerun retrieval or inference.

## Full research workflow (not run by this package)

The source-of-truth research pipeline remains under `benchmarks/sample_retrieval_geometry/pipeline/`. It requires the shared Bridge assets documented in `provenance.json`, including the 510,709-sample ARCHS4 embedding corpus, recount3 query embeddings, GTEx embeddings, manifests, canonical 15,165-gene ordering, and frozen checkpoint.

The relevant full commands are:

```bash
python benchmarks/sample_retrieval_geometry/pipeline/run_benchmark.py
python benchmarks/sample_retrieval_geometry/pipeline/run_absolute_similarity.py
python benchmarks/sample_retrieval_geometry/pipeline/run_anisotropy_analysis.py
python benchmarks/sample_retrieval_geometry/pipeline/run_literature_postprocessing.py
python benchmarks/sample_retrieval_geometry/pipeline/run_gtex_biological_retrieval.py
```

These commands are expensive research entrypoints and should not be used merely to render this notebook. Their original logs are written under `benchmarks/sample_retrieval_geometry/results/<analysis>/run.log`; a typical monitor is:

```bash
tail -f benchmarks/sample_retrieval_geometry/results/gtex_biological_retrieval/run.log
```

Recorded source runs used a GPU for retrieval, up to about 20.1 GiB peak resident memory for the original Stage 0/1 run, and shared assets outside the benchmark directory. See `provenance.json` for hashes, cohort definitions, transform parameters, validation coverage, and limitations.

## Package contents

- `sample_retrieval_geometry.ipynb`: executed human-readable report.
- `data/`: compact validated result tables.
- `figure_data/`: compact data sufficient to redraw every figure.
- `figures/`: PNG and PDF publication figures.
- `provenance.json`: source hashes, cohorts, frozen parameters, software, and limitations.
- `validation.json`: verification status and which claims were recalculated from retained per-query evidence.

## Data-access note

Large Bridge, ARCHS4, recount3, and GTEx assets are not redistributed here. Access to the full research workflow requires the repository's shared asset store and the checkpoint recorded in provenance. The cached publication report is self-contained.
