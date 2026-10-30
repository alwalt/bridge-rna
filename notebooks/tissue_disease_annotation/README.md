# Tissue and disease annotation publication package

This package is the portable, cached report for the approved BridgeRNA tissue and disease annotation benchmark. The frozen research record and source-of-truth pipeline remain under [`benchmarks/tissue_disease_annotation/`](../../benchmarks/tissue_disease_annotation/). The publication scope is supplementary material with a concise main-text reference.

## Lightweight cached report

The executed notebook reads only compact files in this directory. It does not require source expression matrices, model weights, embeddings, inference, training, or network access.

Create a minimal environment from the repository root:

```bash
python -m venv .venv-publication
. .venv-publication/bin/activate
python -m pip install -r notebooks/tissue_disease_annotation/requirements.txt
```

Execute the cached report:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=300 \
  notebooks/tissue_disease_annotation/tissue_disease_annotation.ipynb
```

The notebook resolves files relative to its package or the repository root. The packaged figures and numerical tables can therefore be inspected without the original research assets.

## Publication artifact regeneration and validation

The publication builder recomputes metrics from the saved research predictions, checks them against the frozen fold tables, then writes only aggregate publication artifacts. It does not invoke model inference or classifier fitting.

```bash
python notebooks/tissue_disease_annotation/build_publication.py
```

Prediction-level files are intentionally not duplicated into this package because they include sample identifiers and are not needed to execute the public report. Their hashes and the bounded verification outcome are retained in [`provenance.json`](provenance.json) and [`validation.json`](validation.json).

## Full benchmark reproduction

The research pipeline is [`benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py`](../../benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py), and its frozen configuration is [`benchmarks/tissue_disease_annotation/config.json`](../../benchmarks/tissue_disease_annotation/config.json). From the repository root, the original commands are:

```bash
python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py prepare
python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py validate --device cuda:0
python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py embed --device cuda:0
python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py evaluate
python benchmarks/tissue_disease_annotation/pipeline/run_benchmark.py report
```

These commands are documented for provenance and were **not** run while building this package. They require:

- canonical GTEx v8 and TCGA expression assets at the paths documented in [`provenance.json`](provenance.json);
- the canonical 15,165-gene vocabulary and GENCODE v49 gene lengths;
- Bridge checkpoint run `r7hnr92k`;
- existing TCGA expression and Bridge embedding caches from the TCGA downstream benchmark;
- a CUDA-capable environment for practical GTEx inference.

The recorded original runtime was approximately 54 minutes: 21.7 minutes for GTEx embedding inference and 31.4 minutes for evaluation/reporting. Large source matrices, checkpoints, expression arrays, embeddings, and predictions are not included here. During a full run, monitor progress with:

```bash
tail -f benchmarks/tissue_disease_annotation/results/run.log
```

## Contents

- [`tissue_disease_annotation.ipynb`](tissue_disease_annotation.ipynb): executed public scientific report.
- [`data/`](data/): aggregate metrics, class counts, normalized confusion matrices, and figure data.
- [`figures/`](figures/): publication figures in PNG and PDF.
- [`provenance.json`](provenance.json): source hashes, checkpoint, parameters, dependencies, approval, and limitations.
- [`validation.json`](validation.json): prediction-level verification summary.
- [`validation.log`](validation.log): compact validation log.
- [`build_publication.py`](build_publication.py): bounded cached-evidence validator and figure builder.

## Scientific boundary

This benchmark shows high linear accessibility of tissue and cancer identity from frozen Bridge representations. It does not show superiority over PCA or raw expression, and possible GTEx/TCGA exposure through the public ARCHS4 pretraining corpus prevents interpreting it as guaranteed sample-unseen external generalization.
