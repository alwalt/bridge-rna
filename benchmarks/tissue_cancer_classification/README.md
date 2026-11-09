# Tissue and cancer classification from frozen BridgeRNA representations

This package is the portable, cached report for the approved BridgeRNA tissue and disease annotation benchmark. The frozen research record and source-of-truth pipeline remain under [`research/tissue_disease_annotation/`](research_archive/tissue_disease_annotation/). The publication scope is supplementary material with a concise main-text reference.

## Lightweight cached report

The executed notebook reads only compact files in this directory. It does not require source expression matrices, model weights, embeddings, inference, training, or network access.

Create a minimal environment from the repository root:

```bash
python -m venv .venv-publication
. .venv-publication/bin/activate
python -m pip install -r benchmarks/tissue_cancer_classification/data/requirements.txt
```

Execute the cached report:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=300 \
  benchmarks/tissue_cancer_classification/tissue_disease_annotation.ipynb
```

The notebook resolves files relative to its package or the repository root. The packaged figures and numerical tables can therefore be inspected without the original research assets.

## Publication artifact regeneration and validation

The publication builder recomputes metrics from the saved research predictions, checks them against the frozen fold tables, then writes only aggregate publication artifacts. It does not invoke model inference or classifier fitting.

```bash
python research/tissue_disease_annotation/pipeline/build_publication.py
```

Prediction-level files are intentionally not duplicated into this package because they include sample identifiers and are not needed to execute the public report. Their hashes and the bounded verification outcome are retained in [`data/provenance.json`](data/provenance.json) and [`data/validation.json`](data/validation.json).

## Full benchmark reproduction

The research pipeline is [`research/tissue_disease_annotation/pipeline/run_benchmark.py`](research_archive/tissue_disease_annotation/pipeline/run_benchmark.py), and its frozen configuration is [`research/tissue_disease_annotation/config.json`](research_archive/tissue_disease_annotation/config.json). From the repository root, the original commands are:

```bash
python research/tissue_disease_annotation/pipeline/run_benchmark.py prepare
python research/tissue_disease_annotation/pipeline/run_benchmark.py validate --device cuda:0
python research/tissue_disease_annotation/pipeline/run_benchmark.py embed --device cuda:0
python research/tissue_disease_annotation/pipeline/run_benchmark.py evaluate
python research/tissue_disease_annotation/pipeline/run_benchmark.py report
```

These commands are documented for provenance and were **not** run while building this package. They require:

- canonical GTEx v8 and TCGA expression assets at the paths documented in [`data/provenance.json`](data/provenance.json);
- the canonical 15,165-gene vocabulary and GENCODE v49 gene lengths;
- Bridge checkpoint run `r7hnr92k`;
- existing TCGA expression and Bridge embedding caches from the TCGA downstream benchmark;
- a CUDA-capable environment for practical GTEx inference.

The recorded original runtime was approximately 54 minutes: 21.7 minutes for GTEx embedding inference and 31.4 minutes for evaluation/reporting. Large source matrices, checkpoints, expression arrays, embeddings, and predictions are not included here. During a full run, monitor progress with:

```bash
tail -f research/tissue_disease_annotation/results/run.log
```

## Contents

- [`tissue_cancer_classification.ipynb`](tissue_cancer_classification.ipynb): executed public scientific report.
- [`data/`](data/): aggregate metrics, class counts, normalized confusion matrices, and figure data.
- [`figures/`](figures/): publication figures in PNG and PDF.
- [`data/provenance.json`](data/provenance.json): source hashes, checkpoint, parameters, dependencies, approval, and limitations.
- [`data/validation.json`](data/validation.json): prediction-level verification summary.
- [`data/validation.log`](data/validation.log): compact validation log.
- [`research/tissue_disease_annotation/pipeline/build_publication.py`](research_archive/tissue_disease_annotation/pipeline/build_publication.py): bounded cached-evidence validator and figure builder.

## Scientific boundary

This benchmark shows high linear accessibility of tissue and cancer identity from frozen Bridge representations. It does not show superiority over PCA or raw expression, and possible GTEx/TCGA exposure through the public ARCHS4 pretraining corpus prevents interpreting it as guaranteed sample-unseen external generalization.

## Directory layout

- `data/`: cached evidence, provenance, validation records, and dependency files.
- `figures/`: report figures and associated plot tables.
- `research_archive/`: local, Git-ignored historical research copies; not required to execute this notebook.
- `tissue_cancer_classification.ipynb`: scientific report.

Run the notebook from this directory with its documented Python dependencies. Historical source paths in provenance describe the original runs and are retained unchanged. Research scripts now live under the local `research/` directory; archive copies preserve their original contents.

## Regenerate figures from saved evidence

Run `.venv/bin/python benchmarks/tissue_cancer_classification/pipeline/plot_classification.py` from the repository root. This uses cached tables only and exports PDF, editable-text SVG, 600-dpi PNG, raw plotted-value CSVs, and styling provenance. Error bars show sample SD across five fixed folds.
