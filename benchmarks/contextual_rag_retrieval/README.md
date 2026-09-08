# Contextual RAG Retrieval

This benchmark asks whether frozen BridgeRNA program and contextual-gene
information improves study-independent sample or perturbation retrieval beyond
raw expression, PCA, and globally pooled embeddings.

The benchmark reuses established cohorts and cached representations only. It
does not retrain BridgeRNA or regenerate contextual token tensors.

## Run

```bash
set -o pipefail
.venv/bin/python benchmarks/contextual_rag_retrieval/pipeline/run_benchmark.py \
  2>&1 | tee benchmarks/contextual_rag_retrieval/results/run.log
.venv/bin/python benchmarks/contextual_rag_retrieval/pipeline/build_notebook.py
```

## Important design constraints

- Sample retrieval excludes the query GSE from candidates.
- Contextual response fingerprints are used for contrasts with a valid
  treatment/control reference; they are not manufactured for single samples.
- The combined response corpus has fewer than 25 independent contrasts, so
  K=25/50/100 all include the full eligible corpus and cannot test distinct
  candidate-stage depths. This is reported rather than hidden.
- ARCHS4/recount3 contextual tokens and Hallmark module fingerprints were not
  cached. Their established raw/global retrieval is reused, while contextual
  reranking is marked unavailable.
- Similarity means a retrieved transcriptomic analogue, not mechanistic or
  causal equivalence.

