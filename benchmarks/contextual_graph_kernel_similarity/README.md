# Contextual Graph-Kernel Similarity

This benchmark compares established higher-order graph similarities with simple
graph metrics and cached non-graph BridgeRNA/expression representations.

BridgeRNA remains frozen. Graph construction is inherited unchanged: layer-12
contextual cosine kNN, primary `k=10`, union symmetrization, with `k=5/20` and
mutual-kNN sensitivity retained in source caches.

## Practical kernel definitions

- WL subtree: two iterations, with separate gene-identity-aware and structural
  initial labels, represented by a fixed deterministic hashing feature map.
- Shortest path: deterministic landmark-sampled shortest-path histogram.
- Graphlet: deterministic sampled 3- and 4-node induced motif histogram.
- Spectral: 16 leading eigenvalues of the normalized weighted adjacency.
- Propagation kernel: unavailable because no reliable local implementation is
  installed; no custom substitute is invented.

Higher-order tissue kernels use a prespecified study-diverse subset (up to ten
samples per tissue from distinct GSEs). This is required for tractability and is
not selected using RR1/RR3.

## Run

```bash
set -o pipefail
.venv/bin/python benchmarks/contextual_graph_kernel_similarity/pipeline/run_benchmark.py \
  2>&1 | tee benchmarks/contextual_graph_kernel_similarity/results/run.log
.venv/bin/python benchmarks/contextual_graph_kernel_similarity/pipeline/build_notebook.py
```

