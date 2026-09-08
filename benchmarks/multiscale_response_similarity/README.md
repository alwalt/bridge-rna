# Multiscale response similarity

This benchmark integrates existing BridgeRNA response comparisons across five
resolutions: conventional expression, global sample embedding, Hallmark/module
response, gene attribution, and contextual-gene graph response. It does not
rerun or fine-tune BridgeRNA.

The central output is a component-wise profile, not a composite biological
equivalence score. Metrics with different scales are never compared by their
raw magnitude across columns.

```bash
.venv/bin/python benchmarks/multiscale_response_similarity/pipeline/build_benchmark.py
.venv/bin/python benchmarks/multiscale_response_similarity/pipeline/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute \
  benchmarks/multiscale_response_similarity/multiscale_response_similarity_benchmark.ipynb \
  --output multiscale_response_similarity_benchmark.ipynb \
  --output-dir benchmarks/multiscale_response_similarity \
  --ExecutePreprocessor.timeout=600
```

## Scope and compatibility

RR1/RR3 provide four complete five-scale stress comparisons. The eight
exercise contrasts provide 28 pairwise profiles; their attribution component
uses the existing signed Top-250 IG vectors and is labeled accordingly.
Controlled T-cell PolyA/Ribo and ARCHS4/recount3 are retained as technical
controls, but they are sample-level perturbation/pairing experiments rather
than pairs of treatment-control response vectors and are therefore not forced
into the five-response table.

## Main result

RR3-39 and RR3-40 are positive across all five resolutions, with RR3-40 the
strongest technical remeasurement. RR1 is discordant across multiple scales,
so its failure is deeper than global cosine geometry alone. Conversely, the
unrelated RR1-original/RR3-39-original pair is a clear false friend: global
BridgeRNA cosine is 0.811, versus expression 0.121, Hallmark 0.510,
attribution 0.258, and contextual-graph agreement 0.012.

In the descriptive relationship-ranking analysis, global BridgeRNA alone and
global+graph both reached AUROC 0.986. Combining all five scales reduced AUROC
to 0.871. Multiscale profiling is therefore most useful for diagnosing the
resolution and credibility of a similarity—not as an automatically superior
composite score.

Primary artifacts are under `results/pairwise_profiles/`,
`results/summary/`, `results/discrimination/`, and `results/figures/`. The
executed notebook is the human-readable report.
