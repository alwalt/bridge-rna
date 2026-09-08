# RR1/RR3 robust-response comparison

This benchmark compares RR1, RR3-39, and RR3-40 under identical definitions of
expression, signed input-gene IG attribution, and contextual-neighborhood
robustness. It reuses exact matched animals and existing frozen outputs.

```bash
.venv/bin/python benchmarks/rr1_rr3_robust_response_comparison/pipeline/build_comparison.py
.venv/bin/python benchmarks/rr1_rr3_robust_response_comparison/pipeline/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute \
  benchmarks/rr1_rr3_robust_response_comparison/rr1_rr3_robust_response_comparison_benchmark.ipynb \
  --output rr1_rr3_robust_response_comparison_benchmark.ipynb \
  --output-dir benchmarks/rr1_rr3_robust_response_comparison \
  --ExecutePreprocessor.timeout=600
```

Technical remeasurement is not independent biological replication. Robust
means concordant across the two specified measurements, not protocol-independent
or causally validated.

## Main result

RR3-40 was the strongest technical replication (global BridgeRNA cosine 0.917,
expression cosine 0.822, Hallmark cosine 0.922), followed by RR3-39 (0.790,
0.652, and 0.775). RR1 reversed globally (-0.807) and at the Hallmark level
(-0.424), although a restricted multiscale core remained: 114 genes had at
least two evidence levels and 26 had all three.

RR1 contained 58 protocol-robust versus 169 direction-reversing pathway
profiles. RR3-39 contained 158 versus 6, and RR3-40 contained 211 versus 0.
RR1 measurement-sensitive expression effects were most strongly enriched for
mRNA processing/splicing. Five multiscale genes were shared across all three:
`GADD45G`, `IGFBP1`, `NRN1`, `SLC41A2`, and `TCIM`.

The executed notebook is the primary human-readable artifact. Machine-readable
tables are under `results/summary/`, `results/genes/`, `results/pathways/`, and
`results/manifest/`; publication figures are under `results/figures/`.

## Interpretation limits

- OSD-168 is a technical remeasurement, not independent biological replication.
- Expression-interaction enrichment uses model-vocabulary expression effects
  and the exact 15,165-gene background; it is distinct from raw-count edgeR.
- Existing signed pathway profiles are not conventional GSEA NES.
- The data support a mixed technical/biological interpretation for RR1, not a
  causal attribution to a single protocol variable.
