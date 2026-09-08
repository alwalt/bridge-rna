# RR1/RR3 contextual-graph case study

This focused case study explains how contextual gene graphs can resolve a false
similarity created by global mean pooling. It reuses the frozen layer-12 graph
caches and the fixed `k=10` union-kNN definition from
`frozen_sample_embedding_readout`. RR1/RR3 did not select graph parameters.

The case study does not rerun BridgeRNA and does not reinterpret RR1 as a
successful technical replication. RR1 remains a known cross-protocol failure.
The central test is whether true RR3 same-material remeasurements rank above the
unrelated RR1-original/RR3-39-original pair.

```bash
.venv/bin/python benchmarks/rr1_rr3_contextual_graph_case_study/pipeline/build_case_study.py
.venv/bin/jupyter nbconvert --to notebook --execute \
  benchmarks/rr1_rr3_contextual_graph_case_study/rr1_rr3_contextual_graph_case_study.ipynb \
  --output rr1_rr3_contextual_graph_case_study.ipynb \
  --output-dir benchmarks/rr1_rr3_contextual_graph_case_study
```

Graph scores and vector cosines have different scales, so evidence is based on
ranking/discrimination. A contextual edge is not automatically regulatory or
causal.

## Result

Global Bridge mean pooling assigned the unrelated RR1-original/RR3-39-original
pair a cosine of 0.811, slightly above the true RR3-39 technical remeasurement
(0.790). The prespecified contextual graph ranked both true RR3 remeasurements
(RR3-39: 0.319; RR3-40: 0.399) above the false friend (0.012). RR1 itself
remained poorly reproducible (0.183), so the graph is not presented as a
correction or rescue.

The post hoc Top-250 neighborhood annotation implicated lipogenic/cholesterol,
bile-acid/fatty-acid, nuclear-receptor, and translation-related programs. These
are descriptive pathway annotations, not evidence that contextual edges are
physical or causal gene interactions.

## Outputs

- `results/global_mean_vs_graph.csv`: primary comparison.
- `results/sample_manifest.csv`: exact samples and contrast membership.
- `results/per_gene_neighborhood_discrimination.csv`: all-gene localization.
- `results/top250_neighborhood_enrichment.csv`: enrichment with the canonical
  15,165 genes as background.
- `results/figures/`: publication-ready PNG/PDF figures.
- `rr1_rr3_contextual_graph_case_study.ipynb`: executed human-readable report.
