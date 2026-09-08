# RR1 library-preparation interaction

This benchmark asks whether the RR1 OSD-48 to OSD-168 reversal resembles a
common PolyA→ribodepletion displacement or whether the measurement transition
modifies the estimated FLT−GC contrast itself. It reuses frozen Task 3/4
embeddings, exact animal mappings, contextual graphs, and gene attributions.

The comparison is observational with respect to RR1: OSD, library selection,
read layout/length/depth, preservation, and processing changed together.
Therefore results are described as a **protocol-associated interaction**, not a
causal library-preparation effect.

```bash
.venv/bin/python benchmarks/rr1_library_prep_interaction/pipeline/build_benchmark.py
.venv/bin/python benchmarks/rr1_library_prep_interaction/pipeline/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute \
  benchmarks/rr1_library_prep_interaction/rr1_library_prep_interaction_benchmark.ipynb \
  --output rr1_library_prep_interaction_benchmark.ipynb \
  --output-dir benchmarks/rr1_library_prep_interaction \
  --ExecutePreprocessor.timeout=600
```

## Main result

The matched RR1 analysis uses four FLT and five GC animals represented in both
OSD-48 and OSD-168. FLT and GC protocol displacements are nearly parallel in
expression/global/program spaces (cosine 0.962–0.996), demonstrating a strong
common shift. Their difference is nevertheless 35–41% of the mean displacement
magnitude and is tied only by the observed allocation and its complement among
126 exhaustive condition-label permutations (`p = 2/126 = 0.0159`).

The global FLT−GC response reverses (cosine -0.804). The independent T-cell
PolyA→Ribo mean direction captures 51.9% of the interaction energy, but removing
it improves the response only to -0.709. L12 mean+SD and Hallmark modules also
disagree, whereas signed input-gene IG retains positive cosine despite weak
rank reproducibility. The result is therefore mixed: a
large generic protocol displacement plus a condition-dependent interaction and
residual RR1 instability.

The Top-250 expression-interaction genes are strongly enriched for mRNA
processing and splicing. This localizes sensitivity but does not establish that
the affected biology is artifactual or that library selection alone caused it.

## Protocol-robust core

The follow-up analysis identifies 179 mutual Top-500 same-direction expression
genes, 246 mutual Top-500 same-direction IG genes, and 476 Top-500 concordant
local graph neighborhoods. There are 114 genes supported by at least two levels
and 26 supported by all three. Enrichment separates a robust hepatic
lipid/fatty-acid, peroxisomal, PPARα, cholesterol/transport, circadian, and
mitochondrial response from the mRNA-processing/splicing programs that dominate
the protocol×flight interaction.

Supporting artifacts are under `results/robust_core/`. These results establish
concordance across the two RR1 measurements, not a protocol-independent or
causal spaceflight response.
