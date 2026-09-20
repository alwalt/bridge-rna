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

RR1 measurement-sensitive expression effects were most strongly enriched for
mRNA processing/splicing. Five multiscale genes were shared across all three:
`GADD45G`, `IGFBP1`, `NRN1`, `SLC41A2`, and `TCIM`.

The primary T-cell-independent signed-IG pathway analysis identifies 125 robust
profiles in RR1, 178 in RR3-39, and 200 in RR3-40. Twenty-two are robust in all
three. They group into lipid/carnitine/bile metabolism, amino-acid and
central-carbon metabolism, ER/integrated stress, immune/complement regulation,
stress/metabolic transcription, fluid/water transport, and IGF signaling.

The executed notebook is the primary human-readable artifact. Machine-readable
tables are under `results/summary/`, `results/genes/`, `results/pathways/`, and
`results/manifest/`; publication figures are under `results/figures/`.

Primary pathway artifacts are:

- `results/figures/independent_robust_pathway_upset.{png,pdf}`
- `results/figures/independent_conserved_core_dotplot.{png,pdf}`
- `results/figures/independent_core_three_method_heatmap.{png,pdf}`
- `results/pathways/independent_conserved_core_full.csv`
- `results/pathways/independent_conserved_core_modules.csv`
- `results/pathways/independent_core_three_method_matrix.csv`
- `results/pathways/independent_core_pathway_validation.csv`
- `results/pathways/spaceflight_literature_evidence.csv`
- `results/summary/independent_pathway_upset_intersections.csv`
- `results/summary/independent_core_method_overlap.csv`
- `results/summary/independent_core_pathway_validation_summary.csv`

### Conventional versus BridgeRNA interpretation

Twenty of the 22 exact pathways were testable by the existing full-stratum
edgeR ranked GSEA in all three cohorts. Only two had conventional support in
any cohort: Regulation of Complement Cascade in RR1 (NES 1.688, FDR 0.0473)
and Glyoxylate Metabolism and Glycine Degradation in RR3-39 (NES 1.693, FDR
0.0398). None was conventionally significant in all three. The other 18 fully
testable pathways are reproducible IG reprioritizations with weak/absent exact-
pathway GSEA support; Fluid Transport and Water Transport remain ambiguous
because those exact terms were not in the conventional GSEA collection.

A focused literature audit classifies lipid/carnitine/bile and amino-acid/
central-carbon metabolism as previously reported in rodent spaceflight liver;
ER/integrated stress, immune/complement, and stress/metabolic transcription
have related precedent. Fluid/water transport and IGF-receptor signaling remain
hypothesis-generating. These labels are conservative module-level annotations,
not formal novelty claims. BridgeRNA therefore reprioritizes a technically
reproducible pathway core rather than merely restating conventional GSEA.

Conventional matched expression provides stronger evidence for exploratory
RR1/RR3 directional opposition than full-stratum edgeR. That contrast is kept
secondary because it depends on sample matching and estimator.

The former 29-pathway/six-module analysis used contextual responses projected
into a controlled T-cell PolyA/ribo PC1-2 reference. It is retained only as a
secondary technical-control artifact and is not part of the main biological
figures or conclusions.

## Attribution-only analysis

Attribution-only genes have reproducible signed IG across technical
remeasurement but do not satisfy the robust-expression criterion. There were
200 in RR1, 271 in RR3-39, and 247 in RR3-40; 152 recurred in at least two
cohorts and 37 recurred in all three. These genes had smaller expression
effects and substantially poorer expression ranks, yet were coherently enriched
for xenobiotic, bile-acid, lipid/fatty-acid, amino-acid/catabolic, complement,
peroxisomal, and oxidative-phosphorylation biology.

Contextual-graph support was limited: 10.0% for RR1, 6.3% for RR3-39, and 9.3%
for RR3-40, lower than for expression-only genes under the stringent graph
criterion. Existing deletion sweeps are not gene-set-compatible with this
analysis. Thus, the result supports additional reproducible model-derived
ranking structure, but not a separately validated causal gene program.

Supporting files:

- `results/genes/*_attribution_only.csv`
- `results/genes/shared_attribution_only_genes.csv`
- `results/pathways/attribution_only_enrichment.csv`
- `results/summary/attribution_vs_expression_summary.csv`
- `results/summary/attribution_only_graph_support.csv`
- `results/figures/attribution_vs_expression_rank.{png,pdf}`
- `results/figures/attribution_only_graph_support.{png,pdf}`

## Native-transcriptome sensitivity

`pipeline/analyze_native_transcriptome_sensitivity.py` runs a matched-animal,
raw-count conventional control for RR1, RR3-39, and RR3-40. Each original and
OSD-168 remeasurement is analyzed with edgeR and ranked GSEA in both its
fullest mapped native gene universe and the exact BridgeRNA vocabulary. It does
not load BridgeRNA embeddings or rerun IG.

The paired native analyses retain 13,678–15,362 common tested genes, compared
with 10,587–11,140 under Bridge-vocabulary restriction. Restriction modestly
raises response cosine in RR1 (0.621 to 0.713), RR3-39 (0.610 to 0.663), and
RR3-40 (0.562 to 0.691). In contrast, original-versus-remeasurement pathway-NES
Spearman is stable: native/restricted values are 0.494/0.523, 0.786/0.782, and
0.873/0.871, respectively.

Thresholded pathway calls are more sensitive. Native/restricted robust-pathway
counts are 5/15 (RR1), 248/230 (RR3-39), and 43/65 (RR3-40). Native RR1 has nine
significant direction-reversing pathways, versus zero in the restricted arm;
restriction therefore makes RR1 appear somewhat more stable. Among the 22
signed-IG pathways, the native analysis gives robust conventional support to
Glycine/Serine/Threonine Metabolism and Glyoxylate/Glycine Degradation in
RR3-39; the restricted analysis retains only the latter. No member is
conventionally robust in RR1 or RR3-40.

The direct universe audit finds 3/5, 219/248, and 38/43 native robust hits are
shared with the restricted arm for RR1, RR3-39, and RR3-40. Restriction gains
12, 11, and 27 robust calls while losing 2, 29, and 5, respectively. This is
not a uniform increase: RR3-39 loses more calls than it gains. Within each
measurement, native/restricted NES Spearman ranges from 0.934 to 0.969 and no
pathway significant in both universes reverses NES direction. Native full-
transcriptome GSEA robustly supports only 2/22 signed-IG pathways, compared with
1/22 under restriction.

The sensitivity analysis does not materially overturn the main interpretation:
RR1 remains least reproducible at the pathway-profile level, RR3 responses are
more coherent, and most of the 22 signed-IG pathways remain model-prioritized
rather than conventional GSEA rediscoveries. However, individual pathway calls
and apparent expression-level reproducibility are vocabulary-sensitive.
The explicit decision is **partial contribution, not explanation**: vocabulary
restriction affects marginal calls but does not explain recovery of the
22-pathway BridgeRNA signed-IG core.

Artifacts are under `results/native_transcriptome_sensitivity/`, including the
full edgeR/GSEA tables, 22-pathway validation, native-only significant pathways,
summary tables, and PNG/PDF figures. The run log is
`results/native_transcriptome_sensitivity_run.log`.

## Interpretation limits

- OSD-168 is a technical remeasurement, not independent biological replication.
- Expression-interaction enrichment uses model-vocabulary expression effects
  and the exact 15,165-gene background; it is distinct from raw-count edgeR.
- Existing signed pathway profiles are not conventional GSEA NES.
- Signed IG pathway directions describe attribution contribution, not pathway
  expression up/down regulation.
- The 22-pathway set is selected by signed-IG profile robustness; its pathway-
  profile support is therefore true by construction, while DE/GSEA and robust-
  gene enrichment are independent checks.
- The data support a mixed technical/biological interpretation for RR1, not a
  causal attribution to a single protocol variable.
