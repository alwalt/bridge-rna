# BridgeRNA benchmark synthesis

This document summarizes the executed benchmark notebooks as of 2026-09-07.
It distinguishes direct comparative wins from useful but exploratory findings
and negative results. Values are read from saved benchmark outputs rather than
recomputed or hard-coded into analysis pipelines.

## Where BridgeRNA is strongest

### 1. Masked whole-transcriptome reconstruction

The clearest direct model win is expression reconstruction. On the primary
15,101-gene shared-vocabulary TCGA benchmark, BridgeRNA outperforms
BulkFormer-50M and BulkFormer-147M at every tested partial-mask ratio. Pearson
correlations for BridgeRNA are 0.956, 0.952, 0.939, 0.899, and 0.679 at 15%,
30%, 50%, 70%, and 90% masking. At 50% masking the BulkFormer values are 0.692
and 0.541; at 90% they are 0.475 and 0.090. BridgeRNA also generalizes strongly
to strict unseen-sample/unseen-study ARCHS4 human and mouse holdouts. At 50%
masking Pearson is 0.941 in human and 0.968 in mouse; at 90% it remains 0.675
and 0.690. This is the strongest evidence that the frozen model learned a
useful global expression prior.

### 2. Robust sample identity across independent preprocessing

For 562 unseen-study ARCHS4/recount3 pairs, median expression agreement is
already high (Pearson 0.968), but BridgeRNA embeddings are extremely stable
(median paired cosine 0.9992). Within the 562-pair database, exact-sample
Recall@1/5/10 is 58.7%/84.2%/92.0%; 98.2% of Top-1 neighbors share the GSE.
Against all 510,709 human ARCHS4 embeddings, median exact rank is 2 and
Recall@100 is 91.8%. This supports processing robustness and preservation of
fine sample/study identity, not batch correction.

### 3. Informative reduced gene panels

Frozen model-derived gene panels consistently beat size-matched random panels
and mapped L1000 in external reconstruction. For the original 921-gene panels,
mouse-selected Top-921 reaches Pearson 0.600 on OSDR mouse versus 0.516 ± 0.015
for random and 0.511 for L1000. Human Top-921 reaches 0.596 on TCGA versus
0.529 ± 0.015 random and 0.566 L1000. A separately derived conserved
gene-inference Top-1000 also beats random and L1000 on TCGA and OSDR. In the
GTEx L1000-style evaluation it yields 3,235 well-inferred genes (23.0%) versus
3,030 (21.5%) for mapped L1000, with median matched-gene Pearson 0.573 versus
0.497. The advantage is modest in the L1000-style criterion but consistent.

### 4. Coordinated perturbation-response geometry

In the exploratory eight-study exercise analysis, BridgeRNA organizes
study-level response vectors much more strongly than raw expression/PCA. Its
response retrieval R@1 is 87.5% and MRR 0.891, versus R@1 62.5% and MRR
0.745/0.766 for raw/PCA. BridgeRNA exercise-response cosines form strongly
coherent opposing groups, whereas conventional edgeR response geometry shows
weak within-group separation. Because the groups were originally discovered in
BridgeRNA space, all samples have sample/study exposure, and only eight studies
are available, this is a strong hypothesis-generating result rather than an
independent benchmark win.

The corresponding attribution analysis adds interpretability: model-ranked
gene deletion changes the latent response roughly 7–34 times more than random
deletion. At 1,000 deleted genes, Axis A/B retain 0.640/0.746 of signal versus
0.971/0.973 under random deletion. High-IG/low-DE genes remain coherently
enriched for muscle structural and contractile programs, showing that the model
prioritizes coordinated biology not captured simply by the largest marginal DE
effects.

### 5. Useful contextual/module structure beyond global mean pooling

The best task-agnostic frozen readout is layer-12 Hallmark module mean+SD. It
reaches study-disjoint tissue macro F1 0.881 ± 0.027, slightly above raw
expression (0.872) and close to PCA (0.888). It also reduces the RR1/RR3-39
false-friend cosine from 0.811 for global mean to 0.510 while preserving
positive RR3-39/RR3-40 replication (0.775/0.922). This suggests that keeping
biological modules separate exposes information lost by one global average.

Contextual-gene graph fingerprints provide another useful readout. Controlled
same-RNA PolyA/Ribo graph retrieval reaches 90.0–96.3% R@1, same-tissue graphs
have 2.29-fold higher study-separated edge Jaccard, and degree+strength gives
study-disjoint tissue macro F1 0.862 ± 0.029. In the held-out NASA stress case,
signed graph responses rank the true RR3 technical replications above RR1 and
place the unrelated RR1/RR3 false friend last. These graphs do not beat PCA or
module pooling universally, but they expose gene-organization information that
global pooling obscures.

### 6. A technically informative diagnostic representation

Task 3/4 shows that BridgeRNA is sensitive to biologically relevant and
technical transformations in a structured way. RR3 same-animal response
remeasurements reproduce strongly (cosines 0.790 and 0.917), while the compound
RR1 PolyA-to-ribodepletion/resequencing transition reverses (about -0.804).
Independent same-RNA T-cell PolyA/Ribo experiments define a stable PC1–2
reference that captures about 0.953 of the squared RR1 replication discrepancy,
while RR3 responses remain reproducible despite nonzero technical alignment.
This supports BridgeRNA as a sensitive *technical-confounding profiler* when a
controlled reference exists. It does not show batch invariance or prove that
the aligned dimensions are purely technical.

### 7. Competitive compressed clinical representation

On five-cohort TCGA classification, the 512-D BridgeRNA representation reaches
weighted F1 0.9830, close to BulkFormer-147M (0.9861), and above BulkFormer-50M
(0.9796), despite BridgeRNA's smaller model and smaller 15,165-gene vocabulary.
Full 25,150-gene expression remains best at 0.9904. Pooled survival C-index is
also competitive (0.7597 versus 0.7620 for BulkFormer-147M), but BridgeRNA is
weaker on within-cancer weighted/macro survival endpoints. The win is compact,
competitive transfer—not state of the art over raw expression or the larger
model.

## Notebook-by-notebook verdict

| Notebook | Verdict for BridgeRNA |
|---|---|
| `tcga_imputation` | **Clear win:** best frozen reconstruction across shared mask ratios and strong strict ARCHS4 holdouts. |
| `paired_recount3` | **Clear strength:** near-identical embeddings and strong exact-sample retrieval across pipelines. |
| `landmark_gene_sufficiency` | **Win:** model-selected panels beat random/L1000 and transfer across human/mouse datasets. |
| `tcga_downstream` | **Competitive:** compact representation nearly matches larger models/raw expression for classification; not best for survival. |
| `cross_species_exercise_response` | **Promising exploratory strength:** much clearer response geometry and meaningful attribution; not independent/strict validation. |
| `osdr_batch_effect_representation` | **Mixed strength:** reproducible within-study response modes and attribution, but absolute embeddings retain technical structure and one RR1 response reverses. |
| `library_prep_disentanglement` | **Methodological strength:** controlled technical references diagnose instability and biological overlap; PCA/raw often outperform global BridgeRNA for generic tissue utility. |
| `mouse_encode` | **Information present, not dominant:** cross-species tissue information is accessible with kNN/probes, but raw/PCA usually retrieve tissues better. |
| `frozen_sample_embedding_readout` | **Qualified win:** Hallmark module pooling slightly exceeds raw tissue F1 and fixes some false similarity; global mean remains suboptimal. |
| `contextual_gene_network_utility` | **No comparative win:** contextual edges mostly recapitulate coexpression; raw correlation is stronger on pathway-pair recovery. |
| `contextual_rag_retrieval` | **No win:** raw expression leads general sample retrieval; global BridgeRNA leads the tiny response task, but contextual reranking hurts it. |
| `contextual_graph_kernel_similarity` | **Negative:** higher-order graph kernels are costly and do not improve retrieval over simpler metrics. |
| `gene_property_prediction` | **Pending:** audit complete; no BridgeRNA evaluation yet. BulkFormer essentiality is a sample-conditioned dependency task, not fixed-gene classification. |

## Overall model profile

BridgeRNA is strongest when the question uses what it was pretrained to learn:

1. reconstruct a partially observed bulk transcriptome;
2. recognize the same biological sample across processing pipelines;
3. identify compact gene panels that preserve global state;
4. represent within-study perturbation directions and attribute those directions
   to coordinated gene programs;
5. expose contextual gene/module organization when token structure is retained.

It is less convincing as a universal 512-D sample embedding for arbitrary
cosine retrieval, tissue classification, survival prediction, or graph-kernel
matching. Global mean pooling compresses the representation too aggressively,
and raw expression/PCA frequently retain more directly accessible tissue and
sample information. The most defensible positioning is therefore a strong
frozen **transcriptome reconstruction and contextual response model**, with
useful compact transfer representations—not a general replacement for PCA or
a batch-corrected biological embedding.
