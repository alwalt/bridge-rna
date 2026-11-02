# Human + mouse ARCHS4 Hallmark-state retrieval report

**Benchmark status: COMPLETE.** This report is the multispecies component of the closed Hallmark-state / ARCHS4 retrieval benchmark.

## Executive result

Adding 429,746 compatible mouse ARCHS4 samples fundamentally changes OSDR retrieval. The combined atlas contains 940,455 samples, but mouse OSDR queries retrieve almost exclusively mouse neighbors: at top 10, mouse fractions are 99.53% for Bridge-512 and 100% for both Hallmark methods. Thus, the combined atlas organizes first by species and then by tissue/state. The earlier human-only results remain useful specifically as a forced cross-species analog benchmark; they are not representative of unconstrained retrieval when mouse references are available.

No model was retrained. All 940,455 Bridge embeddings were reused from the authoritative frozen cache. Mouse Hallmark values are explicitly **zero-shot cross-species predictions from the human GTEx-trained head**.

## Complete reference atlas

| Reference | Samples | Bridge-512 | Hallmark-50 | Search implementation |
|---|---:|---|---|---|
| Human | 510,709 | Existing frozen cache/index | Existing frozen database | Existing exact cosine index |
| Mouse | 429,746 | Existing frozen cache; new normalized index | Frozen probe applied unchanged | New exact cosine index |
| Combined | 940,455 | No duplicate matrix | No duplicate matrix | Exact merge of human and mouse top-k scores |

All 15,165 canonical symbols are unique and map one-to-one to a mouse gene. Cache-to-manifest joins have no missing samples or species mismatches, and both species have zero non-finite and zero-norm embeddings.

## OSDR representative-sample retrieval

One FLT and one GC sample were chosen for each of 64 studies by a deterministic rule: lexicographically first eligible matched group, then lexicographically first sample ID. Selection never used retrieval performance.

At top 10, combined-reference results are:

| Method | Mouse fraction | Tissue coherence | Mean top-5 signal coverage | Mean top-10 signal coverage |
|---|---:|---:|---:|---:|
| Bridge-512 cosine | 0.9953 | 0.5984 | 0.8145 | 0.8450 |
| Hallmark-50 cosine | 1.0000 | 0.6352 | 0.8511 | 0.8844 |
| Signed top-10 Hallmark | 1.0000 | 0.6055 | 0.7873 | 0.9510 |

Adding mouse references improves top-10 tissue coherence relative to the forced human reference: Bridge rises from 0.5602 to 0.5984, Hallmark cosine from 0.0633 to 0.6352, and signed overlap from 0.1883 to 0.6055. This is the clearest practical benefit of correcting the atlas.

Only one representative query has a human rank-1 combined hit: OSD-561 GC cerebellum retrieves GSM2627056/GSE98894, annotated as a human small-intestine neuroendocrine tumor liver metastasis. The metadata do not support a convincing cerebellar functional analog, so it is retained as an anomalous match rather than biological evidence.

The former human analogs are largely displaced: only 1.33% of prior Bridge top-10 human hits remain within the combined top 100, and none of the prior Hallmark cosine or signed-overlap top-10 hits remain. Strong forced cross-species analogs therefore do not generally stay highly ranked when same-species alternatives are available.

## FLT versus GC

Under the unchanged strict rule (positive reciprocal-rank effect, BH q ≤ 0.10, FLT prevalence ≥ 0.75, GC prevalence ≤ 0.25), the combined atlas yields 556 strict hits across 18 matched groups. Every strict hit is mouse: 244 Bridge, 212 Hallmark cosine, and 100 signed-overlap hits. The legacy human-only analysis had 435 strict hits across 24 groups. Adding mouse references increases the number of hit-level discoveries but concentrates them into fewer groups and replaces the previous human analog landscape rather than simply supplementing it.

These are differential retrieval neighborhoods, not evidence that spaceflight causes the retrieved disease or perturbation.

## Largest-study per-sample heatmaps

The largest eligible liver experiment is OSD-379: 70 left-liver-lobe samples, evenly split between 35 GC and 35 FLT. The largest muscle count is tied at 20 samples for OSD-665, OSD-666, and OSD-770; the predeclared lexicographic tie-break selects OSD-665, with 10 GC and 10 FLT extensor digitorum longus samples.

OSD-379 shows no category meeting the descriptive replicate-separation rule in either Bridge or Hallmark space. OSD-665 shows no Bridge category separation, but Hallmark retrieval shifts toward the skeletal-muscle/exercise category in FLT (standardized FLT−GC difference +0.695) and away from the broad drug/toxicant category (−1.052). These are descriptive reference-state shifts, not newly tested inferential claims.

## TCGA positive control

The frozen publication comparator is McClure et al., *Landscape of Genetic Alterations Underlying Hallmark Signature Changes in Cancer Reveals TP53 Aneuploidy-driven Metabolic Reprogramming* (Cancer Research Communications, 2023). The paper analyzed 8,603 tumors from 31 tumor types, started with 50 MSigDB Hallmarks, excluded KRAS-signaling-down and spermatogenesis for low variation, and used ssGSEA 2.0 rank normalization with weight 0.75 and NES output. Its highlighted relationship was a proliferative/glycolytic program associated with TP53/aneuploidy, particularly in squamous tumors and basal/basal-squamous BRCA/BLCA subtypes.

The local matched positive control contains 9,816 TCGA tumors from 32 types. The available conventional profiles are deterministic rank-mean ssGSEA-like scores on the frozen vocabulary, not the paper's exact ssGSEA 2.0 NES; mutation/subtype results are therefore not claimed as reproduced.

Nevertheless, cancer-level functional organization is strongly reproduced:

| Metric | Result |
|---|---:|
| Mean per-cancer conventional vs Bridge-Hallmark profile Pearson | 0.756 |
| Cancer-similarity matrix Spearman, conventional vs Bridge-Hallmark | 0.862 |
| Cancer-similarity matrix Spearman, conventional vs Bridge-512 | 0.388 |
| Nearest-cancer agreement | 0.625 |
| Median pathway-wise cancer-ranking Spearman | 0.824 |
| Clustering ARI, conventional vs Bridge-Hallmark, k=8 | 0.805 |
| Clustering ARI, conventional vs Bridge-512, k=8 | 0.177 |

This independently supports the intended interpretation: Hallmark-50 captures established cancer functional organization substantially better than the global 512-D geometry, including cross-tissue functional proximity. It does not validate every individual ARCHS4 retrieval.

## Global ARCHS4 manifold

Conservative tissue labels cover 57.55% of the 940,455 samples. UMAP was fit on a deterministic 30,000-sample species×tissue-stratified subset and transformed all samples. t-SNE uses exactly that subset. Tissue- and species-colored figures share identical coordinates.

| Space | Tissue silhouette | Species silhouette | Tissue NN purity | Species NN purity |
|---|---:|---:|---:|---:|
| Hallmark-50 | −0.051 | 0.236 | 0.654 | 0.9993 |
| Bridge-512 | −0.126 | 0.066 | 0.644 | 0.9976 |

Hallmark-50 modestly strengthens tissue organization, but it also strengthens species separation. Cross-species nearest-neighbor fractions within named tissues are generally below 0.1% in Hallmark space. Therefore, corresponding human and mouse tissues do **not** substantially mix in this atlas. The first Hallmark PCA axes are dominated by combinations of heme metabolism, unfolded-protein response, cholesterol homeostasis, PI3K/AKT/mTOR, MYC, DNA repair, E2F, and cell-cycle programs; effective rank is 14.76.

## Integrated answers

1. Previous OSDR conclusions change substantially: same-species mouse references replace nearly all forced human analogs.
2. Combined top-10 mouse fractions are 99.53% for Bridge and 100% for both Hallmark methods.
3. Bridge retrieval organizes strongly by species, with moderate tissue coherence within species.
4. Hallmark improves tissue/state coherence but does not create global cross-species mixing; species separation is stronger in Hallmark space.
5. Previously interesting mouse→human analogs generally do not remain highly ranked.
6. FLT and GC differ reproducibly in 18 groups under the unchanged strict test, but all strict combined-atlas hits are mouse.
7. OSD-379 shows no reproducible category separation; OSD-665 shows a Hallmark-specific FLT shift toward muscle/exercise references.
8. TCGA is a positive control for cross-tissue functional organization: Bridge Hallmark-50 closely reproduces conventional Hallmark cancer geometry.
9. The global manifold supports tissue-functional structure but rejects broad human↔mouse mixing.

## Limitations

ARCHS4 metadata are incomplete free text. Mouse Hallmark scores use a human-trained probe and are not species-calibrated. The publication positive control uses an available rank-based Hallmark implementation rather than exact ssGSEA 2.0 NES. UMAP/t-SNE are visualizations, not inferential tests. Retrieval similarity does not establish mechanism, disease equivalence, or causality.

## Final operational conclusion

Use Bridge-512 for global transcriptomic context, Hallmark-50 for complementary functional-state hypotheses, and species-conditioned indexes when within-species state is the question. Do not treat the combined human+mouse nearest neighbor as a neutral cross-species search: species NN purity is 0.9993 in Hallmark space and 0.9976 in Bridge space. Human-only OSDR retrieval remains valuable only when explicitly labeled as forced mouse→human analog search.

The final TCGA validation, LUAD leakage audit and cross-space analysis are consolidated in `FINAL_REPORT.md`. Contextual pathway retrieval, improved cross-species alignment, subtype-specific cancer retrieval, and drug-reversal search are future work rather than incomplete benchmark tasks.
