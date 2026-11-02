# Scientific report: TCGA → ARCHS4 transcriptomic and functional-state retrieval

## Scope and frozen system

This benchmark queries all 510,709 human ARCHS4 samples with 9,816 patient-independent TCGA primary tumors spanning 32 cancer types. Bridge-512 and Hallmark-50 retrieval are exact cosine searches over the full corpus. The Hallmark scores are the unchanged frozen GTEx-trained linear Ridge readout applied after its frozen 512-feature scaler; they are continuous Bridge Hallmark predictions, not probabilities and not GSEA enrichment scores. Bridge and the Hallmark system were never trained or fine-tuned here.

The expression baseline is intentionally narrower: cancer-centroid queries against an authoritative deterministic 40,000-sample cache in the exact 15,165-gene natural-log1p(TPM) space. All three methods were also evaluated on this identical candidate set. Full individual expression retrieval would require roughly 76 trillion feature products and was not scientifically necessary for the requested conventional baseline.

## Main findings

1. **Hallmark retrieval does not reproduce Bridge retrieval.** Individual-query Bridge/Hallmark median Jaccard@100 was 0.111 (median overlap 20/100); centroid Jaccard@100 was 0.124 (22.0/100). At top 10, the corresponding medians were 0.053 and 0.053. Thus Hallmark-50 materially reorganizes neighborhoods rather than acting as a lossless compression of Bridge-512.

2. **Bridge better preserved coarse tissue metadata.** At individual-query top 100, median organ-keyword agreement was 0.460 for Bridge and 0.400 for Hallmark. Median unique-study counts were 31 and 34, respectively. Keyword matching is incomplete metadata evidence, not ground truth.

3. **Hallmark retrieval preserved functional profiles while leaving the global neighborhood.** Across cancer centroids, the median Spearman correlation between the TCGA centroid profile and the mean profile of its Hallmark top 100 was 0.960 (IQR 0.935–0.975). 74.6% of Hallmark centroid top-100 matches were absent from the Bridge top 100. Within Hallmark-retrieved candidates, the median cancer-wise correlation between Hallmark and Bridge similarities was only 0.164.

4. **The Hallmark cosine definition matters.** Relative to raw frozen scores, median centroid Jaccard@100 was 0.290 after centering and 0.262 after standardization. Raw cosine remains the prespecified primary metric; centered and standardized results are sensitivity analyses, not post-hoc metric selection.

5. **Matches were fairly robust to a small number of dominant pathways.** Median top-100 Jaccard after removing the strongest one and three standardized query dimensions was 0.896 and 0.794. This argues against every functional match being a one-coordinate artifact, while the observed losses show that dominant programs still influence ranking.

6. **Expression retrieval is related to both modes but identical to neither.** On the common 40k candidate set, median Jaccard@100 was 0.205 for Bridge/expression, 0.156 for Hallmark/expression, and 0.133 for Bridge/Hallmark. This comparison is centroid-only and should not be generalized to individual-query expression retrieval.

7. **Pathway preservation varies by program and cancer.** The five strongest pathway-wise cross-cancer replications were Apoptosis, Mitotic Spindle, Hypoxia, Fatty Acid Metabolism, Estrogen Response Early; the five weakest were Adipogenesis, Coagulation, Apical Surface, Myogenesis, Spermatogenesis. These are retrieval consistency results, not mechanistic claims.

8. **Hub behavior is non-negligible.** The most frequent Bridge and Hallmark samples appeared in 714 and 663 individual top-100 lists, respectively, despite a null expectation near 1.92 appearances under uniform retrieval. Study concentration and hub tables must accompany any interpretation of particular matches.

## Answers to the benchmark questions

- Hallmark-50 does not reproduce Bridge-512: their median top-100 overlap is modest.
- Bridge-512 has better coarse organ-keyword coherence; Hallmark-50 returns slightly more unique studies at top 100.
- Hallmark retrieval finds high-functional-similarity samples outside the global Bridge neighborhood. Representative examples and their pathway contributions are saved, but metadata alone cannot establish biological equivalence.
- Functional matches are driven by query-specific combinations of Hallmark dimensions, recorded per pair, and are mostly stable to removing one dominant dimension.
- The scoped expression baseline occupies another partially overlapping neighborhood.
- Cancer types and pathways vary materially; the per-cancer and per-pathway tables preserve negative results.
- Universal neighbor and study effects are substantial enough that individual examples require caution.
- The evidence supports exposing **global transcriptomic similarity** and **functional-state similarity** as distinct search modes, provided the product labels the Hallmark scores correctly, exposes pathway contributions, and warns about domain shift and hub/study effects.

## Controls and limitations

Random-query and independently shuffled-Hallmark controls, raw/centered/standardized metrics, dominant-dimension removal, exact duplicate-ID checks, finite-value checks, study concentration, and universal-neighbor frequencies were run. ARCHS4 labels are heterogeneous free text; organ/disease keyword metrics have imperfect sensitivity and specificity. The GTEx-trained head is transported into tumors and ARCHS4 without recalibration. Cosine on raw Hallmark scores is origin-sensitive. The expression baseline uses only 40,000 cached candidates and cancer centroids. Literature validation was not automated, so exploratory matches are described as candidates, not mechanisms or discoveries.

## Consolidated independent validation

An independent analysis in the sibling Hallmark-state worktree used the identical frozen head (SHA-256 `f3f6c2ddc75d872b1e78af189f9286ff2be64ee43e90047916fca5ffe0538931`) and the same 9,816 TCGA samples. This evidence was copied without recomputation and is indexed by `references/hallmark_state_retrieval_agent/COPY_MANIFEST.csv`.

The mean 50-Hallmark cancer profiles predicted by Bridge agreed with independently expression-derived, deterministic rank-mean Hallmark profiles: mean cancer-profile Pearson correlation was **0.756**. More importantly, cancer-to-cancer geometry was substantially closer to the conventional Hallmark geometry in Hallmark-50 space than in raw Bridge-512 space (similarity-matrix Spearman **0.862** versus **0.388**). Across pathways, the median correlation of cancer rankings was **0.824**. This is the strongest independent support that the frozen head recovers interpretable Hallmark structure in TCGA. The comparator is an ssGSEA-like rank-mean reconstruction on the 15,165-gene vocabulary, not exact ssGSEA 2.0 NES.

## Extended retrieval utility and negative results

The extended 33-cancer audit showed that the alternative functional neighborhood is not uniformly superior for disease lookup. At patient-level top 50, Bridge-512 had higher macro exact-disease precision (**0.338** versus **0.305**), tissue precision (**0.424** versus **0.395**), and malignancy precision (**0.777** versus **0.707**) than Hallmark cosine. At cancer-centroid top 10 the two were much closer: exact disease precision was **0.488** for Bridge and **0.482** for Hallmark, and tissue precision was **0.506** and **0.512**, respectively. Hallmark cosine had broader signed pathway-signal coverage (**0.791** versus **0.707**), while a signed top-10 overlap mode increased signal coverage to **0.888** but reduced exact-disease precision to **0.388**. These results support complementary search modes, not replacement of Bridge retrieval.

Specific centroid examples illustrate the tradeoff without proving mechanism: Hallmark retrieval returned more exact disease matches than Bridge for UVM (8/10 versus 6/10) and CHOL (9/10 versus 6/10), and LUAD retrieved related LUSC samples. These are metadata-compatible examples, not discoveries.

## Cross-study and cross-species boundary

In 1,471 OSDR mouse samples spanning 88 eligible groups, Hallmark cosine produced higher replicate-neighbor consistency than Bridge-512 (macro top-10 neighbor Jaccard **0.613–0.629** versus **0.366–0.386**). This gain co-occurred with stronger hub behavior and reduced tissue contextualization, so it should not be interpreted as universally better biological retrieval. Among 435 strict flight-enriched hits from 24 groups, 233 were found by Bridge, 179 by Hallmark cosine, and 23 by signed Hallmark overlap.

When the reference atlas was expanded from 510,709 human ARCHS4 samples to a combined 940,455-sample human/mouse atlas, retrieval was overwhelmingly species-pure: mouse fractions at top 10 were **99.53%** for Bridge and **100%** for both Hallmark modes. For Hallmark cosine, tissue coherence rose from **0.063** in the forced human-only search to **0.635** in the combined atlas. The frozen pathway coordinate system can therefore be computed across species, but nearest-neighbor search remains dominated by species/domain structure when same-species references are available.

## Revised conclusion

The consolidated evidence supports the frozen GTEx-trained 512-to-50 head as an interpretable, independently validated projection of Bridge embeddings. It preserves conventional Hallmark cancer structure and creates a retrieval neighborhood that is demonstrably different from Bridge-512. Its benefit is clearest for pathway-state comparison, signal coverage, and some centroid-level examples; Bridge remains stronger for several patient-level tissue and disease metadata criteria. Cross-species results establish portability of the coordinate system but also reveal strong domain separation. The scientifically defensible product claim is therefore two complementary modes—global transcriptomic similarity and functional-state similarity—not a claim that Hallmark retrieval is generally more accurate or mechanistically definitive.
