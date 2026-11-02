# Publication review: frozen Hallmark-state representation and retrieval

## Review record

- **Benchmark:** `benchmarks/tcga_archs4_state_retrieval/`
- **Primary source revision:** `cbd48b7d2bd0eb87bc71225de6f58f6b74924ea6`
- **Consolidated agent source:** `we-are-starting-a-new-self-contained-benchmark/benchmarks/hallmark_state_retrieval/`, also at revision `cbd48b7d2bd0eb87bc71225de6f58f6b74924ea6`
- **Consolidation manifest:** [COPY_MANIFEST.csv](references/hallmark_state_retrieval_agent/COPY_MANIFEST.csv)
- **Publication status:** approved by the author for main-paper treatment of the frozen Hallmark method, independent TCGA validation, retrieval results, and qualified cross-species use
- **Scope boundary:** this review concerns the sample-level frozen Bridge `512 → 50` Hallmark readout. The contextual gene-membership decoder and Integrated Gradients are separate methods and did not generate these retrieval profiles.

The consolidated evidence includes the current exact TCGA→ARCHS4 retrieval benchmark plus the earlier agent's frozen artifacts, complete source pipeline, independent TCGA pathway positive control, 33-cancer retrieval evaluation, signal-matching audit, focused LUAD leakage audit, OSDR atlas analysis, and combined human/mouse ARCHS4 analysis. Large deterministic matrices and raw top-100 caches were not duplicated; compact source tables, manifests, and reports needed to audit all claims were copied unchanged.

## Scientific objective

The combined benchmark asks three distinct questions:

1. **Representation validity:** can a frozen linear readout recover independently calculated Hallmark organization from Bridge's pooled 512-dimensional sample representation?
2. **Retrieval utility:** does Hallmark-50 provide a useful and interpretable search view distinct from Bridge-512, and does it improve disease, tissue, or pathway-state matching?
3. **Cross-species boundary:** when the same human-trained readout is applied unchanged to mouse samples, does unconstrained retrieval yield cross-species analogs or primarily species-conditioned neighbors?

The intended contribution is a qualified two-view system:

- **Bridge-512:** global transcriptomic/contextual similarity.
- **Hallmark-50:** similarity after supervised projection toward 50 expression-derived Hallmark targets.

The evidence does not support replacing Bridge-512 with Hallmark-50. It supports Hallmark-50 as an independently validated, interpretable secondary coordinate system.

## Frozen Hallmark method

### Training targets and cohort

The operational head was trained using 4,260 GTEx samples from 920 donors and 12 broad tissues. For each sample:

1. rank all 15,165 canonical genes by expression;
2. divide ranks by 15,165;
3. average normalized ranks among the canonical members of each of 50 MSigDB Hallmarks; and
4. standardize each Hallmark target across GTEx.

These are deterministic **GTEx-standardized rank-mean Hallmark targets**. They are not running-sum ssGSEA/GSVA scores, GSEA normalized enrichment scores, probabilities, causal pathway activities, or gene contributions.

### Model and inference

Each sample was represented by the frozen 512-dimensional Bridge embedding obtained by unweighted mean pooling of final-layer contextual gene states over the exact 15,165-gene order. A GTEx `StandardScaler` and 50-output Ridge regression were fitted. One global alpha was selected from `0.01, 0.1, 1, 10, 100` using five-fold donor-grouped cross-validation; `alpha=0.1` was selected. The scaler and Ridge head were refit on all GTEx samples and frozen.

Inference is:

```text
natural log1p(TPM), exact canonical gene order
→ frozen Bridge
→ unweighted final-layer mean-pooled 512-D sample embedding z
→ (z - GTEx mean) / GTEx scale
→ frozen Ridge weights and bias
→ 50 continuous Bridge Hallmark scores
```

The consolidated head SHA-256 is `f3f6c2ddc75d872b1e78af189f9286ff2be64ee43e90047916fca5ffe0538931`, identical in both worktrees. The Bridge checkpoint SHA-256 is `f3491beaa697a0408dc1daeb6f8d648566bd46bafc6e44938d723e7ffc325541`.

## Datasets and evaluations

| Evaluation | Cohort/reference | Scientific role |
|---|---|---|
| GTEx probe validation | 4,260 samples, 920 donors, 12 tissues | Donor-disjoint recovery of expression-derived targets and frozen-head fitting. |
| TCGA zero-shot transfer | 9,816 patient-independent primary tumors, 32 cancers | Tests transport of the GTEx head without TCGA fitting. |
| TCGA conventional-pathway positive control | Same 9,816 tumors and 32 cancers | Independently compares Bridge Hallmark cancer profiles with expression-derived rank-mean profiles. |
| Human ARCHS4 retrieval | 510,709 unique human samples | Full-corpus Bridge-512 versus Hallmark-50 search for individual TCGA tumors and cancer centroids. |
| Extended TCGA retrieval | 9,942 samples, 33 cancers including LAML | Evaluates exact disease, tissue, malignancy, related-cancer, and signed-signal retrieval. |
| Common-universe expression baseline | 32 cancer centroids against 40,000 ARCHS4 samples | Scoped conventional natural-`log1p(TPM)` retrieval comparison. |
| OSDR reference-atlas analysis | 1,471 mouse samples, 88 matched FLT/GC groups | Tests replicate reproducibility, differential retrieval, and functional analog hypotheses. |
| Combined ARCHS4 atlas | 510,709 human + 429,746 mouse = 940,455 samples | Tests species conditioning and whether same-species references displace forced mouse→human analogs. |

The current QA verifies unique patient/sample identifiers, finite values, exact matrix dimensions, and no exact TCGA/ARCHS4 identifier overlap. The focused LUAD audit further examined 200 method-specific centroid top-100 rows: no TCGA barcode, explicit reuse hit, or exact TCGA query identifier was found; 128 rows had affirmative independent-cohort evidence and 72 remained metadata-insufficient.

## Validated results

### 1. The frozen head recovers independent pathway organization

| Result | Value | Interpretation | Evidence |
|---|---:|---|---|
| GTEx donor-disjoint target recovery | median Pearson `0.994` across 50 targets | Strong recovery of the training-domain rank-mean targets. | Generating benchmark transfer summary. |
| GTEx→TCGA zero-shot transfer | median Pearson `0.852`, range `0.305–0.944`; all 50 positive | The frozen healthy-tissue map transports to cancer without TCGA fitting. | Generating benchmark transfer summary. |
| Conventional-expression versus Bridge-Hallmark cancer profiles | mean Pearson `0.756` across 32 cancers | Independent evidence that Bridge Hallmark profiles recover broad conventional pathway organization. | [provenance_and_summary.json](references/hallmark_state_retrieval_agent/results/tcga_pan_cancer_positive_control/provenance_and_summary.json) |
| Conventional versus Bridge-Hallmark cancer-similarity geometry | Spearman `0.862` | Hallmark projection recovers conventional cancer relationships. | Same positive-control summary. |
| Conventional versus raw Bridge-512 cancer geometry | Spearman `0.388` | Named Hallmark projection aligns with conventional pathway geometry more strongly than uninterpreted raw sample geometry. | Same positive-control summary. |
| Pathway-wise ranking across cancers | median Spearman `0.824` | Individual Hallmark cancer rankings broadly reproduce expression-derived rankings. | [pathway_cancer_ranking_concordance.csv](references/hallmark_state_retrieval_agent/results/tcga_pan_cancer_positive_control/pathway_cancer_ranking_concordance.csv) |

The conventional profiles are deterministic rank-mean scores on the frozen 15,165-gene vocabulary, not an exact reproduction of the cited paper's ssGSEA 2.0 NES. This validates broad functional organization, not exact paper replication or clinical validity.

### 2. Hallmark retrieval creates a distinct neighborhood

| Result | Value | Interpretation | Evidence |
|---|---:|---|---|
| Individual Bridge/Hallmark overlap@100 | median Jaccard `0.111`; overlap `20/100`; IQR `0.064–0.176`; `n=9,816` | Hallmark-50 is not a lossless compression of Bridge-512 retrieval. | [individual_method_overlap.csv](results/individual_method_overlap.csv) |
| Cancer-centroid overlap@100 | median Jaccard `0.124`; overlap `22/100`; IQR `0.093–0.220`; `n=32` | Neighborhood reorganization persists after centroid averaging. | [centroid_method_overlap.csv](results/centroid_method_overlap.csv) |
| Query-to-retrieved Hallmark profile | median Spearman `0.960`, IQR `0.935–0.975` | Hallmark retrieval preserves its query coordinates, although this is selection-conditioned. | [functional_profile_preservation.csv](results/functional_profile_preservation.csv) |
| Hallmark top-100 outside Bridge top-100 | `74.6%` | Most Hallmark neighbors lie outside the global Bridge neighborhood. | [centroid_functional_discordance.csv](results/centroid_functional_discordance.csv) |
| Common 40k candidate universe | median Jaccard@100: Bridge/expression `0.205`, Hallmark/expression `0.156`, Bridge/Hallmark `0.133` | Expression, Bridge, and Hallmark retrieval are related but nonidentical. | [expression_subset_method_overlap.csv](results/expression_subset_method_overlap.csv) |

### 3. Retrieval utility is context-dependent, not generally superior

For individual-patient queries across 33 cancers, macro confirmed precision@50 favored Bridge:

| Criterion | Hallmark-50 | Bridge-512 |
|---|---:|---:|
| Exact disease | `0.305` | `0.338` |
| Tissue | `0.395` | `0.424` |
| Malignancy | `0.707` | `0.777` |
| Disease or same-tissue related cancer | `0.313` | `0.353` |

These values are preserved in [macro_hypothesis_metrics.csv](references/hallmark_state_retrieval_agent/results/tcga33_retrieval/macro_hypothesis_metrics.csv). They show that the 50-dimensional projection discards information useful for individual disease and context identification.

Cancer-centroid retrieval was approximately tied:

- exact disease P@10: Hallmark `0.482`, Bridge `0.488`;
- tissue P@10: Hallmark `0.512`, Bridge `0.506`;
- malignancy P@10: Hallmark `0.855`, Bridge `0.864`.

Narrow positive examples were UVM (`8/10` exact Hallmark hits versus `6/10` Bridge), CHOL (`9/10` versus `6/10`), and LUAD retrieval of several LUSC samples sharing lung-cancer programs. These are examples of functional reranking, not a general performance advantage.

Whole-vector Hallmark cosine increased signed top-10 pathway coverage from Bridge's `0.707` to `0.791` while retaining nearly tied centroid disease precision. A signed top-10 overlap rule increased pathway coverage further to `0.888` but reduced exact-disease P@10 to `0.388`. Thus stronger program matching trades away identity coherence; sparse matching is appropriate as an annotation or secondary reranking, not the primary search metric.

### 4. OSDR and multispecies evidence establishes the cross-species boundary

Hallmark top-10 neighborhoods were more reproducible among OSDR biological replicates than Bridge neighborhoods: neighbor Jaccard was approximately `0.613–0.629` for Hallmark cosine versus `0.366–0.386` for Bridge. Hallmark also increased signed pathway coverage, while Bridge retained much more tissue context. Hub concentration partially inflated the Hallmark reproducibility.

Across 88 matched FLT/GC groups, 435 strict FLT-enriched ARCHS4 samples occurred in 24 groups after 1,000 within-group label permutations and the frozen prevalence/FDR rule: 233 Bridge hits, 179 Hallmark-cosine hits, and 23 signed-overlap hits. These were group-specific shifts, not a universal spaceflight state.

Adding 429,746 mouse references changed the interpretation decisively:

- combined top-10 mouse fraction: Bridge `99.53%`, Hallmark cosine `100%`, signed Hallmark overlap `100%`;
- species nearest-neighbor purity: Bridge `0.9976`, Hallmark `0.9993`;
- Hallmark top-10 tissue coherence increased from `0.0633` in forced human-only retrieval to `0.6352` with the combined atlas.

Therefore, unconstrained combined retrieval is primarily species-conditioned. Human-only OSDR retrieval is a deliberately forced mouse→human analog analysis, not neutral cross-species nearest-neighbor search. The strongest coherent Hallmark hypotheses involved adipose/lipid programs and trained-muscle analogs; many other matches reflected generic UPR, heme, proliferation, or tumor programs.

## Positive findings

1. The frozen `512 → 50` head recovers conventional expression-derived Hallmark organization across independent TCGA cancers.
2. Hallmark-50 supplies a reproducible, named biological coordinate system that is substantially more aligned with conventional pathway geometry than raw Bridge-512.
3. At cancer-centroid resolution, Hallmark retrieval retains disease/tissue coherence approximately comparable to Bridge while increasing named pathway-signal coverage.
4. Selected UVM, CHOL, LUAD/LUSC, adipose, and muscle examples demonstrate the intended functional-reranking use case.
5. Hallmark and Bridge retrieval expose complementary neighborhoods, enabling global-context search plus functional-program inspection.
6. The unchanged head can be applied cross-species, but the resulting search must be species-conditioned or explicitly labeled as forced analogy.

## Negative findings and limitations

1. Hallmark-50 generally underperforms Bridge-512 for individual-patient disease, tissue, and malignancy identity.
2. Higher signed-program coverage does not guarantee biological equivalence; generic proliferation, DNA-repair, UPR, heme, and stress programs generate cross-context matches.
3. Same-space Hallmark-profile preservation is expected from the retrieval objective and is not independent validation; the conventional TCGA comparison supplies the stronger evidence.
4. ARCHS4 annotations are incomplete free text. Confirmed precision is conservative, but unannotated samples are not proven negatives.
5. Universal-neighbor and study effects are substantial. The largest individual-query hubs occurred hundreds of times.
6. The human ARCHS4 corpus mixes Bridge training, validation, and unseen samples, so aggregate retrieval is not a strict unseen-sample evaluation.
7. The expression baseline is centroid-only on 40,000 cached candidates.
8. Exact-identifier and focused LUAD audits reduce but do not eliminate the possibility of biological duplicate material under unrelated accessions.
9. The mouse Hallmark scores are zero-shot predictions from a human GTEx-trained head and are not species-calibrated.
10. Similarity does not establish mechanism, causal pathway activation, disease equivalence, or therapeutic relevance.

## Publication recommendation

**Recommendation: include the frozen Hallmark method and independent TCGA validation in the main manuscript; present retrieval as a complementary two-view result, with detailed per-cancer, signal-matching, OSDR, hub, and multispecies controls in supplementary material.**

The main-text contribution should be:

1. a method schematic for GTEx rank-mean target construction and frozen `512 → 50` inference;
2. independent TCGA evidence that Bridge Hallmark profiles recover conventional pathway organization (`0.756`, `0.862`, `0.824`);
3. evidence that Hallmark retrieval creates a distinct, interpretable neighborhood while centroid identity performance remains approximately tied to Bridge;
4. the negative individual-patient result showing Bridge remains the default global/contextual retrieval mode; and
5. the cross-species result that unconstrained human+mouse retrieval is overwhelmingly species-conditioned.

The defensible conclusion is not that Hallmark retrieval is better. It is that the frozen head converts an opaque global representation into a biologically validated pathway coordinate system that provides a useful secondary functional-state view.

## Proposed figures and tables

1. Frozen Hallmark training and inference schematic.
2. Conventional versus Bridge-Hallmark TCGA cancer-profile concordance, with Bridge-512 geometry as a comparator.
3. Bridge/Hallmark neighborhood overlap for individual and centroid queries.
4. Individual versus centroid retrieval tradeoff: disease/tissue/malignancy precision and signed pathway coverage.
5. Representative UVM, CHOL, and LUAD functional-reranking examples with explicit metadata caveats.
6. Combined human/mouse species-composition result demonstrating the cross-species boundary.
7. Supplementary hub, leakage, pathway-dominance, OSDR reproducibility, and per-cancer tables.

## Approval status

The author explicitly approved main-paper inclusion and requested consolidation of the earlier agent's relevant artifacts, scripts, data, review, outline, and notebook evidence into this worktree. This approval does not imply that Hallmark retrieval may be described as generally superior or mechanistically causal.
