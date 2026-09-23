# BridgeRNA direction-aware drug-reversal benchmark

## Executive result

Adding transcriptional direction changes the pharmacological interpretation materially, but it does **not** establish a general Bridge advantage over DE.

Bridge and DE often assign different—and sometimes opposite—directions to the same drug. Bridge produced more prespecified cross-study-consistent reversers in radiation (29 versus 17) and bone loss (79 versus 16), whereas DE produced many more in muscle atrophy (74 versus 9). Across all six conditions, no drug-level competitive enrichment survived BH correction over the approximately 13,000 eligible compound consensuses (all `q≈1`). The top-1% lists are therefore descriptive screening sets, not discoveries with experiment-wide significance.

The strongest defensible conclusion is **complementarity**: Bridge exposes directionally coherent hypotheses that DE does not prioritize, but the effect is disease- and context-dependent, and the available LINCS cell contexts substantially limit biological interpretation.

## Frozen design and perturbation resource

No upstream preprocessing, embeddings, condition vectors, Integrated Gradients, DE, modules, DisGeNET, ChEMBL enrichment, or earlier benchmark result was recomputed. Input hashes and the exact contrast inventory are recorded in [condition_signature_provenance.json](results/condition_signature_provenance.json).

The resource audit selected public LINCS/L1000 GSE92742 Level-5 COMPZ/MODZ before reversal outcomes were inspected. It contains 473,647 replicate-collapsed signatures; the frozen primary slice contains 112,495 exemplar compound signatures, 20,231 perturbagen IDs, 71 cell lines, and 8,831 Bridge-canonical BING genes (898 measured landmarks and 7,933 high-confidence inferred genes). The downloaded 20-GB matrix passed gzip validation and its published SHA-512 (`6a3115...208a2a`). Full audit counts, doses, times, replicates, controls, gene coverage, and prespecified-drug coverage are in [lincs_resource_audit.csv](results/lincs_resource_audit.csv), [lincs_context_inventory.csv.gz](results/lincs_context_inventory.csv.gz), and [lincs_resource_audit_provenance.json](results/lincs_resource_audit_provenance.json).

Level-5 values are moderated z-score signatures standardized to plate controls. They permit signed drug-response testing, but they are not raw treatment and vehicle expression profiles. Each cell/dose/time context was retained separately before drug-level medians were calculated.

Bridge latent reversal was **not run**: L1000 Level-5 values (978 measured plus inferred genes) cannot reproduce the validated 15,165-gene log1p(TPM) encoder input. Encoding them would produce an uninterpretable latent cosine. This critical validity boundary was frozen in [analysis_contract.json](analysis_contract.json).

## Reversal statistic and controls

The primary statistic is the negative rank-weighted cosine between a frozen signed top-500 condition module and an individual LINCS signature. Positive values indicate transcriptional opposition; negative values indicate reinforcement. A drug consensus is the median across contexts and requires at least three signatures in at least two cell lines. The exploratory candidate rule is positive median, at least 60% positive contexts, and top 1% among eligible drug consensuses. BH q-values are always reported and are not part of the screening rule.

The full signed-ranking sensitivity, Bridge-only/shared/DE-only partitions, stable recurrent programs, random-drug null, 2,000 sign shuffles, 200 expression-matched modules, and cell/dose/time outputs are saved in [results/](results/). Expression-matched modules preserve source-expression decile and measured-versus-inferred L1000 status. Prioritized null tests are explicitly post-ranking validation and do not replace the global multiple-testing result.

## Unified six-condition result

Counts below are descriptive top-1% screens. `Cross-study reproducible` requires a positive score and top-decile competitive p in all three independent space-condition studies. Literature support was only counted from the frozen de Weerd highlighted list or the prior radiation literature audit; bone and muscle were not systematically literature-audited here.

| Condition | Bridge reversal candidates | DE reversal candidates | Bridge-only candidates | Cross-study reproducible (Bridge / DE) | Target-supported Bridge | Literature-supported Bridge |
|---|---:|---:|---:|---:|---:|---:|
| Multiple sclerosis | 131 | 130 | 130 | NA | 0 | 1 |
| Crohn's disease | 130 | 131 | 128 | NA | 0 | 0 |
| Systemic lupus erythematosus | 131 | 131 | 131 | NA | 0 | 2 |
| Radiation injury | 131 | 131 | 128 | 29 / 17 | 0 | 0 |
| Bone loss | 131 | 131 | 128 | 79 / 16 | 1 | not audited |
| Muscle atrophy | 131 | 131 | 131 | 9 / 74 | 0 | not audited |

The machine-readable table is [unified_six_condition_table.csv](results/unified_six_condition_table.csv). Because a top-1% rule nearly fixes the first two counts by design, the important results are score direction, context consistency, cross-study consistency, null behavior, and target/literature triangulation—not the raw candidate totals.

## Published disease validation

### Multiple sclerosis

Only ibrutinib of the five highlighted drugs has an eligible LINCS signature. It weakly opposed Bridge (`median=0.0288`, 4/6 contexts positive) but reinforced DE (`median=-0.1582`, 0/6 positive). Its competitive p-values were 0.129 and 0.977, respectively, and neither result survived FDR. Muromonab, daclizumab, zanubrutinib, and alemtuzumab are absent; they are not counted as failures.

The strongest prioritized Bridge null-supported named compounds were hydroxytyrosol and calyculin; DE's included BG-1012. These are screening hypotheses, not validation of therapeutic relevance.

### Crohn's disease

Glucosamine reinforced both programs. VX-702 reinforced Bridge (`-0.0153`, 5/14 contexts positive) but opposed DE (`0.0966`, 11/14 positive); the auxiliary DE consensus passed both prioritized nulls, but its global competitive p was 0.074 and q was approximately 1. Zinc, zinc acetate, and dilmapimod are absent.

Prioritized Bridge null-supported compounds included vinblastine and ABC-294640; several DE compounds also passed both auxiliary nulls. The result does not favor Bridge in Crohn's.

### Systemic lupus erythematosus

All five highlighted drugs are represented. Bridge classified enzastaurin and sunitinib as reversing, while DE classified fostamatinib and cladribine as reversing; gemcitabine and the remaining method/drug combinations were mixed. None was globally significant. Enzastaurin's Bridge expression-matched result was suggestive (`q=0.0697`) but its sign-shuffle q was 0.130.

Bridge prioritized fludroxycortide and diltiazem passed both auxiliary nulls; DE prioritized pentoxyverine and radicicol did as well. Thus, despite Bridge's previously stronger DisGeNET recovery in SLE, direction-aware pharmacology remains complementary rather than decisively superior.

Complete drug-by-method classifications are in [published_drug_direction_audit.csv](results/published_drug_direction_audit.csv), and prioritized nulls are in [prioritized_candidate_nulls.csv](results/prioritized_candidate_nulls.csv).

## Space-health extension

### Radiation injury

Bridge produced 29 cross-study-consistent screens versus 17 for DE. Radicicol was the clearest named Bridge example (`consensus=0.0933`, positive in all three datasets), while oxamflatin passed both prioritized nulls on the stable recurrent Bridge program. However, all three prioritized DE compounds also passed both auxiliary nulls, and no radiation compound was globally FDR-significant. The earlier finding of strong context dependence therefore remains important.

The prespecified target hypotheses were directionally informative but not statistically enriched:

- Pentoxifylline/PDE5A: modest Bridge opposition (`0.0302`, 12/17 positive), essentially null/mixed for DE; global `p=0.344`, `q≈1`.
- Sitagliptin/DPP4: modest Bridge opposition (`0.0357`, 8/13 positive), mixed for DE; `p=0.298`, `q≈1`.
- KU-0060648/DNA-PK: weak Bridge opposition (`0.0189`, 35/50 positive) but DE reinforcement (`-0.0747`, 15/50 positive).
- NU-7441: Bridge mixed; DE reinforcing.
- ERBB2 compounds were perturbagen-ID/context dependent. Neratinib opposed Bridge but reinforced DE; afatinib and lapatinib changed classification across exact LINCS perturbagen records.
- Omipalisib, RG-547, THZ1, JQ1, and molibresib lacked eligible exact signatures.

These results help distinguish transcriptional direction but do not resolve whether an intervention is radioprotective or radiosensitizing. The exact audit is [radiation_hypothesis_direction_audit.csv](results/radiation_hypothesis_direction_audit.csv).

### Bone loss

Bridge produced 79 cross-study-consistent screens versus 16 for DE. N-bromoacetyltryptamine and etoposide opposed the stable recurrent Bridge program and passed both prioritized nulls. Etoposide is cytotoxic and illustrates why reversal is not efficacy. Of the Bridge top-1% candidates, only fluphenazine also had a frozen expanded-ChEMBL module overlap (HTR7), and that target enrichment was nonsignificant (`best p=0.375`, `q=1`).

### Muscle atrophy

The direction-aware result reverses the impression of a universal Bridge advantage: Bridge had only 9 cross-study-consistent screens versus 74 for DE, and none of three prioritized Bridge compounds passed either auxiliary null. This is particularly important because the earlier target-overlap Jaccard was high but ERBB2/PDE5A-driven. Directionality does not corroborate a broad Bridge muscle-reversal advantage.

## Bridge versus DE and complementarity

For identical drugs, the median Bridge-minus-DE score was positive in MS (`+0.0086`), SLE (`+0.0177`), bone (`+0.0031`), and muscle (`+0.0069`), but negative in Crohn's (`-0.0270`) and radiation (`-0.0079`). The drug rankings were weakly anticorrelated in five conditions (`Spearman -0.15` to `-0.36`) and weakly positively correlated only in bone (`0.076`). With tens of thousands of drugs, paired Wilcoxon p-values are very small but effect sizes are tiny; these tests demonstrate systematic representational differences, not practical superiority.

Top-500 versus full-ranking drug scores were moderately correlated for Bridge (approximately 0.50–0.72) and more strongly for DE (0.78–0.90), so Bridge candidates are more module-definition-sensitive. Exact comparisons are in [bridge_vs_de_paired_comparison.csv](results/bridge_vs_de_paired_comparison.csv), [full_ranking_reversal_sensitivity.parquet](results/full_ranking_reversal_sensitivity.parquet), and [partition_reversal_summary.csv](results/partition_reversal_summary.csv).

## Target pharmacology integration

LINCS-to-ChEMBL identity mapping used full InChIKey first and a unique normalized preferred name second. Of 51,383 LINCS perturbagens in metadata, 1,423 map to the frozen >10-target expanded-ChEMBL universe. Among top-1% Bridge screens, direct frozen module-target overlap occurred only for bone fluphenazine/HTR7; no Bridge screen in the other five conditions had such support. DE had a few target-supported screens, but every corresponding target-enrichment FDR was nonsignificant.

This sparse intersection shows that transcriptional reversal and direct target enrichment are largely orthogonal evidence layers here. The complete classifications—target overlap plus reversal, reinforcement, mixed direction, reversal without direct overlap, and insufficient identity—are in [target_reversal_integration.parquet](results/target_reversal_integration.parquet) and [lincs_to_expanded_chembl_mapping.csv.gz](results/lincs_to_expanded_chembl_mapping.csv.gz).

## Cell-line, dose, and time dependence

Most eligible drug consensuses span multiple cell lines, but biologically relevant contexts are sparse. The two-cell minimum retained 340 compounds in immune-line subsets for MS/SLE, 28 in the radiation FIBRNPC/HT29 subset, and only four for stable recurrent radiation programs. Crohn's HT29, bone CD34/ASC, and muscle SKB subsets could not satisfy a two-cell-line rule and therefore support no context-relevant consensus claim.

Even highlighted drugs varied substantially by context: ibrutinib's Bridge scores ranged from `-0.0209` to `0.0855`; SLE sunitinib ranged from `-0.127` to `0.102`; and fostamatinib from `-0.141` to `0.142`. Exact signature and cell/dose/time tables are [prioritized_context_scores.csv.gz](results/prioritized_context_scores.csv.gz), [prioritized_cell_dose_time_sensitivity.csv.gz](results/prioritized_cell_dose_time_sensitivity.csv.gz), and [prioritized_context_sensitivity_summary.csv](results/prioritized_context_sensitivity_summary.csv).

## Overall answer

Directionality improves interpretation by separating opposition, reinforcement, and context-dependent responses that target overlap alone cannot distinguish. It identifies plausible Bridge-only hypotheses—most clearly in the bone and radiation stable programs—and reveals cases where a target-supported compound would reinforce rather than reverse a condition signature.

It does **not** show that Bridge generally outperforms DE. Global drug-level FDR is null; relevant human cell contexts are limited; L1000 uses inferred genes and cancer-heavy panels; Bridge's rankings are more module-size sensitive; and muscle favors DE strongly. The evidence supports Bridge as a complementary hypothesis generator, not a validated therapeutic-ranking system.

The four evidence layers remain separate:

1. reproducible condition biology (frozen prior benchmark),
2. targetability (frozen expanded ChEMBL),
3. transcriptional opposition (this analysis), and
4. independent literature evidence (frozen prior audits).

No compound satisfies all four layers with experiment-wide statistical support in this benchmark.

