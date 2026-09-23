# BridgeRNA Drug Discovery and Pharmacological Reversal Benchmark

## Scientific questions

This benchmark asked three progressively stronger questions:

1. Does the frozen Bridge representation identify condition-associated biology beyond genes prioritized by conventional differential expression (DE)?
2. Is that biology reproducible across independent terrestrial and spaceflight/OSDR datasets, externally supported, and connected to pharmacological targets?
3. Do experimentally observed drug perturbations oppose those condition-associated states in expression space or in the frozen Bridge latent space?

These are distinct evidence levels. Reproducible biology does not imply druggability; target overlap does not establish a beneficial direction; and transcriptional opposition does not establish therapeutic efficacy.

The benchmark froze the Bridge checkpoint, canonical 15,165-gene order, preprocessing, signed attribution procedure, matched DE analyses, and primary top-500 module size before drug evaluation. ARCHS4 exposure was treated as provenance: GSE184119 and GSE113165 are training-exposed, GSE211204 is partially exposed and has an exposure-excluded sensitivity subset, and the remaining primary space-condition cohorts are confirmed unseen. Results in exposed cohorts support reproducibility but are not claims of strict out-of-pretraining generalization.

## Conditions and frozen contrasts

The nine-dataset space-condition design is recorded in [dataset_matrix.csv](results/dataset_matrix.csv) and its freeze rationale in [README.md](README.md).

| Condition | Role | Dataset | Frozen contrast |
|---|---|---|---|
| Radiation injury | terrestrial discovery | GSE297090 | 1 Gy gamma versus matched gamma mock and 1 Gy proton versus matched proton mock, analyzed separately with donor/organoid-state blocking |
| Radiation injury | terrestrial replication | GSE184119 | 10 Gy versus 0 Gy within six normal human cell types; cell type retained in the model |
| Radiation injury | OSDR | OSD-993 / GSE297560 | each 4 Gy particle exposure versus its shipped particle control, plus gamma versus shipped gamma control; modalities kept separate |
| Bone loss | terrestrial discovery | GSE189524 | osteoporosis versus non-osteoporosis mesenchymal stromal cells after frozen transcript-to-gene aggregation and donor-level paired-library averaging |
| Bone loss | terrestrial replication | GSE276529 | glucocorticoid-induced osteoporosis cortical femur versus age-matched non-GIOP cortical femur |
| Bone loss | OSDR | OSD-867 / GSE273868 | ISS versus synchronous ground control, separately at weeks 1 and 2, in human bone-marrow mesenchymal stromal cells |
| Muscle atrophy | terrestrial discovery | GSE211204 | paired day-10 unilateral lower-limb suspension versus baseline, with the prespecified exposure-excluded sensitivity subset |
| Muscle atrophy | terrestrial replication | GSE113165 | paired day-5 bed rest versus pre-bed-rest, retaining age/sex strata |
| Muscle atrophy | OSDR | OSD-684 / GSE234465 | spaceflight versus ground, separately in young- and old-donor engineered myobundles |

The positive-control disease analyses used GSE138614 active multiple-sclerosis lesions, GSE101794 Crohn's ileum, and GSE72509 SLE blood. Their exact sample sets, contrasts, covariates, coverage, and exposure provenance are frozen in the de Weerd replication outputs.

## Evaluation 1 — Biological validity

The methodological replication compared frozen Bridge and DE top-500 modules against exact-CUI DisGeNET v7 disease references using a common mapped gene universe. The primary results in [disease_gene_recovery.csv](../deweerd_replication/results/evaluation/disease_gene_recovery.csv) were:

| Disease | Method | Known genes recovered | Odds ratio | Raw p | FDR |
|---|---:|---:|---:|---:|---:|
| MS active lesion | Bridge | 72 | 1.770 | 2.55×10⁻⁵ | 3.06×10⁻⁵ |
| MS active lesion | DE | 87 | 2.245 | 5.86×10⁻¹⁰ | 1.76×10⁻⁹ |
| Crohn's ileum | Bridge | 70 | 1.950 | 2.22×10⁻⁶ | 3.33×10⁻⁶ |
| Crohn's ileum | DE | 79 | 2.275 | 2.26×10⁻⁹ | 4.53×10⁻⁹ |
| SLE blood | Bridge | 106 | 2.760 | 2.36×10⁻¹⁶ | 1.42×10⁻¹⁵ |
| SLE blood | DE | 72 | 1.673 | 1.22×10⁻⁴ | 1.22×10⁻⁴ |

Both methods recovered significant known biology. DE was stronger for MS and Crohn's; Bridge was substantially stronger for SLE. This is not an exact numerical reproduction of de Weerd et al.: their filtered DisGeNET reference sets were not published or reconstructable, so the documented DisGeNET v7 exact-CUI analysis is a methodological replication. The full interpretation is in [integrated_benchmark_summary.md](../deweerd_replication/results/integrated_benchmark_summary.md).

## Evaluation 2 — Drug-target recovery

No authorized DrugBank export was available. The reproducible approximation used ChEMBL 37 parent-normalized clinical compounds with human single-protein targets from mechanisms plus high-confidence binding evidence. The expanded graph contains 4,389 drugs, 2,046 targets, and 16,986 edges; 366 drugs have more than 10 targets and enter the primary right-sided Fisher tests. By comparison, de Weerd reported 328 eligible DrugBank drugs and a 16,600-protein background. ChEMBL therefore approximates the number of tested drugs but not DrugBank's edge semantics or background. The frozen construction and audit are in [drug_target_universe_qc.json](../deweerd_replication/results/evaluation2_expanded/drug_target_universe_qc.json) and [evaluation2_report.md](../deweerd_replication/results/evaluation2_expanded/evaluation2_report.md).

Only five of the 15 highlighted de Weerd drugs were represented and eligible under the prespecified greater-than-10-target rule: two in MS, none in Crohn's, and three in SLE. Their pooled Bridge-minus-DE rank-percentile difference was −0.0093 (exact sign-flip p=0.75). Bridge therefore did **not** preferentially recover the highlighted drugs. Drugs absent from the resource were not counted as Bridge failures, and the ChEMBL result must not be presented as an exact DrugBank replication. This negative result is retained without threshold tuning.

## Evaluation 3 — Cross-study space-condition biology

The strongest benchmark result is gene-level complementarity. Partitioning every frozen top-500 module into Bridge-only, shared, and DE-only genes showed that 43.5% of Bridge-only genes recurred in another independent dataset of the same condition, versus 11.7% of DE-only genes. Mean terrestrial–OSDR Jaccard was 0.159 for Bridge-only and 0.016 for DE-only genes. Bridge-only genes were also more influential under frozen representation deletion: mean absolute axis-effect change was 0.0635 (0.0143 per 100 genes), versus 0.0246 (0.00549 per 100 genes) for DE-only genes. Targetable fractions were similar—16.3% Bridge-only and 14.6% DE-only—so Bridge recurrence was not simply a consequence of more ChEMBL connectivity. Source values are in [partition_overall_summary.csv](results/expanded_chembl_sensitivity/partition_overall_summary.csv); the complete sensitivity analysis is in [expanded_sensitivity_report.md](results/expanded_chembl_sensitivity/expanded_sensitivity_report.md).

These results support complementarity, not replacement: DE captures large marginal changes, whereas Bridge can prioritize additional learned or contextual biology that transfers more consistently between studies.

### Bone loss

Bone Bridge-only genes reproduced across independent cohorts (recurrence 42.3%; terrestrial–OSDR Jaccard 0.173), overlapped osteoporosis-associated genes (148 overlaps; minimum corrected q=1.25×10⁻⁶), and converged on coherent extracellular-matrix biology. Their targetable fraction was 16.6%, with three significant drug enrichments in the partition analysis. This establishes reproducible, externally supported, partly targetable biology; it does not establish that the connected compounds would reverse bone loss.

### Muscle atrophy

Muscle Bridge-only genes showed the highest recurrence (50.3%) and terrestrial–OSDR Jaccard (0.175), with significant skeletal-muscle-atrophy reference enrichment (59 overlaps; minimum q=5.24×10⁻⁵). Recurrent genes include LAMB2, ITGA7, DES, NEB, TPM2, ACTA1, ERBB2, PDE5A, and PDE4B. The expanded-ChEMBL top-25 drug Jaccard was 0.164 for Bridge and 0.0068 for DE. This is far smaller than the original restrictive-ChEMBL Bridge Jaccard of 0.714, demonstrating drug-universe sensitivity while preserving a strong qualitative Bridge–DE separation. The restrictive result was largely amplified by ERBB2 and PDE5A recurrence rather than a broad significant drug program; corrected enrichment of the two-gene universal set was not significant. Details are in [summary.md](results/muscle_target_decomposition/summary.md) and [expanded_sensitivity_report.md](results/expanded_chembl_sensitivity/expanded_sensitivity_report.md).

### Radiation injury

The initial drug-level radiation convergence was weak, but the context analysis exposed strong gene-level stability. Across matched contrast pairs, Bridge Jaccard exceeded DE in 18/21; the mean paired Bridge-minus-DE difference was +0.105 (Wilcoxon p=1.83×10⁻⁵). The cross-study core contains 237 Bridge genes versus 35 DE genes. A strict core of 15 genes occurs in all seven Bridge radiation modules, while zero genes occur in all seven DE modules. In the matched terrestrial NHDF-to-OSDR fibroblast comparisons, Bridge Jaccard was 0.186–0.208, versus 0.017–0.022 for DE, with all Bridge overlaps exceeding matched random expectation (empirical p=0.0005).

Radiation response is strongly dependent on tissue, cell type, modality, dose, and time. The earlier weak drug convergence therefore masked substantially stronger gene-level reproducibility. Stable prioritization is also not identical to a stable signed response: only a subset of the 15 universal Bridge genes retains the same sign across every contrast. Radiation should be benchmarked by matched context and cross-context core separately. Full results are in [REPORT.md](results/radiation_context_sensitivity/REPORT.md).

## Pharmacological targetability

The frozen radiation cores connect recurrent biology to PDE5A, DPP4, PRKDC/DNA-PK, CDK7, ERBB2, and extracellular-matrix mechanisms. The literature audit in [REPORT.md](results/radiation_pharmacology/REPORT.md) separates normal-tissue mitigation from tumor radiosensitization:

- Pentoxifylline–PDE5A has human combination-trial evidence for radiation-fibrosis mitigation, but pentoxifylline alone was ineffective in a small placebo-controlled trial; the enrichment is a single-target result with q=1.
- DPP4 has target-level animal evidence through selective inhibition, but the ranked luteolin/resveratrol mappings do not match the mechanisms reported in their radiation studies.
- Omipalisib–PRKDC/DNA-PK and CDK7 inhibition have mechanistically compatible radiosensitization evidence and may imply normal-tissue hazard rather than protection.
- ERBB2-directed mappings are recurrent but commonly hub-driven and directionally ambiguous.
- ECM-associated compounds such as ocriplasmin connect multiple core genes, yet lack evidence sufficient to infer benefit.

Molibresib is a cautionary example: its multiple ChEMBL overlaps are binding relationships to ACTN4/LMNB1/RUNX1/SMC4/TCOF1/UBTF rather than its canonical BET mechanism. Counting those overlaps as independent BET support would be mechanistically misleading. In every condition, ChEMBL overlap establishes pharmacological connectivity—not beneficial direction, transcriptional reversal, or efficacy.

## Secondary direction-aware analysis — LINCS/L1000

LINCS is explicitly **secondary**. Its Level-5 signatures contain 978 directly measured landmark genes plus inferred genes and z-score normalization, which does not satisfy the validated full-transcriptome natural-log `log1p(TPM)` Bridge encoder contract. It therefore supports signed gene-ranking reversal only, not valid latent reversal.

The prespecified screen found more Bridge cross-study reversers than DE for radiation (29 versus 17) and bone (79 versus 16), but fewer for muscle (9 versus 74). No compound survived global BH correction across approximately 13,000 eligible consensuses; Bridge and DE frequently disagreed on direction. These results indicate context dependence and complementary prioritization, not universal Bridge superiority. Resource details, all nulls, and cell-context limitations are in [DIRECTION_AWARE_REVERSAL_REPORT.md](../direction_aware_reversal/DIRECTION_AWARE_REVERSAL_REPORT.md).

## Primary direction-aware analysis — GSE264130 RNA-seq

The primary reversal benchmark used only experimentally observed drug-treated and matched DMSO PLATE-Seq RNA-seq from benchmark #6: 372 drugs in DIPG6 and SF8628, 744 drug/cell contexts, 24 hours, one study-selected dose per drug/cell context, and 15,028 Bridge-compatible genes. Treated and matched-control expression was encoded through the frozen Bridge model, and equivalence checks gave a maximum control-embedding difference of zero. No predicted drug responses, chemical fingerprints, or molecular embeddings were used. The handoff is frozen in [benchmark6_handoff_audit.json](../gse264130_reversal/results/benchmark6_handoff_audit.json).

Fourteen Bridge and 11 DE expression screens met the prespecified unadjusted both-cell rule. Within a condition, the methods shared no candidates. No expression or latent result survived BH correction (minimum q=0.9973), and no drug passed strict terrestrial-discovery–terrestrial-replication–OSDR validation. Expression–latent Spearman agreement ranged from −0.506 to +0.061, while GSE264130–LINCS agreement was also weak (approximately −0.13 to +0.18 for Bridge across the two cell contexts). Cell context and representation materially changed conclusions.

Examples remain hypotheses:

- Bone pacritinib–MAP4K4 combined both-cell Bridge expression and latent opposition with direct target overlap, but ChEMBL enrichment was nonsignificant, OSDR latent direction reinforced rather than reversed, and LINCS supplied no match.
- Bone OTX015 opposed Bridge in both expression and latent screens, but lacked frozen direct-target support and also failed OSDR latent replication.
- Radiation bortezomib and palbociclib showed both-cell Bridge expression opposition and same-direction LINCS evidence, but no latent or strict cross-study support.
- SLE gemcitabine and cladribine showed latent opposition and LINCS directional agreement, while Bridge expression was reinforcing or mixed.

The definitive primary report is [GSE264130_PRIMARY_REVERSAL_REPORT.md](../gse264130_reversal/GSE264130_PRIMARY_REVERSAL_REPORT.md), with the independent review in [INDEPENDENT_METHOD_REVIEW.md](../gse264130_reversal/INDEPENDENT_METHOD_REVIEW.md).

## Integrated interpretation

The evidence hierarchy is:

`reproducible Bridge biology`
→ `external biological support`
→ `pharmacological targetability`
→ `observed RNA-seq reversal`
→ `latent reversal`
→ `cross-assay LINCS replication`
→ `literature support`

Evidence is strongest in the first three layers. Bridge identifies reproducible condition-associated biology beyond genes prioritized by conventional DE, particularly for bone loss and muscle atrophy and for radiation once biological context is accounted for. These complementary programs include pharmacologically actionable targets. However, target overlap does not reliably translate into directionally consistent drug reversal: neither the primary GSE264130 RNA-seq analysis nor the secondary LINCS analysis produced globally significant, cross-study-validated therapeutic candidates.

This negative reversal result is not a failure of the biological representation. It establishes a boundary between identifying reproducible/actionable biology—which is supported—and predicting robust therapeutic countermeasures, which is not established by the available perturbation contexts.

## Relationship to differential expression

Bridge should not replace DE. DE captures strong marginal expression changes and outperformed Bridge in MS and Crohn's disease-gene recovery and in the LINCS muscle-reversal screen. Bridge identifies additional learned/contextual programs; Bridge-only genes showed markedly greater cross-study and terrestrial–OSDR recurrence, and Bridge was stronger for SLE disease-gene recovery. Which representation is more informative depends on the endpoint. Direction-aware reversal does not demonstrate universal Bridge superiority, and using both methods exposes agreement, complementarity, and contradiction that either alone would hide.

## Limitations

- Each space condition has only three independent datasets, with heterogeneous tissues, exposures, doses, and timepoints.
- ARCHS4 pretraining exposure affects some terrestrial cohorts; exposed-cohort results are reproducibility evidence, not strict generalization.
- de Weerd's exact filtered DisGeNET reference sets and licensed DrugBank target universe were unavailable.
- ChEMBL connectivity and hub targets can amplify families of drugs without broad biological support.
- GSE264130 contains only two diffuse intrinsic pontine glioma cell contexts, which are remote from most benchmark tissues.
- No drug reversal survived global FDR correction, and none passed strict three-study space-condition validation.
- Expression–latent and RNA-seq–LINCS agreement were weak.
- Radiation directionality is context specific; recurrent rank does not guarantee recurrent sign.
- Target overlap and transcriptional opposition do not demonstrate therapeutic efficacy, safety, dose feasibility, or tissue delivery.

## Final conclusion

Bridge provides reproducible biological information complementary to DE, including substantial terrestrial-to-spaceflight transfer and pharmacologically targetable programs in bone loss, muscle atrophy, and context-matched radiation injury. The benchmark does **not** provide robust evidence for a specific therapeutic reversal candidate. Direction-aware validation was essential: it prevented recurrent target overlap from being overinterpreted as treatment potential and defined the current evidence boundary for BridgeRNA drug discovery.
