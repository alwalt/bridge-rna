# BridgeRNA drug-discovery benchmark — concise summary

## Question and design

The benchmark asked whether frozen BridgeRNA condition programs (1) recover biology beyond matched differential expression, (2) reproduce across independent terrestrial and OSDR cohorts and connect to pharmacological targets, and (3) are opposed by experimentally observed drug perturbations. It covered radiation injury, bone loss, and muscle atrophy across nine frozen datasets, with MS, Crohn's disease, and SLE as de Weerd positive controls.

## Strongest findings

- Bridge-only genes recurred in another same-condition dataset at 43.5%, versus 11.7% for DE-only genes.
- Terrestrial–OSDR gene Jaccard was 0.159 for Bridge-only versus 0.016 for DE-only genes.
- Deleting Bridge-only genes changed the frozen representation axis more than deleting DE-only genes (0.0635 versus 0.0246 mean absolute change); targetable fractions were similar (16.3% versus 14.6%).
- Bone Bridge-only genes reproduced across studies, were enriched for osteoporosis-associated genes, and formed coherent extracellular-matrix programs.
- Muscle Bridge-only genes reproduced across terrestrial and OSDR cohorts; expanded-ChEMBL drug Jaccard was 0.164 for Bridge versus 0.0068 for DE. The original restrictive-ChEMBL Bridge estimate (0.714) was strongly universe dependent.
- In radiation, Bridge exceeded DE in 18/21 matched comparisons (mean paired Jaccard difference +0.105; Wilcoxon p=1.83×10⁻⁵). Bridge had a 237-gene cross-study core and 15 genes in all seven contrasts, versus 35 and zero for DE.
- DisGeNET v7 recovery favored DE in MS and Crohn's but Bridge in SLE, supporting endpoint-dependent complementarity rather than universal superiority.
- ChEMBL identified targetable Bridge biology, but Bridge did not preferentially recover de Weerd's highlighted drugs under the prespecified greater-than-10-target rule.

## Primary conclusion and major negative result

Bridge identifies reproducible condition-associated biology beyond genes prioritized by DE, with strong terrestrial-to-OSDR transfer. Portions of that biology are pharmacologically connected. However, targetability did not yield robust direction-aware countermeasures: in observed GSE264130 RNA-seq, 14 Bridge and 11 DE expression screens passed an unadjusted both-cell rule, but none survived BH correction (minimum q=0.9973) or strict three-study validation. Secondary LINCS results were likewise globally nonsignificant and context dependent.

Bridge and DE are complementary. The benchmark supports biological prioritization and target hypothesis generation, not validated therapies.

## Major limitations

Only three datasets represent each space condition; tissues and exposures are heterogeneous; some terrestrial cohorts were ARCHS4-exposed; exact de Weerd DrugBank and filtered DisGeNET resources were unavailable; ChEMBL hub effects are substantial; GSE264130 has only two glioma cell lines; and expression, latent, and LINCS reversal agreed weakly.

## Definitive reports

- [Full scientific synthesis](FINAL_SCIENTIFIC_REPORT.md)
- [de Weerd replication](../deweerd_replication/results/integrated_benchmark_summary.md)
- [Expanded ChEMBL sensitivity](results/expanded_chembl_sensitivity/expanded_sensitivity_report.md)
- [Radiation-context sensitivity](results/radiation_context_sensitivity/REPORT.md)
- [Radiation pharmacology](results/radiation_pharmacology/REPORT.md)
- [Secondary LINCS reversal](../direction_aware_reversal/DIRECTION_AWARE_REVERSAL_REPORT.md)
- [Primary GSE264130 reversal](../gse264130_reversal/GSE264130_PRIMARY_REVERSAL_REPORT.md)
