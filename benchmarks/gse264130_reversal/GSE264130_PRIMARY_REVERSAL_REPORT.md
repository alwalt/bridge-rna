# GSE264130 primary direction-aware drug-reversal benchmark

## Executive conclusion

Experimentally observed GSE264130 RNA-seq perturbations do identify drugs that oppose frozen Bridge condition programs, but the evidence is exploratory and strongly context dependent. Across the six conditions, 14 Bridge-expression screens and 11 DE-expression screens met the prespecified unadjusted rule in both DIPG6 and SF8628. The two sets had **zero drugs in common within any condition**. No expression or latent result survived BH correction (`minimum q=0.9973`), and no drug satisfied the strict both-cell, terrestrial-discovery, terrestrial-replication, and OSDR criterion.

Bridge therefore provides reversal information beyond DE in the literal sense that it identifies different relationships, but this benchmark does not establish that those relationships are more valid. Expression-space and Bridge-latent reversal agreed poorly, and independent GSE264130–LINCS agreement was weak.

## Benchmark #6 handoff audit

The source was the finalized benchmark #6 worktree at commit `5d23557`. No benchmark #6 analysis or prediction model was rerun. The following observed assets were verified directly:

- 372 drugs and 744 drug/cell contexts;
- DIPG6 and SF8628, one study-selected dose per drug/cell pair, all at 24 hours;
- two independently plated treated replicates per context;
- six DMSO wells per cell line and plate, 24 DMSO wells total;
- plate-matched `treated − DMSO` deltas averaged only after plate-level subtraction;
- 15,028 observed genes in the canonical 15,165-gene Bridge vocabulary, with 137 canonical genes absent;
- Entrez counts converted to RPK using the released gene lengths, then TPM and natural `log1p`;
- exact gene ordering verified from Bridge token IDs;
- a 744 × 15,028 observed expression-delta matrix and a 1,536 × 15,028 observed sample-expression matrix.

The handoff had valid observed expression vectors and four control-centroid embeddings but did **not** contain treated-sample or drug-delta embeddings. It explicitly stated that latent vectors had not been exported. No predicted response, Morgan fingerprint, MoLFormer embedding, or prediction output was used. Details and source-file hashes are in [benchmark6_handoff_audit.json](results/benchmark6_handoff_audit.json).

## Frozen design

All condition biology came from the previously frozen signatures and condition vectors. The drug data did not alter Bridge attribution, top-500 modules, Bridge-only programs, recurrent programs, radiation cores, DE, DisGeNET, ChEMBL, or previous convergence results.

Expression reversal is the negative rank-weighted cosine between the observed drug delta and each frozen signed top-500 module. Positive values mean transcriptional opposition. Each contrast and cell line was scored separately. The empirical p-value is the drug-label/random-drug null rank among all 372 assayed drugs in the same cell line and frozen contrast. BH correction was applied within contrast, method, and cell line.

The descriptive both-cell screen requires a positive condition-level score and empirical `p≤0.05` in both cell lines. This threshold was frozen before scoring and is not an FDR claim.

For latent reversal, all 1,536 observed samples were passed through the frozen encoder using benchmark #6's exact validated inference path: float32, batch size one, `model.encode(normalize=False)`, no autocast. The four archived benchmark #6 control embeddings were reproduced bit-for-bit. For each plate, the treated embedding minus the mean of six matched DMSO embeddings was calculated, and the two plate-level deltas were averaged. Negative condition/drug cosine is reported as positive latent reversal.

The full contract is [analysis_contract.json](analysis_contract.json), and encoder equivalence is recorded in [encoder_equivalence.json](results/encoder_equivalence.json).

## Primary expression-space result

| Condition | Bridge candidates | DE candidates | Shared | Bridge-only | DE-only |
|---|---:|---:|---:|---:|---:|
| Multiple sclerosis | 2 | 2 | 0 | 2 | 2 |
| Crohn's disease | 1 | 1 | 0 | 1 | 1 |
| Systemic lupus erythematosus | 2 | 3 | 0 | 2 | 3 |
| Radiation injury | 4 | 1 | 0 | 4 | 1 |
| Bone loss | 3 | 1 | 0 | 3 | 1 |
| Muscle atrophy | 2 | 3 | 0 | 2 | 3 |

Bridge both-cell expression screens were:

- MS: voreloxin, voxtalisib.
- Crohn's: panobinostat.
- SLE: leuprolide, pacritinib.
- Radiation: bortezomib, LY2835219/abemaciclib, PLX51107, palbociclib.
- Bone: OTX015, PLX51107, pacritinib.
- Muscle: EPZ-5676/pinometostat, ezatiostat.

DE produced a completely disjoint set. This is evidence of complementarity, but zero overlap and absent FDR support also mean that neither set should be treated as confirmed.

Exact per-contrast, per-cell scores, usable gene counts, empirical p-values, FDR, and direction are in [expression_reversal_by_contrast_cell.parquet](results/expression_reversal_by_contrast_cell.parquet). All primary screens are in [all_primary_candidates.csv](results/all_primary_candidates.csv).

## Latent reversal

Observed latent both-cell screens were:

- MS: tirapazamine, flutamide, RO4929097.
- Crohn's: none.
- SLE: cytarabine, gemcitabine, cladribine.
- Radiation: none.
- Bone: pacritinib, brivanib, OTX015, BI-2536, ixabepilone.
- Muscle: pacritinib, BI-2536, brivanib.

Only **OTX015 and pacritinib in bone** overlapped Bridge expression and latent condition-level screens. Even these failed cross-study replication: both opposed some terrestrial bone vectors but reinforced the OSDR bone vector in latent space. Thus, the agreement is not evidence of a universal bone-loss reversal.

Across all drugs and contrast vectors, expression/latent Spearman correlations ranged from approximately `−0.51` to `+0.06`; sign agreement ranged from 15% to 56%. SLE showed the strongest disagreement (`r=−0.44` in DIPG6 and `−0.51` in SF8628). The two readouts capture substantially different aspects of the frozen representation.

Exact latent results are in [latent_reversal_by_contrast_cell.parquet](results/latent_reversal_by_contrast_cell.parquet), with agreement statistics in [expression_latent_agreement.csv](results/expression_latent_agreement.csv).

## Cross-study condition replication

No drug passed the prespecified strict rule in expression or latent space for radiation, bone, or muscle. That rule required positive reversal and empirical `p≤0.10` in both GSE264130 cell lines for all three independent condition datasets.

At the weaker sign-only level:

- Radiation expression: 7 Bridge versus 1 DE drugs opposed all three datasets in both cells.
- Bone expression: 4 Bridge versus 2 DE.
- Muscle expression: 4 Bridge versus 60 DE.
- Latent sign-only consistency was 0 for bone, 1 for muscle, and 153 for radiation, but none met the per-dataset empirical threshold. The broad radiation latent sign tendency is therefore a global geometric orientation, not selective drug evidence.

These findings reinforce the earlier conclusions: radiation is context dependent, and muscle does not show a consistent Bridge advantage.

Dataset/cell results are in [expression_reversal_by_dataset_cell.csv.gz](results/expression_reversal_by_dataset_cell.csv.gz) and [latent_reversal_by_dataset_cell.csv.gz](results/latent_reversal_by_dataset_cell.csv.gz).

## Cellular-context dependence

Bridge expression signs differed between DIPG6 and SF8628 for approximately:

- 49% of MS drugs,
- 45% of Crohn's drugs,
- 25% of SLE drugs,
- 50% of radiation drugs,
- 54% of bone drugs,
- 51% of muscle drugs.

The two glioma lines cannot be treated as interchangeable replicates. Scores were never averaged before individual-cell testing. Their shared disease origin also limits relevance to immune, intestinal, skeletal-muscle, bone, and normal-tissue radiation biology.

## Published and previously highlighted drugs

Only ibrutinib was assayed among the five highlighted MS drugs. Its Bridge expression direction was mixed between cells, DE reinforced in both, and latent opposition lacked the both-cell empirical threshold.

No highlighted Crohn's drug was assayed.

For SLE, gemcitabine, enzastaurin, sunitinib, and cladribine were assayed; fostamatinib was absent. Gemcitabine and cladribine were strong latent screens in both cells, but Bridge expression either reinforced or was mixed. Enzastaurin and sunitinib did not produce consistent Bridge reversal.

Among the radiation hypotheses, afatinib, lapatinib, and neratinib were assayed. None passed an expression or latent both-cell screen. Pentoxifylline, DPP4 inhibitors, the named DNA-PK/CDK7 inhibitors, JQ1, and molibresib were not assayed and were not counted as failures.

The exact audit is [highlighted_drug_audit.csv](results/highlighted_drug_audit.csv).

## ChEMBL target integration

Frozen expanded-ChEMBL target overlap was sparse among the GSE264130 candidate union:

- Bone pacritinib: MAP4K4 overlap; ChEMBL enrichment `q=1`.
- Bone ixabepilone: TUBA1B/TUBA1C/TUBB3 overlap; best `q=0.381`.
- Crohn's panobinostat: PIK3C3 overlap; `q=1`.
- SLE pacritinib: CSF1R overlap; `q=1`.

Target overlap therefore does not independently validate the observed direction. Pacritinib in bone is the only drug with Bridge expression reversal, latent reversal, and direct frozen target overlap, but it fails OSDR latent replication, lacks significant ChEMBL enrichment, and was not matched in the frozen LINCS analysis.

## Secondary LINCS cross-assay comparison

Between 166 and 172 drugs per condition were name-matched across GSE264130 and the frozen LINCS analysis. Bridge rank correlations were weak (`−0.125` to `+0.180`), direction agreement ranged from about 39% to 59%, and top-20 overlap ranged from 0 to 6 drugs. This is not strong cross-assay replication.

Examples of same-direction Bridge opposition include radiation bortezomib and palbociclib and bone brivanib/BI-2536. Panobinostat's Crohn's Bridge expression result opposed the frozen LINCS direction. SLE gemcitabine and cladribine agreed with LINCS in latent direction but not with Bridge expression direction.

No composite score was created. Detailed matches are in [rnaseq_lincs_drug_agreement.csv.gz](results/rnaseq_lincs_drug_agreement.csv.gz) and [rnaseq_lincs_agreement_summary.csv](results/rnaseq_lincs_agreement_summary.csv).

## Strongest hypotheses—without efficacy claims

1. **Bone pacritinib–MAP4K4:** both-cell Bridge expression and latent opposition plus target overlap. Weakened by nonsignificant ChEMBL enrichment, absent strict cross-study support, and OSDR latent reinforcement.
2. **Bone OTX015/BET-family perturbation:** both-cell expression and latent opposition. Weakened by absent direct frozen target support, no LINCS name match, and OSDR latent reinforcement.
3. **Radiation bortezomib and palbociclib:** both-cell Bridge expression opposition with same-direction LINCS evidence. Neither has latent or strict cross-study support in this analysis.
4. **SLE gemcitabine and cladribine:** latent opposition, LINCS direction agreement, and prior highlighted-drug/literature status. Bridge expression does not agree, so the evidence is representation dependent.
5. **Crohn's panobinostat–PIK3C3:** both-cell Bridge expression opposition and target overlap, but LINCS points the other way and ChEMBL enrichment is nonsignificant.

These are mechanistic hypotheses in two diffuse-midline-glioma cell lines, not therapeutic recommendations.

## Answers to the final questions

1. **Do observed drugs reverse Bridge programs?** Yes, a small number meet the unadjusted both-cell screen; none is FDR significant.
2. **Beyond DE?** Yes descriptively: all 14 Bridge screens are Bridge-only within their respective condition. There is no evidence they are globally more significant or more cross-study reproducible.
3. **Does latent agree with expression?** Generally no. Only two bone drugs overlap at condition level, and neither replicates across all bone datasets.
4. **Which reproduce across terrestrial and OSDR states?** None under the frozen strict statistical rule.
5. **Cell dependence?** High: roughly one quarter to over one half of Bridge drug signs differ between the two cells, depending on condition.
6. **RNA-seq–LINCS agreement?** Weak, with small rank correlations and low top-20 overlap.
7. **Does ChEMBL agree?** Only sparsely, and none of the supporting target enrichments is FDR significant.
8. **Strongest combined evidence?** Pacritinib and OTX015 in bone, bortezomib/palbociclib in radiation, and gemcitabine/cladribine in SLE—each with important discordant layers.
9. **Does directionality improve interpretation?** Yes. It exposes reinforcement, cell dependence, expression/latent disagreement, and failures of cross-study replication that target overlap alone concealed. It narrows rather than strengthens most earlier drug claims.

## Bottom line

Direction-aware observed RNA-seq is a materially stronger audit than target overlap alone. Its main contribution here is negative discrimination: it shows that many apparently attractive target or LINCS relationships are context dependent, representation dependent, or nonreplicating. Bridge adds complementary drug-response information beyond DE, but no candidate currently has sufficient multi-layer, cross-study, and multiplicity-corrected support to be called validated.
