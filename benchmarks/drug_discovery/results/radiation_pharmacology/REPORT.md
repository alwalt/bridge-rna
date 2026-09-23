# Pharmacology of the reproducible Bridge radiation programs

## Executive conclusion

The frozen gene-level result is substantially stronger than the pharmacology layer. None of the four fixed Bridge gene sets produces a drug enrichment passing BH FDR across the 366 eligible expanded-ChEMBL drugs. The broader core's best association is ocriplasmin (p=2.27e-4, q=0.083), while the matched-fibroblast minimum q is 0.977 and the other two sets have q=1. The mappings are therefore hypothesis generators, not therapeutic predictions.

The most interpretable candidates are not necessarily the highest ranked. Ocriplasmin and bacterial collagenase reflect a multi-gene extracellular-matrix program but lack radiation-specific evidence. Pentoxifylline/PDE5A has human combination-trial evidence for radiation fibrosis mitigation despite a weak enrichment rank. Omipalisib/PRKDC, CDK7 inhibitors, and ERBB2-directed drugs have target-compatible evidence mainly as tumor radiosensitizers, implying possible harm to irradiated normal tissue rather than protection.

## Fixed sets and targetability

| Fixed Bridge set | Genes | ChEMBL-targetable | Connected under >10-target rule | Drugs with overlap | Multi-gene drugs | Minimum p | Minimum q |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict radiation core | 15 | 2 | 1 | 1 | 0 | 0.035 | 1 |
| Cross-study radiation core | 237 | 47 | 23 | 58 | 8 | 0.000227 | 0.083 |
| All-three-study Bridge-only | 84 | 15 | 10 | 22 | 1 | 0.0368 | 1 |
| Matched fibroblast program | 258 | 43 | 22 | 116 | 18 | 0.00314 | 0.977 |

All ChEMBL-targetable genes, including those excluded by the >10-target drug rule, are saved in `targetable_genes_by_set.csv`. Exact fixed memberships are in `fixed_gene_set_membership.csv.gz`.

## Strict radiation core

| Rank | Drug | Supporting Bridge genes | Support | Targets | Overlap | p | q |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | OCRIPLASMIN | LAMB2 | single recurrent target | 36 | 1 | 0.035 | 1 |

Only CPT1A and LAMB2 are ChEMBL-targetable; only LAMB2 connects to an eligible drug. Ocriplasmin is therefore a single-target strict-core result despite its five-gene support in the broader core.

## Cross-study radiation core

| Rank | Drug | Supporting Bridge genes | Support | Targets | Overlap | p | q |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | OCRIPLASMIN | COL1A2;COL4A2;LAMB1;LAMB2;LAMC2 | multiple recurrent targets | 36 | 5 | 0.000227 | 0.083 |
| 2 | TOSEDOSTAT | ANPEP;MMP2 | multiple recurrent targets | 20 | 2 | 0.0384 | 1 |
| 3 | COLLAGENASE CLOSTRIDIUM HISTOLYTICUM | COL1A2;COL4A2 | multiple recurrent targets | 24 | 2 | 0.0536 | 1 |
| 4 | MOLIBRESIB | ACTN4;LMNB1;RUNX1;SMC4;TCOF1;UBTF | multiple recurrent targets | 226 | 6 | 0.144 | 1 |
| 5 | OMIPALISIB | PRKDC | single recurrent target | 11 | 1 | 0.159 | 1 |
| 6 | LUTEOLIN | DPP4 | single recurrent target | 12 | 1 | 0.172 | 1 |
| 7 | RG-547 | CDK7 | single recurrent target | 12 | 1 | 0.172 | 1 |
| 8 | RESVERATROL | DPP4 | single recurrent target | 13 | 1 | 0.185 | 1 |
| 9 | ANG1005 | LRP1 | single recurrent target | 14 | 1 | 0.198 | 1 |
| 10 | ANVATABART OPADOTIN | ERBB2 | single recurrent target | 14 | 1 | 0.198 | 1 |
| 11 | DISITAMAB VEDOTIN | ERBB2 | single recurrent target | 14 | 1 | 0.198 | 1 |
| 12 | TRASTUZUMAB BOTIDOTIN | ERBB2 | single recurrent target | 14 | 1 | 0.198 | 1 |
| 13 | TRASTUZUMAB EMTANSINE | ERBB2 | single recurrent target | 14 | 1 | 0.198 | 1 |
| 14 | SORAFENIB | CDK7;ERBB2 | multiple recurrent targets | 54 | 2 | 0.207 | 1 |
| 15 | AFATINIB | ERBB2 | single recurrent target | 15 | 1 | 0.211 | 1 |

Ocriplasmin is supported by five ECM genes and is the only nominally strong multi-target result. Tosedostat and collagenase have two targets; molibresib has six binding-supported overlaps, but those overlaps do not represent its canonical BET target mechanism. Most ERBB2, CDK7, DPP4, PDE5A and PRKDC drugs are single-hub results.

## All-three-study recurrent Bridge-only genes

| Rank | Drug | Supporting Bridge genes | Support | Targets | Overlap | p | q |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | MOLIBRESIB | LMNB1;RUNX1;RUVBL1;UBTF | multiple recurrent targets | 226 | 4 | 0.0368 | 1 |
| 2 | OMIPALISIB | PRKDC | single recurrent target | 11 | 1 | 0.0593 | 1 |
| 3 | ENTINOSTAT | KDM1A | single recurrent target | 12 | 1 | 0.0645 | 1 |
| 4 | LUTEOLIN | DPP4 | single recurrent target | 12 | 1 | 0.0645 | 1 |
| 5 | RG-547 | CDK7 | single recurrent target | 12 | 1 | 0.0645 | 1 |
| 6 | RESVERATROL | DPP4 | single recurrent target | 13 | 1 | 0.0697 | 1 |
| 7 | PACLITAXEL | KDM1A | single recurrent target | 14 | 1 | 0.0748 | 1 |
| 8 | DACTOLISIB | PRKDC | single recurrent target | 17 | 1 | 0.0902 | 1 |
| 9 | RONICICLIB | CDK7 | single recurrent target | 20 | 1 | 0.105 | 1 |
| 10 | DINACICLIB | CDK7 | single recurrent target | 21 | 1 | 0.11 | 1 |
| 11 | ZOTIRACICLIB | CDK7 | single recurrent target | 22 | 1 | 0.115 | 1 |
| 12 | AT-7519 | CDK7 | single recurrent target | 25 | 1 | 0.13 | 1 |
| 13 | PENTOXIFYLLINE | PDE5A | single recurrent target | 25 | 1 | 0.13 | 1 |
| 14 | DIPYRIDAMOLE | PDE5A | single recurrent target | 27 | 1 | 0.139 | 1 |
| 15 | ABEMACICLIB | CDK7 | single recurrent target | 33 | 1 | 0.168 | 1 |

Molibresib is the sole multi-target drug in this set. CDK7 alone generates 12 eligible drug mappings, so much of the remaining list is hub amplification. Pentoxifylline and dipyridamole are single-PDE5A mappings.

## Matched terrestrial–OSDR fibroblast program

| Rank | Drug | Supporting Bridge genes | Support | Targets | Overlap | p | q |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | OCRIPLASMIN | COL1A2;COL6A1;COL6A2;LAMB2 | multiple recurrent targets | 36 | 4 | 0.00314 | 0.977 |
| 2 | COLLAGENASE CLOSTRIDIUM HISTOLYTICUM | COL1A2;COL6A1;COL6A2 | multiple recurrent targets | 24 | 3 | 0.00756 | 0.977 |
| 3 | CAPIVASERTIB | LIMK1;LIMK2 | multiple recurrent targets | 13 | 2 | 0.0199 | 0.977 |
| 4 | ANVATABART OPADOTIN | ERBB2;TUBB6 | multiple recurrent targets | 14 | 2 | 0.0229 | 0.977 |
| 5 | DISITAMAB VEDOTIN | ERBB2;TUBB6 | multiple recurrent targets | 14 | 2 | 0.0229 | 0.977 |
| 6 | ENCORAFENIB | LIMK1;LIMK2 | multiple recurrent targets | 14 | 2 | 0.0229 | 0.977 |
| 7 | TRASTUZUMAB BOTIDOTIN | ERBB2;TUBB6 | multiple recurrent targets | 14 | 2 | 0.0229 | 0.977 |
| 8 | TRASTUZUMAB EMTANSINE | ERBB2;TUBB6 | multiple recurrent targets | 14 | 2 | 0.0229 | 0.977 |
| 9 | DABRAFENIB | LIMK1;LIMK2 | multiple recurrent targets | 34 | 2 | 0.114 | 0.977 |
| 10 | DASATINIB ANHYDROUS | ERBB2;LIMK1;LIMK2 | multiple recurrent targets | 82 | 3 | 0.164 | 0.977 |
| 11 | OMIPALISIB | PRKDC | single recurrent target | 11 | 1 | 0.172 | 0.977 |
| 12 | PAZOPANIB | LIMK1;LIMK2 | multiple recurrent targets | 46 | 2 | 0.184 | 0.977 |
| 13 | RG-547 | CDK7 | single recurrent target | 12 | 1 | 0.186 | 0.977 |
| 14 | MOLIBRESIB | ACTN4;KPNA6;LMNB1;RUNX1;SMC4;TCOF1 | multiple recurrent targets | 226 | 6 | 0.188 | 0.977 |
| 15 | AZINTUXIZUMAB VEDOTIN | TUBB6 | single recurrent target | 13 | 1 | 0.2 | 0.977 |

This program contains 258 genes: the frozen NHDF top-500 intersected with at least one frozen OSD-993 fibroblast module. Its leading signals are multi-gene ECM proteases, LIMK1/2 binding edges, ERBB2/TUBB6 antibody-drug conjugates, and a large TUBB6 microtubule-drug family. No enrichment survives FDR.

## Literature audit

The audit covered the union of the top five nonzero drugs from each fixed set, plus pentoxifylline, resveratrol and afatinib because recurrent PDE5A, DPP4 and ERBB2 targets have direct radiation relevance. Searches included radioprotection, post-exposure mitigation, injury, fibrosis, senescence, DNA-damage response, oxidative stress and normal-tissue toxicity. Empty exact-drug searches are retained in the raw search snapshots; absence means no radiation-specific evidence was found in this search, not proof that none exists.

| Drug | Evidence scope/class | Radiation interpretation | Target compatibility |
|---|---|---|---|
| OCRIPLASMIN | exact; 6 no radiation-specific evidence found | no radiation-specific evidence | LAMB2/collagen/laminin hydrolysis is directionally interpretable as ECM degradation, but benefit versus tissue damage is unknown |
| COLLAGENASE CLOSTRIDIUM HISTOLYTICUM | exact; 6 no radiation-specific evidence found | no radiation-specific evidence | collagen hydrolysis matches ECM targets but does not establish safe reversal of irradiated tissue fibrosis |
| TOSEDOSTAT | exact; 6 no radiation-specific evidence found | no radiation-specific evidence | ANPEP inhibition is curated; MMP2 is binding-only; no radiation direction established |
| MOLIBRESIB | class-level only; 3 animal radiation model | potentially protective/mitigating at BET-class level | JQ1 reduced radiation lung fibrosis through BET/BRD4, but the ChEMBL overlap is LMNB1/RUNX1/RUVBL1/UBTF rather than BRD4 and no exact molibresib radiation study was found |
| OMIPALISIB | exact; 4 cell radiation model | potentially sensitizing/harmful | exact drug suppresses irradiation-induced DNA-PKcs phosphorylation/NHEJ; ChEMBL overlap is PRKDC |
| ENTINOSTAT | exact; 6 no radiation-specific evidence found | no radiation-specific evidence | ranking edge is KDM1A binding, not entinostat's canonical HDAC mechanism |
| LUTEOLIN | exact; 3 animal radiation model; 4 cell radiation model | direction unclear/context dependent | normal vascular protection was attributed to HSPB1/SLC7A11/GPX4 and tumor radiosensitization to p38/ROS; neither directly validates the ChEMBL DPP4 overlap |
| RG-547 | CDK7-class only; 3 animal radiation model; 4 cell radiation model | potentially sensitizing/harmful | CDK7 inhibition impairs DNA-repair transcription and sensitizes tumor cells/xenografts; no exact RG-547 radiation study found |
| RESVERATROL | exact; 3 animal radiation model | direction unclear/context dependent | resveratrol protected hematopoietic stem cells via SIRT1/NOX4/ROS, not the ranked DPP4 edge |
| CAPIVASERTIB | exact; 6 no radiation-specific evidence found | no radiation-specific evidence | ranked LIMK1/2 edges are binding evidence and do not represent capivasertib's canonical AKT mechanism |
| ANVATABART OPADOTIN | exact; 6 no radiation-specific evidence found | no radiation-specific evidence | ERBB2 binding/TUBB6 payload mechanism is plausible for tumor cytotoxicity, not normal-tissue mitigation |
| DISITAMAB VEDOTIN | exact; 1 human clinical | potentially sensitizing/harmful | HER2-directed ADC plus microtubule-disrupting payload aligns with ERBB2/TUBB6 overlap; clinical evidence concerns antitumor combination, not injury mitigation |
| PENTOXIFYLLINE | exact combination evidence; 1 human clinical | potentially protective/mitigating | PDE5A inhibition is directionally compatible with increased cyclic-nucleotide signaling, but clinical benefit was shown with vitamin E and pentoxifylline alone was ineffective in the small placebo-controlled trial |
| AFATINIB | exact; 3 animal radiation model; 4 cell radiation model | potentially sensitizing/harmful | ERBB2/ErbB inhibition is compatible with reduced DNA repair and tumor radiosensitization |

### Independent evidence highlights

- **Pentoxifylline/PDE5A:** two small randomized human studies reported reduced or prevented radiation fibrosis with pentoxifylline plus vitamin E. The 2003 placebo-controlled study also found that either component alone was ineffective, so this is combination-specific evidence rather than proof for PDE5A inhibition alone [R1, R2].
- **Omipalisib/PRKDC:** exact-drug cell work showed suppression of radiation-induced DNA-PKcs phosphorylation and non-homologous end joining, producing radiosensitization [R4]. This agrees with established DNA-PK inhibitor biology [R16], but supports sensitization rather than normal-tissue protection.
- **CDK7:** genetic and chemical inhibition impaired DNA-repair transcription and sensitized medulloblastoma cells and xenografts to radiation [R5]. This supports the recurrent CDK7 target, not RG-547 specifically.
- **ERBB2/TUBB6:** afatinib, trastuzumab and HER2-directed microtubule ADCs sensitize tumor models to radiation [R11–R14]. A six-patient phase-II first stage combined disitamab vedotin, anti-PD-1 and radiotherapy with tolerable short-term safety and antitumor responses [R15]. None demonstrates normal-tissue mitigation.
- **Luteolin and resveratrol:** both have independent radiation-model evidence, including normal-tissue protection, but the reported mechanisms (HSPB1/SLC7A11/GPX4 and SIRT1/NOX4) do not validate their ChEMBL DPP4 overlap. Luteolin also radiosensitized tumor models, making direction context dependent [R6–R8].
- **DPP4 target:** sitagliptin protected mouse intestinal and hematopoietic systems after irradiation [R9, R10]. Sitagliptin is not an eligible >10-target ranked drug here; this is target-level external support only.
- **BET class:** JQ1 attenuated radiation lung fibrosis in an animal model while sensitizing tumor cells [R3]. Molibresib lacks exact radiation evidence, and its ranked Bridge overlaps are not BET-family targets.

## Hub versus multi-target structure

- **Genuine multi-target program mappings:** ocriplasmin (five cross-study ECM genes; four matched-fibroblast ECM genes), collagenase (two/three collagen genes), and the matched-fibroblast LIMK1/2 pairs. These are the clearest program-level ChEMBL mappings, although none is FDR significant.
- **Multi-overlap but mechanism-misaligned:** molibresib's six/four binding overlaps do not represent its canonical BET targets; treating this as six independent mechanistic supports would be misleading.
- **Single-hub amplification:** CDK7 creates 12 eligible mappings; ERBB2 creates numerous kinase inhibitors and ADCs; TUBB6 creates a large microtubule-drug family; PRKDC, PDE5A and DPP4 generate smaller single-target families.
- **Study convergence:** every gene in the all-three-study Bridge-only set is independently prioritized across all three radiation studies, but a drug attached to one such gene remains single-target evidence. Study recurrence and multi-target pharmacology are separate axes.

## Prioritized drug–target hypotheses

1. **Pentoxifylline–PDE5A / cyclic-nucleotide signaling:** highest translational interest because the target recurs across all three studies and human combination trials support fibrosis mitigation. Limitations: rank 13 in the all-three-study set, one overlapping gene, q=1, and benefit cannot be assigned to pentoxifylline or PDE5A alone.
2. **Omipalisib–PRKDC/DNA-PK:** strongest exact drug–overlap-target–radiation mechanism. It should be framed as a radiosensitization hypothesis and potential normal-tissue hazard, not a radioprotector; q=1.
3. **CDK7 inhibitors–DNA-repair transcription:** CDK7 is recurrent Bridge-only across all three studies and target-level tumor radiation evidence is strong. Exact ranked drugs require validation, and inhibition is expected to sensitize rather than protect.
4. **Ocriplasmin–laminin/collagen ECM program:** strongest multi-target enrichment and reproduced in matched fibroblasts. It is statistically suggestive only (cross-study q=0.083), has no located radiation-specific evidence, and extracellular proteolysis could mitigate or worsen injury.
5. **DPP4-axis modulation:** DPP4 is recurrent Bridge-only across all three studies, and a selective DPP4 inhibitor protects irradiated mice. The ranked luteolin/resveratrol edges are target-incompatible with their reported radiation mechanisms, so a prospective selective-DPP4 analysis is more defensible than interpreting those drugs as validated hits.
6. **ERBB2-directed inhibition/ADCs:** recurrent in the broader and fibroblast programs with substantial tumor radiosensitization evidence. This is an antitumor sensitization hypothesis with possible normal-tissue risk; it does not support radiation-injury mitigation.
7. **BET inhibition and radiation fibrosis:** animal evidence makes the class scientifically interesting, but molibresib's ChEMBL overlap targets do not match BET biology and no exact-drug evidence was found. This is a low-confidence bridge from the current enrichment.

No item above demonstrates therapeutic efficacy. Drug-target overlap does not establish whether the Bridge gene should be activated or inhibited, whether drug exposure reaches the relevant tissue, or whether antitumor sensitization is safe for normal tissue.

## Reproducibility files

- `bridge_gene_drug_mechanism_mapping.csv.gz`: exact Bridge gene → eligible drug → ChEMBL evidence/action → radiation contexts.
- `drug_support_compact.csv`: exact drug → supporting genes/datasets → p/FDR/rank.
- `drug_enrichment_all.csv.gz` and `drug_enrichment_nonzero.csv`: all 366 eligible-drug tests and nonzero subset.
- `fixed_gene_set_membership.csv.gz` and `targetable_genes_by_set.csv`: frozen definitions and all targetable genes.
- `literature_evidence_audit.csv` and `literature_references.csv`: classifications and primary references.
- `literature_search_provenance.json`: candidate selection, evidence hierarchy, search date and limitations.
- `literature_search_europepmc_raw.json`, `literature_search_targeted_raw.json`, `literature_search_exact_raw.json`, and `literature_selected_records_raw.json`: exact saved search records.
- `provenance.json`: input hashes, statistics and protected-step audit.

## References

- **R1.** [Randomized, placebo-controlled trial of combined pentoxifylline and tocopherol for regression of superficial radiation-induced fibrosis](https://pubmed.ncbi.nlm.nih.gov/12829674/) (2003; PMID 12829674).
- **R2.** [Randomized trial of pentoxifylline and vitamin E after breast irradiation](https://pubmed.ncbi.nlm.nih.gov/22846413/) (2013; PMID 22846413).
- **R3.** [Pharmacological targeting of BET proteins attenuates radiation-induced lung fibrosis](https://pubmed.ncbi.nlm.nih.gov/29343723/) (2018; PMID 29343723).
- **R4.** [The PI3K/mTOR inhibitor omipalisib suppresses nonhomologous end joining and sensitizes cancer cells to radio- and chemotherapy](https://pubmed.ncbi.nlm.nih.gov/34330845/) (2021; PMID 34330845).
- **R5.** [Transcriptional control of DNA repair networks by CDK7 regulates sensitivity to radiation in MYC-driven medulloblastoma](https://pubmed.ncbi.nlm.nih.gov/33910002/) (2021; PMID 33910002).
- **R6.** [Luteolin target HSPB1 regulates endothelial cell ferroptosis to protect against radiation vascular injury](https://pubmed.ncbi.nlm.nih.gov/39392831/) (2024; PMID 39392831).
- **R7.** [Luteolin acts as a radiosensitizer in non-small cell lung cancer cells](https://pubmed.ncbi.nlm.nih.gov/25586525/) (2015; PMID 25586525).
- **R8.** [Resveratrol ameliorates irradiation-induced long-term hematopoietic stem-cell injury in mice](https://pubmed.ncbi.nlm.nih.gov/23124026/) (2013; PMID 23124026).
- **R9.** [Sitagliptin alleviates radiation-induced intestinal injury](https://pubmed.ncbi.nlm.nih.gov/35620578/) (2022; PMID 35620578).
- **R10.** [Sitagliptin mitigates total-body-irradiation-induced hematopoietic injury in mice](https://pubmed.ncbi.nlm.nih.gov/32774687/) (2020; PMID 32774687).
- **R11.** [Afatinib promotes radiosensitivity in nasopharyngeal carcinoma](https://pubmed.ncbi.nlm.nih.gov/36678534/) (2023; PMID 36678534).
- **R12.** [Afatinib radiosensitizes head and neck squamous-cell carcinoma cells](https://pubmed.ncbi.nlm.nih.gov/28423495/) (2017; PMID 28423495).
- **R13.** [Anti-tubulin drugs conjugated to anti-ErbB antibodies selectively radiosensitize](https://pubmed.ncbi.nlm.nih.gov/27698471/) (2016; PMID 27698471).
- **R14.** [Trastuzumab enables radiosensitization in HER2-positive breast cancer](https://pubmed.ncbi.nlm.nih.gov/35205763/) (2022; PMID 35205763).
- **R15.** [Disitamab vedotin, toripalimab and radiotherapy for bladder preservation: proof-of-concept study](https://pubmed.ncbi.nlm.nih.gov/40774226/) (2025; PMID 40774226).
- **R16.** [Preclinical evaluation of the DNA-PK inhibitor NU7441](https://pubmed.ncbi.nlm.nih.gov/16707462/) (2006; PMID 16707462).
