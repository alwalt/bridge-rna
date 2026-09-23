#!/usr/bin/env python3
"""Build literature audit and report for the frozen radiation pharmacology layer."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/radiation_pharmacology"

REFERENCES = [
    ("R1", "12829674", "Randomized, placebo-controlled trial of combined pentoxifylline and tocopherol for regression of superficial radiation-induced fibrosis", 2003),
    ("R2", "22846413", "Randomized trial of pentoxifylline and vitamin E after breast irradiation", 2013),
    ("R3", "29343723", "Pharmacological targeting of BET proteins attenuates radiation-induced lung fibrosis", 2018),
    ("R4", "34330845", "The PI3K/mTOR inhibitor omipalisib suppresses nonhomologous end joining and sensitizes cancer cells to radio- and chemotherapy", 2021),
    ("R5", "33910002", "Transcriptional control of DNA repair networks by CDK7 regulates sensitivity to radiation in MYC-driven medulloblastoma", 2021),
    ("R6", "39392831", "Luteolin target HSPB1 regulates endothelial cell ferroptosis to protect against radiation vascular injury", 2024),
    ("R7", "25586525", "Luteolin acts as a radiosensitizer in non-small cell lung cancer cells", 2015),
    ("R8", "23124026", "Resveratrol ameliorates irradiation-induced long-term hematopoietic stem-cell injury in mice", 2013),
    ("R9", "35620578", "Sitagliptin alleviates radiation-induced intestinal injury", 2022),
    ("R10", "32774687", "Sitagliptin mitigates total-body-irradiation-induced hematopoietic injury in mice", 2020),
    ("R11", "36678534", "Afatinib promotes radiosensitivity in nasopharyngeal carcinoma", 2023),
    ("R12", "28423495", "Afatinib radiosensitizes head and neck squamous-cell carcinoma cells", 2017),
    ("R13", "27698471", "Anti-tubulin drugs conjugated to anti-ErbB antibodies selectively radiosensitize", 2016),
    ("R14", "35205763", "Trastuzumab enables radiosensitization in HER2-positive breast cancer", 2022),
    ("R15", "40774226", "Disitamab vedotin, toripalimab and radiotherapy for bladder preservation: proof-of-concept study", 2025),
    ("R16", "16707462", "Preclinical evaluation of the DNA-PK inhibitor NU7441", 2006),
]

AUDIT = [
    # drug, exact/class, evidence class, direction, target compatibility, refs, conclusion
    ("OCRIPLASMIN", "exact", "6 no radiation-specific evidence found", "no radiation-specific evidence", "LAMB2/collagen/laminin hydrolysis is directionally interpretable as ECM degradation, but benefit versus tissue damage is unknown", "", "Top statistical association; unsupported as a radiation intervention."),
    ("COLLAGENASE CLOSTRIDIUM HISTOLYTICUM", "exact", "6 no radiation-specific evidence found", "no radiation-specific evidence", "collagen hydrolysis matches ECM targets but does not establish safe reversal of irradiated tissue fibrosis", "", "Mechanistically legible ECM hit without radiation evidence."),
    ("TOSEDOSTAT", "exact", "6 no radiation-specific evidence found", "no radiation-specific evidence", "ANPEP inhibition is curated; MMP2 is binding-only; no radiation direction established", "", "Unsupported radiation candidate."),
    ("MOLIBRESIB", "class-level only", "3 animal radiation model", "potentially protective/mitigating at BET-class level", "JQ1 reduced radiation lung fibrosis through BET/BRD4, but the ChEMBL overlap is LMNB1/RUNX1/RUVBL1/UBTF rather than BRD4 and no exact molibresib radiation study was found", "R3", "Interesting BET-class hypothesis, but the ranking targets and literature mechanism do not align."),
    ("OMIPALISIB", "exact", "4 cell radiation model", "potentially sensitizing/harmful", "exact drug suppresses irradiation-induced DNA-PKcs phosphorylation/NHEJ; ChEMBL overlap is PRKDC", "R4;R16", "Strongest exact drug-target-radiation mechanistic match, but it is a radiosensitizer."),
    ("ENTINOSTAT", "exact", "6 no radiation-specific evidence found", "no radiation-specific evidence", "ranking edge is KDM1A binding, not entinostat's canonical HDAC mechanism", "", "Low-confidence target interpretation."),
    ("LUTEOLIN", "exact", "3 animal radiation model; 4 cell radiation model", "direction unclear/context dependent", "normal vascular protection was attributed to HSPB1/SLC7A11/GPX4 and tumor radiosensitization to p38/ROS; neither directly validates the ChEMBL DPP4 overlap", "R6;R7", "Independent radiation evidence exists, but not through the ranked DPP4 edge."),
    ("RG-547", "CDK7-class only", "3 animal radiation model; 4 cell radiation model", "potentially sensitizing/harmful", "CDK7 inhibition impairs DNA-repair transcription and sensitizes tumor cells/xenografts; no exact RG-547 radiation study found", "R5", "Target-level support for CDK7, not exact-drug support."),
    ("RESVERATROL", "exact", "3 animal radiation model", "direction unclear/context dependent", "resveratrol protected hematopoietic stem cells via SIRT1/NOX4/ROS, not the ranked DPP4 edge", "R8", "Protective evidence exists, but target compatibility is weak."),
    ("CAPIVASERTIB", "exact", "6 no radiation-specific evidence found", "no radiation-specific evidence", "ranked LIMK1/2 edges are binding evidence and do not represent capivasertib's canonical AKT mechanism", "", "Unsupported and target-mismatched radiation candidate."),
    ("ANVATABART OPADOTIN", "exact", "6 no radiation-specific evidence found", "no radiation-specific evidence", "ERBB2 binding/TUBB6 payload mechanism is plausible for tumor cytotoxicity, not normal-tissue mitigation", "", "No radiation-specific evidence found."),
    ("DISITAMAB VEDOTIN", "exact", "1 human clinical", "potentially sensitizing/harmful", "HER2-directed ADC plus microtubule-disrupting payload aligns with ERBB2/TUBB6 overlap; clinical evidence concerns antitumor combination, not injury mitigation", "R13;R15", "Early combination evidence; no normal-tissue protective claim."),
    ("PENTOXIFYLLINE", "exact combination evidence", "1 human clinical", "potentially protective/mitigating", "PDE5A inhibition is directionally compatible with increased cyclic-nucleotide signaling, but clinical benefit was shown with vitamin E and pentoxifylline alone was ineffective in the small placebo-controlled trial", "R1;R2", "Best human mitigation signal, but combination-specific and not FDR-enriched."),
    ("AFATINIB", "exact", "3 animal radiation model; 4 cell radiation model", "potentially sensitizing/harmful", "ERBB2/ErbB inhibition is compatible with reduced DNA repair and tumor radiosensitization", "R11;R12", "Target-compatible antitumor radiosensitizer, not a normal-tissue mitigator."),
]


def fmt(value):
    if pd.isna(value):
        return "—"
    return f"{value:.3g}"


def main():
    support = pd.read_csv(OUT / "drug_support_compact.csv")
    membership = pd.read_csv(OUT / "fixed_gene_set_membership.csv.gz")
    summary = pd.read_csv(OUT / "gene_set_summary.csv")
    refs = pd.DataFrame(REFERENCES, columns=["reference_id", "pmid", "title", "year"])
    refs["url"] = "https://pubmed.ncbi.nlm.nih.gov/" + refs.pmid + "/"
    refs.to_csv(OUT / "literature_references.csv", index=False)

    audit = pd.DataFrame(AUDIT, columns=["drug_name", "evidence_scope", "evidence_class",
                                        "radiation_direction", "target_direction_compatibility",
                                        "reference_ids", "audit_conclusion"])
    ranking = (support[support.drug_name.isin(audit.drug_name)]
               .pivot_table(index="drug_name", columns="gene_set", values="rank", aggfunc="min")
               .reset_index())
    audit = audit.merge(ranking, on="drug_name", how="left")
    audit.to_csv(OUT / "literature_evidence_audit.csv", index=False)

    target_lists = []
    for set_name, group in membership.groupby("gene_set"):
        target_lists.append((set_name,
                             "; ".join(group[group.chembl_targetable].gene),
                             "; ".join(group[group.eligible_drug_connected].gene)))
    pd.DataFrame(target_lists, columns=["gene_set", "all_chembl_targetable_genes",
                                        "eligible_drug_connected_genes"]).to_csv(
                                            OUT / "targetable_genes_by_set.csv", index=False)

    lines = ["# Pharmacology of the reproducible Bridge radiation programs", "",
             "## Executive conclusion", "",
             "The frozen gene-level result is substantially stronger than the pharmacology layer. None of the four fixed Bridge gene sets produces a drug enrichment passing BH FDR across the 366 eligible expanded-ChEMBL drugs. The broader core's best association is ocriplasmin (p=2.27e-4, q=0.083), while the matched-fibroblast minimum q is 0.977 and the other two sets have q=1. The mappings are therefore hypothesis generators, not therapeutic predictions.", "",
             "The most interpretable candidates are not necessarily the highest ranked. Ocriplasmin and bacterial collagenase reflect a multi-gene extracellular-matrix program but lack radiation-specific evidence. Pentoxifylline/PDE5A has human combination-trial evidence for radiation fibrosis mitigation despite a weak enrichment rank. Omipalisib/PRKDC, CDK7 inhibitors, and ERBB2-directed drugs have target-compatible evidence mainly as tumor radiosensitizers, implying possible harm to irradiated normal tissue rather than protection.", "",
             "## Fixed sets and targetability", "",
             "| Fixed Bridge set | Genes | ChEMBL-targetable | Connected under >10-target rule | Drugs with overlap | Multi-gene drugs | Minimum p | Minimum q |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"strict_radiation_core":"Strict radiation core", "cross_study_radiation_core":"Cross-study radiation core",
              "all_three_study_bridge_only":"All-three-study Bridge-only", "matched_fibroblast_program":"Matched fibroblast program"}
    for row in summary.itertuples():
        lines.append(f"| {labels[row.gene_set]} | {row.gene_count} | {row.chembl_targetable_gene_count} | {row.eligible_connected_gene_count} | {row.eligible_drugs_with_overlap} | {row.multi_gene_drugs} | {fmt(row.minimum_pvalue)} | {fmt(row.minimum_fdr)} |")
    lines += ["", "All ChEMBL-targetable genes, including those excluded by the >10-target drug rule, are saved in `targetable_genes_by_set.csv`. Exact fixed memberships are in `fixed_gene_set_membership.csv.gz`.", ""]

    sections = [
        ("strict_radiation_core", "Strict radiation core", 20),
        ("cross_study_radiation_core", "Cross-study radiation core", 15),
        ("all_three_study_bridge_only", "All-three-study recurrent Bridge-only genes", 15),
        ("matched_fibroblast_program", "Matched terrestrial–OSDR fibroblast program", 15),
    ]
    for set_name, title, number in sections:
        subset = support[support.gene_set.eq(set_name)].sort_values("rank").head(number)
        lines += [f"## {title}", "", "| Rank | Drug | Supporting Bridge genes | Support | Targets | Overlap | p | q |", "|---:|---|---|---|---:|---:|---:|---:|"]
        if not len(subset):
            lines.append("| — | No eligible drug overlap | — | — | — | — | — | — |")
        for row in subset.itertuples():
            lines.append(f"| {row.rank} | {row.drug_name} | {row.supporting_genes} | {row.support_type} | {row.target_count} | {row.overlap_count} | {row.pvalue:.3g} | {row.fdr:.3g} |")
        if set_name == "strict_radiation_core":
            lines += ["", "Only CPT1A and LAMB2 are ChEMBL-targetable; only LAMB2 connects to an eligible drug. Ocriplasmin is therefore a single-target strict-core result despite its five-gene support in the broader core."]
        elif set_name == "cross_study_radiation_core":
            lines += ["", "Ocriplasmin is supported by five ECM genes and is the only nominally strong multi-target result. Tosedostat and collagenase have two targets; molibresib has six binding-supported overlaps, but those overlaps do not represent its canonical BET target mechanism. Most ERBB2, CDK7, DPP4, PDE5A and PRKDC drugs are single-hub results."]
        elif set_name == "all_three_study_bridge_only":
            lines += ["", "Molibresib is the sole multi-target drug in this set. CDK7 alone generates 12 eligible drug mappings, so much of the remaining list is hub amplification. Pentoxifylline and dipyridamole are single-PDE5A mappings."]
        else:
            lines += ["", "This program contains 258 genes: the frozen NHDF top-500 intersected with at least one frozen OSD-993 fibroblast module. Its leading signals are multi-gene ECM proteases, LIMK1/2 binding edges, ERBB2/TUBB6 antibody-drug conjugates, and a large TUBB6 microtubule-drug family. No enrichment survives FDR."]
        lines.append("")

    lines += ["## Literature audit", "", "The audit covered the union of the top five nonzero drugs from each fixed set, plus pentoxifylline, resveratrol and afatinib because recurrent PDE5A, DPP4 and ERBB2 targets have direct radiation relevance. Searches included radioprotection, post-exposure mitigation, injury, fibrosis, senescence, DNA-damage response, oxidative stress and normal-tissue toxicity. Empty exact-drug searches are retained in the raw search snapshots; absence means no radiation-specific evidence was found in this search, not proof that none exists.", "",
              "| Drug | Evidence scope/class | Radiation interpretation | Target compatibility |",
              "|---|---|---|---|"]
    for row in audit.itertuples():
        lines.append(f"| {row.drug_name} | {row.evidence_scope}; {row.evidence_class} | {row.radiation_direction} | {row.target_direction_compatibility} |")

    lines += ["", "### Independent evidence highlights", "",
              "- **Pentoxifylline/PDE5A:** two small randomized human studies reported reduced or prevented radiation fibrosis with pentoxifylline plus vitamin E. The 2003 placebo-controlled study also found that either component alone was ineffective, so this is combination-specific evidence rather than proof for PDE5A inhibition alone [R1, R2].",
              "- **Omipalisib/PRKDC:** exact-drug cell work showed suppression of radiation-induced DNA-PKcs phosphorylation and non-homologous end joining, producing radiosensitization [R4]. This agrees with established DNA-PK inhibitor biology [R16], but supports sensitization rather than normal-tissue protection.",
              "- **CDK7:** genetic and chemical inhibition impaired DNA-repair transcription and sensitized medulloblastoma cells and xenografts to radiation [R5]. This supports the recurrent CDK7 target, not RG-547 specifically.",
              "- **ERBB2/TUBB6:** afatinib, trastuzumab and HER2-directed microtubule ADCs sensitize tumor models to radiation [R11–R14]. A six-patient phase-II first stage combined disitamab vedotin, anti-PD-1 and radiotherapy with tolerable short-term safety and antitumor responses [R15]. None demonstrates normal-tissue mitigation.",
              "- **Luteolin and resveratrol:** both have independent radiation-model evidence, including normal-tissue protection, but the reported mechanisms (HSPB1/SLC7A11/GPX4 and SIRT1/NOX4) do not validate their ChEMBL DPP4 overlap. Luteolin also radiosensitized tumor models, making direction context dependent [R6–R8].",
              "- **DPP4 target:** sitagliptin protected mouse intestinal and hematopoietic systems after irradiation [R9, R10]. Sitagliptin is not an eligible >10-target ranked drug here; this is target-level external support only.",
              "- **BET class:** JQ1 attenuated radiation lung fibrosis in an animal model while sensitizing tumor cells [R3]. Molibresib lacks exact radiation evidence, and its ranked Bridge overlaps are not BET-family targets.", ""]

    lines += ["## Hub versus multi-target structure", "",
              "- **Genuine multi-target program mappings:** ocriplasmin (five cross-study ECM genes; four matched-fibroblast ECM genes), collagenase (two/three collagen genes), and the matched-fibroblast LIMK1/2 pairs. These are the clearest program-level ChEMBL mappings, although none is FDR significant.",
              "- **Multi-overlap but mechanism-misaligned:** molibresib's six/four binding overlaps do not represent its canonical BET targets; treating this as six independent mechanistic supports would be misleading.",
              "- **Single-hub amplification:** CDK7 creates 12 eligible mappings; ERBB2 creates numerous kinase inhibitors and ADCs; TUBB6 creates a large microtubule-drug family; PRKDC, PDE5A and DPP4 generate smaller single-target families.",
              "- **Study convergence:** every gene in the all-three-study Bridge-only set is independently prioritized across all three radiation studies, but a drug attached to one such gene remains single-target evidence. Study recurrence and multi-target pharmacology are separate axes.", ""]

    lines += ["## Prioritized drug–target hypotheses", "",
              "1. **Pentoxifylline–PDE5A / cyclic-nucleotide signaling:** highest translational interest because the target recurs across all three studies and human combination trials support fibrosis mitigation. Limitations: rank 13 in the all-three-study set, one overlapping gene, q=1, and benefit cannot be assigned to pentoxifylline or PDE5A alone.",
              "2. **Omipalisib–PRKDC/DNA-PK:** strongest exact drug–overlap-target–radiation mechanism. It should be framed as a radiosensitization hypothesis and potential normal-tissue hazard, not a radioprotector; q=1.",
              "3. **CDK7 inhibitors–DNA-repair transcription:** CDK7 is recurrent Bridge-only across all three studies and target-level tumor radiation evidence is strong. Exact ranked drugs require validation, and inhibition is expected to sensitize rather than protect.",
              "4. **Ocriplasmin–laminin/collagen ECM program:** strongest multi-target enrichment and reproduced in matched fibroblasts. It is statistically suggestive only (cross-study q=0.083), has no located radiation-specific evidence, and extracellular proteolysis could mitigate or worsen injury.",
              "5. **DPP4-axis modulation:** DPP4 is recurrent Bridge-only across all three studies, and a selective DPP4 inhibitor protects irradiated mice. The ranked luteolin/resveratrol edges are target-incompatible with their reported radiation mechanisms, so a prospective selective-DPP4 analysis is more defensible than interpreting those drugs as validated hits.",
              "6. **ERBB2-directed inhibition/ADCs:** recurrent in the broader and fibroblast programs with substantial tumor radiosensitization evidence. This is an antitumor sensitization hypothesis with possible normal-tissue risk; it does not support radiation-injury mitigation.",
              "7. **BET inhibition and radiation fibrosis:** animal evidence makes the class scientifically interesting, but molibresib's ChEMBL overlap targets do not match BET biology and no exact-drug evidence was found. This is a low-confidence bridge from the current enrichment.", "",
              "No item above demonstrates therapeutic efficacy. Drug-target overlap does not establish whether the Bridge gene should be activated or inhibited, whether drug exposure reaches the relevant tissue, or whether antitumor sensitization is safe for normal tissue.", ""]

    lines += ["## Reproducibility files", "",
              "- `bridge_gene_drug_mechanism_mapping.csv.gz`: exact Bridge gene → eligible drug → ChEMBL evidence/action → radiation contexts.",
              "- `drug_support_compact.csv`: exact drug → supporting genes/datasets → p/FDR/rank.",
              "- `drug_enrichment_all.csv.gz` and `drug_enrichment_nonzero.csv`: all 366 eligible-drug tests and nonzero subset.",
              "- `fixed_gene_set_membership.csv.gz` and `targetable_genes_by_set.csv`: frozen definitions and all targetable genes.",
              "- `literature_evidence_audit.csv` and `literature_references.csv`: classifications and primary references.",
              "- `literature_search_provenance.json`: candidate selection, evidence hierarchy, search date and limitations.",
              "- `literature_search_europepmc_raw.json`, `literature_search_targeted_raw.json`, `literature_search_exact_raw.json`, and `literature_selected_records_raw.json`: exact saved search records.",
              "- `provenance.json`: input hashes, statistics and protected-step audit.", "",
              "## References", ""]
    for row in refs.itertuples():
        lines.append(f"- **{row.reference_id}.** [{row.title}]({row.url}) ({row.year}; PMID {row.pmid}).")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
