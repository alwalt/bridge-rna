#!/usr/bin/env python3
"""Create evidence-gap tables and a report from the Organ recovery experiment."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
OUT=RESULTS/"organ_recovery"
DATA_ROOT=ROOT.parents[1]/"data"
DB=DATA_ROOT/"archs4"/"bridge_metadata.sqlite"
ANATOMY_CATEGORIES={"tissue","organism part","organ","source organ","brain region","tissue type",
                    "tissue origin","tissue source","source tissue"}
STRUCTURED=re.compile(r"(?:^|[;|\n])\s*(?:tissue|organ|anatomical (?:site|source)|body site|sample source)\s*[:=]",re.I)


def pct(n:int,d:int)->str: return f"{100*n/d:.2f}%"


def main()->None:
    missing=pd.read_parquet(OUT/"missing_organ_recovery_predictions.parquet")
    gaps=missing[missing.mapping_status.isin(["ambiguous","unknown"])].copy()
    gaps["has_unmapped_gemma_anatomy"]=False
    gaps["has_structured_anatomy_key"]=False
    gaps["has_nonempty_sample_source"]=False
    gaps["sample_source_text"]=""
    gaps["sample_title"]=""
    unknown_gsms=gaps.loc[gaps.mapping_status.eq("unknown"),"gsm"].tolist()
    gemma=set(); structured=set(); nonempty=set(); sources={}; titles={}
    con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
    for start in range(0,len(unknown_gsms),700):
        batch=unknown_gsms[start:start+700]; marks=','.join('?'*len(batch))
        for gsm,source,chars,title in con.execute(
            f"SELECT gsm,source_name_ch1,characteristics_ch1,title FROM archs4_metadata WHERE gsm IN ({marks})",batch):
            sources[gsm]=(source or '')[:500];titles[gsm]=(title or '')[:500]
            if (source or '').strip(): nonempty.add(gsm)
            if STRUCTURED.search(chars or '') or STRUCTURED.search(source or ''): structured.add(gsm)
        for gsm,cat,val in con.execute(
            f"SELECT gsm,category,raw_value FROM gemma_annotations WHERE gsm IN ({marks})",batch):
            if (cat or '').casefold() in ANATOMY_CATEGORIES and (val or '').strip(): gemma.add(gsm)
    mask=gaps.mapping_status.eq("unknown")
    gaps.loc[mask,"has_unmapped_gemma_anatomy"]=gaps.loc[mask,"gsm"].isin(gemma)
    gaps.loc[mask,"has_structured_anatomy_key"]=gaps.loc[mask,"gsm"].isin(structured)
    gaps.loc[mask,"has_nonempty_sample_source"]=gaps.loc[mask,"gsm"].isin(nonempty)
    gaps.loc[mask,"sample_source_text"]=gaps.loc[mask,"gsm"].map(sources).fillna('')
    gaps.loc[mask,"sample_title"]=gaps.loc[mask,"gsm"].map(titles).fillna('')
    gaps["evidence_gap_class"]="ambiguous_recognized_candidates"
    u=gaps.mapping_status.eq("unknown")
    gaps.loc[u&gaps.has_unmapped_gemma_anatomy,"evidence_gap_class"]="unmapped_explicit_gemma_anatomy"
    gaps.loc[u&~gaps.has_unmapped_gemma_anatomy&gaps.has_structured_anatomy_key,"evidence_gap_class"]="structured_anatomy_outside_dictionary"
    gaps.loc[u&~gaps.has_unmapped_gemma_anatomy&~gaps.has_structured_anatomy_key&gaps.has_nonempty_sample_source,
             "evidence_gap_class"]="unrecognized_sample_source_text"
    gaps.loc[u&~gaps.has_unmapped_gemma_anatomy&~gaps.has_structured_anatomy_key&~gaps.has_nonempty_sample_source,
             "evidence_gap_class"]="no_sample_source_or_recognized_anatomy"
    gaps.to_parquet(OUT/"ambiguous_unknown_samples.parquet",index=False)
    gap_summary=gaps.groupby(["species","mapping_status","evidence_gap_class"],as_index=False).size().rename(columns={"size":"n"})
    gap_summary.to_csv(OUT/"anatomy_evidence_gap_summary.csv",index=False)

    val=pd.read_parquet(OUT/"deterministic_validation_predictions.parquet")
    d=val[(val.condition=="D")&(val.mapping_status=="resolved")]
    strict=d[d.evidence_source.eq("Gemma")|d.evidence_field.eq("source_name_ch1")]
    sample_direct=d[~d.evidence_field.str.contains("gse_|protocol|extract",na=False)]
    missing_resolved=missing[missing.mapping_status.eq("resolved")]
    def exact_phrase(row):
        phrases=[x.strip().casefold() for x in (row.matched_phrase or '').split('|')]
        return (row.proposed_canonical_organ or '').casefold() in phrases
    exact_direct=int(missing_resolved.apply(exact_phrase,axis=1).sum())
    synonym_additional=len(missing_resolved)-exact_direct
    pd.DataFrame([{"resolved_total":len(missing_resolved),"exact_geometa_phrase_n":exact_direct,
        "alias_or_uberon_phrase_additional_n":synonym_additional,
        "alias_share_of_resolved":synonym_additional/len(missing_resolved)}]).to_csv(
            OUT/"synonym_mapping_gain.csv",index=False)
    strict_missing=missing_resolved[missing_resolved.evidence_source.eq("Gemma")|missing_resolved.evidence_field.eq("source_name_ch1")]
    total=940455; existing=404266
    laya=pd.read_csv(OUT/"laya_shortlist_predictions.csv")
    short=laya[laya["mode"].eq("shortlist")]
    candidate=pd.read_csv(OUT/"candidate_recall_analysis.csv").iloc[0]
    deterministic=pd.read_csv(OUT/"deterministic_validation_summary.csv")
    no_direct=int((missing.mapping_status.eq("unknown")).sum())
    strict_no_source=int(((gaps.evidence_gap_class=="no_sample_source_or_recognized_anatomy") & gaps.mapping_status.eq("unknown")).sum())
    no_machine_evidence=int((gaps.mapping_status.eq("unknown") & gaps.evidence_gap_class.eq("unrecognized_sample_source_text")).sum())
    unmapped_raw=int((gaps.mapping_status.eq("unknown") & gaps.evidence_gap_class.isin([
        "unmapped_explicit_gemma_anatomy","structured_anatomy_outside_dictionary"])).sum())
    report=f"""# Conservative Organ recovery audit

## Outcome

The current database has **{total:,}** samples, **{existing:,}** canonical Organ labels ({pct(existing,total)}), and **{total-existing:,}** missing canonical labels. Missing canonical metadata is not the same as missing evidence: the direct GEOMeta/UBERON synonym system found one or more recognized anatomical candidates in **{len(missing)-no_direct:,}** currently missing samples. It resolved **{len(missing_resolved):,}**, marked **{(missing.mapping_status=='ambiguous').sum():,}** ambiguous, and found no recognized 54-category alias in **{no_direct:,}**.

The last number is an intentionally conservative lexical gap, not proof that all free text is biologically empty. Among these unknowns, **{unmapped_raw:,}** retain explicit but unmapped Gemma/structured anatomy evidence. The remaining **{no_machine_evidence:,}** have no machine-recognized direct or structured anatomy, although all have a non-empty ARCHS4 sample-source string. Thus zero records are literally devoid of source text, but this audit cannot honestly claim those strings are usable anatomy without broader ontology mapping or manual review. The evidence-gap table preserves them for that review.

## Existing Organ state

- Human: 384,203 present; 126,506 missing.
- Mouse: 20,063 present; 409,683 missing.
- Existing source: 383,647 GEOMeta; 20,619 Gemma; zero promoted deterministic/model labels.
- All proposed labels in this experiment are separate files; the SQLite database was opened read-only.

## Vocabulary and ontology

All 54 observed GEOMeta Organ labels are inventoried. **52/54** have a reliable UBERON correspondence. `Head and Neck` and `Nasal` remain deliberately unmapped because neither denotes one clean anatomical structure. Brain subcategories and several wording-different categories are marked `category_correspondence`, not falsely called exact. The synonym table separates direct anatomical aliases from any future cell-type-to-organ inference; no such inference was used here.

## Held-out deterministic validation (5,000 GEOMeta samples)

| Evidence condition | Coverage | Precision among mapped | Ambiguous | Unknown |
|---|---:|---:|---:|---:|
"""
    names={"A":"ARCHS4 only","B":"+ GEO GSM extras","C":"+ GSE context","D":"+ Gemma anatomy"}
    for row in deterministic.itertuples(index=False):
        report+=f"| {names[row.condition]} | {row.coverage:.2%} | {row.accuracy_among_mapped:.2%} | {row.ambiguity_rate:.2%} | {row.unknown_rate:.2%} |\n"
    report+=f"""

The high-precision subset (Gemma anatomy or ARCHS4 `source_name_ch1`) mapped {len(strict):,}/5,000 at **{strict.correct.mean():.2%} precision**. A broader sample-direct subset mapped {len(sample_direct):,}/5,000 at **{sample_direct.correct.mean():.2%} precision**. Protocol and study-only evidence performed poorly (roughly 54–86% in observed strata) and should support/disambiguate, not generate production labels.

## Recovery on currently missing samples

- Resolved: **{len(missing_resolved):,}** ({pct(len(missing_resolved),len(missing))}).
- Ambiguous: **{(missing.mapping_status=='ambiguous').sum():,}** ({pct((missing.mapping_status=='ambiguous').sum(),len(missing))}).
- Unknown/no recognized direct alias: **{no_direct:,}** ({pct(no_direct,len(missing))}).
- Human resolved: 60,713; mouse resolved: 281,517. Mouse recovery is **not independently validated** because GEOMeta ground truth is human-only.
- Accepting every resolved proposal would raise apparent Organ coverage to **{pct(existing+len(missing_resolved),total)}**, but is not recommended at 92.73% held-out precision.
- Restricting to the observed 98.04%-precision gate would propose {len(strict_missing):,} additions and raise coverage to **{pct(existing+len(strict_missing),total)}**; it still needs independent organ-stratified and mouse validation before promotion.

## Targeted Laya result

The deterministic system produced 140 ambiguous held-out cases. Candidate recall was **{candidate.candidate_recall:.2%}** with a mean of {candidate.mean_candidate_count:.2f} labels (range {int(candidate.min_candidate_count)}–{int(candidate.max_candidate_count)}). Shortlisted Laya accuracy was **{short.correct.mean():.2%}** overall and **{short.loc[short.true_in_shortlist,'correct'].mean():.2%}** when the truth was in the shortlist; it returned Unknown for **{short.selected_label.eq('Unknown').mean():.2%}**. Flat 54-way Laya on the same hard cases scored **{laya[laya['mode'].eq('flat_54')].correct.mean():.2%}**. Scores are model scores, not calibrated probabilities.

Adding Laya to deterministic mapping increased full-cohort coverage from 70.98% to 72.14%, while precision among labeled samples fell from 92.73% to 92.07%. Laya is therefore **not necessary or suitable for promotion** under this protocol.

## Answers and recommendation

1. **Canonical-label gap:** 536,189 samples.
2. **Evidence gap:** 179,884 missing-label samples have no recognized direct 54-category alias. Of these, 29,982 have explicit unmapped structured/Gemma anatomy and 149,902 have only unrecognized sample-source text. The latter is the strict machine-recognized evidence gap; “truly no usable evidence” cannot be established automatically merely because all source strings are non-empty.
3. **Deterministic accuracy:** 96.20% at 55.32% coverage from ARCHS4-only; 92.73% at 70.98% with all sources. A strict source/Gemma gate reached 98.04% precision at 37.68% validation coverage.
4. **UBERON/synonym benefit:** it supplies normalized IDs and reviewed aliases for 52 categories. Among 342,230 resolved proposals, 214,279 contained the literal GEOMeta category phrase and 127,951 required an enabled alias/UBERON-preferred phrase. This lexical increment does not by itself validate the labels.
5. **GEO context:** GSM extras add 4.62 percentage points of coverage but reduce precision 1.48 points; study context adds another 11.26 points but reduces precision another 2.62 points. Study-only inference should be rejected.
6. **Ambiguity:** 14,075 currently missing samples are deterministically ambiguous.
7. **Laya:** shortlist accuracy remains only 21.43%; it does not solve the ambiguous tail.
8. **Deployment implication:** at most the 14,075 ambiguous proposals are immediate Laya candidates, but current evidence says they should remain Unknown instead. The 179,884 lexical unknowns lack a justified shortlist.

Next, manually adjudicate a species- and organ-stratified sample of the strict 210,809 proposed additions, especially mouse, and establish per-organ acceptance rules. Promote nothing until each accepted rule clears a predeclared precision threshold (for example >=99%) on an independent validation set. Keep GEO study text as retrieval/Laya context, not label-generating evidence, and do not scale Laya.

## Files

The result directory contains the complete vocabulary, UBERON mapping, synonym dictionary, 5,000-sample predictions and summaries, all currently missing-sample proposals, ambiguity/evidence-gap tables, bounded Laya predictions, candidate recall, score bins, and strategy comparison. See `README.md` for regeneration commands.
"""
    (ROOT/"ORGAN_RECOVERY_REPORT.md").write_text(report)
    con.close()


if __name__=="__main__": main()
