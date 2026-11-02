# Conservative Organ recovery audit

## Outcome

The current database has **940,455** samples, **404,266** canonical Organ labels (42.99%), and **536,189** missing canonical labels. Missing canonical metadata is not the same as missing evidence: the direct GEOMeta/UBERON synonym system found one or more recognized anatomical candidates in **356,305** currently missing samples. It resolved **342,230**, marked **14,075** ambiguous, and found no recognized 54-category alias in **179,884**.

The last number is an intentionally conservative lexical gap, not proof that all free text is biologically empty. Among these unknowns, **29,982** retain explicit but unmapped Gemma/structured anatomy evidence. The remaining **149,902** have no machine-recognized direct or structured anatomy, although all have a non-empty ARCHS4 sample-source string. Thus zero records are literally devoid of source text, but this audit cannot honestly claim those strings are usable anatomy without broader ontology mapping or manual review. The evidence-gap table preserves them for that review.

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
| ARCHS4 only | 55.32% | 96.20% | 0.66% | 44.02% |
| + GEO GSM extras | 59.94% | 94.73% | 1.06% | 39.00% |
| + GSE context | 71.20% | 92.11% | 2.58% | 26.22% |
| + Gemma anatomy | 70.98% | 92.73% | 2.80% | 26.22% |


The high-precision subset (Gemma anatomy or ARCHS4 `source_name_ch1`) mapped 1,884/5,000 at **98.04% precision**. A broader sample-direct subset mapped 2,661/5,000 at **97.41% precision**. Protocol and study-only evidence performed poorly (roughly 54–86% in observed strata) and should support/disambiguate, not generate production labels.

## Recovery on currently missing samples

- Resolved: **342,230** (63.83%).
- Ambiguous: **14,075** (2.63%).
- Unknown/no recognized direct alias: **179,884** (33.55%).
- Human resolved: 60,713; mouse resolved: 281,517. Mouse recovery is **not independently validated** because GEOMeta ground truth is human-only.
- Accepting every resolved proposal would raise apparent Organ coverage to **79.38%**, but is not recommended at 92.73% held-out precision.
- Restricting to the observed 98.04%-precision gate would propose 210,809 additions and raise coverage to **65.40%**; it still needs independent organ-stratified and mouse validation before promotion.

## Targeted Laya result

The deterministic system produced 140 ambiguous held-out cases. Candidate recall was **89.29%** with a mean of 2.16 labels (range 2–5). Shortlisted Laya accuracy was **21.43%** overall and **24.00%** when the truth was in the shortlist; it returned Unknown for **58.57%**. Flat 54-way Laya on the same hard cases scored **16.43%**. Scores are model scores, not calibrated probabilities.

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
