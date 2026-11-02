---
name: save-review
description: Condense a user-specified benchmark review, scientific report, Markdown file, or conversation review into a separate MANUSCRIPT_OUTLINE.md with Methods, Results, and Discussion bullets. Use for paper drafting from existing evidence, without reassessment or computation.
---

# Save review as a manuscript outline

Turn an existing scientific review into a succinct outline the author can write around. Save it separately from the full review: `benchmark-review` produces `PUBLICATION_REVIEW.md`; this skill produces `MANUSCRIPT_OUTLINE.md`. Never overwrite the full review or other source documents.

## Input and destination

- Read the user-specified benchmark review, scientific report, or Markdown file in full. A review explicitly selected from the current conversation is also a valid source. Incorporate explicit subsequent corrections without generating a new scientific assessment.
- Resolve the corresponding existing `benchmarks/<benchmark_name>/` directory in the current worktree and read applicable AGENTS.md instructions. If only a benchmark is specified, use its `PUBLICATION_REVIEW.md` when available. Ask for the source or destination when ambiguous or unavailable; do not reconstruct missing material from assumptions or raw benchmark results.
- Save to `benchmarks/<benchmark_name>/MANUSCRIPT_OUTLINE.md`. If this outline already exists, read it before replacing it and inspect differences in scope and source. Preserve user-authored additions unless replacement is explicit; ask if conflicting versions cannot be reconciled from the supplied source. Do not overwrite an unreadable file.

## Style and scientific fidelity

- Use succinct, scientifically precise bullets intended as an outline the author can write around. Preserve the report's terminology, dataset names, definitions, scientific interpretation, and evidence strength.
- Retain exact values and supplied precision, units, evaluation axes, class counts, sample/seed/fold counts, uncertainty, and baseline definitions for every metric included. Do not invent missing values, recalculate statistics, or change aggregation definitions.
- Select the findings most relevant to the requested drafting scope; this is a concise derivative, not a complete copy. Retain qualifications, negative findings, and QC issues that materially affect interpretation of the selected findings. Keep source citations or artifact references close to the claims they support.
- Distinguish descriptive ratios from information-retention or statistical claims. Do not turn speculative biological explanations into established findings or infer significance from descriptive differences.
- Avoid filler, promotional language, and lengthy prose. Do not write a full manuscript section unless explicitly requested; the short suggested passage required below is part of this workflow. Do not infer publication approval from saving an outline.

## Required output

Include a short title and a source reference or relative Markdown link, followed by three concise outline sections in this order:

1. **Methods** or **Supplementary Methods**: 4–7 bullets covering objective, data, representations, protocol, metrics, and controls where supplied.
2. **Results** or **Supplementary Results**: 4–7 bullets covering exact values, baseline comparisons, notable positive and negative observations, and QC where supplied.
3. **Discussion** or **Supplementary Discussion**: 4–6 bullets covering meaning, strengths, tradeoffs, limitations, and a defensible conclusion.

Use the supplementary heading set when requested or when the source explicitly recommends supplementary placement; otherwise use Methods, Results, Discussion. Keep the heading set consistent. Do not add unsupported content to meet bullet counts; flag missing source information briefly within the relevant section. Prefer bullets over tables for this drafting outline.

After the three outline sections, add one placement-specific drafting aid:

- **Suggested main-text reference** for a supplementary study: 1–3 sentences stating the supported finding and pointing readers to the supplementary analysis. Retain material qualifications; do not invent supplement figure/table numbers.
- **Suggested Results paragraph** for a main-text study: one concise paragraph summarizing the principal supported results with exact supplied values, evaluation context, and necessary caveats. Do not add a new interpretation or write a full manuscript section.

Use the publication placement specified by the user, otherwise the source review's explicit recommendation. If neither establishes main-text versus supplementary placement (including a recommendation to exclude), ask which placement is intended before choosing a drafting aid; do not silently promote the study. Keep the outline sections in the order above and place this drafting aid last.

## Save and confirm

- Do not rerun analyses, execute notebooks, download data, or regenerate the scientific assessment. Do not modify source files, `PUBLICATION_REVIEW.md`, benchmark results, or other skills. The only permitted benchmark write is `MANUSCRIPT_OUTLINE.md`.
- Read the saved outline back and compare it with the source for changed numbers, unsupported interpretations, lost material caveats, correct heading order, and bullet counts. Fix transcription and formatting issues without reassessing the science.
- Confirm the saved path with a clickable link and briefly identify the source used. Do not stage, commit, push, merge, or create a PR.
