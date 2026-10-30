---
name: save-review
description: Save a scientific benchmark review already generated in the current conversation as PUBLICATION_REVIEW.md in the corresponding benchmark directory. Use to preserve an existing assessment, not to conduct a new review or rerun analyses.
---

# Save review

Save the complete scientific review from the current agent conversation to the corresponding benchmark's `PUBLICATION_REVIEW.md`. Use the conversation as the scientific source of truth; this is a preservation and formatting task, not a new assessment.

## Resolve the source and destination

- Identify the full review and any explicit subsequent corrections in the conversation. Preserve all substantive review content, including objectives, hypotheses, datasets, methods, baselines, results, tables, positive and negative findings, limitations, scientific significance, and publication recommendations.
- Resolve the corresponding existing `benchmarks/<benchmark_name>/` directory in the current worktree and read applicable AGENTS.md instructions. If the source review is unavailable, incomplete, or ambiguous, or the destination cannot be identified, request the missing text or benchmark path before writing. Do not reconstruct missing content from memory, benchmark artifacts, or assumptions.
- If `PUBLICATION_REVIEW.md` exists, read its full contents before overwriting. Compare it with the conversation review. Preserve the conversation's assessment; do not silently drop substantive content present only in the existing file or combine conflicting assessments. If the intended replacement is unclear, explain the discrepancy and ask which version to retain. If the existing file cannot be read, do not overwrite it.

## Preserve scientific meaning

- Keep original terminology, dataset names, definitions, condition directions, comparison axes, numerical values, precision, units, class counts, sample/seed/fold counts, and uncertainty wherever supplied. Retain tables, artifact references, and citations.
- Report metrics with their evaluation axes, class counts, and uncertainty where supplied. Do not invent missing context, calculate new statistics, or change aggregation definitions.
- Distinguish descriptive ratios from information-retention or statistical claims. Keep exploratory interpretations and speculative biological explanations explicitly qualified; do not promote them to established findings.
- Preserve the recommendation, evidence strength, caveats, and approval status. Saving the review does not constitute approval or scientific validation.
- Do not rerun analyses, execute notebooks, download data, inspect results to generate a new scientific assessment, or invoke another benchmark workflow. Do not modify source files, research outputs, or other skills. The only benchmark write permitted is the requested `PUBLICATION_REVIEW.md`. Do not stage, commit, push, merge, or create a PR.

## Format and save

- Use a descriptive title and clear Markdown headings organized around the review's content: objective, datasets, methods, results, findings, limitations, and publication recommendation as applicable. Do not manufacture sections containing scientific claims absent from the source.
- Use succinct, scientifically precise bullets as an outline an author can write around, with GitHub-compatible tables for tabular evidence. Condense wording only when every substantive point and qualification is retained. Prefer completeness over brevity when they conflict.
- Avoid filler, promotional language, and lengthy prose. Do not expand the review into a manuscript section unless explicitly requested.
- Write the document to `benchmarks/<benchmark_name>/PUBLICATION_REVIEW.md`. Read it back and compare against the conversation review for omissions, changed numbers, broken tables, or altered interpretations. Correct formatting or transcription errors only; do not revise the scientific assessment.
- Confirm the saved file path with a clickable link in the conversation. Briefly state that the existing review was preserved without rerunning analyses, and disclose any source ambiguity that remains.
