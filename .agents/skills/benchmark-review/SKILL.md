---
name: benchmark-review
description: Review an existing BridgeRNA benchmark against saved evidence and recommend manuscript, supplement, or exclusion placement. Use before publication packaging; save PUBLICATION_REVIEW.md in the benchmark directory and summarize it in chat for user approval.
---

# Benchmark review

Given an existing benchmark directory, assess its scientific contribution using existing evidence without running expensive computations. The only permitted benchmark file change is creating or updating PUBLICATION_REVIEW.md in that benchmark directory; preserve all existing results, notebooks, code, manifests, figures, logs, and provenance. Resolve the directory in the current worktree and read applicable AGENTS.md instructions. If the intended benchmark is ambiguous, request its path. Inspect existing assets before suggesting downloads or recomputation.

## Evidence review

Read the primary notebook, README, pipeline, cohort manifests, saved predictions and metrics, figures, logs, and provenance as available. Review the objective, testable hypothesis, datasets and biological replication, preprocessing, methodology, baselines, metrics, results, and scientific significance. Distinguish intended protocol from what actually ran.

Trace headline claims and plotted values to specific saved artifacts and pipeline definitions. Use bounded read-only calculations in memory to verify counts, summaries, uncertainty, and metric definitions; do not execute notebooks or scripts that write files. Do not download data, run inference, fit models, regenerate splits, or launch full pipelines. Mark unavailable evidence as unverified rather than inferring success from prose or logs.

Check sample and study exposure, leakage, pairing and strata, discovery versus evaluation separation, frozen cohorts/panels, baseline comparability, seeds and replicate counts, and technical versus biological replication. Where applicable verify canonical 15,165-gene ordering, natural log1p(TPM), species-correct annotations, mapping losses, masked-only reconstruction scoring, eligible enrichment backgrounds, and correctly fitted PCA. Preserve benchmark-specific declared protocols; identify deviations rather than silently redefining them.

Separate effect size from statistical significance, held-out performance from exploratory selection, and biological evidence from representation robustness. Do not equate clustering with better biology or imply batch correction without direct evidence. Assess null and negative findings as scientific results.

## Deliverable and approval boundary

Save every scientific review to `<benchmark_directory>/PUBLICATION_REVIEW.md` before requesting approval. If that file already exists, read its full contents before replacing it. Reconcile its claims and unresolved issues with current evidence; retain still-valid context and explain material changes or superseded conclusions. Do not discard previous qualifications silently. If it cannot be read, stop before overwriting it.

Use clear Markdown headings, readable tables for datasets and validated results, and scientific explanations connecting evidence to interpretation. Include:

- **Benchmark objective:** scientific question, testable hypothesis, and intended contribution.
- **Datasets and cohort:** sources, sample/study counts, biological versus technical replication, splits, exposure, and gene coverage where applicable.
- **Methods and evaluation:** preprocessing, frozen model/checkpoint, baselines, protocol actually evaluated, metrics, and statistical/aggregation definitions.
- **Validated results:** a table of important saved results with effect sizes, uncertainty and replicate counts where available, verification status, and supporting artifact paths. Separate validated, discrepant, and unverified claims. Display rounding is allowed; do not alter raw stored values.
- **Positive and negative findings:** explain supported gains, null or adverse findings, and their scientific significance without overstating them.
- **Limitations and unresolved issues:** confounding, leakage risks, comparability, missing evidence, and boundaries on interpretation.
- **Publication recommendation:** main manuscript, supplementary material, or exclusion, with scientific reasoning and prerequisites. Distinguish weak evidence from redundancy or scope.
- **Proposed publication scope and approval status:** cohorts/protocols, claims, tables, and figures to package. Mark the proposal pending approval unless explicit approval for this scope already exists; saving a review does not itself approve publication.

Record the reviewed benchmark path, source Git revision when available, and evidence inspected. Use relative artifact links where practical. Confirm the Markdown file was saved successfully. Then display a concise summary in the agent conversation with a link to PUBLICATION_REVIEW.md, key validated findings, major limitations, publication recommendation, and approval request. Do not substitute the chat summary for the saved report.

End by asking the user to approve that concrete scope or request revisions, then stop. Do not create the publication notebook, modify other research artifacts, commit, or push. Approval must be explicit; silence is not approval. Identify this skill's SKILL.md path and its approval requirement when explaining the stop. A later approved packaging request can proceed without asking again for the same scope.
