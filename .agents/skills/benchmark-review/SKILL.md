---
name: benchmark-review
description: Review an existing BridgeRNA benchmark against saved evidence and recommend manuscript, supplement, or exclusion placement. Use before publication packaging; produce a read-only scientific report for user approval.
---

# Benchmark review

Given an existing benchmark directory, assess its scientific contribution without changing files or running expensive computations. Resolve the directory in the current worktree and read applicable AGENTS.md instructions. If the intended benchmark is ambiguous, request its path. Inspect existing assets before suggesting downloads or recomputation.

## Evidence review

Read the primary notebook, README, pipeline, cohort manifests, saved predictions and metrics, figures, logs, and provenance as available. Review the objective, testable hypothesis, datasets and biological replication, preprocessing, methodology, baselines, metrics, results, and scientific significance. Distinguish intended protocol from what actually ran.

Trace headline claims and plotted values to specific saved artifacts and pipeline definitions. Use bounded read-only calculations in memory to verify counts, summaries, uncertainty, and metric definitions; do not execute notebooks or scripts that write files. Do not download data, run inference, fit models, regenerate splits, or launch full pipelines. Mark unavailable evidence as unverified rather than inferring success from prose or logs.

Check sample and study exposure, leakage, pairing and strata, discovery versus evaluation separation, frozen cohorts/panels, baseline comparability, seeds and replicate counts, and technical versus biological replication. Where applicable verify canonical 15,165-gene ordering, natural log1p(TPM), species-correct annotations, mapping losses, masked-only reconstruction scoring, eligible enrichment backgrounds, and correctly fitted PCA. Preserve benchmark-specific declared protocols; identify deviations rather than silently redefining them.

Separate effect size from statistical significance, held-out performance from exploratory selection, and biological evidence from representation robustness. Do not equate clustering with better biology or imply batch correction without direct evidence. Assess null and negative findings as scientific results.

## Deliverable and approval boundary

Return a concise scientific report in chat, without saving a file:

- Scientific question and hypothesis; datasets, cohort size, and protocol actually evaluated.
- Main positive and negative findings, with raw saved effect sizes, uncertainty and replicate counts where available, and artifact paths supporting each important claim.
- Verification status, discrepancies, missing evidence, limitations, and scientific significance.
- One recommendation: main manuscript, supplementary material, or exclusion, with a reason and any prerequisites. Distinguish exclusion for weak evidence from exclusion for redundancy or scope.
- Proposed publication scope: which cohort/protocol, claims, tables, and figures to package, plus unresolved issues.

End by asking the user to approve that concrete scope or request revisions, then stop. Do not create the publication notebook, modify research artifacts, commit, or push. Approval must be explicit; silence is not approval. Identify this skill's SKILL.md path and its approval requirement when explaining the stop. A later approved packaging request can proceed without asking again for the same scope.
