---
name: benchmark-notebook
description: Package an approved BridgeRNA benchmark as an executed public notebook under notebooks/ using validated cached evidence, regenerate metrics and figures cheaply, then commit and push the existing agent branch. Use after scientific review approval.
---

# Benchmark notebook

Create a portable publication package for an approved benchmark in the current worktree. Read applicable AGENTS.md instructions and the research directory before editing. Confirm explicit user approval of the benchmark and publication scope in the conversation or a supplied review record; an invocation explicitly approving a named benchmark is sufficient. If approval or scope is missing, inspect available evidence read-only and ask for the missing approval before writing. Cite this SKILL.md's requirement when explaining that stop. Do not require repeat approval already given.

## Preserve the frozen research record

Treat benchmarks/<benchmark_name>/ and source data as read-only. Keep the original primary research notebook intact. Do not train, fine-tune, run model inference, download large data, rerun expensive benchmarks, change rankings/splits/panels/labels, or optimize the approved protocol. Reuse existing metric and plotting code when safe, without invoking pipeline entrypoints with costly or mutating side effects. Do not merge branches, create PRs, or switch away from the agent's existing branch.

## Publication package

Create or update notebooks/<benchmark_name>/ with one primary <benchmark_name>.ipynb, a README, compact cached tables/predictions as appropriate, figures, provenance, and dependency documentation. Put any needed reusable publication helpers beneath this package. Preserve prior materially different publication protocols in clearly named subdirectories rather than overwriting them.

Explain the scientific question and hypothesis, datasets and exact cohort, preprocessing and gene coverage, frozen model and checkpoint, methods, baselines, metrics and chance levels, positive and negative results, limitations, and scientific significance. Finish with a short conclusion answering the question. Use the approved interpretation and cite relevant sources; label literature-only comparisons and differing sample/gene universes explicitly.

Inventory existing results before copying. Package only validated compact artifacts needed to reproduce the report. Never duplicate large datasets, checkpoints, embeddings, H5 files, archives, or large prediction arrays. For large dependencies, provide relative/configurable references and retrieval/rerun instructions; the default report must execute from packaged compact cached results. If prediction-level data are too large to package, validate them locally in bounded chunks, retain compact derived metrics and provenance, and document the external source. Do not include identifiable or restricted sample data without established publication permission; use aggregate results where needed.

Use portable paths resolved from the package or discovered repository root, never machine-specific absolute paths, external worktree symlinks, or hidden current-working-directory assumptions. Document tested dependency versions and a minimal environment/install command; reuse an available environment without committing it.

## Validate evidence and render

Recalculate metrics from saved predictions wherever available, using the established implementation, eligible samples/genes, labels, masks, folds, pairing, and seeds. Verify sample ordering, dimensions, vocabulary, split overlap, and relevant invariants. Compare recalculated values against published cached summaries with justified numerical tolerances. If predictions are unavailable, verify summaries against their saved source tables and explicitly state that prediction-level verification was unavailable. Never invent values, silently repair research outputs, or hide substantive discrepancies; resolve them within approved evidence or report the blocking issue before publication.

Regenerate publication-quality figures from validated saved numerical results; save PNG and PDF with compact CSV/Parquet figure data. Show units, readable labels, consistent ordering, chance levels, uncertainty and individual observations when appropriate. Preserve full numerical precision in stored tables; round only display values. Report mean, SD and count for repeated experiments and retain audit-level records where practical.

Record source artifact paths and hashes, source Git commit, exact cohort/manifest reference, parameters, seeds, checkpoint, preprocessing, metric definitions, dependency versions, verification outcomes, approval scope, and limitations in provenance JSON. Avoid hashing huge inputs unnecessarily; reuse trustworthy recorded hashes and label their origin.

Execute all notebook cells in order from a fresh kernel using cached results only, with no network, model inference, or original research writes required. Save the executed notebook. Verify tables/figures and artifact references, and inspect the rendered outputs for readable plots and consistent conclusions. Keep validation logs compact and report failures honestly.

In README document the lightweight cached-report execution command and separately the exact full-benchmark commands, dependencies, data/checkpoint acquisition and access prerequisites, compute/storage expectations where known, and monitoring guidance. Do not run the expensive commands now. Preserve original manifests and cite the research pipeline as the source of truth.

## Commit and push

After successful validation, inspect git status, the complete intended diff, and large/untracked files. Stage only this task's publication package and necessary scoped helpers; leave unrelated changes untouched. Confirm the current branch and its remote destination. Commit with a descriptive message and push the agent's existing branch using a normal push, setting an upstream for the same branch if needed. This skill invocation authorizes that commit and push; do not request redundant permission. Never force-push, rebase, merge, create a new branch, or create a PR. If detached HEAD, an ambiguous remote, a rejected push, or missing credentials prevents completion, preserve the validated work and report the precise blocker without destructive recovery.

Report the notebook/package paths, verification performed, material limitations, commit SHA, and push status. Leave PR creation and merging to the user.
