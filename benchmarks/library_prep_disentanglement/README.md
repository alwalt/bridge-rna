# Task 4 — Library-prep disentanglement

This standalone benchmark asks whether library-associated variation can be
separated from biological variation in **frozen** BridgeRNA sample embeddings.
It was motivated by Task 3, but it neither modifies nor trains on Task 3.

Start with [`library_prep_disentanglement_benchmark.ipynb`](library_prep_disentanglement_benchmark.ipynb).
Pipeline scripts are the reproducible source of truth; the notebook only reads
saved results.

## Controlled evidence and current scope

The ARCHS4 audit is conservative. `molecule_ch1 = total RNA` is not called
rRNA-depleted unless another metadata field explicitly describes rRNA removal.
ARCHS4 is classified **OBSERVATIONAL** because it lacks authoritative same-RNA
pair identifiers, so it is not used to supervise the primary model.

Controlled resources verified from their deposited metadata:

| Dataset | Design | Role |
|---|---|---|
| Chen et al. 2020, DOI 10.1038/s41597-020-00719-4 | 40 donors; the same naïve CD4 T-cell RNA processed by PolyA selection and Ribo-Zero | Train |
| Zhao et al. 2018, SRP127360 | pooled blood and colon source RNA; four technical libraries per protocol | Completely held-out test |

GSE150097 is retained only as a validation *candidate*. Its public metadata
contains both protocols but does not provide a defensible cross-protocol
same-RNA mapping for every sample. It is not silently promoted to paired data.
Consequently, the exploratory run uses a fixed, predeclared epoch count and no
validation-driven model selection. This is more conservative than splitting a
single study and claiming study-disjoint validation, but the external test has
only two biological source RNAs and cannot support definitive generalization.

## Reproduce

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/audit_archs4_library_prep.py

Rscript benchmarks/library_prep_disentanglement/pipeline/download_srp127360_recount3.R

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/prepare_controlled_data.py \
  --include-srp127360 --device cuda:0 --batch-size 4

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/characterize_bridge.py \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/chen_2020_tcells \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/zhao_2018_srp127360

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/run_task4.py \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/chen_2020_tcells \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/zhao_2018_srp127360 \
  --device cuda:0 2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_disentanglement/run.log

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/evaluate_task3_challenge.py \
  --device cuda:0 2>&1 | tee benchmarks/library_prep_disentanglement/results/task4g_task3_challenge/run.log
```

## Data and output policy

- Official downloads, TPM matrices, and embeddings live under ignored `work/`.
- Compact audit tables, metrics, logs, figures, and provenance live in `results/`.
- Input is natural `log1p(TPM)` in the canonical 15,165-gene order.
- No NASA/OSDR sample is used for training, model selection, or tuning.
- The held-out test is reported as exploratory because its biological N is two.
- Use “library-associated” unless controlled evidence supports a causal claim.

The neural decomposition is compared with original Bridge, linear removal,
no-pair-loss, no-adversarial-loss, shuffled-label, and shuffled-pair controls.
Success requires held-out library suppression in FE, library retention in RE,
improved same-RNA retrieval, and preservation of Task 3 RR3 controls—not merely
changing the sign of the RR1 cosine.

## Controlled-subspace follow-up

The follow-up diagnostic tests whether the RR1 protocol transition aligns with
the controlled T-cell PolyA→Ribo displacement before attempting another neural
correction. It uses cached frozen embeddings and does not retrain anything:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_controlled_library_subspace.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_followup_controlled_subspace_run.log
```

Compact outputs and figures are under
`results/task4_followup_controlled_subspace/`. The SVD removal is explicitly a
diagnostic; it is not presented as a production correction.

## Held-out OSDR response robustness

The response-robustness analysis projects the unchanged OSDR sample embeddings
away from 0, 1, 2, 3, 5, or 10 dimensions of the independently fitted T-cell
basis, then reconstructs the fixed FLT−GC responses. It never fits to OSDR or
changes Task 3 sample memberships or mode labels.

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_osdr_response_robustness.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_response_robustness_run.log
```

Results are under `results/task4_response_robustness/`. At PC1–5, RR1 changes
from −0.804 to +0.195 and is more affected than 500 random five-dimensional
removals. This is not a successful general correction: RR3-39 declines from
0.790 to 0.497, median 14-contrast response preservation is 0.523, response
matrix Spearman preservation is 0.559, fixed-label silhouette falls from 0.763
to 0.200, and ARI falls to 0.116. RR3-40 remains comparatively stable
(0.917→0.894). Thus, the controlled basis identifies a technically sensitive
RR1 component but substantially reorganizes broader response geometry.

Limitations: the basis comes from one 40-donor T-cell study; the independent
blood/colon effects reverse orientation; OSDR is held-out but small; and an
orthogonal residual cannot be interpreted as purified biological signal.

## Gene-level technical-replication diagnostic

This follow-up applies signed Integrated Gradients to RR1, RR3-39, and RR3-40
original/remeasured responses. The original response direction is fixed within
each pair, preventing self-orientation from hiding a reversal. The controlled
T-cell signature is fitted independently and OSDR is never used to define it.

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_technical_replication_gene_attributions.py \
  --devices cuda:0 cuda:1 2>&1 | \
  tee benchmarks/library_prep_disentanglement/results/task4_gene_attribution_diagnostic_run.log
```

If the raw signed attributions are already complete, figures/tables can be
regenerated without IG using `--reuse-attributions`. Outputs are under
`results/task4_gene_attribution_diagnostic/`.

RR1 shares 53 Top-100 attribution genes between measurements, but has weak
genome-wide signed agreement (Spearman 0.194); 51/53 shared genes retain sign.
RR3 shares 71–76 genes and has signed Spearman near 0.60. Controlled-signature
overlap is somewhat larger for RR1 (11, 49, and 125 genes at Top-100/250/500)
than RR3-39 (10/37/94) or RR3-40 (9/39/100), but the difference is modest.
Conventional expression also shows lower RR1 reproducibility (cosine 0.369)
than RR3 (0.652/0.822). Therefore BridgeRNA accentuates and reorganizes an
existing expression discrepancy; it does not create one absent from expression.

Enrichment uses the exact 15,165-gene background. Shared RR1 and reproducible
RR3 genes emphasize hepatic metabolic programs. Measurement-specific and
controlled-overlap sets had no significant coherent enrichment. These genes are
associative attributions, not causal technical or biological effectors.

## Simple correction comparison

The final methodological comparison asks whether controlled SVD removal is
actually preferable to simpler corrections. It evaluates no correction, mean
direction projection, SVD PC1–1/2/3/5, paired additive residualization, and the
existing FE representation without retraining:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/compare_simple_corrections.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_simple_correction_comparison_run.log
```

Outputs are under `results/task4_simple_correction_comparison/`. Corrections
are fitted leave-one-donor-out for controlled-pair evaluation and once on all
40 controlled donors for held-out OSDR application. AUROC below 0.5 is reported
without silently flipping it; orientation-free AUROC and accuracy proximity to
chance distinguish systematic inversion from genuine loss of predictability.

No method meets all predefined goals. Mean-direction/PC1 projection preserves
the response matrix (Spearman ~0.99; ARI 1.0) but RR1 remains negative (~−0.70).
Removing two components makes RR1 positive (+0.221), while response-matrix
preservation falls to 0.619, median response preservation to 0.541, RR3-39 to
0.480, and ARI to 0.272. PC1–5 yields RR1 +0.195 but further reduces matrix
preservation to 0.559 and ARI to 0.116. Paired additive residualization improves
controlled pairing and preserves all within-study responses exactly, because
the protocol offset cancels in FLT−GC; it therefore cannot change RR1. Existing
FE worsens RR1 and is not a successful alternative.

The controlled basis is consequently recommended for diagnostic
quantification rather than routine correction. A residual is not pure biology,
and systematic held-donor classifier inversion is not evidence that library
information has been erased.

## Replication-discrepancy decomposition (diagnostic only)

This final follow-up quantifies—but does not remove—the component of each NASA
technical-replication discrepancy aligned with the independently learned
40-donor T-cell library-associated basis:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/decompose_replication_discrepancies.py
```

Outputs are under `results/task4_discrepancy_decomposition/`. RR1 has 53.9% of
its squared discrepancy aligned with controlled PC1 and 96.0% within PC1–5.
The corresponding PC1–5 values are 22.6% for RR3-39 and 56.0% for RR3-40. All
three exceed 1,000 same-dimensional random subspaces (empirical one-sided
`p=0.001`), so controlled-subspace alignment is not uniquely RR1, although RR1
is the strongest and most concentrated discrepancy.

The controlled T-cell shifts are internally consistent, whereas the two
held-out pooled-blood/colon source shifts reverse orientation and NASA
discrepancies vary in alignment. The evidence therefore does not support one
universal additive PolyA→Ribo vector; a context-dependent library-associated
transformation is more consistent with the available observations.

The preservation audit uses cached OSDR metadata and the Lai Polo design. It
finds no fully crossed same-material design that independently identifies
preservation, library selection, and their interaction. OSD-48 C13/C14 varies
preservation across different animals; OSD-48 C14/OSD-168 varies a broader
library/sequencing workflow on matched source material; OSD-168 ERCC contrasts
hold library fixed. A minimal decisive follow-up would cross preservation ×
library method on aliquots of the same RNA in multiple biological contexts,
while holding sequencing workflow fixed.

These are alignment fractions, not causal percentages of technical effect.
The aligned component is not proven pure technical signal and the residual is
not purified biology. The Lai Polo source supporting the preservation/library
interaction motivation is DOI `10.1016/j.isci.2020.101733`.

## Technical-subspace donor robustness

The Technical Alignment Score robustness analysis re-estimates the controlled
reference from donor resamples without recomputing embeddings or applying any
correction:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_technical_subspace_robustness.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_technical_subspace_robustness_run.log
```

Outputs are under `results/task4_technical_subspace_robustness/`. The analysis
uses 1,000 paired-donor bootstraps, 250 repeated 32/8 donor splits, and a
40-fold leave-one-donor-out check. PC1 is highly stable (median projection
similarity 0.9995; median angle 1.24°), and PC1–2 remains stable (projection
similarity 0.9925; largest angle 7.01°). The full PC1–5 span is less stable
(median projection similarity 0.8584; largest angle 45.60°), reflecting weak
secondary directions that can rotate under resampling.

RR1's PC1–5 Technical Alignment Score remains highly reproducible: bootstrap
median 0.9566, SD 0.0054, and 95% interval 0.9430–0.9637 versus 0.9597 using all
40 donors. RR1 > RR3-40 > RR3-39 in every bootstrap at every tested k. The
large RR1 increase beyond PC1 is almost entirely PC2: PC1 contributes 0.5390
and PC2 contributes 0.4141, whereas PCs 3–5 together contribute only ~0.0066.

Within-experiment donor generalization is also strong. Repeated 32/8 held-out
donor median alignment is 0.9792 at k=1 and 0.9936 at k=5, with nearly
identical leave-one-out results. Accordingly, this score is a stable diagnostic
of similarity to the characterized T-cell PolyA→Ribo transformation. It is not
a causal percentage attributed to library preparation, a universal reference
across tissues, a pure technical component, or evidence of batch correction.

## Technical Confounding Profiler prototype

The profiler is a reporting layer over cached response vectors, technical-basis
robustness results, random-subspace controls, and gene attributions. It uses the
stable controlled T-cell PC1–2 span as the operational library-associated
reference:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/build_technical_confounding_profiler.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_confounding_profiler_run.log
```

Outputs are under `results/task4_confounding_profiler/`. RR1 combines opposing
response reproducibility (`R=-0.804`) with high technical alignment (`T=0.953`).
RR3-39 has `R=0.790`, `T=0.141`; RR3-40 has `R=0.917`, `T=0.466`. Thus,
response reproducibility and technical alignment provide distinct information:
technical-associated structure can occur in a discrepancy even when the main
response remains reproducible.

The biological-impact module reports existing signed-IG agreement and gene-set
results without rerunning attribution. RR1 attribution Spearman is 0.194, with
53 shared Top-100 genes and 96.2% sign agreement among those genes, consistent
with extensive reweighting/reranking rather than simple shared-gene reversal.
The profiler remains diagnostic. Its T-cell reference is not established as
universal across tissues, and alignment is neither causal attribution nor batch
correction.

The notebook's primary display now uses separate bar graphs for Response
Reproducibility (`R`), PC1–2 Technical Alignment (`T`), and global Biological
Preservation. At the PC1–5 operating point that makes RR1 positive, response-
matrix preservation is 0.559 and mode ARI is 0.116. This correlation is not a
percentage of biology. Bootstrap/random/reference-stability details and IG
results are secondary evidence.

The contextual-gene extension uses frozen inference on the exact 34 cached Task
3 log1p(TPM) inputs and streams contrast means without saving a multi-gigabyte
per-sample contextual tensor:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/analyze_contextual_gene_reproducibility.py \
  --device cuda:0 --batch-size 1 2>&1 | \
  tee benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/contextual_gene_run.log
```

RR1 median gene-context reproducibility is −0.125, with 61.6% of genes showing
reversed contextual responses. RR3-39 and RR3-40 medians are 0.783 and 0.864,
with reversal fractions 7.5% and 6.6%. RR1's contextually unstable Top-100 has
little overlap with high-IG genes, so contextual instability and input
influence are complementary. Exploratory ranked GSEA associates RR1 contextual
instability with RNA-processing/splicing, chromatin, and DNA-repair programs;
fatty-acid oxidation ranks toward relative contextual stability. These scores
do not prove altered gene regulation or make the pathways technical artifacts.

### Contextual-gene robustness audit

The contextual result was audited without repeating BridgeRNA inference:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/audit_contextual_gene_robustness.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/contextual_robustness/run.log
```

Outputs are under `results/task4_confounding_profiler/contextual_robustness/`.
The audit verifies the exact 34 samples, FLT-minus-GC direction, 15,165 genes,
512 contextual dimensions, and absence of zero response vectors. It adds the
symmetric response magnitude `sqrt(norm_A * norm_B)` and normalized discrepancy
`norm(A-B)/(norm(A)+norm(B)+epsilon)`.

RR1 remains strongly abnormal after excluding the lowest 10% of genes by
response magnitude: median contextual cosine is -0.089 and 57.8% of retained
genes remain reversed, versus medians 0.793 and 0.878 for RR3-39 and RR3-40.
RR1 median normalized discrepancy is 0.766 (0.756 after filtering), versus
0.374 and 0.297 for RR3. The contextual instability result is therefore
classified **ROBUST**, rather than a low-norm cosine artifact.

The initial 250-permutation GSEA had an empirical probability floor near
0.004. The audit reruns enrichment only, using 1,000 permutations and two
rankings: normalized discrepancy and negative cosine after removing the bottom
10% by response magnitude. GO BP, KEGG, and Reactome sets are intersected with
the tested BridgeRNA vocabulary (10--500 represented genes). RNA
processing/splicing, chromatin organization/remodeling, and DNA repair/metabolic
processes remain robustly enriched toward RR1 contextual instability under both
rankings. Fatty-acid beta-oxidation and related hepatic metabolic programs
remain toward relative contextual stability.

RR3-39 visual/phototransduction terms are driven by a small, redundant set of
low-expression genes; epidermal terms are likewise low-expression and fail the
two-ranking robustness criterion. These are retained in the machine-readable
audit but are not interpreted as liver biology. Enrichment is limited to
programs represented in BridgeRNA's 15,165-gene universe and does not describe
the complete mouse transcriptome.

### Final controlled contextual-gene validation

The final planned Task 4 analysis independently tests the RR1 contextual
pathway hypotheses in the 40-donor Chen et al. same-RNA T-cell experiment:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/analyze_controlled_gene_context.py \
  --device cuda:0 --batch-size 2 2>&1 | \
  tee benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/controlled_gene_context_run.log
```

Results are under
`results/task4_confounding_profiler/controlled_gene_context/`. The frozen
checkpoint and established count-to-gene-length-TPM-to-natural-log1p pipeline
were used for 80 libraries (40 authoritative same-RNA PolyA/Ribo pairs), 15,165
model genes, and 512-dimensional contextual gene representations.

For every gene, the analysis calculates donor-specific Ribo-minus-PolyA
contextual displacement. Its primary sensitivity score is median displacement
magnitude multiplied by positive mean leave-one-donor-out directional
consistency; `norm(mean donor displacement)` provides a second magnitude-aware
ranking. Median leave-one-out consistency is 0.959. Across 250 donor
bootstraps, median ranking Spearman is 0.993 and median Top-500 overlap is
485/500. A 1,000-replicate sign-flip control gives empirical `p=0.001` for
median directional consistency.

Using the same 15,165-gene universe, pathway resources, size limits, and 1,000
GSEA permutations as the RR1 robustness audit, RNA processing/splicing,
chromatin organization/remodeling, and DNA repair/metabolism are independently
supported under both controlled rankings. However, RR1-versus-controlled gene
ranking Spearman is only 0.163. Top-N overlap is not significant at 100, 250,
or 500 genes and becomes modestly enriched only at Top-1000 (85 observed versus
65.9 expected; `p=0.0088`). Representative pathway leading edges do not share
individual genes. The result is therefore **PARTIAL CONCORDANCE** at the
pathway-family level, not strong same-gene concordance.

Fatty-acid/lipid metabolism is not consistently enriched as controlled
technical sensitivity. Its behavior is mixed in T cells, so the liver-specific
relative stability result cannot be generalized across tissues. RR1 changed
multiple workflow variables beyond library selection; these controlled results
strengthen but do not causally prove the PolyA/Ribo hypothesis.

This completes the planned Task 4 analyses. The benchmark can be frozen for
paper use with conservative diagnostic language: the analysis does not show
batch correction, purified biology, a universal T-cell technical reference, or
that PolyA/Ribo alone caused RR1 instability.

### Conventional expression baseline

The final baseline uses raw counts and edgeR quasi-likelihood models rather
than differential testing on `log1p(TPM)`:

```bash
.venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/analyze_conventional_expression_baseline.py \
  2>&1 | tee \
  benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/conventional_expression_baseline_run.log
```

Controlled T cells use a paired `~ donor + library_prep` model. RR1 uses the
exact nine animal-matched OSD-48/OSD-168 pairs with animal blocking and a
measurement-by-flight-status interaction, reporting
`(FLT-GC)_OSD48 - (FLT-GC)_OSD168`. edgeR tests 11,373 expressed T-cell genes
and 11,916 RR1 genes from the 15,165-gene input universe.

The controlled conventional-versus-contextual gene rankings correlate at
Spearman 0.581 and strongly overlap at every Top-N cutoff, showing that
BridgeRNA preserves much of the large controlled library-selection expression
effect. RR1 correlation is only 0.075, although top-ranked overlap remains
enriched (31/100, 124/500, and 248/1000), indicating substantial contextual
reorganization rather than independence from expression.

Conventional T-cell DE identifies RNA processing/splicing, chromatin, and DNA
repair. The conventional RR1 interaction identifies RNA processing but not
chromatin or DNA repair at FDR < 0.05. Conventional expression therefore
reproduces one of three predefined cross-context families and no exact
significant pathways, whereas the BridgeRNA contextual analysis reproduces all
three families and nine exact significant pathways. At a 10% rank threshold,
902 RR1 genes have strong contextual but weaker conventional instability; this
conclusion persists at 5% and 20% thresholds.

The result is classified **AMPLIFICATION/REORGANIZATION OF CONVENTIONAL
SIGNAL**. BridgeRNA exposes coherent RR1 contextual pathway organization beyond
the standard interaction ranking, but it retains substantial conventional
signal and does not establish a wholly novel or causal biological program.
Outputs are under
`results/task4_confounding_profiler/conventional_expression_baseline/`.

### Expression-adjusted contextual sensitivity

The final gene-level control models contextual sensitivity conditional on the
continuous conventional edgeR statistic rather than subtracting Top-N DE genes:

```bash
.venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/analyze_expression_adjusted_context.py \
  2>&1 | tee \
  benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/expression_adjusted_context_run.log
```

Within each experiment, robust LOWESS (`frac=0.20`, three robustifying
iterations) predicts contextual sensitivity from `log1p(sqrt(edgeR QL F))`.
The positive residual means that contextual representation changed more than
expected from conventional expression behavior. Percentile-rank LOWESS is an
independent sensitivity analysis. The matched tables contain 11,373 controlled
T-cell and 11,916 RR1 genes.

Residuals are effectively uncorrelated with the fitted expression statistic
(Spearman 0.011 in T cells and -0.003 in RR1), are not dominated by the lowest
expression or contextual-magnitude deciles, and remain stable after excluding
the lowest 5--20% by both features. Residual GSEA uses the same pathway files,
gene-set limits, tested universe, and 1,000 permutations as the preceding
analyses.

RNA processing/splicing remains expression-adjusted in both experiments.
Chromatin and DNA repair remain strongly expression-adjusted in RR1 but not in
the controlled T cells, where their contextual enrichment is explained by
conventional expression. Cross-context residual gene Spearman is 0.005, with no
Top-100 overlap and no significant Top-250/500/1000 overlap. The surviving RNA
pathway concordance is therefore driven by different genes.

A 1,000-shuffle competitive family control supports nonrandom broad-family
rank concentration, but the stricter residual GSEA criterion is primary. The
result is classified **PARTIAL ADDITIONAL CONTEXTUAL ORGANIZATION**: BridgeRNA
contains coherent context organization not predicted by the per-gene edgeR
statistic, particularly within RR1, while part of the original cross-context
three-family result is conventional-expression-associated. Outputs are under
`results/task4_confounding_profiler/expression_adjusted_context/`.
