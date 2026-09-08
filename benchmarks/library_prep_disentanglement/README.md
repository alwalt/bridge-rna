# Task 4 — Library-prep disentanglement

This standalone benchmark asks whether library-associated variation can be
separated from biological variation in **frozen** BridgeRNA sample embeddings.
It was motivated by Task 3, but it neither modifies nor trains on Task 3.

Start with [`library_prep_disentanglement_benchmark.ipynb`](library_prep_disentanglement_benchmark.ipynb).
Pipeline scripts are the reproducible source of truth; the notebook only reads
saved results.

## Full-transcriptome versus Bridge-vocabulary conventional control

`pipeline/analyze_full_vs_bridge_vocab_expression.py` compares conventional
raw-count edgeR and ranked GSEA for the unchanged 11 independent mouse-liver
FLT-vs-GC contrasts under two gene universes: all expressed genes recoverable
from each source count matrix and the exact 15,165-gene BridgeRNA vocabulary.
It reuses the prior vocabulary-restricted results, maps full count rows with the
local GENCODE mouse annotation, and uses the same GO BP, KEGG, Reactome,
filtering, permutation, and seed settings in both arms. BridgeRNA embeddings do
not enter this control.

```bash
set -o pipefail
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_full_vs_bridge_vocab_expression.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_full_vs_bridge_vocab_expression_run.log
```

Outputs are stored in `results/task4_full_vs_bridge_vocab_expression/`. The
1-vs-1 RR1-CASIS 22-day contrast is descriptive only, and RR3 41-day remains
severely underpowered.

Across the 11 contrasts, the restricted arm retains a median 75.1% of genes
tested in the full arm. Full-versus-restricted pathway NES profiles remain
strongly concordant (median Spearman 0.961; range 0.943–0.987), with 100%
direction agreement among exact pathways significant in both arms. The median
fraction of significant pathways shared relative to their union is 59.9%.
RR3 39-day retains stronger RNA-processing enrichment than RR3 40-day in both
the full (best NES 2.545 vs 2.086) and restricted (2.270 vs 2.004) analyses.
Vocabulary restriction weakens some marginal calls but does not explain the
overall difference between conventional and contextual results.

## RR3 PC1–2 functional-overlap diagnostic

`pipeline/analyze_rr3_functional_overlap.py` decomposes the matched RR3 39-day
and 40-day OSD-137/OSD-168 responses into components parallel and orthogonal to
the frozen controlled T-cell PolyA/Ribo PC1–2 reference. It reuses cached
inputs/embeddings, runs zero-baseline Integrated Gradients for the four response
components, uses the established pathway resources, and evaluates 5,000 paired
animal bootstraps plus the existing 500 matched random 2D subspaces.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/analyze_rr3_functional_overlap.py \
  --devices cuda:0 cuda:1 2>&1 | tee \
  benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/rr3_functional_overlap/run.log
```

The source audit corrects two potentially confusing comparisons: PC1–2 removal
changes RR3-39/RR3-40 cosines to 0.480/0.876; 0.497/0.894 are PC1–5 results.
The 40-day technical comparison is matched 2/2 and differs from the 3/2
biological edgeR cohort because F5 lacks an OSD-168 remeasurement.

At the point estimate, parallel-component replication is 0.995 (39d) and 0.998
(40d), while orthogonal replication is 0.480 and 0.876. RR3-39 RNA-processing
genes modestly favor the parallel attribution ranking, but primary component
GSEA is nonsignificant and chromatin/DNA-response families show comparable
preferences. The paired-bootstrap interval for the RR3-39 subtraction effect
spans zero. The conservative decision is **D: unstable/underpowered**; the
controlled reference contains reproducible biological-response information and
must not be treated as a technical-only correction space.

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

### Independent biological replication of contextual programs

This targeted follow-up evaluates 11 stratified mouse-liver FLT-minus-GC
contrasts (84 samples) from OSD-47, OSD-48, OSD-137, OSD-173, OSD-242, and
OSD-245. OSD-168 is excluded from independent recurrence because it is a
technical remeasurement of RR1/RR3 material; it is retained only for the final
technical/biological triangulation.

For each gene and contrast, the contextual ranking is
`L2(mean(h_FLT) - mean(h_GC))`, matching the existing Task 4 contextual-response
magnitude. Ranked GSEA uses the exact 15,165-gene universe, GO BP/KEGG/Reactome,
10--500 gene-set limits, and 1,000 permutations. The conventional comparator
uses raw counts, TMM, and a separate robust edgeR quasi-likelihood model per
contrast. The single 1-vs-1 RR1-CASIS 22-day stratum is explicitly flagged as a
descriptive fixed-BCV edgeR ranking because it has no residual degrees of
freedom.

```bash
.venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/analyze_independent_biological_replication.py \
  --device cuda:0 --batch-size 1 \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_confounding_profiler/independent_biological_replication_run.log
```

Outputs are under
`results/task4_confounding_profiler/independent_biological_replication/`.
Recurrence is counted by both contrast and independent OSD so multiple strata
from one mission are not mistaken for independent biological replication.

High-contextual-response enrichment (positive NES on the nonnegative response-
magnitude ranking) recurs for chromatin organization/remodeling in two
independent OSDs, DNA repair/metabolism in two, and hepatic lipid/metabolic
programs in all six. RNA-processing high-response enrichment occurs in only one
OSD, despite broader recurrence in conventional edgeR. Conventional edgeR also
supports chromatin and DNA-response recurrence across five and four independent
OSDs, respectively. Median pairwise gene-rank Spearman is 0.296 versus median
pathway-NES Spearman 0.446, consistent with some program-level recurrence despite
changing gene rankings.

The decision is **B with important limitations**: chromatin/DNA-response biology
is not adequately described as an RR1-only technical artifact, but BridgeRNA
high-context evidence is sparse and one DNA-supporting OSD is the unreplicated
1-vs-1 RR1-CASIS stratum. This supports provisional recurrent spaceflight-
associated biology that is technically vulnerable in RR1, not causal separation
of biological and technical effects.

### Exploratory biological–technical latent overlap

This no-inference follow-up projects the same 11 independent FLT-minus-GC
Bridge response vectors into the controlled same-RNA T-cell PolyA/Ribo
uncentered PC1--2 reference. Conventional response intensity is RMS edgeR
log2FC among each contrast's `filterByExpr`-tested Bridge genes. PC1 and PC1--5,
program-specific RMS effects, exclusion of groups with fewer than two animals
per arm, and 1,000 matched random 2D subspaces are sensitivity analyses.

RMS log2FC correlates with absolute PC1--2-aligned magnitude (Spearman 0.745,
nominal p=0.0085) but not aligned fraction (0.264, p=0.433). Excluding the 1/2
and 1/1 contrasts gives 0.667 and 0.067, respectively. The magnitude association
is not unusual relative to random 2D subspaces (empirical p=0.314). Program-
specific RMS effects likewise associate with aligned magnitude, not fraction.
The result is **Outcome 1 with an Outcome 5 qualification**: stronger biological
responses have larger absolute projections, but do not increasingly occupy the
controlled reference, and the scaling is not specific to that reference.

Outputs are under
`results/task4_confounding_profiler/biological_technical_overlap/`.

The nested RR3-only follow-up compares the OSD-137 39-, 40-, and 41-day
responses. Total, aligned, orthogonal, and RMS-log2FC magnitudes are
non-monotonic (lower at day 40 and highest at day 41). PC1--2 aligned fraction
is 0.678, 0.733, and 0.735, respectively: a numerical increase followed by an
effective plateau, not a defensible time trend with only three points and a
1-FLT/2-GC day-41 group. Day 41 was correctly omitted from OSD-168 technical
replication because OSD-168 contains no remeasurement of animals F6/G6/G7.
Outputs are under `biological_technical_overlap/rr3_timecourse/`.

### RR1 preservation-context diagnostic

This follow-up reuses four validated, unpooled RR1 contrasts: OSD-47 CASIS
21d (2 FLT/2 GC), OSD-47 CASIS 22d (1/1), OSD-48 NASA upon-euthanasia
(2/2), and OSD-48 NASA carcass (5/5). Source/API metadata confirm OSD-47
flight liver dissection on orbit and distinct OSD-48 `Upon euthanasia` and
`Carcass` strata. Both studies report PolyA, TruSeq stranded RNA, single-end
50-bp HiSeq 3000 sequencing at UC Davis; this is not a PolyA-versus-ribo
comparison.

PC1--2 aligned fractions are 0.045, 0.789, 0.562, and 0.938, respectively.
The OSD-47 22d estimate is unreplicated. In direct OSD-48-to-OSD-168
technical replication, whole-response, PC1--2-parallel, and outside-PC1--2
cosines are -0.804, -0.974, and 0.221. For RR3-39 they are 0.790, 0.995,
and 0.480; for RR3-40 they are 0.917, 0.998, and 0.876. Fractions of
replication-discrepancy energy in PC1--2 are 0.953, 0.141, and 0.466.

The decision is **D -- underpowered/confounded**. The results are consistent
with preservation-context association but cannot isolate preservation from
study, strain, age, duration, animal, or protocol differences. PC1--2 is a
controlled PolyA/Ribo-associated reference, not a technical-only space.
Outputs are under
`results/task4_confounding_profiler/rr1_preservation_context/`.

### PolyA/Ribo-oriented spaceflight responses

The unchanged PC1--2 reference is oriented with the controlled paired mean
`mean(z_Ribo - z_PolyA)`, rather than arbitrary SVD signs. The matched RR1
carcass response switches from PolyA-directed in OSD-48 (directional cosine
-0.785; signed projection -0.707) to Ribo-directed in OSD-168 (+0.626;
+0.268). RR3-39 remains PolyA-directed (-0.693 to -0.619), while RR3-40
remains Ribo-directed (+0.820 to +0.780). Thus the RR1 reversal follows the
independently oriented reference, whereas RR3 shows that occupancy can remain
biologically reproducible. Direction is associative and does not identify the
library used, causality, or artifact. Outputs are under
`results/task4_confounding_profiler/polya_ribo_directionality/`.

### Final Biological Confounding Profiler

The final profiler combines all 14 validated Task 3 FLT-minus-GC responses,
the exact RR1/RR3 matched remeasurements, controlled-reference occupancy and
directionality, conventional edgeR/GSEA, contextual response, expression-
adjusted context excess, and independent biological recurrence. Its central
principle is **technical sensitivity identifies vulnerability, not artifact**.

RR1 carcass is classified measurement-vulnerable: whole-response and PC1--2
component reproducibility are -0.804 and -0.974. RR3-39 and RR3-40 are
reproducible despite technical overlap: whole-response cosines are 0.790 and
0.917 and PC1--2 cosines are 0.995 and 0.998. RNA processing is biologically
supported but technically vulnerable in RR1. Chromatin and DNA-response
programs recur independently and are candidate biological organization, not
proven biology. Lipid/metabolic programs recur across all six independent
OSDs but lack a program-specific matched-remeasurement test. The profiler also
retains the exact 596-gene RR1 context-excess/non-significant-DE set.

Reproduce without inference or retraining:

```bash
.venv/bin/python \
  benchmarks/library_prep_disentanglement/pipeline/build_bridge_confounding_profiler.py
```

Outputs are under
`results/task4_confounding_profiler/final_profiler/`.

### RNA-processing genes: shared machinery or latent convergence?

The gene-level follow-up defines 1,108 Bridge-vocabulary RNA-processing genes
from the existing GO BP, KEGG, and Reactome collections. It reuses the cached
40-donor controlled contextual tensor and regenerates only the six exact RR1/
RR3 contextual response tensors needed for cross-experiment 512-D gene-vector
cosines.

Gene-rank relationships are weak overall: controlled sensitivity versus RR1
instability is 0.134, controlled sensitivity versus reproducible RR3-39 is
0.088, and RR3-39 versus RR3-40 reproducible scores correlate 0.119. However,
specific Top-set overlaps exceed matched RNA-gene expectations. Controlled
T-cell/RR3-39 overlap is significant at Top 5%, 10%, and 20%; RR3-39/RR3-40
overlap is highly enriched at all three thresholds. Their Top-10% sets share
34 genes versus 11.1 expected, even though the median direct gene-vector cosine
between timepoints is -0.0065. CNOT3, RBM14, and ZFC3H1 lie in the primary
Top-10% controlled-sensitive, RR1-unstable, and reproducible-RR3 intersection.

The conclusion is a **mixed model**: some shared RNA-processing machinery
recurs, while different gene-level contextual transformations also converge
on a higher-order RNA-processing-associated latent organization. Outputs are
under `results/task4_confounding_profiler/rna_processing_gene_analysis/`.
# Task 4 technical-component decomposition

The component-level extension characterizes all 40 identifiable uncentered
components of the controlled 40-donor T-cell PolyA→Ribo difference matrix. It
reuses frozen sample embeddings, cached contextual-gene displacements, fixed
Task 3 response vectors, and the exact 15,165-gene enrichment universe.

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_technical_component_decomposition.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_technical_component_decomposition_run.log
```

Outputs are under `results/task4_technical_component_decomposition/`. PCs 1–5
explain 99.52% of the controlled displacement, but none met the conservative
exploratory criteria for a technical-only component: each showed pathway
coherence or appreciable independent response overlap. Removing the full
PC1–5 prefix eliminated controlled library predictability and improved paired
cross-library retrieval, but preserved only 0.559 Spearman correlation of the
Task 3 response-cosine matrix. NASA-informed component labels and correction
results are exploratory, not independent validation.

## RR3 cohort audit

`pipeline/audit_rr3_cohorts.py` audits the exact OSD-137 39-, 40-, and 41-day
animal cohorts using cached authoritative OSDR metadata, frozen embeddings,
and existing contextual/expression results. Outputs are under
`results/task4_confounding_profiler/rr3_cohort_audit/`.

The 39-day response remains negative and the 40-day response positive under
every valid leave-one-animal-out deletion. The PC1–2 reversal is driven more
strongly by the FLT cohort shift than by the GC shift. However, duration,
animal identity, and collection/euthanasia cohort are inseparable, and several
collection-order variables are unavailable. The result is therefore classified
as **mixed/unresolved**, not evidence of a one-day biological transition.

## RR1/RR3 sample-state diagnostic

`pipeline/analyze_rr1_rr3_sample_state.py` tests whether the crossed RR1/RR3
similarities inside the controlled PolyA/Ribo-associated PC1–2 are explained
by carcass handling or RNA quality. Results are under
`results/task4_confounding_profiler/rr1_rr3_sample_state/`.

## T-cell-signature-selective filtering

`pipeline/analyze_selective_tcell_signature_filter.py` tests a conservative
alternative to whole-PC1–2 subtraction. It defines all technical gene sets
from the independent 40-donor paired T-cell PolyA/Ribo experiment and removes
only those genes' mean-pooled contextual contributions inside the unchanged
PC1–2 reference. Outputs are under
`results/task4_selective_tcell_signature_filter/`.

The result is outcome B of the prespecified test. Top-500 and Top-1000
filtering improve RR1 technical cosine only from -0.8042 to -0.7937 and
-0.7819, respectively. These changes are larger than matched random panels
but do not resolve the reversal. They preserve the independent RR1/RR3
relationships because only a small response contribution is removed. Whole
PC1–2 subtraction improves RR1 to +0.2208 but collapses those relationships.
The controlled T-cell signature is strongly RNA-processing-associated, while
the shared RR1/RR3 PC1–2 organization uses mostly different genes and retains
hepatic lipid/peroxisomal/PPARalpha/bile and small-molecule programs.

This supports using controlled technical perturbations as diagnostic
references, not treating their occupied latent directions as biologically
empty or safely removable. The selective analysis is response-level and is
not a deployable correction method.

## PC1–2 contextual-attribution mechanisms

`pipeline/analyze_pc12_attribution_mechanisms.py` compares the full 15,165 × 2
per-gene PC1/PC2 contribution profiles that generate the controlled T-cell,
RR1, and RR3 responses. It also compares signed pathway-attribution profiles
and raw-expression responses. No correction, retraining, or new inference is
performed. Results are under `results/task4_pc12_attribution_mechanisms/`.

Latent direction and mechanism are not equivalent. RR1 carcass and RR3-39
have PC1–2 cosine 0.993 but gene-attribution cosine 0.665, rank correlation
0.175, and pathway-profile Pearson 0.020. RR1 euthanasia and RR3-40 have
latent cosine 0.991 but attribution cosine 0.334 and pathway-profile Pearson
-0.054. The RR1 OSD-48/OSD-168 reversal is distributed across opposing and
reweighted gene contributions rather than localized to the controlled T-cell
Top-N RNA-processing signature. These results support a shared sensitive
latent space used by partially different mechanisms, not technical-only PCs.

## Matched-ribodepletion RR1/RR3 diagnostic

`pipeline/analyze_matched_ribo_rr1_rr3.py` asks whether the original OSD-48
PolyA RR1 relationships with ribodepleted RR3 survive when RR1 is replaced by
its OSD-168 ribodepleted measurement. It preserves the RR3-39/RR3-40 cohort
definitions and reports geometry, contextual attribution, pathway profiles,
and animal-bootstrap uncertainty separately. Outputs are under
`results/task4_matched_ribo_rr1_rr3/`.

The original relationship does not survive. OSD-48 carcass RR1 is aligned
with RR3-39 (0.813) and opposed to RR3-40 (-0.830), whereas OSD-168 RR1 is
opposed to RR3-39 (-0.852) and aligned with RR3-40 (0.790). Attribution
profiles switch consistently, but pathway agreement remains weak. RR3-40's
same-library technical replication is much stronger (full cosine 0.919,
attribution cosine 0.829, pathway Pearson 0.930) than RR1's cross-protocol
replication. This is evidence of measurement-sensitive RR1 geometry, not
proof that library selection alone caused the difference.

## RR1/RR3 sample-level PC1–2 audit

`pipeline/audit_rr1_rr3_samples_pc12.py` provides the sample-first audit behind
the aggregate comparisons. It projects 54 OSD-48/137/168 profiles into the
unchanged controlled T-cell PolyA/Ribo-sensitive PC1–2 reference, preserves
authoritative metadata and missing fields, separates baseline from response,
and reports animal-bootstrap uncertainty and metadata confounding. Outputs are
under `results/task4_rr1_rr3_sample_pc12_audit/`.

RR1 exhibits both a global dataset/protocol shift and a condition-dependent
change. OSD-48 carcass has ΔPC1/ΔPC2 +0.733/+0.571; OSD-168 has
-0.440/-0.467. The FLT centroid shifts about -1.173/-1.038 farther than the GC
centroid across the two measurement contexts, so a global shift alone cannot
explain the reversal. RR3-40 instead retains nearly identical components
(-0.109/-0.089 versus -0.118/-0.090). OSD, library selection, read setup,
sample state, preservation, and library kit are structurally confounded in
RR1, preventing assignment of causality to any one factor.

Carcass status and RIN are insufficient explanations. The shared PC1–2
high-contribution gene sets are enriched for hepatic lipid, fatty-acid,
peroxisomal, bile, and small-molecule programs, but overlap the controlled
T-cell Top-500 signature by only 0–2 genes. This supports a shared latent space
containing both protocol-associated and biological/sample-state organization,
not an identical gene-level PolyA/Ribo artifact.

## RR3-39/RR3-40 four-state diagnostic

`pipeline/analyze_rr3_four_state.py` decomposes the two RR3 responses into
GC39, FLT39, GC40, and FLT40 using nine audited animals, the unchanged T-cell
PC1–2 reference, full 512-D embeddings, and cached per-animal contextual
tensors. Outputs are under `results/task4_rr3_39_40_four_state/`.

GC39/GC40 differ by 0.186 in full space, whereas FLT39/FLT40 differ by 0.358.
The Δ39/Δ40 cosine is -0.567 in full space, -0.995 within PC1–2, and +0.457
outside PC1–2. Their Top-500 contextual sets overlap by 69 genes but only
34.8% retain coordinate direction; shared hepatic metabolic families are
partly oppositely organized. However, leave-one-animal-out cosine ranges from
-0.747 to +0.157 and the bootstrap interval spans zero. Duration, animal, and
collection cohort are inseparable, so the result cannot be described as a
one-day time-course effect or a resolved physiological mechanism.

## Strict paired RR1/RR3 technical replication

`pipeline/analyze_paired_technical_replication.py` verifies exact animal
correspondence and reconstructs RR1, RR3-39, and RR3-40 FLT-minus-GC responses
from the same animals in the original and OSD-168 measurements. Outputs are
under `results/task4_rr1_rr3_paired_technical_replication/`.

RR1 uses four matched FLT animals (M25/M26/M28/M30) and five matched GC
animals (M36-M40); unmatched M27 and M29 are excluded. RR3-39 uses F1/F2 and
G1/G2, and strict RR3-40 uses F3/F4 and G3/G5; F5 is retained only as a
secondary full-stratum sensitivity analysis. Strict response cosines are
-0.804, +0.790, and +0.917, respectively.

The paired decomposition shows that RR1 FLT and GC technical displacements
are nearly parallel (cosine 0.996), but differ greatly in magnitude
(3.984 versus 2.666; differential displacement norm 1.353). RR3 differential
displacements are much smaller (0.144 and 0.073). RR3 also preserves its
contextual attribution and pathway profiles, whereas RR1 does not. These
results diagnose a condition-dependent response to a compound measurement
transition; they do not isolate PolyA/ribodepletion as causal or establish a
pure biological correction.

## Empirical PC1–2 biological-contrast null

`pipeline/analyze_pc12_empirical_null.py` evaluates occupancy of the unchanged
controlled T-cell PolyA/Ribo-sensitive PC1–2 plane across 20 curated biological
contrasts with at least two samples per arm (8 exercise and 12 spaceflight),
plus two explicitly underpowered secondary contrasts. Outputs are under
`results/task4_pc12_empirical_biological_null/`.

The median response-energy occupancy is 72.9%. RR3 GC39→GC40 is low (8.7%,
10th percentile), RR3 FLT39→FLT40 is ordinary (62.0%, 35th percentile), and
RR3-40 FLT−GC values near 69–73% are typical. OSD-48 RR1 carcass is the largest
observation (93.8%; empirical p=1/21), while OSD-168 RR1 is high but not unique
(88.6%; 85th percentile). The fixed plane captures 55.3% of global centered
variance in 40,000 ARCHS4 embeddings; one direction is close to the global
top-PC space and the other is not. Results therefore support partial generic
high-variance structure with context-dependent reuse, not a technical-only,
RNA-processing, or universally coordinated-program plane.

## PC1/PC2-specific geometry and attribution

`pipeline/analyze_pc1_pc2_specific_geometry.py` decomposes the unchanged
controlled T-cell PC1–2 plane into its two fixed directions, using the existing
20-contrast empirical null, 40,000 ARCHS4 embeddings, and cached contextual
response tensors. Outputs are under `results/task4_pc1_pc2_specific_geometry/`.

PC1 and PC2 capture 30.4% and 24.9% of global centered ARCHS4 variance, and
median biological-response occupancy is 42.3% and 31.6%, respectively. RR1
carcass occupancy is split across both PCs (58.4%/35.4%), and both signed
coordinates reverse after OSD-168 remeasurement. RR3-40 preserves both PCs,
predominantly PC1. PC-specific contextual attribution reproduces strongly for
RR3-40 technical replication but is heterogeneous across unrelated contexts.
The supported interpretation is reusable, globally prominent geometry with
context-dependent molecular realization—not fixed technical or biological PC
identities.

## Multi-layer response reproducibility

`pipeline/analyze_multilayer_reproducibility.py` compares full-space latent
geometry, existing response-specific Integrated Gradients, pathway-attribution
profiles, and conventional expression across three strict same-animal technical
remeasurements and twelve same-species/same-tissue cross-response negatives.
Outputs are under `results/task4_multilayer_reproducibility/`.

Geometry alone gives descriptive ROC AUC 0.694. Attribution and pathway
profiles each separate the three designed pairs from the compact null (AUC
1.0), while the unweighted combined score gives 0.972. Conventional expression
also gives 1.0, so this does not demonstrate a BridgeRNA advantage. One
unrelated RR1/RR3-39 pair is a geometric false friend (latent cosine 0.811,
attribution cosine 0.258, pathway Pearson 0.220). With only three same-material
positives and no independent-cohort positives, these results concern technical
measurement correspondence and cannot establish general biological
reproducibility.

## Attribution versus conventional expression

`pipeline/analyze_attribution_vs_expression.py` compares full-response IG with
conventional expression change for nine contrasts and constructs out-of-fold
expression-adjusted residual attribution. Outputs are under
`results/task4_attribution_vs_expression/`.

Absolute attribution and expression-change ranks are strongly coupled (median
Spearman 0.683), while a simple signed linear model explains little attribution
variance out of fold (median R² 0.031). Residual-attribution cosine remains
0.734, 0.869, and 0.694 for RR1, RR3-39, and RR3-40. Nevertheless, only one
residual GSEA result reaches FDR < 0.05, and residual similarity does not improve
the already perfect descriptive separation achieved by conventional expression
in the three-positive compact benchmark. The conclusion is mixed: BridgeRNA
weighting is not identical to expression change, but a coherent, independently
validated biological residual has not been demonstrated.

## PCA versus frozen BridgeRNA audit

`pipeline/analyze_pca_vs_bridgerna.py` compares conventional PCA of the exact
15,165-gene `log1p(TPM)` input with the frozen 512-dimensional representation.
Outputs are under `results/task4_pca_vs_bridgerna/`, and the executed notebook
contains the full tables, figures, caveats, and interpretation.

The global reference contains 40,000 ARCHS4 samples; PCA was fitted on 32,029
study-disjoint training samples. BridgeRNA is substantially more concentrated
than expression PCA (PC1+2: 65.2% versus 23.7%; participation ratio 2.98 versus
approximately 22.60). Geometry is related but reorganized (distance Spearman
0.747; 10-NN overlap 47.3%; normalized Procrustes R² 0.582). The controlled
T-cell PolyA/Ribo displacement is already highly concentrated in expression
(PC1+2 94.8%) and becomes 99.1% concentrated in BridgeRNA.

PCA-15165 matches BridgeRNA for RR3-39/40 response reproducibility, but
BridgeRNA amplifies the RR1 reversal and the RR1/RR3-39 geometric false friend.
On 1,000 held-out TCGA samples, predefined PCA reconstruction is comparable at
50% masking and substantially stronger at 90% masking. PCA currently uses one
deterministic seed whereas the reused BridgeRNA result summarizes ten. A
same-sample global PCA-FULL matrix was unavailable and was not manufactured;
OSDR full-vocabulary PCA is a labeled transductive secondary analysis.

For these endpoints, outcome C is the closest description—BridgeRNA amplifies
low-dimensionality in ways that can degrade similarity—but broader utility is
mixed because some existing linear tissue readouts favor BridgeRNA. This does
not establish that PCA is universally superior or anisotropy intrinsically bad.

## PCA versus BridgeRNA biological generalization

`pipeline/analyze_pca_vs_bridgerna_generalization.py` evaluates tissue identity
using identical labels, samples, splits, and linear classifiers. Results are in
`results/task4_pca_vs_bridgerna_generalization/`. The conservatively mapped
ARCHS4 cohort contains 3,272 human samples from 1,678 GSEs and 14 tissues.

In five-fold study-disjoint evaluation, macro F1 is 0.888 ± 0.016 for
training-selected PCA-15165, 0.872 ± 0.021 for raw expression, and 0.780 ±
0.028 for BridgeRNA. Random-split macro F1 is 0.942, 0.951, and 0.857,
respectively. At k=10, tissue-neighborhood purity is 0.818, 0.832, and 0.774.
BridgeRNA remains lower throughout the exploratory learning curve, and removing
its top 1, 2, or 5 PCs does not rescue performance.

The 40,000-sample source reference contains at most two samples per GSE.
Consequently, per-study LOSO estimates and within-tissue study-ID prediction
would be statistically uninformative and are explicitly marked unavailable.
PCA-FULL is also unavailable on the identical global ARCHS4 sample set. Existing
GTEx-human/ENCODE-mouse results are retained only as a qualified external
source-plus-species secondary comparison.

For this endpoint, the supported classification is outcome D: destructive
compression. The current frozen representation does not demonstrate tissue
generalization beyond conventional expression PCA. This conclusion is scoped
to the evaluated cohort and does not establish universal PCA superiority.

## Information-collapse localization

`pipeline/analyze_information_collapse.py` streams the frozen 12-layer model
once and caches six compact pooling summaries at the input embedding and every
layer. Results are under `results/task4_information_collapse/`.

### Frozen readout selection

`pipeline/analyze_frozen_readout_selection.py` compares prespecified,
label-free pooling summaries from cached frozen contextual tokens at layers
4–9 and 12 using the exact five GSE-disjoint tissue folds. The strongest fixed
readout was layer-12 mean+SD (1,024D; macro F1 0.8332 ± 0.0254), improving on
the standard final mean (0.7998) but remaining below raw expression (0.8716)
and selected PCA (0.8883). Fold-local unsupervised PCA projection did not
improve the selected summaries. Results and full provenance are in
`results/task4_frozen_readout_selection/`.

`pipeline/analyze_frozen_readout_response_safety.py` applies that selected
readout to the unchanged strict OSDR technical-remeasurement contrasts. It
reproduces the canonical mean-pooling controls (RR1 −0.8067, RR3-39 0.7900,
RR3-40 0.9166) and obtains mean+SD cosines of −0.7881, 0.7667, and 0.9141,
respectively. The improved readout therefore leaves the principal Task 4
diagnosis intact: it neither conceals the RR1 reversal nor materially damages
the strongly reproducible RR3-40 response. The canonical 512-D mean remains
the compatibility output; mean+SD is the best frozen readout identified by
this bounded search, not a replacement checkpoint.

Mean pooling is nearly rank-one before the Transformer. Tissue macro F1 rises
through the network, peaks at layer 7 (0.810 with the common ridge probe), and
ends at 0.800; tissue 10-NN purity peaks near 0.820 in layers 5–7 and falls to
0.774 at layer 12. Final-layer mean+SD improves macro F1 to 0.836, but remains
below raw expression (0.872) and PCA (0.888). The evidence supports mixed
readout loss and modest late-layer degradation, not a purely pooling-driven or
monotonic Transformer collapse.
