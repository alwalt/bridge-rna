#!/usr/bin/env python3
"""Build the compact, executable collaborator-facing benchmark notebook."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.11"}
nb.cells = [
    nbf.v4.new_markdown_cell("""# Frozen Bridge sample-retrieval geometry

This is the human-readable Stage 0–1 report for the independent
`sample_retrieval_geometry` benchmark. It loads saved results only: it does not
regenerate embeddings, recompute the 562 × 510,709 search, or train a model.
See `REPORT.md` for equations, detailed provenance, limitations, and the final
recommendation."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

def find_here(start):
    for p in (start.resolve(), *start.resolve().parents):
        c = p if p.name == 'sample_retrieval_geometry' else p / 'benchmarks' / 'sample_retrieval_geometry'
        if (c / 'config.json').is_file(): return c
    raise FileNotFoundError('sample_retrieval_geometry not found')

HERE = find_here(Path.cwd()); RESULTS = HERE / 'results'
comparison = pd.read_csv(RESULTS / 'ranked_method_comparison.csv')
distributions = pd.read_csv(RESULTS / 'cosine_distributions.csv')
density = pd.read_csv(RESULTS / 'density_stratified_metrics.csv')
hubness = pd.read_csv(RESULTS / 'hubness_summary.csv')
set_metrics = pd.read_csv(RESULTS / 'set_retrieval_metrics.csv')
provenance = json.loads((RESULTS / 'provenance.json').read_text())
BIO = RESULTS / 'biological_validation'
composition = pd.read_csv(BIO / 'composition_summary.csv')
label_qc = pd.read_csv(BIO / 'biological_label_summary.csv')
density_models = pd.read_csv(BIO / 'density_failure_models.csv')
source_comparison = pd.read_csv(BIO / 'main_comparison_source_label.csv')
tissue_comparison = pd.read_csv(BIO / 'main_comparison_tissue_label.csv')
set_biology = pd.read_csv(BIO / 'set_method_biological_summary.csv')
diversity = pd.read_csv(BIO / 'retrieval_diversity_summary.csv')
AUDIT = RESULTS / 'recount3_audit'
funnel = pd.read_csv(AUDIT / 'original_cohort_funnel.csv')
macro = pd.read_csv(AUDIT / 'study_macro_retrieval_metrics.csv')
capped = pd.read_csv(AUDIT / 'capped_study_retrieval_metrics.csv')
cohort_inventory = pd.read_csv(AUDIT / 'bridge_evaluation_cohort_inventory.csv')
expansion = pd.read_csv(AUDIT / 'expansion_inventory.csv')
ABS = RESULTS / 'absolute_similarity'
cosine_by_rank = pd.read_csv(ABS / 'exact_pair_cosine_by_rank.csv')
cosine_micro_macro = pd.read_csv(ABS / 'paired_cosine_micro_macro_summary.csv')
expanded_micro = pd.read_csv(ABS / 'expanded_ranked_metrics_micro.csv')
expanded_macro = pd.read_csv(ABS / 'expanded_ranked_metrics_study_macro.csv')
random_exact = pd.read_csv(ABS / 'random_vs_exact_cosine.csv')
random_percentiles = pd.read_csv(ABS / 'exact_cosine_random_empirical_percentiles.csv')
near_counts = pd.read_csv(ABS / 'near_exact_neighbor_counts_by_rank.csv')
poor_rank_frequency = pd.read_csv(ABS / 'high_similarity_poor_rank_frequency.csv')
native_scores = pd.read_csv(ABS / 'method_native_score_by_rank.csv')
poor_rank_examples = pd.read_csv(ABS / 'high_similarity_poor_rank_examples.csv')
ANISO = RESULTS / 'anisotropy'
anisotropy = pd.read_csv(ANISO / 'anisotropy_summary.csv')
pca_summary = pd.read_csv(ANISO / 'pca_summary.csv')
geometry_distributions = pd.read_csv(ANISO / 'cosine_distribution_summaries.csv')
geometry_retrieval = pd.read_csv(ANISO / 'geometry_vs_local_corrections.csv')
geometry_macro = pd.read_csv(ANISO / 'retrieval_metrics_study_macro.csv')
geometry_biology = pd.read_csv(ANISO / 'biological_neighborhood_preservation.csv')
geometry_density = pd.read_csv(ANISO / 'query_density_summary.csv')
geometry_hubness = pd.read_csv(ANISO / 'hubness_summary.csv')
geometry_separation = pd.read_csv(ANISO / 'similarity_separation_metrics.csv')
LIT = RESULTS / 'literature_postprocessing'
literature_main = pd.read_csv(LIT / 'main_comparison.csv')
literature_distributions = pd.read_csv(LIT / 'cosine_distribution_summaries.csv')
literature_diagnostics = pd.read_csv(LIT / 'alignment_uniformity.csv')
literature_macro = pd.read_csv(LIT / 'retrieval_metrics_study_macro.csv')
literature_biology = pd.read_csv(LIT / 'biological_neighborhood_preservation.csv')
literature_hubness = pd.read_csv(LIT / 'hubness_summary.csv')
whitening_diagnostics = pd.read_csv(LIT / 'whitening_diagnostics.csv')
GTEX = RESULTS / 'gtex_biological_retrieval'
gtex_cohort = pd.read_csv(GTEX / 'gtex_cohort_summary.csv')
gtex_annotation = pd.read_csv(GTEX / 'archs4_annotation_summary.csv')
gtex_harmonization = pd.read_csv(GTEX / 'tissue_harmonization.csv')
gtex_micro = pd.read_csv(GTEX / 'retrieval_metrics_micro.csv')
gtex_macro = pd.read_csv(GTEX / 'retrieval_metrics_tissue_macro.csv')
gtex_per_tissue = pd.read_csv(GTEX / 'per_tissue_p10_comparison.csv')
gtex_comparison = pd.read_csv(GTEX / 'technical_identity_vs_biological_retrieval.csv')
gtex_control = pd.read_csv(GTEX / 'gtex_to_gtex_control.csv')
gtex_density = pd.read_csv(GTEX / 'density_by_tissue.csv')
gtex_raw_scores = pd.read_csv(GTEX / 'raw_cosine_by_tissue_match_status.csv')
CTRL = RESULTS / 'representation_metric_controls'
control_assets = json.loads((CTRL / 'asset_audit.json').read_text())
control_micro = pd.read_csv(CTRL / 'micro_metrics.csv')
control_macro = pd.read_csv(CTRL / 'study_macro_metrics.csv')
control_separation = pd.read_csv(CTRL / 'separation_metrics.csv')
control_pca = pd.read_csv(CTRL / 'pca_provenance.csv')
control_sensitivity = pd.read_csv(CTRL / 'metric_sensitivity_summary.csv')
GCTRL = RESULTS / 'gtex_representation_controls'
gctrl_identity = pd.read_csv(GCTRL / 'identity_vs_biological_retrieval.csv')
gctrl_micro = pd.read_csv(GCTRL / 'retrieval_metrics_micro.csv')
gctrl_macro = pd.read_csv(GCTRL / 'retrieval_metrics_tissue_macro.csv')
gctrl_tissue = pd.read_csv(GCTRL / 'per_tissue_metrics.csv')
gctrl_control = pd.read_csv(GCTRL / 'gtex_to_gtex_control.csv')
gctrl_overlap = pd.read_csv(GCTRL / 'neighborhood_overlap_per_query.csv')
gctrl_runtime = pd.read_csv(GCTRL / 'runtime.csv')
CONTEXT = RESULTS / 'contextual_metadata'
context_coverage = pd.read_csv(CONTEXT / 'annotation_coverage.csv')
context_confidence = pd.read_csv(CONTEXT / 'annotation_confidence_qc.csv')
context_metrics = pd.read_csv(CONTEXT / 'contextual_agreement.csv')
context_examples = pd.read_csv(CONTEXT / 'representative_examples.csv')
assert provenance['queries'] == 562 and provenance['references'] == 510709
print(f"Loaded completed results: {provenance['queries']} queries × {provenance['references']:,} references")"""),
    nbf.v4.new_markdown_cell("""## Cosine range

The query-to-corpus distribution is exact over all 287,018,458 comparisons.
The ARCHS4-to-ARCHS4 distribution is a deterministic two-million-pair estimate;
its minimum is observed in that sample and is not the global minimum over all
~130 billion possible pairs."""),
    nbf.v4.new_code_cell("""display(distributions.pivot(index='statistic', columns='distribution', values='cosine').round(6))
display(plt.imread(RESULTS / 'figures/archs4_pairwise_cosine.png'))
plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Ranked retrieval comparison

Only methods with a mathematically defined global ordering appear here.
`source_r1_known` is exact normalized `source_name_ch1` agreement—not an
inferred tissue, disease, treatment, or condition label."""),
    nbf.v4.new_code_cell("""cols = ['method','exact_r1','exact_r5','exact_r10','exact_r100','mrr','median_rank','same_gse_r1','source_r1_known','runtime_seconds']
display(comparison[cols].style.format({c:'{:.2%}' for c in ['exact_r1','exact_r5','exact_r10','exact_r100','same_gse_r1','source_r1_known']}).format({'mrr':'{:.4f}','runtime_seconds':'{:.2f}'}).hide(axis='index'))"""),
    nbf.v4.new_code_cell("""display(plt.imread(RESULTS / 'figures/ranked_method_comparison.png'))
plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Density stratification and set methods

The query strata are prespecified tertiles of mean cosine to ten nearest
references. Reciprocal-kNN and SNN are evaluated as returned sets rather than
being forced into global R@K/MRR."""),
    nbf.v4.new_code_cell("""primary = density[density.method.isin(['Cosine','CSLS(k=10)','LocalScaling(k=10)','NICDM(k=10)'])]
display(primary[['method','density_stratum','exact_r1','exact_r10','mrr','same_gse_r1']].style.format({c:'{:.2%}' for c in ['exact_r1','exact_r10','same_gse_r1']}).format({'mrr':'{:.4f}'}).hide(axis='index'))
display(set_metrics.style.format({'exact_set_recall':'{:.2%}','same_gse_set_recall':'{:.2%}','mean_returned':'{:.1f}','empty_fraction':'{:.2%}'}).hide(axis='index'))"""),
    nbf.v4.new_code_cell("""display(plt.imread(RESULTS / 'figures/density_stratified_retrieval.png'))
plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Hubness, diagnostics, and interpretation

Cosine has strong hub concentration. Local corrections reduce maximum N_k and
top-hub assignment share, but their modest exact-GSM gains trade against
same-GSE/source Top-1 agreement. Objective per-query gains, losses, unchanged
cases, and dense failures are saved in `rank_change_diagnostics.csv`.

The evidence supports offering NICDM or local scaling as a retrieval-time
reranker alongside cosine. It does not yet justify representation learning:
Stages 2–4 were not implemented or run."""),
    nbf.v4.new_code_cell("""display(hubness[hubness.k.isin([1,100])][['method','k','references_selected','max_nk','skewness','gini','top_10_hubs_assignment_fraction']].style.format({'skewness':'{:.2f}','gini':'{:.4f}','top_10_hubs_assignment_fraction':'{:.2%}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ['hubness_comparison.png', 'density_vs_cosine_rank.png']):
    ax.imshow(plt.imread(RESULTS / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Calibration and rank-change diagnostics

The global empirical null is monotone and cannot change cosine ranking. Its
P-values nevertheless differ across density strata. Rank-change distributions
show the full balance of gains and losses rather than selected anecdotes; the
objective representative cases remain available in the saved diagnostics table."""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, name in zip(axes, ['empirical_null_density_calibration.png', 'rank_change_distributions.png']):
    ax.imshow(plt.imread(RESULTS / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""# Stage 1A–1H: composition-controlled biological validation

**Primary composition result.** The benchmark is not balanced: GSE234297 and
its `peripheral blood` source contribute 144/562 queries (25.6%), and the five
largest studies contribute 51.1%. Of 187 dense queries, 122 (65.2%) come from
that single study/source. Cell type, cell line, disease, treatment, and
primary-versus-cultured status are not available as curated corpus-wide fields
and were not inferred from free text."""),
    nbf.v4.new_code_cell("""display(composition.style.format({'largest_fraction':'{:.1%}','top5_fraction':'{:.1%}','effective_n_shannon':'{:.1f}','effective_n_inverse_simpson':'{:.1f}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ['query_composition.png', 'density_by_study.png']):
    ax.imshow(plt.imread(BIO / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_code_cell("""display(plt.imread(BIO / 'figures/density_by_source.png'))
plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Does density remain associated after composition controls?

The dense-minus-sparse failure difference is +6.8 percentage points overall
but −1.7 points after excluding GSE234297/peripheral blood. Conditional models
show no precise independent density association. The earlier dense-stratum
failure pattern is therefore largely composition-driven in this cohort."""),
    nbf.v4.new_code_cell("""display(density_models.style.format({'odds_ratio_per_density_sd':'{:.2f}','ci_low':'{:.2f}','ci_high':'{:.2f}','p_value':'{:.3f}'}).hide(axis='index'))
display(plt.imread(BIO / 'figures/density_failure_effects.png')); plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Biological-label quality and retrieval

Exact normalized `source_name_ch1` is structured and complete but heterogeneous
across submitters. The conservative inherited tissue mapping covers only 162
queries—144 blood and 18 brain—so tissue results are a limited sensitivity
analysis. The exact paired GSM is excluded below; cross-study results also
exclude every reference from the query GSE."""),
    nbf.v4.new_code_cell("""display(label_qc.style.format({'largest_query_class_fraction':'{:.1%}','query_missing_fraction':'{:.1%}','reference_missing_fraction':'{:.1%}'}).hide(axis='index'))
display(source_comparison.style.format({c:'{:.1%}' for c in ['exact_r1','exact_r10','biological_p1','biological_p10','cross_study_biological_p1','cross_study_biological_p10']}).format({'hubness_gini_n100':'{:.3f}'}).hide(axis='index'))
display(tissue_comparison.style.format({c:'{:.1%}' for c in ['exact_r1','exact_r10','biological_p1','biological_p10','cross_study_biological_p1','cross_study_biological_p10']}).format({'hubness_gini_n100':'{:.3f}'}).hide(axis='index'))
display(plt.imread(BIO / 'figures/biological_retrieval_comparison.png')); plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Density-stratified biology, set retrieval, and diversity

Cosine preserves the strongest within-corpus source neighborhoods. Local
normalization improves exact identity and reduces study redundancy but lowers
source precision. In the limited blood/brain task, corrections improve
cross-study tissue retrieval. SNN maximum-score sets are too broad to serve as
selective retrieval results."""),
    nbf.v4.new_code_cell("""display(set_biology.style.format({'mean_returned':'{:.1f}','nonempty_fraction':'{:.1%}','mean_set_precision':'{:.1%}','any_same_label':'{:.1%}'}).hide(axis='index'))
display(diversity.style.format({'unique_gse_top10':'{:.2f}','unique_source_top10':'{:.2f}','largest_gse_fraction_top10':'{:.1%}','largest_source_fraction_top10':'{:.1%}'}).hide(axis='index'))
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, name in zip(axes, ['density_stratified_biological_retrieval.png', 'retrieval_diversity.png']):
    ax.imshow(plt.imread(BIO / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Objective-specific recommendation

- **Technical identity:** NICDM, with local scaling close behind.
- **Within-corpus biological source:** cosine.
- **Cross-study biology:** not resolved by available metadata; corrections win
  the limited blood/brain tissue sensitivity analysis, but exact source labels
  do not transfer across submitters.

The density effect is largely driven by one peripheral-blood study, and metadata
are insufficient for a trustworthy cell-state objective. Representation
correction is not warranted from these results. Stages 2–4 remain unimplemented."""),
    nbf.v4.new_markdown_cell("""# recount3 discovery audit and study balancing

The upstream run started from every predefined human, unseen, mapped-single
Bridge sample in both exposure cohorts (84,896 candidates). It traversed the
full contemporaneous human recount3 SRA project catalog, but only 8,199/8,803
project metadata files were readable; 604 failures prevent a strict claim of
exhaustiveness. The preserved tables—not the later-edited upstream config—show
that 574 pairs were processed: 562 unseen-study and 12 unseen-sample/seen-study.

The audit below is metadata-only. It reused saved ranks and the existing
recount3 metadata restart cache; it downloaded no expression and generated no
embeddings."""),
    nbf.v4.new_code_cell("""display(funnel)
display(cohort_inventory)
display(expansion)"""),
    nbf.v4.new_markdown_cell("""## Balanced exact-pair retrieval

Study-macro evaluation gives every one of the 40 studies equal weight. Local
scaling and NICDM retain modest R@5/R@10 advantages, while CSLS no longer
improves on cosine. Cap-5 and cap-10 subsets are deterministic and were chosen
without using density or retrieval outcomes."""),
    nbf.v4.new_code_cell("""macro_cols = ['method','studies','macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_mrr']
display(macro[macro_cols].style.format({c:'{:.1%}' for c in macro_cols if c.startswith('macro_r')}).format({'macro_mrr':'{:.4f}'}).hide(axis='index'))
display(capped.style.format({c:'{:.1%}' for c in ['r_at_1','r_at_5','r_at_10','r_at_100']}).format({'mrr':'{:.4f}'}).hide(axis='index'))
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, name in zip(axes, ['study_macro_retrieval.png', 'balanced_density_sensitivity.png']):
    ax.imshow(plt.imread(AUDIT / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Expansion decision

Within the original mapped-single eligibility criteria, the readable recount3
snapshot contains no unprocessed candidates. A broader cached-metadata
intersection identifies 257 unseen-study and one seen-study candidate, but all
258 have ambiguous multi-GSE manifest mappings and were deliberately outside
the original protocol. The 257 unseen-study candidates add neural/brain-heavy
contexts and 43 component GSEs not in the 562, but one composite GSE label still
accounts for 42.8%.

Keep the strict 562 and report study-macro metrics alongside micro metrics.
Consider the 257 only as a separately named, manually adjudicated expansion;
do not merge them into the existing benchmark. No expression processing,
embedding generation, or representation learning was performed."""),
    nbf.v4.new_markdown_cell("""# Absolute similarity versus retrieval uniqueness

Bridge cosine and retrieval rank answer different questions. Exact-pair cosine
measures absolute agreement between independent ARCHS4 and recount3 processing.
Rank measures whether other samples in the 510,709-reference corpus are even
closer. The analysis below reuses the saved score cache and random-pair sample;
it performs no embedding inference or new retrieval search."""),
    nbf.v4.new_code_cell("""display(cosine_micro_macro.style.format({c:'{:.6f}' for c in cosine_micro_macro.columns if 'cosine' in c or 'study_mean' in c}).hide(axis='index'))
display(cosine_by_rank[['rank_group','n','mean','median','sd','q1','q3','minimum','maximum']].style.format({c:'{:.6f}' for c in ['mean','median','sd','q1','q3','minimum','maximum']}).hide(axis='index'))
display(plt.imread(ABS / 'figures/exact_cosine_vs_rank.png')); plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Expanded micro and study-macro retrieval

Micro metrics give every one of 562 queries equal weight. Study-macro metrics
first score each GSE and then give every one of 40 studies equal weight.
R@1,000 represents only 0.1958% of the reference corpus and is included to
describe the tail rather than as a primary success criterion."""),
    nbf.v4.new_code_cell("""micro_cols=['method','r_at_1','r_at_5','r_at_10','r_at_100','r_at_1000','mrr','median_rank','mean_rank','median_percentile_rank']
display(expanded_micro[micro_cols].style.format({c:'{:.1%}' for c in ['r_at_1','r_at_5','r_at_10','r_at_100','r_at_1000']}).format({'mrr':'{:.4f}','mean_rank':'{:.2f}','median_percentile_rank':'{:.6f}'}).hide(axis='index'))
macro_cols=['method','macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_r_at_1000','macro_mrr']
display(expanded_macro[macro_cols].style.format({c:'{:.1%}' for c in macro_cols if c.startswith('macro_r')}).format({'macro_mrr':'{:.4f}'}).hide(axis='index'))"""),
    nbf.v4.new_markdown_cell("""## Random-pair placement and near-tied neighborhoods

The exact-pair median lies at the empirical 99.9936th percentile of the saved
two-million random ARCHS4-pair distribution. Yet poor-rank queries can have
thousands of references close to their exact-pair cosine. Counts below use
`reference cosine >= exact cosine - delta`, including closer references and
the exact counterpart."""),
    nbf.v4.new_code_cell("""display(random_exact.style.format({c:'{:.6f}' for c in ['mean','median','sd','q1','q3','minimum','maximum','p1','p5','p95','p99']}).hide(axis='index'))
display(random_percentiles.style.format({'cosine':'{:.6f}','random_pair_empirical_percentile':'{:.4%}','random_pairs_exceeding_fraction':'{:.4%}'}).hide(axis='index'))
display(poor_rank_frequency.style.format({'fraction_all_queries':'{:.2%}','fraction_among_poor_rank':'{:.2%}'}).hide(axis='index'))
display(poor_rank_examples.style.format({'exact_pair_cosine':'{:.6f}','top1_cosine':'{:.6f}','top1_minus_exact_cosine':'{:.6f}'}).hide(axis='index'))
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
for ax, name in zip(axes, ['random_vs_exact_cosine.png','near_exact_neighbors_by_rank.png']):
    ax.imshow(plt.imread(ABS / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Native ranking scores

Each row is grouped by that method's own exact-counterpart rank. Only the
cosine method's native score is Bridge cosine. CSLS, local-scaling, and NICDM
values are transformed ranking criteria, whose numerical scales should not be
compared with cosine or with one another."""),
    nbf.v4.new_code_cell("""display(native_scores[['method','rank_group','native_score_n','native_score_mean','native_score_median','native_score_q1','native_score_q3','bridge_cosine_median']].style.format({c:'{:.6f}' for c in ['native_score_mean','native_score_median','native_score_q1','native_score_q3','bridge_cosine_median']}).hide(axis='index'))"""),
    nbf.v4.new_markdown_cell("""## Interpretation

Mean/median exact-pair cosine is 0.997806/0.999218, and equal-study weighting
still gives a macro mean of 0.996569. Cross-pipeline representation agreement
is therefore broadly strong. Forty-six queries rank below 100 despite this;
33 still exceed cosine 0.99 and four exceed 0.999. For rank >1,000 queries, the
median neighborhood contains 5,944.5 references above exact−0.001 and 14,889
above exact−0.005. These are primarily retrieval-uniqueness failures in crowded
regions, not simple representation-agreement failures. Native local-scaling,
NICDM, and CSLS scores remain ranking criteria—not cosine similarities."""),
    nbf.v4.new_markdown_cell("""# ARCHS4 anisotropy and common components

The mean and PCA basis below were fit once on all 510,709 ARCHS4 references;
the same transform was then applied unchanged to recount3 queries. The squared
norm of the mean unit direction is 0.878774—essentially the observed random-pair
mean cosine 0.878748. Most samples therefore share a large global direction.
Mean-centering moves the random median from 0.912954 to −0.002471."""),
    nbf.v4.new_code_cell("""display(anisotropy.style.format({c:'{:.6f}' for c in anisotropy.columns if c not in ['references','embedding_dim']}).hide(axis='index'))
display(pca_summary.style.format('{:.6f}').hide(axis='index'))
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, name in zip(axes.ravel(), ['raw_random_cosine_histogram.png','raw_random_cosine_ecdf.png','raw_vs_centered_cosine.png','pca_explained_variance.png']):
    ax.imshow(plt.imread(ANISO / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Geometry, separation, and exact retrieval

Centered PC1 explains 57.7% of variance and the covariance participation ratio
is only 2.91 effective dimensions. Centering alone degrades retrieval, but
removing ten centered PCs improves exact R@1 to 37.2%, R@10 to 85.4%, and
study-macro R@10 to 84.5%, exceeding NICDM/local scaling in this paired task.
PC count was evaluated diagnostically here and must be frozen before external
biological evaluation."""),
    nbf.v4.new_code_cell("""display(geometry_distributions[['representation','distribution','n','mean','median','sd','p5','p95','p99']].style.format({c:'{:.6f}' for c in ['mean','median','sd','p5','p95','p99']}).hide(axis='index'))
display(geometry_separation.style.format({c:'{:.6f}' for c in geometry_separation.columns if c != 'representation'}).hide(axis='index'))
display(geometry_retrieval.style.format({c:'{:.1%}' for c in ['r_at_1','r_at_10','r_at_100','r_at_1000','macro_r_at_10']}).format({'mrr':'{:.4f}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, name in zip(axes, ['pc_removal_cosine_distributions.png','retrieval_comparison.png']):
    ax.imshow(plt.imread(ANISO / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Biological preservation, density, and hubness

Centered-minus-PC10 also improves the available same-GSE, exact-source, and
limited blood/brain tissue metrics. Hubness is mixed: Top-1 concentration
falls, while Top-10 concentration rises. GSE234297 remains unusually dense
through PC2 but becomes less dense than other studies after PC5/PC10; common
components redistribute rather than uniformly eliminate local density."""),
    nbf.v4.new_code_cell("""display(geometry_biology.style.format({c:'{:.1%}' for c in ['same_gse_top1','same_gse_top10','source_p_at_1','source_p_at_10','tissue_p_at_1','tissue_p_at_10']}).hide(axis='index'))
display(geometry_density.style.format({c:'{:.6f}' for c in geometry_density.columns if c != 'representation'}).hide(axis='index'))
display(geometry_hubness.style.format({'gini':'{:.6f}','top10_hubs_assignment_fraction':'{:.2%}','skewness':'{:.2f}'}).hide(axis='index'))
display(plt.imread(ANISO / 'figures/hubness_density_comparison.png')); plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Recommendation before GTEx

Retain raw cosine for interpretable absolute Bridge agreement. For technical
identity retrieval, carry centered-minus-PC10 forward as the leading frozen
reference-derived geometry, alongside raw cosine and NICDM/local scaling. A
future GTEx→ARCHS4 benchmark should freeze these three views before evaluation
and choose by biological and cross-study retrieval—not retune PC count on GTEx.
No representation learning or Bridge fine-tuning was performed."""),
    nbf.v4.new_markdown_cell("""# Literature-grounded post-processing

This extension freezes ABTT at D=10 and compares it with full-dimensional
BERT-style whitening. All means, PCs, and whitening parameters were fitted on
the 510,709 ARCHS4 references only and then applied unchanged to recount3. The
same retained two-million random-pair indices are used throughout. No component
count or whitening setting was tuned against recount3 retrieval."""),
    nbf.v4.new_code_cell("""display(literature_distributions[['method','distribution','n','mean','median','sd','p1','p5','p95','p99','p99.9']].style.format({c:'{:.6f}' for c in ['mean','median','sd','p1','p5','p95','p99','p99.9']}).hide(axis='index'))
display(whitening_diagnostics.style.format('{:.6g}').hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, name in zip(axes, ['random_pair_distributions.png','whitening_covariance_spectrum.png']):
    ax.imshow(plt.imread(LIT / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Alignment and uniformity

Alignment is mean squared distance for the 562 normalized exact pairs; lower
is closer. Uniformity is `log E exp(-2||x-y||²)` over the fixed random pairs;
more negative is more uniform. Whitening sacrifices absolute pair closeness
but contracts the random distribution much more, producing the strongest
exact-versus-random AUROC. Uniformity is a diagnostic, not a sole objective."""),
    nbf.v4.new_code_cell("""display(literature_diagnostics.style.format({c:'{:.6f}' for c in literature_diagnostics.columns if c != 'method'}).hide(axis='index'))
display(plt.imread(LIT / 'figures/alignment_uniformity.png')); plt.axis('off'); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Retrieval and study-macro comparison

Whitening is strongest for exact technical identity: R@1 46.62%, R@10 94.48%,
and study-macro R@10 94.12%. ABTT D=10 exactly reproduces the frozen prior
result (37.19%, 85.41%, and 84.46%). Retrieval-score corrections remain
separate: they change ranking criteria without defining a transformed Bridge
representation."""),
    nbf.v4.new_code_cell("""display(literature_main.style.format({c:'{:.1%}' for c in ['r_at_1','r_at_10','r_at_100','r_at_1000','macro_r_at_10']}).format({'mrr':'{:.4f}','alignment':'{:.4f}','uniformity':'{:.4f}'}).hide(axis='index'))
display(literature_macro[['method','macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_r_at_1000','macro_mrr']].style.format({c:'{:.1%}' for c in ['macro_r_at_1','macro_r_at_5','macro_r_at_10','macro_r_at_100','macro_r_at_1000']}).format({'macro_mrr':'{:.4f}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, name in zip(axes, ['retrieval_micro.png','retrieval_study_macro.png']):
    ax.imshow(plt.imread(LIT / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Biological preservation and hubness

ABTT has the strongest conservative source precision; whitening has the
strongest same-GSE metrics and limited tissue P@10. The tissue sensitivity is
blood/brain dominated. Whitening minimizes Top-1 hub concentration, whereas
NICDM/local scaling minimize Top-10 hub concentration. No method dominates
every objective."""),
    nbf.v4.new_code_cell("""display(literature_biology.style.format({c:'{:.1%}' for c in ['same_gse_top1','same_gse_top10','source_p_at_1','source_p_at_10','tissue_p_at_1','tissue_p_at_10']}).hide(axis='index'))
display(literature_hubness.style.format({'top10_hubs_assignment_fraction':'{:.2%}','gini':'{:.6f}','skewness':'{:.2f}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, name in zip(axes, ['biological_preservation.png','hubness_comparison.png']):
    ax.imshow(plt.imread(LIT / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Frozen candidates for independent GTEx validation

Advance raw cosine, frozen ABTT D=10, full 512-D ARCHS4-fitted whitening,
NICDM k=10, and local scaling k=10. Raw cosine preserves direct Bridge-score
interpretability; ABTT is the source-label leader; whitening is the technical-
identity and isotropy leader; local corrections are retrieval-only comparators.
Mean-centering is diagnostic and CSLS remains supplemental. GTEx was not
examined, and no Bridge parameters or learned post-processors were trained."""),
    nbf.v4.new_markdown_cell("""# Independent GTEx → ARCHS4 tissue retrieval

This independent validation reuses 6,314 authoritative GTEx v11 samples and
their existing frozen Bridge embeddings. Frozen ARCHS4-derived transformations
are applied unchanged. Explicit GTEx-origin ARCHS4 records are excluded before
search, and no method is tuned on GTEx."""),
    nbf.v4.new_code_cell("""display(gtex_cohort.style.format({'largest_tissue_fraction':'{:.1%}'}).hide(axis='index'))
display(gtex_annotation.style.format({'coverage':'{:.1%}','largest_labeled_tissue_fraction':'{:.1%}','gtex_represented_coverage_of_full_corpus':'{:.2%}','eligible_gtex_tissue_coverage_of_full_corpus':'{:.2%}'}).hide(axis='index'))
display(gtex_harmonization.style.format({'labeled_prevalence':'{:.3%}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, name in zip(axes, ['gtex_tissue_composition.png','archs4_tissue_coverage.png']):
    ax.imshow(plt.imread(GTEX / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Micro and tissue-macro retrieval

Annotated precision counts only confidently labeled retrieved references.
Hit@K uses all K positions; Unknown@K makes annotation missingness explicit.
Whitening leads tissue-macro P@10 (91.21%), ABTT narrowly leads micro annotated
P@10 (95.39%), and raw has the highest absolute Hit@10 (26.38%). Roughly
72–76% of Top-10 results are unlabeled under the conservative mapping."""),
    nbf.v4.new_code_cell("""display(gtex_micro[gtex_micro.k.eq(10)][['method','annotated_precision','absolute_hit_rate','unknown_rate','annotated_precision_ci_low','annotated_precision_ci_high']].style.format({c:'{:.1%}' for c in ['annotated_precision','absolute_hit_rate','unknown_rate','annotated_precision_ci_low','annotated_precision_ci_high']}).hide(axis='index'))
display(gtex_macro[gtex_macro.k.eq(10)].style.format({c:'{:.1%}' for c in ['annotated_precision','absolute_hit_rate','unknown_rate']}).format({'fold_enrichment':'{:.1f}×'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, name in zip(axes, ['micro_precision_by_k.png','tissue_macro_precision_by_k.png']):
    ax.imshow(plt.imread(GTEX / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Per-tissue behavior, confusions, and density

Brain and liver reach 100% annotated P@10 for all methods. Raw is weakest for
stomach and adipose; whitening substantially improves both, while breast and
spleen decline. Dominant labeled confusions include adipose↔breast and
pancreas↔stomach. Whitening exposes rather than removes between-tissue density
differences. Whole Blood is absent from this cached GTEx cohort."""),
    nbf.v4.new_code_cell("""display(gtex_per_tissue.style.format({c:'{:.1%}' for c in gtex_per_tissue.columns if c.endswith('_p_at_10') or c.endswith('_unknown_at_10')}).hide(axis='index'))
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, name in zip(axes, ['per_tissue_p10_heatmap.png','tissue_confusion_whitening_top10.png','local_density_by_tissue.png']):
    ax.imshow(plt.imread(GTEX / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Controlled GTEx retrieval and technical-versus-biological comparison

After removing self and all same-donor samples, GTEx→GTEx tissue-macro P@10 is
90.61% raw, 93.35% ABTT, and 94.86% whitening. This agrees with the independent
ARCHS4 macro ordering. High raw cosine for both same- and different-tissue
neighbors confirms that decisions occur in the dense extreme tail."""),
    nbf.v4.new_code_cell("""display(gtex_comparison.style.format({c:'{:.1%}' for c in ['recount3_exact_r_at_10','recount3_macro_r_at_10','gtex_archs4_tissue_p_at_10','gtex_tissue_macro_p_at_10','absolute_hit_at_10','unknown_at_10']}).format({'tissue_macro_enrichment_at_10':'{:.1f}×'}).hide(axis='index'))
display(gtex_control[gtex_control.k.eq(10)].style.format({'micro_precision':'{:.1%}','tissue_macro_precision':'{:.1%}'}).hide(axis='index'))
display(gtex_raw_scores.style.format({c:'{:.6f}' for c in ['mean','median','p5','p95']}).format({'median_random_pair_percentile':'{:.3%}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, name in zip(axes, ['technical_vs_biological.png','gtex_to_gtex_control.png']):
    ax.imshow(plt.imread(GTEX / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Recommendation and stop

Use whitening for exact identity and as the primary tissue-macro geometry.
Retain raw cosine for interpretable absolute similarity and labeled-hit yield,
and ABTT D=10 as a prespecified biological sensitivity. Advance all three
frozen representations to future cross-species testing; do not select only
whitening from human-only evidence. NICDM/local scaling remain baselines but
did not improve GTEx tissue retrieval. No cross-species analysis was run."""),
    nbf.v4.new_markdown_cell("""# Retrospective representation × metric controls

This section factors frozen representation, distance metric, and covariance
normalization on the unchanged 562-query × 510,709-reference recount3 task.
The full raw-expression control uses the exact 15,165-gene `log1p(TPM)`
contract reconstructed from existing ARCHS4 counts; all PCA parameters are fit
on ARCHS4 references only. Completed GTEx results are loaded but not rerun or
modified."""),
    nbf.v4.new_code_cell("""display(pd.DataFrame([control_assets]).T.rename(columns={0:'value'}))
display(control_pca.style.format({'variance_explained_512':'{:.2%}','eigenvalue_floor':'{:.3e}','participation_ratio':'{:.1f}'}).hide(axis='index'))"""),
    nbf.v4.new_markdown_cell("""## Retrieval and study-macro controls

Cosine/Pearson are similarities; Euclidean, Manhattan, and Canberra are
distances and are ranked in ascending order. Published raw-Bridge and
Bridge-whitened cosine results remain the authoritative rows in the summary."""),
    nbf.v4.new_code_cell("""cols=['representation','metric','r_at_1','r_at_10','r_at_100','r_at_1000','mrr','median_rank','runtime_seconds']
display(control_micro[cols].style.format({c:'{:.2%}' for c in ['r_at_1','r_at_10','r_at_100','r_at_1000']}).format({'mrr':'{:.4f}','runtime_seconds':'{:.1f}'}).hide(axis='index'))
display(control_macro[['representation','metric','macro_r_at_1','macro_r_at_10','macro_r_at_100','macro_r_at_1000','macro_mrr']].style.format({c:'{:.2%}' for c in ['macro_r_at_1','macro_r_at_10','macro_r_at_100','macro_r_at_1000']}).format({'macro_mrr':'{:.4f}'}).hide(axis='index'))"""),
    nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, name in zip(axes, ['r_at_10.png','separation_auroc.png','best_baseline_vs_bridge_whitened.png']):
    ax.imshow(plt.imread(CTRL / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Attribution and limitations

The tables separate what can be recovered from ordinary expression geometry,
fixed 512-PC compression, PCA whitening, raw Bridge, and the already-frozen
Bridge whitening transform. Random-pair AUROC measures exact-pair separation
without conflating it with rank. The full-expression cache is derived and
ignored; no raw data, embeddings, GTEx analyses, or learned model parameters
were changed. See `REPORT.md` for the numerical attribution and storage/runtime
trade-offs."""),
    nbf.v4.new_markdown_cell("""# Frozen GTEx representation controls

The five conventional controls were fixed by the recount3 analysis before
this extension: expression Canberra/Pearson, PCA-512 Canberra, and PCA-white
cosine/Canberra. They use the unchanged 6,064-query, 13-harmonized-tissue,
510,376-reference benchmark. Existing Bridge rankings are reused exactly; no
GTEx fitting or method selection occurs here."""),
    nbf.v4.new_code_cell("""display(gctrl_identity.style.format({c:'{:.2%}' for c in ['recount3_exact_r_at_10','gtex_archs4_tissue_macro_p_at_10','gtex_gtex_tissue_macro_p_at_10']}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, name in zip(axes, ['identity_vs_biology.png','compression_vs_biology.png']):
    ax.imshow(plt.imread(GCTRL / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Primary and different-donor tissue retrieval

PCA-white cosine leads GTEx→ARCHS4 tissue-macro P@10 (92.08%), narrowly above
PCA Canberra (91.59%) and Bridge-white cosine (91.21%). In the fully labeled
different-donor control, expression Pearson leads (96.12%) and every frozen
conventional control exceeds Bridge whitening (94.86%). Unknown ARCHS4 labels
remain explicit and affect absolute hit yield."""),
    nbf.v4.new_code_cell("""display(gctrl_macro[gctrl_macro.k.eq(10)].style.format({c:'{:.2%}' for c in ['annotated_precision','absolute_hit_rate','unknown_rate']}).format({'fold_enrichment':'{:.1f}×'}).hide(axis='index'))
display(gctrl_control[gctrl_control.k.eq(10)].style.format({'micro_precision':'{:.2%}','tissue_macro_precision':'{:.2%}'}).hide(axis='index'))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, name in zip(axes, ['gtex_archs4_macro_by_k.png','gtex_gtex_macro_by_k.png']):
    ax.imshow(plt.imread(GCTRL / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Tissue-specific errors and neighborhood overlap

Brain and liver are perfect across methods. Representation choice matters most
for adipose, breast, spleen, and stomach. Expression/PCA and Bridge whitening
share only about four to five of ten neighbors on average, so similar tissue
scores often arise from different sample neighborhoods."""),
    nbf.v4.new_code_cell("""p10 = gctrl_tissue[gctrl_tissue.k.eq(10)].pivot(index='tissue', columns='method', values='annotated_precision')
display(p10.style.format('{:.1%}'))
display(gctrl_overlap.groupby('method').agg(mean_shared=('shared_neighbors','mean'),mean_jaccard=('jaccard','mean'),zero_overlap=('shared_neighbors',lambda x:(x==0).mean())).style.format({'mean_shared':'{:.2f}','mean_jaccard':'{:.3f}','zero_overlap':'{:.1%}'}))
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, name in zip(axes, ['per_tissue_p10.png','tissue_confusion_comparison.png','neighborhood_overlap.png']):
    ax.imshow(plt.imread(GCTRL / 'figures' / name)); ax.axis('off')
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## Conclusion and stop

Bridge does not outperform the frozen conventional controls on this tissue
task. Full expression remains strongest for exact identity; expression
Pearson wins the fully labeled GTEx control; and matched-dimensional PCA
slightly leads Bridge whitening on GTEx→ARCHS4 tissue-macro retrieval. Bridge
whitening still exposes useful biology in a compact learned representation,
but broad human-tissue organization is largely explainable by linear
transcriptomic structure. No cross-species or perturbation analysis was run."""),
]
nb.cells.extend([
    nbf.v4.new_markdown_cell("""# Exploratory contextual-metadata audit

The union of cached Top-100 hits is deduplicated before annotation: 24,938 GSMs
are each classified once using frozen explicit-evidence rules over GEO source,
title, and characteristics. Conflicts and missing evidence remain unknown.
No embedding or retrieval calculation is repeated. No cached corpus-wide GSE
title/summary/design asset was available."""),
    nbf.v4.new_code_cell("""display(context_coverage.style.format({'fraction':'{:.2%}'}).hide(axis='index'))
display(context_confidence.style.format({'fraction':'{:.2%}'}).hide(axis='index'))"""),
    nbf.v4.new_markdown_cell("""## Tissue-conditioned context

GTEx explicitly supports a primary-tissue query context. It does not provide
sample-level healthy-state evidence, so `normal_healthy` is a descriptive proxy
rather than authoritative agreement. Precision uses confidently annotated hits;
unknown fractions remain explicit and are the dominant limitation."""),
    nbf.v4.new_code_cell("""ctx = context_metrics[(context_metrics.k == 10) & context_metrics.tissue_conditioned]
display(ctx[['method','axis','retrieved_hits','confident_hits','agreement_precision','absolute_agreement_rate','unknown_fraction']].style.format({c:'{:.2%}' for c in ['agreement_precision','absolute_agreement_rate','unknown_fraction']}).hide(axis='index'))"""),
    nbf.v4.new_markdown_cell("""## Auditable examples and conclusion

Examples use the lowest query index whose Top-1 is tissue-matched and has at
least one confident context annotation. Original metadata and the exact matched
rule/evidence are retained in the saved table. Coverage is too sparse to support
disease-state claims or method tuning; this remains a high-precision exploratory
annotation layer."""),
    nbf.v4.new_code_cell("""cols=['method','gtex_sample','query_tissue','rank','gsm','gse','retrieved_tissue','state','material','source_name_ch1','title','state_evidence','material_evidence']
display(context_examples[context_examples['rank'] <= 3][cols].head(18))"""),
])
nbf.write(nb, HERE / "sample_retrieval_geometry.ipynb")
print(HERE / "sample_retrieval_geometry.ipynb")
