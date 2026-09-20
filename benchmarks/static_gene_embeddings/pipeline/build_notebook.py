#!/usr/bin/env python3
"""Build the human-readable static gene-embedding benchmark notebook."""

from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# BridgeRNA static gene embeddings

This first embedding benchmark asks whether the frozen, pre-contextual gene identity
vectors show interpretable organization. All numerical analysis is produced by
`pipeline/run_static_embeddings.py`; this notebook only loads saved outputs and
renders the report.

The analysis uses the canonical 15,165-gene order and the checkpoint tensor
`model_state_dict.gene_embedding.weight` (15,165 × 512). Average expression is the
arithmetic mean TPM across the **25,370 locally cached human samples that belong
to the model's training manifest**. This frozen reference subset is documented in
`training_expression_cohort.parquet`; it is not presented as the full 640,000-sample
training mean.""")
code("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Image

HERE = Path.cwd()
if HERE.name != 'static_gene_embeddings':
    HERE = HERE / 'benchmarks/static_gene_embeddings'
RESULTS = HERE / 'results/static_embeddings'
analysis = pd.read_csv(RESULTS / 'gene_embedding_analysis.csv')
clusters = pd.read_csv(RESULTS / 'cluster_summary.csv')
go = pd.read_csv(RESULTS / 'go_enrichment.csv')
kegg = pd.read_csv(RESULTS / 'kegg_enrichment.csv')
provenance = json.loads((RESULTS / 'provenance.json').read_text())
assert len(analysis) == 15165 and analysis.gene.nunique() == 15165
assert set(analysis.cluster) == set(range(1, 11))
print(f"Genes: {len(analysis):,}; static dimensions: {provenance['embedding_shape'][1]}; "
      f"expression-reference samples: {provenance['expression']['samples']:,}")""")

md("""## Protocol and invariants

- t-SNE is run on a 50-component PCA representation of the static vectors with
  perplexity 30 and seed 42. It is a visualization only: global t-SNE distances
  are not interpreted.
- K-means is fit to the original, unscaled 512-dimensional vectors with `k=10`,
  50 initializations, and seed 42. Displayed cluster IDs are scikit-learn labels
  plus one; IDs are arbitrary names, not ranks.
- GO Biological Process and KEGG are tested independently by one-sided
  hypergeometric over-representation. The background is the canonical model
  vocabulary intersected with each library; BH correction is applied across all
  eligible terms within each cluster and library.
- TPM is reconstructed with `expm1` from an existing canonical natural
  `log1p(TPM)` cache before taking the arithmetic gene mean.""")
code("""display(clusters.style.format({
    'mean_average_training_tpm': '{:,.2f}',
    'median_average_training_tpm': '{:,.2f}',
}))
display(pd.DataFrame({
    'quantity': ['training-reference samples', 'full training manifest', 'coverage',
                 'GO background', 'KEGG background'],
    'value': [provenance['expression']['samples'], provenance['expression']['full_training_samples'],
              f"{provenance['expression']['coverage_fraction']:.2%}",
              provenance['enrichment']['go']['annotated_background_genes'],
              provenance['enrichment']['kegg']['annotated_background_genes']]
}))""")

md("""## Static embedding map colored by training expression

The color scale uses $\log_{10}(\mathrm{mean\ TPM}+0.001)$ for visibility, while
the saved table retains the unrounded arithmetic mean TPM. Any visible expression
gradient is descriptive association, not evidence that the embedding is merely an
expression lookup.""")
code("display(Image(filename=str(RESULTS / 'figures/tsne_by_training_tpm.png'))) ")
code("""analysis[['average_training_tpm', 'log10_average_training_tpm']].describe(
    percentiles=[.01, .1, .25, .5, .75, .9, .99])""")

md("""## K-means clusters displayed in t-SNE space

K-means operates in 512 dimensions; the following panel only projects those fixed
assignments onto t-SNE. Apparent separation or overlap in two dimensions is not the
clustering objective.""")
code("display(Image(filename=str(RESULTS / 'figures/tsne_by_cluster.png'))) ")

md("""## GO Biological Process enrichment for all clusters

The table reports the five lowest-FDR terms per cluster, including nonsignificant
results when a cluster has fewer than five discoveries. Full tested tables and
overlap genes are retained in `go_enrichment.csv`.""")
code("""go_top = (go.sort_values(['cluster', 'fdr', 'p_value'])
          .groupby('cluster', group_keys=False).head(5)
          [['cluster', 'term', 'overlap_size', 'term_size', 'fdr', 'significant']])
display(go_top.style.format({'fdr': '{:.2e}'}))
print('Significant GO terms per cluster:')
display(go.groupby('cluster').significant.sum().rename('FDR < 0.05').to_frame().T)""")

md("""### Requested GO comparison: clusters 6 and 9

Bars show the ten lowest-FDR terms in each selected cluster. The dashed line is
FDR 0.05; gray bars do not cross it.""")
code("display(Image(filename=str(RESULTS / 'figures/go_clusters_6_9.png'))) ")
code("""display(go.loc[go.cluster.isin([6, 9])].sort_values(['cluster','fdr'])
        .groupby('cluster', group_keys=False).head(10)
        [['cluster','term','overlap_size','cluster_annotated_size','term_size','fdr','overlap_genes']]
        .style.format({'fdr':'{:.2e}'}))""")

md("""## KEGG enrichment for all clusters

The same frozen cluster assignments and library-specific measurable background are
used. Full tested results are retained in `kegg_enrichment.csv`.""")
code("""kegg_top = (kegg.sort_values(['cluster', 'fdr', 'p_value'])
            .groupby('cluster', group_keys=False).head(5)
            [['cluster', 'term', 'overlap_size', 'term_size', 'fdr', 'significant']])
display(kegg_top.style.format({'fdr': '{:.2e}'}))
print('Significant KEGG terms per cluster:')
display(kegg.groupby('cluster').significant.sum().rename('FDR < 0.05').to_frame().T)""")

md("""### Requested KEGG comparison: clusters 2 and 7

Bars show the ten lowest-FDR terms in each selected cluster, with the same multiple-
testing threshold as above.""")
code("display(Image(filename=str(RESULTS / 'figures/kegg_clusters_2_7.png'))) ")
code("""display(kegg.loc[kegg.cluster.isin([2, 7])].sort_values(['cluster','fdr'])
        .groupby('cluster', group_keys=False).head(10)
        [['cluster','term','overlap_size','cluster_annotated_size','term_size','fdr','overlap_genes']]
        .style.format({'fdr':'{:.2e}'}))""")

md("""## Conclusion and limitations

This benchmark establishes a reproducible descriptive map of BridgeRNA's static
gene embedding and tests whether its deterministic k-means partitions concentrate
known GO and KEGG annotations. Enrichment supports functional organization only
where multiple-testing-adjusted results are present; it does not establish causal
biology or contextual behavior.

Major limitations are that t-SNE is nonlinear and local, k-means imposes spherical
Euclidean partitions, cluster numbers are arbitrary, gene-set databases are
incomplete and overlapping, and mean training TPM currently covers the 25,370
model-training samples available in the shared expression cache rather than all
640,000 training samples and contains no mouse samples. The next contextual-embedding subtask should use the same
gene vocabulary and explicitly separate sample context from this static baseline.""")
code("""print(f"Adjusted enrichment discoveries: GO={int(go.significant.sum())}; "
      f"KEGG={int(kegg.significant.sum())}. Thus this static k-means analysis "
      "does not provide FDR-controlled functional enrichment evidence for any cluster.")""")
code("pd.DataFrame({'limitation': provenance['limitations']})")

md("""# Extended static-space characterization

The following exploratory extension compares the unchanged static tensor, row-wise
L2 normalization, and post-hoc row-wise LayerNorm. LayerNorm is only a geometric
transformation of the checkpoint tensor; it is **not** evidence about a model
trained with embedding-level normalization. The screening profile uses three
seeds, K = 5/10/20, 10 annotation nulls, and one shuffled-vector null, so its tail
probabilities are deliberately coarse.""")
code("""CHAR = HERE / 'results/static_characterization'
static_summary = pd.read_csv(CHAR / 'representation_summary.csv')
static_neighbors = pd.read_csv(CHAR / 'neighborhood_functional_overlap.csv')
static_agreement = pd.read_csv(CHAR / 'neighborhood_representation_agreement.csv')
display(static_summary[['representation','pca_participation_ratio','pcs_for_90pct',
 'silhouette','calinski_harabasz','cluster_stability_ari','significant_go_terms',
 'significant_kegg_terms']].style.format({
 'pca_participation_ratio':'{:.1f}','silhouette':'{:.4f}',
 'calinski_harabasz':'{:.2f}','cluster_stability_ari':'{:.4f}'}))""")
code("display(Image(filename=str(CHAR / 'pca_variance_spectra.png'))) ")
code("display(Image(filename=str(CHAR / 'geometry_projections.png'))) ")
md("""The embeddings are nearly full-dimensional: approximately 442–443 PCs are
needed for 90% variance, and the participation ratio is approximately 494–495 of
512. Silhouette scores are effectively zero and repeated K-means solutions have
ARI near 0.005. Thus the visually intermixed projections agree with the quantitative
absence of stable global partitions; the conclusion does not depend on t-SNE or
UMAP appearance.""")
code("""display(static_neighbors[['representation','library','k','shared_pair_fraction',
 'null_mean','fold_over_null','empirical_p']].style.format({
 'shared_pair_fraction':'{:.4f}','null_mean':'{:.4f}','fold_over_null':'{:.3f}',
 'empirical_p':'{:.3f}'}))
display(static_agreement.style.format({'mean_neighbor_jaccard':'{:.3f}',
                                       'sd_neighbor_jaccard':'{:.3f}'}))""")
md("""Static local topology is also close to random (roughly 0.97–1.02× null).
L2 and post-hoc LayerNorm produce nearly the same neighbors (Jaccard ≈ 0.90–0.92),
whereas either shares only ≈ 0.21–0.25 with raw Euclidean neighborhoods. The few
global L2 KEGG or LayerNorm GO discoveries therefore occur without stable global
clusters or a broad local functional effect and should be treated as isolated
exploratory signals.""")

md("""# Contextual-depth follow-up

This lightweight follow-up retains **individual gene tokens**—no sample-level mean
pooling. It uses one deterministic primary tumor from each of eight TCGA cohorts
(GBM, BRCA, LUAD, KIRC, LIHC, PRAD, COAD, and SKCM). L0 is the pure, sample-invariant
`gene_embedding.weight`; L1, L6, and L12 are the per-gene representations after the
corresponding transformer layer. Neighbors are other genes within the same sample,
which prevents trivial retrieval of the same gene from another sample.

Raw cosine and gene-wise mean-centered cosine are evaluated. Top-PC removal was
omitted from this lightweight screen. GO/KEGG use the same 15,165-gene universe at
every depth. Pair-retrieval negatives are annotation-degree-quartile matched.""")
code("""CTX = HERE / 'results/contextual_depth'
depth = pd.read_csv(CTX / 'layerwise_summary.csv')
contexts = pd.read_csv(CTX / 'selected_gene_context_dependence.csv')
examples = pd.read_csv(CTX / 'selected_gene_context_neighborhoods.csv')
display(pd.read_csv(CTX / 'sample_manifest.csv')[['cohort','sample_id','sample_type']])
display(Image(filename=str(CTX / 'contextual_depth_summary.png')))""")
code("""order=['L0_static','L1_early','L6_middle','L12_final']
q=depth[(depth.geometry=='mean_centered_cosine') & (depth.k==25)].copy()
neighbor=q.pivot(index='layer',columns='library',values='neighbor_fold')
auc=q.pivot(index='layer',columns='library',values='related_pair_auroc')
table=pd.DataFrame(index=order)
table['GO neighbor enrichment']=neighbor['GO']
table['KEGG neighbor enrichment']=neighbor['KEGG']
table['GO related-pair AUROC']=auc['GO']
table['KEGG related-pair AUROC']=auc['KEGG']
table['Null expectation']='1.0× neighbors; AUROC 0.5'
display(table.style.format({c:'{:.3f}' for c in table.columns if c!='Null expectation'}))""")
md("""The depth trend supports the stated hypothesis in this exploratory cohort.
For mean-centered cosine at k=25, GO neighborhood enrichment rises from ~1.00×
at L0 to 1.12×, 1.59×, and 2.21× at L1/L6/L12. KEGG rises from ~1.01× to
1.25×, 2.04×, and 3.20×. Related-pair AUROC increases more modestly—from near
chance at L0 to approximately 0.542 (GO) and 0.562 (KEGG) at L12. Raw cosine
shows the same ordering but weaker intermediate-layer effects, indicating that
anisotropy removal helps expose rather than create the depth trend.""")
code("""display(depth[(depth.k==25)][['layer','geometry','library','neighbor_fold',
 'neighbor_fold_sd','related_pair_auroc','auroc_sd','pair_smd']]
 .sort_values(['geometry','library','layer']).style.format({
 'neighbor_fold':'{:.3f}','neighbor_fold_sd':'{:.3f}','related_pair_auroc':'{:.3f}',
 'auroc_sd':'{:.3f}','pair_smd':'{:.3f}'}))""")

md("""## Same-gene neighborhoods across contexts

The examples below quantify how top-10 centered-cosine neighborhoods for the same
anchor change across the eight tumors. Cross-context Jaccard below one demonstrates
context dependence; GO/KEGG coherence records whether changing neighbors still
share an annotation with the anchor. These examples are descriptive and are not
tissue-specific enrichment tests.""")
code("""display(contexts.style.format({
 'mean_cross_context_neighbor_jaccard':'{:.3f}','go_coherent_fraction':'{:.3f}',
 'kegg_coherent_fraction':'{:.3f}'}))
display(examples[(examples.layer=='L12_final') & examples.anchor_gene.isin(['CD3D','ESR1','TP53'])]
 [['cohort','anchor_gene','rank','neighbor_gene','anchor_log1p_tpm','shares_go','shares_kegg']]
 .sort_values(['anchor_gene','cohort','rank']).head(90))""")

md("""## Integrated interpretation

The exploratory evidence supports a progression from **static gene identity to
contextual functional organization**. The static table is high-dimensional,
globally unstable under K-means, and locally near random under GO/KEGG overlap.
Functional topology then increases monotonically across L1, L6, and L12 in eight
diverse expression contexts, under both raw and mean-centered cosine. The stronger
neighborhood effects than pairwise AUROC suggest that contextualization produces
useful local topology more clearly than broad global separation.

This is a screening result, not a definitive validation: it uses eight tumors,
10 null replicates, and annotation-overlap endpoints. A confirmatory analysis
should freeze a larger tissue-balanced cohort, increase null replicates, include
healthy tissues, and validate the depth trend with pathway-disjoint or held-out
annotations.""")

md("""# Balanced GTEx replication and matched TCGA comparison

The contextual experiment is extended to two frozen, balanced cohorts:

- **GTEx:** 10 biologically distinct normal tissues × 20 samples = 200 samples.
- **TCGA:** 10 tumor contexts × 20 primary tumors = 200 samples.

Both arms use the canonical 15,165-gene order, species-correct GENCODE v49
exon-length TPM followed by natural `log1p`, checkpoint `r7hnr92k`, layers
L0/L1/L6/L12, raw and mean-centered cosine, k = 10/25/50, identical annotation
backgrounds/null controls, and identical clustering and visualization seeds.
GTEx and TCGA are matched analytically, although normal tissues and tumors are not
paired biological conditions.""")
code("""MATCH = HERE / 'results/matched_gtex_tcga'
matched = pd.read_csv(MATCH / 'gtex_tcga_summary.csv')
dependence = pd.read_csv(MATCH / 'context_dependence_summary.csv')
matched_manifest = pd.read_csv(MATCH / 'cohort_manifest.csv')
display(matched_manifest.groupby(['dataset','context']).size().rename('samples').to_frame())
display(Image(filename=str(MATCH / 'gtex_tcga_matched_summary.png')))""")

md("""## Functional neighborhood replication

The table uses mean-centered cosine and k=25. L0 is evaluated once per dataset
because the pure lookup vector is sample-invariant; contextual layers summarize
all 200 samples per dataset.""")
code("""layer_order=['L0_static','L1_early','L6_middle','L12_final']
mq=matched[(matched.geometry=='mean_centered_cosine') & (matched.k==25)].copy()
display(mq[['dataset','layer','library','neighbor_fold','neighbor_fold_sd','samples_x']]
 .sort_values(['dataset','library','layer']).style.format({
 'neighbor_fold':'{:.3f}','neighbor_fold_sd':'{:.3f}'}))""")
md("""GTEx reproduces the TCGA depth trend closely. At k=25, GTEx GO enrichment
increases from 1.00× at L0 to 1.11×, 1.57×, and 2.23× at L1/L6/L12; KEGG
increases from 1.01× to 1.23×, 2.00×, and 3.25×. The matched TCGA values are
1.12×/1.59×/2.22× for GO and 1.25×/2.04×/3.21× for KEGG. Thus the effect
replicates in normal tissues and is not specific to the earlier eight-tumor screen.""")

md("""## Genome-wide context dependence

For every gene and layer, unit-normalized contextual tokens are accumulated across
all samples. Mean same-gene cosine is calculated exactly for pairs within a tissue
or tumor context and for pairs across contexts. This is a gene-level test, not
sample-level tissue clustering.""")
code("""display(dependence.style.format({
 'within_context_cosine':'{:.4f}','across_context_cosine':'{:.4f}',
 'within_minus_across':'{:.4f}','genes_positive_difference':'{:.1%}'}))""")
md("""L0 has no context dependence, as required for a sample-invariant lookup.
At every contextual layer, all genes have higher average within-context than
across-context similarity. Separation grows with depth and is larger in GTEx:
0.013, 0.093, and 0.152 at L1/L6/L12, compared with 0.005, 0.046, and 0.078
in TCGA. This suggests that normal-tissue identity produces stronger context-
specific gene geometry than the selected tumor contexts.""")

md("""## Contextual gene clustering

Within each sample, centered unit-normalized gene tokens are clustered with
spherical k-means (`k=10`) under two seeds. Silhouette and ARI quantify geometric
separation and initialization stability. Enrichment is calculated from the same
assignments against the model gene universe; `significant_terms` is the mean sum
of FDR-significant terms across the ten clusters per sample.""")
code("""cq=(mq.groupby(['dataset','layer'],as_index=False)
    .agg(silhouette=('silhouette','first'),stability_ari=('stability_ari','first'),
         mean_significant_go=('significant_terms',lambda x: x[mq.loc[x.index,'library'].eq('GO')].mean()),
         mean_significant_kegg=('significant_terms',lambda x: x[mq.loc[x.index,'library'].eq('KEGG')].mean())))
display(cq.style.format({'silhouette':'{:.4f}','stability_ari':'{:.3f}',
 'mean_significant_go':'{:.1f}','mean_significant_kegg':'{:.1f}'}))""")
md("""Global gene clusters remain only weakly separated—the L12 silhouette is
approximately 0.048—but become substantially more reproducible after
contextualization (ARI ≈ 0.42–0.46 versus ≈ 0.004 at L0) and carry many adjusted
GO/KEGG associations. This is consistent with overlapping functional modules:
functionally coherent local and cluster structure can increase without producing
well-separated spherical clouds.""")

md("""## Matched visualizations

A fixed set of 2,000 genes is used in both datasets. For each layer, GTEx and TCGA
are fit jointly with PCA-50 and jointly projected with identical t-SNE and UMAP
parameters and seed. Colors are fixed clusters from the average of the matched
GTEx and TCGA gene representations. These panels are qualitative only.""")
code("display(Image(filename=str(MATCH / 'matched_contextual_projections.png'))) ")

md("""## Cross-cohort conclusion

The balanced GTEx cohort independently reproduces the L0→L12 increase in GO and
KEGG neighborhood organization seen in TCGA. Static L0 remains sample-invariant,
near-null, and unstable under clustering. Contextual representations progressively
develop functional neighborhoods, more stable gene modules, and context-specific
same-gene states. GTEx and TCGA have remarkably similar functional-depth curves,
but GTEx develops stronger tissue-specific same-gene separation.

Together, the evidence supports the hypothesis that the lookup table primarily
provides gene identity, while the transformer combines that identity with the
sample expression state to create biologically organized, context-dependent gene
representations. The result remains exploratory because annotation overlap is not
an independent downstream phenotype, tumor and normal contexts are not paired,
and the clustering model imposes spherical modules on overlapping biology.""")

md("""# Cohort-level contextual gene modules and TPM controls

The preceding within-sample analysis asks whether a particular expression state
contains gene modules. This complementary analysis averages each gene's 512-D
token across the same 200 balanced samples, separately for GTEx and TCGA, and then
clusters the resulting 15,165 original vectors. K-means is fit in 512 dimensions;
t-SNE and UMAP are display-only.

For every dataset/layer, ordinary Euclidean K-means uses `k=10`, five seeds, and
10 initializations per seed. Stability is mean pairwise ARI. GO and KEGG use the
same canonical model background and BH correction within cluster/library.""")
code("""MOD = HERE / 'results/contextual_gene_modules'
modules = pd.read_csv(MOD / 'contextual_gene_module_summary.csv')
controls = pd.read_csv(MOD / 'expression_control_summary.csv')
display(modules[['dataset','layer','silhouette','stability_ari','min_cluster_size',
 'max_cluster_size','significant_go_terms','significant_kegg_terms','top_go_term',
 'top_kegg_term']].style.format({'silhouette':'{:.4f}','stability_ari':'{:.3f}'}))""")
md("""Module stability and functional enrichment increase strongly with depth in
both datasets. L0 is unstable (ARI ≈ 0.006) and has only one adjusted GO and KEGG
term. At L12, ARI reaches 0.779 in GTEx and 0.850 in TCGA, with thousands of GO
and hundreds of KEGG discoveries. Silhouette remains low (≈0.02–0.03), again
indicating stable overlapping biological partitions rather than sharply separated
geometric islands.""")

md("""## Matched projections colored by 512-D module

GTEx and TCGA are jointly fit with PCA-50 and jointly projected per layer using
identical t-SNE/UMAP settings. Color is the cohort-specific K-means assignment
obtained from the original 512-D vectors. Cluster numbers are arbitrary and are
not homologous across datasets or layers even when colors match.""")
code("display(Image(filename=str(MOD / 'matched_projections_by_clusters.png'))) ")

md("""## The same projections colored by expression

The coordinates below are identical to the cluster-colored panels; only color is
changed to cohort mean natural `log1p(TPM)`. This reveals how much of the visible
geometry follows expression level.""")
code("display(Image(filename=str(MOD / 'matched_projections_by_expression.png'))) ")

md("""## Quantifying TPM dependence

Four complementary diagnostics are used:

- Spearman correlation between embedding norm and mean `log1p(TPM)`.
- Spearman correlation between pairwise cosine similarity and negative absolute
  expression difference across 200,000 random gene pairs.
- Cluster effect size for TPM (η²) and NMI with expression deciles.
- Mean TPM difference among top-25 embedding neighbors relative to random pairs.
""")
code("""display(modules[['dataset','layer','norm_tpm_spearman',
 'pair_cosine_expression_similarity_spearman','cluster_tpm_eta_squared',
 'cluster_expression_bin_nmi','neighbor_expression_difference_ratio']]
 .style.format({c:'{:.3f}' for c in ['norm_tpm_spearman',
 'pair_cosine_expression_similarity_spearman','cluster_tpm_eta_squared',
 'cluster_expression_bin_nmi','neighbor_expression_difference_ratio']}))""")
md("""TPM is an important organizing variable after contextualization. At L12,
cluster membership explains approximately 70% of GTEx and 77% of TCGA mean-TPM
variance; similar-expression genes also have higher cosine similarity, and top-25
neighbors differ in expression by only 40% (GTEx) or 35% (TCGA) of the random-pair
difference. The modules therefore cannot be described as expression-independent.""")

md("""## Is functional organization reducible to TPM?

Two controls address this directly. First, genes are divided into ten equal-sized
mean-expression bins and those bins are tested for enrichment. Second, mean
`log1p(TPM)` is linearly regressed from every one of the 512 embedding dimensions;
the residual vectors are reclustered with the same K-means parameters. Linear
residualization is conservative but does not remove nonlinear expression effects.""")
code("""display(controls.style.format({'silhouette':'{:.4f}',
                                       'stability_ari':'{:.3f}'}))""")
md("""Expression deciles alone are biologically enriched (382/73 GO/KEGG terms in
GTEx and 493/98 in TCGA), confirming that abundance is a genuine biological
confound. However, the L12 TPM-residual modules retain 2,229/305 significant
GO/KEGG terms in GTEx and 2,192/280 in TCGA, with ARI 0.927 and 0.859. Most of
the functional organization therefore survives removal of the linear TPM axis.

The supported interpretation is nuanced: contextualization organizes genes partly
by expression magnitude, but the reproducible and functionally coherent L12
modules are not primarily reducible to that magnitude. Functional enrichment and
module stability remain strong after explicit TPM residualization in both normal
tissue and tumors.""")

nb['cells'] = cells
nb['metadata'] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}
nbf.write(nb, HERE / "static_gene_embeddings.ipynb")
print(HERE / "static_gene_embeddings.ipynb")
