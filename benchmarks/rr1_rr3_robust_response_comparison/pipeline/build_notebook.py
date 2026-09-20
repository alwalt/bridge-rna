#!/usr/bin/env python3
"""Build the publication-facing RR1/RR3 robust-response notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks/rr1_rr3_robust_response_comparison"
NOTEBOOK = BENCH / "rr1_rr3_robust_response_comparison_benchmark.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
cells = []
M = cells.append
M(nbf.v4.new_markdown_cell("""# RR1/RR3 protocol-robust response comparison

This notebook asks what survives technical remeasurement in RR1 and how that compares with RR3-39 and RR3-40. Every response is **mean(FLT) − mean(GC)**. It reuses matched animals and frozen outputs; BridgeRNA inference and Integrated Gradients were not rerun.

“Robust” below means concordant across the specified original/remeasurement pair. It does **not** mean protocol-independent, causal, or independently biologically replicated."""))
M(nbf.v4.new_code_cell("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path.cwd()
if ROOT.name == 'rr1_rr3_robust_response_comparison': ROOT = ROOT.parents[1]
B = ROOT/'benchmarks/rr1_rr3_robust_response_comparison'
R = B/'results'
pd.set_option('display.max_colwidth', 100)
print('Benchmark:', B)"""))
M(nbf.v4.new_markdown_cell("""## 1. Cohorts and exact technical counterparts

RR1 compares OSD-48 with OSD-168; RR3 compares the matched 39-day and 40-day OSD-137 strata with OSD-168. The table below records sample counts, animals, library preparation, sequencing configuration, preservation, strain, sex, and RIN where available. These are technical remeasurements of matched biological material—not new biological replicates."""))
M(nbf.v4.new_code_cell("""meta = pd.read_csv(R/'manifest/cohort_metadata_summary.csv')
display(meta.style.hide(axis='index').format({'RIN_median':'{:.2f}','RIN_min':'{:.2f}','RIN_max':'{:.2f}'}, na_rep='—'))
animals = pd.read_csv(R/'manifest/exact_matched_animals.csv')
display(animals.groupby(['cohort','measurement','OSD','condition']).agg(n=('sample_id','nunique'), animals=('animal_id', lambda x: ', '.join(x))).reset_index().style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""## 2. Primary comparison

The same definitions are used for all cohorts. Expression and signed IG robustness require mutual Top-500 membership and concordant sign. Graph robustness uses a positive Top-500 per-gene neighborhood-response cosine. Counts supported by ≥2 or all 3 evidence levels are intersections of these prespecified indicators, not learned scores."""))
M(nbf.v4.new_code_cell("""primary = pd.read_csv(R/'summary/primary_comparison.csv', index_col=0)
display(primary.style.format('{:.3f}', na_rep='—'))"""))
M(nbf.v4.new_code_cell("""display(Image(filename=str(R/'figures/multiscale_reproducibility.png'), width=900))
display(Image(filename=str(R/'figures/robust_core_size.png'), width=900))"""))
M(nbf.v4.new_markdown_cell("""**Headline.** RR3-40 is the strongest technical replication: global response cosine 0.917, expression cosine 0.822, Hallmark cosine 0.922, and 172 genes supported by at least two levels. RR3-39 is moderate/good. RR1 reverses globally (−0.807) and at the Hallmark level (−0.424), with weaker expression, IG-rank, and local-graph agreement. Nevertheless, RR1 retains a restricted 114-gene ≥2-level core and 26 genes supported at all three levels.

RR1's signed IG cosine (0.727) should not be read alone: its low IG Spearman (0.194) shows extensive rank reweighting. Cosine can remain positive when a shared high-magnitude component dominates."""))
M(nbf.v4.new_markdown_cell("""## 3. Expression and attribution robustness

Top-N overlap and genome-wide direction agreement complement correlation. The scatter plot highlights the all-three-level genes and the largest expression interactions."""))
M(nbf.v4.new_code_cell("""topn = pd.read_csv(ROOT/'benchmarks/rr1_library_prep_interaction/results/robust_core/rr3_controls/topn_overlap.csv')
display(topn.pivot_table(index=['level','top_n'], columns='cohort', values=['overlap','direction_agreement']).style.format({'direction_agreement':'{:.1%}'}, na_rep='—'))
repro = pd.read_csv(ROOT/'benchmarks/rr1_library_prep_interaction/results/robust_core/rr3_controls/multilevel_reproducibility.csv')
display(repro.style.hide(axis='index').format(precision=3, na_rep='—'))
display(Image(filename=str(R/'figures/original_vs_remeasurement.png'), width=1100))"""))
M(nbf.v4.new_markdown_cell("""## 4. What BridgeRNA adds

Expression supplies the conventional response baseline. IG asks which input genes influence the frozen latent response; contextual graphs ask whether each gene's local learned neighborhood changes reproducibly. These are multiscale confirmations, not proof that BridgeRNA discovered a raw protocol effect."""))
M(nbf.v4.new_code_cell("""contribution = pd.read_csv(R/'summary/model_contribution.csv')
display(contribution.pivot(index='evidence', columns='cohort', values='genes').fillna(0).astype(int).style)
ax = contribution.pivot(index='cohort', columns='evidence', values='genes').plot.bar(figsize=(10,5))
ax.set(xlabel='', ylabel='Genes', title='Model contribution under identical evidence definitions'); ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()"""))
M(nbf.v4.new_markdown_cell("""## 5. Measurement-sensitive component

For each cohort, `Interaction = ΔX_remeasurement − ΔX_original`; the Top-500 absolute interactions were tested by over-representation analysis against the exact 15,165-gene BridgeRNA universe. These are model-vocabulary expression effects—not a replacement for the existing raw-count edgeR analyses."""))
M(nbf.v4.new_code_cell("""sensitive = pd.read_csv(R/'pathways/measurement_sensitive_enrichment.csv')
top_sensitive = (sensitive.query('fdr < 0.05').sort_values(['cohort','fdr']).groupby('cohort').head(8))
display(top_sensitive[['cohort','term','overlap','set_size','fdr']].style.hide(axis='index').format({'fdr':'{:.2e}'}))"""))
M(nbf.v4.new_markdown_cell("""RR1's leading interaction enrichments are consistently **mRNA processing / splicing** terms (FDR down to ~2.3×10⁻¹¹), making this sensitivity especially prominent in RR1. RR3-39 instead has strong light/visual-perception and myogenesis interaction terms; RR3-40 has weaker mitochondrial RNA-processing and immune terms. This does not make RNA processing a pure artifact—it shows that its measured RR1 response is unusually protocol-sensitive."""))
M(nbf.v4.new_markdown_cell("""## 6. T-cell-independent robust pathway core

The primary pathway profile now uses the full signed input-gene IG vectors directly. A pathway is protocol-robust when it is mutually Top-250 by absolute signed-IG pathway score and has concordant sign between the original and remeasurement. No T-cell projection is used.

This yields 125 robust profiles in RR1, 178 in RR3-39, and 200 in RR3-40. **Twenty-two pathways are shared across all three**, replacing the former 29-pathway PC1–2-derived result."""))
M(nbf.v4.new_code_cell("""ind = pd.read_csv(R/'pathways/independent_signed_ig_pathway_classification.csv')
ind_upset = pd.read_csv(R/'summary/independent_pathway_upset_intersections.csv')
ind_core = pd.read_csv(R/'pathways/independent_conserved_core_full.csv')
ind_modules = pd.read_csv(R/'pathways/independent_conserved_core_modules.csv')
display(ind.groupby('cohort').protocol_robust.sum().rename('robust pathways').to_frame().style)
display(ind_upset.query("intersection != 'None'").style.hide(axis='index'))
display(Image(filename=str(R/'figures/independent_robust_pathway_upset.png'), width=900))
display(Image(filename=str(R/'figures/independent_conserved_core_dotplot.png'), width=900))
display(ind_core[['module','source','pathway']].drop_duplicates().style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""The independent 22-pathway core collapses into seven interpretable themes: lipid/carnitine/bile metabolism; amino-acid and central-carbon metabolism; ER/integrated stress; innate immune/complement regulation; stress/metabolic transcription; fluid/water transport; and IGF-receptor signaling. Dot-plot signs describe signed attribution contribution—not pathway up/down regulation."""))
M(nbf.v4.new_markdown_cell("""### Conventional and BridgeRNA support

The matrix separates raw-count edgeR ranked GSEA, enrichment of technically reproducible IG genes, and the robust signed-IG pathway-profile evidence. Scores are scaled within method for color and are never treated as directly comparable units. Because the 22 pathways were selected using the signed-IG pathway profile, that column is true by construction; DE/GSEA and gene-level IG enrichment provide the independent checks."""))
M(nbf.v4.new_code_cell("""three = pd.read_csv(R/'pathways/independent_core_three_method_matrix.csv')
method_overlap = pd.read_csv(R/'summary/independent_core_method_overlap.csv')
display(Image(filename=str(R/'figures/independent_core_three_method_heatmap.png'), width=1150))
display(three.style.hide(axis='index').format({'signed_score':'{:+.3f}'},na_rep='—'))
display(method_overlap.style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""At the collapsed-theme level, conventional support is limited: RR1 supports innate immune/complement regulation and RR3-39 supports amino-acid/central-carbon metabolism. Robust-gene IG enrichment independently supports lipid/carnitine/bile metabolism in all three cohorts; it additionally supports innate immune/complement regulation in RR1 and ER-stress biology in RR3-39. Thus the primary conserved core is BridgeRNA-attribution-derived rather than a restatement of significant conventional GSEA.

Conventional analysis still establishes broader response reproducibility—especially in RR3—and the earlier matched-expression analysis shows exploratory RR1/RR3 opposition. But that opposition is estimator-dependent: matched Δlog1p(TPM) supports it more strongly than full-stratum edgeR. It is therefore retained only as a secondary observation, not a primary pathway conclusion."""))
M(nbf.v4.new_markdown_cell("""### Individual-pathway validation and prior spaceflight evidence

The 22 full signed-IG pathways—not the older T-cell PC1–2-derived set—are checked below against raw-count edgeR ranked GSEA in each cohort. `IG + conventional support` means at least one cohort has exact-pathway GSEA FDR < 0.05. `IG reprioritization` means the pathway was testable in all three cohorts but conventional GSEA was nonsignificant in all three. `Ambiguous` is reserved for incomplete GSEA coverage.

The literature labels are conservative module-level annotations from a focused rodent/spaceflight literature audit. “Potentially novel/hypothesis-generating” means direct evidence was not found in that audit; it is not a formal claim of novelty."""))
M(nbf.v4.new_code_cell("""validation = pd.read_csv(R/'pathways/independent_core_pathway_validation.csv')
validation_summary = pd.read_csv(R/'summary/independent_core_pathway_validation_summary.csv')
display(validation_summary.style.hide(axis='index'))
view = validation[['module','pathway','RR1_GSEA_NES','RR1_GSEA_FDR','RR3_39_GSEA_NES','RR3_39_GSEA_FDR','RR3_40_GSEA_NES','RR3_40_GSEA_FDR','GSEA_recurrence','evidence_classification','prior_spaceflight_evidence','interpretation']]
display(view.style.hide(axis='index').format({c:'{:.3f}' for c in ['RR1_GSEA_NES','RR3_39_GSEA_NES','RR3_40_GSEA_NES']}).format({c:'{:.2e}' for c in ['RR1_GSEA_FDR','RR3_39_GSEA_FDR','RR3_40_GSEA_FDR']}, na_rep='not tested'))
lit = pd.read_csv(R/'pathways/spaceflight_literature_evidence.csv')
display(lit.style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""**Result.** Twenty of 22 pathways were present in the exact conventional GSEA collection for all three cohorts. Only two received conventional FDR support in any cohort: **Regulation of Complement Cascade** in RR1 (NES 1.688, FDR 0.0473) and **Glyoxylate Metabolism and Glycine Degradation** in RR3-39 (NES 1.693, FDR 0.0398). No pathway was conventionally significant in all three cohorts. The other 18 fully testable pathways are reproducible signed-IG reprioritizations with weak/absent exact-pathway GSEA support; Fluid Transport and Water Transport remain ambiguous because the conventional GSEA resource did not test their exact terms.

The literature audit shows that much of the core is biologically plausible rather than unprecedented: lipid transport, triglyceride/fatty-acid metabolism, carnitine, bile, and amino-acid/central-carbon metabolism have direct rodent liver spaceflight precedent. ER/integrated stress, innate immune/complement, and FOXO/RORA-like metabolic-stress regulation have related precedent, but their exact recurrent pathway identities are less firmly established. Fluid/water transport and IGF-receptor signaling are the clearest hypothesis-generating members. This supports the restrained conclusion that BridgeRNA **reprioritizes reproducible pathway structure** that conventional exact-pathway GSEA usually does not elevate; it does not establish that all 18 signals are new or causal."""))
M(nbf.v4.new_markdown_cell("""### Secondary technical PolyA/ribo control

The former 29-pathway/six-module result was derived from contextual gene responses projected into the controlled T-cell PolyA/ribo PC1–2 reference. It is retained in machine-readable outputs as a technical sensitivity analysis, but removed from the main biological figures and conclusions. Its score direction must not be interpreted as pathway up/down regulation."""))
M(nbf.v4.new_markdown_cell("""## 7. Shared and cohort-specific biology"""))
M(nbf.v4.new_code_cell("""bio = pd.read_csv(R/'summary/biological_summary.csv')
shared = json.loads((R/'summary/shared_unique_counts.json').read_text())
display(pd.Series({k:v for k,v in shared.items() if 'pathway' not in k},name='N').to_frame().style)
display(bio.query("feature_type == 'gene' and interpretation == 'shared_all_three'").style.hide(axis='index'))
display(ind_core[['module','source','pathway']].drop_duplicates().style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""Only five ≥2-level genes are shared by all three comparisons: **GADD45G, IGFBP1, NRN1, SLC41A2, and TCIM**. The T-cell-independent pathway core contains 22 signed-IG profiles spanning metabolic, stress-response, immune/complement, transport, and signaling biology. RR3-39 and RR3-40 share a larger robust gene/pathway response than RR1, consistent with better technical replication. Cohort-specific calls remain candidates rather than universal spaceflight markers."""))
M(nbf.v4.new_markdown_cell("""## 8. Attribution-only genes: model priorities missed by expression ranking

An **attribution-only** gene has reproducible signed IG across the original and remeasurement (mutual Top-500 with concordant sign) but does not meet the corresponding robust-expression criterion. This identifies genes consistently influential to the frozen latent response despite not being among the strongest reproducible marginal expression effects. It does not establish causal importance."""))
M(nbf.v4.new_code_cell("""ae = pd.read_csv(R/'summary/attribution_vs_expression_summary.csv')
graph = pd.read_csv(R/'summary/attribution_only_graph_support.csv')
shared_attr = pd.read_csv(R/'genes/shared_attribution_only_genes.csv')
display(ae.style.hide(axis='index').format({'median_abs_expression_effect':'{:.3f}','median_expression_rank':'{:.1f}','median_attribution_rank':'{:.1f}'}))
print('Attribution-only genes recurring in ≥2 cohorts:', len(shared_attr))
print('Attribution-only genes recurring in all 3 cohorts:', int((shared_attr.cohort_count == 3).sum()))
rank_cols=[c for c in shared_attr if c.endswith('_attribution_rank_mean')]
shared_attr['mean_available_IG_rank']=shared_attr[rank_cols].mean(axis=1,skipna=True)
display(shared_attr.sort_values(['cohort_count','mean_available_IG_rank'],ascending=[False,True]).head(25).style.hide(axis='index').format(precision=1,na_rep='—'))
display(Image(filename=str(R/'figures/attribution_vs_expression_rank.png'), width=1100))"""))
M(nbf.v4.new_markdown_cell("""Attribution-only genes have substantially smaller conventional expression effects and much poorer expression ranks than robust-expression genes, by construction, while retaining strong reproducible IG ranks. There are 200, 271, and 247 attribution-only genes in RR1, RR3-39, and RR3-40; 152 recur in at least two cohorts and 37 recur in all three."""))
M(nbf.v4.new_markdown_cell("""### Biological coherence and contextual-graph support"""))
M(nbf.v4.new_code_cell("""enr = pd.read_csv(R/'pathways/attribution_only_enrichment.csv')
display(enr.query('fdr < 0.05').sort_values(['cohort','fdr']).groupby('cohort').head(10)[['cohort','gene_set','term','overlap','set_size','fdr']].style.hide(axis='index').format({'fdr':'{:.2e}'}))
display(graph.style.hide(axis='index').format({'graph_support_fraction':'{:.1%}','median_graph_local_cosine':'{:.3f}'}))
display(Image(filename=str(R/'figures/attribution_only_graph_support.png'), width=800))
display(pd.read_csv(R/'summary/deletion_compatibility.csv').style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""The attribution-only sets are biologically coherent: xenobiotic and bile-acid metabolism recur prominently, with lipid/fatty-acid, amino-acid/catabolic, complement, peroxisomal, and oxidative-phosphorylation programs varying by cohort. The ≥2-cohort set is likewise enriched for xenobiotic metabolism, adipogenesis, complement, bile-acid metabolism, and NAD(P)-related metabolism.

However, contextual-graph confirmation is **not stronger** for attribution-only genes: only 10.0% (RR1), 6.3% (RR3-39), and 9.3% (RR3-40) meet the stringent robust-graph criterion, versus 36.1%, 12.9%, and 14.4% of expression-only genes. Existing deletion sweeps targeted different mode-level IG rankings and cannot validate these exact gene sets.

**Scientist-facing conclusion:** BridgeRNA consistently prioritizes a recurrent, pathway-coherent set of genes that conventional response magnitude does not rank highly. This is evidence of additional model-derived ranking structure, especially around hepatic xenobiotic, bile-acid, lipid, and amino-acid metabolism. The sparse graph support and lack of a compatible attribution-only deletion test prevent the stronger claim that these genes constitute a separately validated causal response program."""))
M(nbf.v4.new_markdown_cell("""## 9. Direct answers

1. **How much RR1 survives?** A restricted core survives: 179 expression genes, 246 IG genes, 476 positive local graph neighborhoods, 114 genes supported by ≥2 levels, and 26 by all three. Its global and Hallmark response directions do not reproduce.
2. **Compared with RR3?** RR3-40 is strongest (381 expression genes; 172 ≥2-level; 31 all-three), followed by RR3-39 at the global/pathway scales. RR1 is markedly less reproducible.
3. **Is RR1 uniquely splicing-sensitive?** It is the clearest cohort for measurement-sensitive mRNA processing/splicing enrichment. Related terms occur elsewhere, so “unique” should not be interpreted as exclusive biology.
4. **What is robust in all three?** Five multiscale genes and 22 T-cell-independent signed-IG pathway profiles, centered on lipid/carnitine/bile and central-carbon metabolism, stress responses, immune/complement regulation, transport, and metabolic transcription.
5. **What is cohort-specific?** 90 RR1-only, 75 RR3-39-only, and 134 RR3-40-only ≥2-level genes under these definitions; these may reflect duration, cohort, or protocol.
6. **Does attribution identify a more stable subset than expression alone?** It supplies orthogonal model evidence and narrows candidates through intersections, but RR1 attribution ranks themselves are weakly reproducible; it does not simply outperform expression.
7. **Does the graph add confirmation?** Yes, at the gene-neighborhood level, but median local agreement remains much lower in RR1 (0.138) than RR3-39/40 (0.340/0.386).
8. **Why does RR3-40 reproduce better?** Numerically it is more concordant at expression, global latent, Hallmark, graph, and pathway levels. The benchmark establishes this association, not a causal mechanism.
9. **Is RR1 failure technical or biological?** Mixed. It coincides with a major protocol transition and strong measurement-sensitive programs, while a smaller concordant biological core remains.
10. **What remains defensible?** A restricted multiscale gene core plus a T-cell-independent signed-IG pathway core involving lipid/carnitine/bile and central-carbon metabolism, stress responses, immune/complement regulation, transport, and metabolic transcription. Claims should remain at the level of concordant attribution."""))
M(nbf.v4.new_markdown_cell("""## Scientist-facing synthesis

1. **Conventional analysis established** that RR3-40 is the strongest technical replication, RR3-39 also reproduces well, and RR1 is substantially more measurement-sensitive. Conventional matched expression retains a restricted RR1 core and suggests exploratory RR1/RR3 directional opposition, but the opposition is weaker in full-stratum edgeR and is therefore estimator-dependent.
2. **BridgeRNA added** reproducible gene-importance and contextual-neighborhood evidence beyond marginal expression ranking. A T-cell-independent signed-IG pathway analysis identifies 22 profiles conserved across all three technical comparisons; robust-gene enrichment independently reinforces lipid/carnitine/bile metabolism across all three and selected immune or stress programs in individual cohorts.
3. **The robust biological conclusion** is that metabolic, stress-response, immune/complement, transport, and regulatory programs retain concordant BridgeRNA attribution across technical remeasurement, while RR1 preserves a much narrower multiscale response than RR3. These are technically reproducible representations—not proof of causal spaceflight pathways or protocol-independent biology."""))
M(nbf.v4.new_markdown_cell("""## 10. Native-transcriptome sensitivity analysis

This final conventional control asks whether restricting RNA-seq analysis to BridgeRNA's 15,165-gene vocabulary materially changes the RR1/RR3 conclusions. It uses raw counts and the exact matched animals for each original/OSD-168 technical-remeasurement pair. edgeR and ranked GSEA are run separately in (1) each dataset's fullest mapped native gene universe and (2) the Bridge vocabulary. BridgeRNA embeddings and IG are not involved.

Because PolyA and ribodepletion expose different measurable transcript populations, the native arms retain their own expression-filtered gene sets. Response reproducibility is therefore computed over genes tested in both measurements within each cohort and universe."""))
M(nbf.v4.new_code_cell("""N = R/'native_transcriptome_sensitivity'
native = pd.read_csv(N/'primary_summary.csv')
audit = pd.read_csv(N/'native_gene_audit.csv')
display(audit.style.hide(axis='index'))
display(native.style.hide(axis='index').format({'cosine':'{:.3f}','spearman':'{:.3f}','direction_agreement':'{:.1%}','NES_spearman':'{:.3f}'}))
display(Image(filename=str(N/'expression_reproducibility_native_vs_vocab.png'), width=900))
display(Image(filename=str(N/'pathway_reproducibility_native_vs_vocab.png'), width=900))"""))
M(nbf.v4.new_markdown_cell("""### Direct gene-universe comparison

For the compact table, a `GSEA hit` is deliberately stringent: the pathway must have FDR < 0.05 in both the original and remeasurement and retain the same NES direction. `Gained` and `lost` therefore describe changes in this protocol-robust conventional set—not every isolated significance call."""))
M(nbf.v4.new_code_cell("""universe = pd.read_csv(N/'gene_universe_summary.csv')
compact = universe[['Cohort','Native_genes','Native_GSEA_hits','Bridge_vocab_GSEA_hits','Shared_hits','Gained_by_restriction','Lost_by_restriction','Pathway_Jaccard','Native_IG_core_overlap']].rename(columns={'Bridge_vocab_GSEA_hits':'15,165 GSEA hits','Native_IG_core_overlap':'IG-core overlap'})
display(compact.style.hide(axis='index').format({'Pathway_Jaccard':'{:.3f}'}))
de_counts = universe[['Cohort','Native_DE_genes_original','Native_DE_genes_remeasurement','Bridge_vocab_DE_genes_original','Bridge_vocab_DE_genes_remeasurement','NES_Spearman_original','NES_Spearman_remeasure','FDR_Spearman_original','FDR_Spearman_remeasure','Direction_reversals_between_universes']]
display(de_counts.style.hide(axis='index').format({c:'{:.3f}' for c in de_counts.columns if 'Spearman' in c}))
display(Image(filename=str(N/'robust_gsea_hits_by_gene_universe.png'), width=850))
changes = pd.read_csv(N/'pathway_gained_lost_shared.csv')
display(changes.query("status != 'shared'").groupby(['cohort','status']).head(10).style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""### The 22 signed-IG pathways under the native conventional universe

This table tests the already frozen 22-pathway set against conventional GSEA in the original and remeasurement separately. It does not use conventional results to redefine the set."""))
M(nbf.v4.new_code_cell("""support22 = pd.read_csv(N/'signed_ig_22_conventional_support.csv')
rows=[]
for _,q in support22.iterrows():
    row={'Module':q.module,'Pathway':q.pathway}
    for cohort,key in [('RR1','RR1'),('RR3-39','RR3_39'),('RR3-40','RR3_40')]:
        for universe,label in [('native','Native'),('bridge_vocab','Bridge vocab')]:
            ok=(q[f'{universe}_{key}_original_FDR']<.05 and q[f'{universe}_{key}_remeasure_FDR']<.05 and q[f'{universe}_{key}_original_NES']*q[f'{universe}_{key}_remeasure_NES']>0)
            row[f'{cohort} {label}']='robust' if ok else 'not robust'
    rows.append(row)
display(pd.DataFrame(rows).style.hide(axis='index'))
native_only = pd.read_csv(N/'native_only_significant_pathways.csv.gz')
native_only_counts = native_only.groupby(['cohort','measurement']).size().rename('native-only significant pathways').reset_index()
display(native_only_counts.style.hide(axis='index'))
display(native_only.sort_values(['cohort','measurement','fdr_native']).groupby(['cohort','measurement']).head(5)[['cohort','measurement','source','pathway','nes_native','fdr_native','nes_bridge_vocab','fdr_bridge_vocab']].style.hide(axis='index').format({'nes_native':'{:+.3f}','fdr_native':'{:.2e}','nes_bridge_vocab':'{:+.3f}','fdr_bridge_vocab':'{:.2e}'},na_rep='not tested'))"""))
M(nbf.v4.new_markdown_cell("""**Sensitivity result.** The restricted analysis retains 10,587–11,140 genes common to each paired response, versus 13,678–15,362 in the native analysis. Restriction modestly increases expression-response reproducibility in all three cohorts: cosine changes from 0.621→0.713 (RR1), 0.610→0.663 (RR3-39), and 0.562→0.691 (RR3-40). Rank correlation changes similarly (0.506→0.568, 0.598→0.627, and 0.613→0.674).

The global GSEA response relationship is much less sensitive to vocabulary: original-versus-remeasurement NES Spearman is 0.494 vs 0.523 for RR1, 0.786 vs 0.782 for RR3-39, and 0.873 vs 0.871 for RR3-40 (native vs restricted). Individual significance calls do move. Native/restricted robust-pathway counts are 5/15 for RR1, 248/230 for RR3-39, and 43/65 for RR3-40. Importantly, native RR1 contains nine significant direction-reversing pathways versus zero after restriction, so vocabulary restriction makes RR1 appear somewhat more stable—it does not create its instability.

For the frozen 22 signed-IG pathways, the native analysis provides reproducible conventional support for two pathways, both in RR3-39: **Glycine, Serine and Threonine Metabolism** and **Glyoxylate Metabolism and Glycine Degradation**. The restricted arm retains only the latter. None of the 22 is conventionally robust in RR1 or RR3-40. Thus the principal conclusion—that most of the 22 are reproducible BridgeRNA-prioritized signals rather than conventional GSEA rediscoveries—survives use of the full native transcriptome.

Native-only calls add interpretable detail, including antiviral/immune and mitochondrial RNA processing in RR1 original, mitochondrial/OXPHOS programs in RR1 remeasurement and RR3-39, and xenobiotic/glutathione/steroid/CoA metabolism in RR3-40. They show that vocabulary restriction can hide individual programs and alter thresholded pathway counts. It does **not** materially change the higher-level result: RR1 remains the least conventionally reproducible pair; RR3-39 and RR3-40 retain substantially more coherent pathway-response profiles; and the 22-pathway signed-IG core is not explained by omitting non-Bridge genes."""))
M(nbf.v4.new_markdown_cell("""### Decision

**Vocabulary effect contributes partially, but does not explain BridgeRNA reprioritization.** Restriction changes thresholded recovery non-uniformly: robust GSEA hits increase 5→15 in RR1 and 43→65 in RR3-40, but decrease 248→230 in RR3-39. Therefore there is no general mechanism by which the 15,165-gene universe simply inflates pathway recovery. Native/restricted NES profiles remain strongly correlated within each measurement (0.934–0.969), with zero significant cross-universe NES direction reversals.

Most decisively, native-transcriptome GSEA supports only 2 of the fixed 22 signed-IG pathways as protocol-robust—and both occur in RR3-39—versus 1 under vocabulary restriction. Retaining additional native genes therefore adds one conventional confirmation, but leaves 20/22 without reproducible conventional support. The BridgeRNA pathway core cannot reasonably be attributed to gene-universe restriction alone."""))
M(nbf.v4.new_markdown_cell("""## 11. Limitations and provenance

- OSD-168 is a technical remeasurement, not an independent biological replicate.
- “Protocol-robust” only applies to the compared measurement pair.
- The integrated core counts unweighted agreement across expression, signed input-gene IG, and contextual graph evidence.
- Expression-interaction ORA is restricted to the 15,165-gene universe; it is distinct from full-transcriptome raw-count edgeR/GSEA.
- Pathway-profile scores are reused outputs and should not be described as conventional NES.
- The former 29-pathway/six-module result is a secondary T-cell PolyA/ribo PC1–2 technical-control result, not the primary biological core.
- No causal assignment of RR1 instability to one protocol variable is possible from this comparison."""))

nb["cells"] = cells
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
