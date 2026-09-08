#!/usr/bin/env python3
"""Build the result-driven Task 4 notebook."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []

def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# Task 4 — Library-prep disentanglement

**Question:** Can library-associated variation be separated from biological variation in frozen BridgeRNA representations?

This is a standalone benchmark motivated by Task 3. BridgeRNA is frozen, and NASA/OSDR data are never used for training, model selection, or hyperparameter tuning. Computation lives in `pipeline/`; this notebook reads saved outputs. We use *library-associated* unless controlled evidence supports a causal claim.""")

code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
HERE = Path.cwd()
if HERE.name != 'library_prep_disentanglement':
    candidate = HERE / 'benchmarks/library_prep_disentanglement'
    if candidate.exists(): HERE = candidate
RESULTS = HERE / 'results'
def read_csv(path): return pd.read_csv(path) if path.exists() else pd.DataFrame()
def read_json(path): return json.loads(path.read_text()) if path.exists() else {}
print('Benchmark:', HERE.resolve())""")

md("""## 1. Data discovery and provenance

ARCHS4 labels are audited conservatively: `total RNA` alone is not treated as proof of rRNA depletion. True same-RNA pairs require authoritative deposited metadata.""")
code("""audit = read_json(RESULTS/'task4a_data_audit/archs4_audit_summary.json')
display(pd.DataFrame([audit]).T.rename(columns={0:'value'}))
display(read_csv(RESULTS/'task4a_data_audit/archs4_explicit_label_summary.csv'))
display(read_csv(RESULTS/'task4a_data_audit/controlled_dataset_audit.csv'))""")
md("""The controlled 40-donor T-cell study is used for training. SRP127360 is completely held out, but contains only two biological source RNAs (pooled blood and colon), with four technical libraries per protocol. GSE150097 is retained as a candidate because its public records do not establish every cross-protocol pair. This exploratory run therefore uses fixed epochs without validation-driven selection.""")

md("## 2. Original frozen BridgeRNA baseline")
code("""baseline = read_json(RESULTS/'task4c_bridge_baseline/summary.json')
display(pd.DataFrame([baseline]).style.format(precision=4))
pairs = read_csv(RESULTS/'task4c_bridge_baseline/bridge_pair_metrics.csv')
if len(pairs):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    ax[0].hist(pairs.paired_cosine, bins=15, color='#386cb0')
    ax[0].set(xlabel='Paired cosine', ylabel='Pairs', title='Same-RNA cross-library cosine')
    ax[1].hist(pairs.euclidean_distance, bins=15, color='#fdb462')
    ax[1].set(xlabel='Euclidean distance', ylabel='Pairs', title='Cross-library displacement')
    plt.show()""")
md("""High pair cosine does not guarantee donor retrieval. A dominant PolyA→Ribo difference component is evidence of reproducible displacement within a study, not proof of a universal causal library effect.""")

md("## 3. FE/RE decomposition and controls")
code("""summary = read_csv(RESULTS/'task4_disentanglement/main_summary.csv')
if summary.empty:
    display(Markdown('*Disentanglement results are not available yet.*'))
else:
    display(summary.style.format(precision=3, na_rep='—'))
    metrics = [c for c in ['auroc','balanced_accuracy','macro_f1','pair_cosine','pair_r1','pair_mrr'] if c in summary]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3*len(metrics), 3.6), layout='constrained')
    for ax, column in zip(np.atleast_1d(axes), metrics):
        ax.barh(summary.representation, summary[column], color='#4daf4a')
        ax.set(title=column.replace('_',' '), xlim=(0,1)); ax.grid(axis='x', alpha=.2)
    plt.show()""")
md("""FE is successful only if held-out library prediction falls while cross-library biological identity improves. RE should retain library-associated information. Linear removal, loss ablations, and shuffled controls test whether the neural decomposition adds value.""")
code("""umap_png = RESULTS/'task4_disentanglement/heldout_representation_umaps.png'
if umap_png.exists():
    image = plt.imread(umap_png)
    plt.figure(figsize=(8,12)); plt.imshow(image); plt.axis('off'); plt.show()""")

md("## 4. Independent Task 3 challenge")
code("""task3 = read_csv(RESULTS/'task4g_task3_challenge/task3_challenge_metrics.csv')
if task3.empty:
    display(Markdown('*Independent Task 3 application is not available yet.*'))
else:
    display(task3.style.format(precision=3))
    pivot = task3.pivot(index='comparison', columns='representation', values='cosine')
    ax = pivot.plot.bar(figsize=(9,4), color=['#377eb8','#4daf4a','#e41a1c'])
    ax.axhline(0,color='black',lw=.8); ax.set(ylabel='Response cosine', title='Independent technical-replication challenge')
    plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.show()""")
md("""Task 3 is an external challenge, not a tuning set. RR1 improvement matters only if RR3 and ERCC/no-ERCC relationships remain robust; changing the RR1 sign alone is not success.""")

md("""## 5. Does RR1 align with a controlled PolyA→Ribo subspace?

This diagnostic uses only the **original frozen BridgeRNA embeddings**. The technical direction and SVD basis are learned from 40 verified same-RNA T-cell pairs; OSDR is used only afterward as an external diagnostic. Difference vectors are consistently defined as `Ribo − PolyA` or `OSD-168 new/Ribo-like − OSD-48 old/PolyA-like`.

The removal basis is the uncentered SVD of donor-wise differences, which retains their shared mean displacement. Centered PCA variance is also saved. This is a mechanism check—not a recommended correction.""")
code("""follow = RESULTS/'task4_followup_controlled_subspace'
diagnostic = read_csv(follow/'concise_summary.csv')
display(diagnostic.style.format({'result':'{:.4f}'}))
dist = read_csv(follow/'controlled_distribution_summary.csv')
display(dist.style.format(precision=4))
for name in ['controlled_library_geometry.png','external_rr1_alignment_and_removal.png','random_subspace_control.png']:
    path = follow/name
    if path.exists():
        img = plt.imread(path); plt.figure(figsize=(11,7)); plt.imshow(img); plt.axis('off'); plt.show()""")

md("""### External orientation, correction specificity, and biological damage

An RR1 sign change is not sufficient evidence. We compare learned removal with 500 deterministic random subspaces per dimension and quantify how much original geometry, neighborhood structure, controlled pair retrieval, and RR3 technical preservation remain.""")
code("""display(read_csv(follow/'independent_source_alignment.csv').style.format(precision=4))
display(read_csv(follow/'task3_response_after_subspace_removal.csv').style.format(precision=4))
display(read_csv(follow/'controlled_geometry_damage.csv').style.format(precision=4))
display(read_csv(follow/'random_subspace_rr1_summary.csv').style.format(precision=4))""")
md("""The controlled T-cell effect is strongly reproducible *within that study*, and individual RR1 old→new displacements substantially project into the T-cell basis. However, held-out pooled blood and colon point in the opposite orientation. Five-component removal changes RR1 strongly and much more than random removal, but also removes a large fraction of embedding energy, disrupts neighborhoods, and weakens RR3-39. Therefore this supports **partial, context-dependent overlap**, not a universal PolyA/Ribo direction or a validated correction. The next correction effort should model the broader protocol transition and acquire additional independent same-RNA studies rather than treating library selection alone as established causality.""")

md("""## 6. Held-out response-geometry robustness

The following analysis removes the independently learned T-cell technical basis from **individual OSDR sample embeddings before** reconstructing each fixed `FLT − GC` response. No OSDR result affects the basis. Original sample memberships, technical-replication mappings, 14 response vectors, ordering, and mode labels are unchanged.

The primary preservation evidence is response-level—not global sample neighborhoods. Results are shown for 0, 1, 2, 3, 5, and 10 removed dimensions, with 500 deterministic random orthonormal controls per nonzero dimension.""")
code("""robust = RESULTS/'task4_response_robustness'
verify = read_csv(robust/'original_metric_verification.csv')
display(verify.style.format(precision=6))
display(read_csv(robust/'technical_replication_cosine_curve.csv').style.format(precision=3))
display(read_csv(robust/'technical_replication_spearman_curve.csv').style.format(precision=3))
img=plt.imread(robust/'figure_a_replication_curve.png');plt.figure(figsize=(11,6));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Response damage and fixed-mode preservation

For an orthogonal projection, `cos(Δz_original, Δz_corrected)` equals the retained norm fraction; both are retained in the raw table for clarity. Mode assignments are never refit. ARI compares unsupervised two-cluster structure with the original fixed labels.""")
code("""display(read_csv(robust/'response_damage_summary.csv').style.format(precision=3))
display(read_csv(robust/'task3_mode_preservation.csv').style.format(precision=3))
for name in ['figure_b_response_matrices.png','figure_c_response_preservation.png']:
    img=plt.imread(robust/name);plt.figure(figsize=(14,7));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Selectivity and absolute-space versus response-space geometry

The controlled basis is compared with at least 500 random bases at every dimension. A selective RR1 change is informative mechanistically, but it is not sufficient if RR3 or the broader response organization is damaged.""")
code("""display(read_csv(robust/'random_subspace_summary.csv').style.format(precision=4))
display(read_csv(robust/'concise_summary.csv').style.format({'result':'{:.4f}'}))
for name in ['figure_d_rr1_random_null.png','figure_e_absolute_vs_response.png']:
    img=plt.imread(robust/name);plt.figure(figsize=(10,5));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Response-robustness conclusion

1. **RR1:** The independently learned basis explains a specific component of the RR1 reversal: PC1–5 removal changes cosine from about −0.804 to +0.195, beyond all 500 matched random removals.
2. **RR3:** Preservation is mixed. RR3-40 remains high, whereas RR3-39 falls materially.
3. **Broader Task 3 geometry:** It is not preserved well after five components. The response-matrix correlation falls to roughly 0.56, fixed-label silhouette drops sharply, and unsupervised agreement with the original modes is low.
4. **Response versus absolute geometry:** Median response preservation is somewhat lower than median absolute sample cosine, though it exceeds Top-10 neighborhood overlap. These results do not show that response geometry is generally immune to the removal.
5. **Usefulness of Δz:** Uncorrected within-study response vectors remain valuable for controlled cross-study technical comparisons, but this particular removal trades a selective RR1 improvement against substantial global response reorganization.
6. **Purified biology:** **No.** We have identified and removed a controlled, technical-associated subspace; neither the residual nor its positive RR1 cosine can be called pure spaceflight biology.""")

md("""## 7. Gene-level basis of RR1 instability

Signed Integrated Gradients are computed independently for the six fixed response measurements using the established zero-expression baseline and frozen BridgeRNA encoder. Within each technical pair, the **original OSD-48 or OSD-137 Δz direction is held fixed** for both measurements. This is essential: orienting each response toward itself would conceal reversal. The controlled signature is independently defined as `IG(mean T-cell Ribo) − IG(mean T-cell PolyA)` along the controlled mean latent displacement.

Genes are ranked by absolute attribution while retaining sign. The tables below verify the latent responses and compare Top-100 overlap, random-overlap enrichment, genome-wide signed Spearman correlation, union-Top-100 correlation, and sign agreement.""")
code("""gene_diag=RESULTS/'task4_gene_attribution_diagnostic'
display(read_csv(gene_diag/'latent_response_verification.csv').style.format(precision=6))
ig_compare=read_csv(gene_diag/'technical_replication_ig_comparison.csv')
display(ig_compare.style.format(precision=4))
for name in ['figure_a_signed_attribution_scatter.png','figure_b_top100_overlap_sign.png']:
    img=plt.imread(gene_diag/name);plt.figure(figsize=(14,6));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Controlled technical signature and expression-level control

The controlled PolyA→Ribo signature is compared with `remeasurement attribution − original attribution` at Top-100, Top-250, and Top-500. These cutoff analyses are sensitivity checks, not independent tests. Conventional expression uses the identical samples and 15,165-gene `log1p(TPM)` inputs.""")
code("""display(read_csv(gene_diag/'controlled_signature_discrepancy_overlap.csv').style.format(precision=4))
display(read_csv(gene_diag/'expression_response_comparison.csv').style.format(precision=4))
for name in ['figure_c_controlled_signature_overlap.png','figure_d_signed_gene_heatmap.png']:
    img=plt.imread(gene_diag/name);plt.figure(figsize=(15,6));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Gene-set interpretation

GO Biological Process, KEGG, and Reactome enrichment uses the exact 15,165-gene model vocabulary as custom background with g:Profiler multiple-testing correction. Empty gene-set/source combinations mean no significant terms, not missing computation. Attribution is associative and is not evidence that a gene causes technical sensitivity.""")
code("""enrichment=read_csv(gene_diag/'enrichment.csv')
if len(enrichment):
    cols=[c for c in ['query','source','native','name','p_value','intersection_size'] if c in enrichment]
    display(enrichment.sort_values('p_value').groupby('query',group_keys=False).head(12)[cols].style.format({'p_value':'{:.2e}'}))
display(read_csv(gene_diag/'interpretation_gene_sets.csv').groupby('gene_set').size().rename('genes').to_frame())""")

md("""### Gene-level conclusion

1. **RR1 is a mixture, dominated by attribution reweighting rather than simple sign reversal.** It retains 53 shared Top-100 genes, 96.2% with the same sign, but genome-wide signed Spearman is only 0.194. Its reversed latent vector therefore arises from broad changes in attribution magnitude and the remaining gene set, not wholesale reversal of the shared leading genes.
2. **RR3 is more reproducible.** RR3-39 and RR3-40 share 71 and 76 Top-100 genes, with genome-wide signed Spearman around 0.60 and nearly complete sign preservation.
3. **Controlled-signature overlap is statistically enriched but only modestly preferential for RR1.** RR1 shares 11/49/125 genes at Top-100/250/500 versus 10/37/94 for RR3-39 and 9/39/100 for RR3-40. This supports partial overlap, not a uniquely PolyA/Ribo-driven RR1 mechanism.
4. **Expression already contains the instability.** RR1 expression cosine is 0.369, below RR3-39 (0.652) and RR3-40 (0.822). BridgeRNA reorganizes and accentuates an existing expression-level discrepancy.
5. **Biology:** Shared RR1 genes and reproducible RR3 genes both emphasize hepatic small-molecule, organic-acid, lipid, bile-secretion, and broader metabolic programs. No significant coherent enrichment was detected for the two opposite-sign RR1 genes, measurement-specific sets, controlled Top-100 signature, or the 11-gene controlled/RR1 intersection.

The technically unstable RR1 response has an interpretable gene-level signature, but these results neither identify causal “batch genes” nor isolate pure biological signal.""")

md("""## 8. Are simple corrections better than controlled SVD projection?

All linear corrections are learned solely from the 40 paired T-cell donors. Controlled-pair evaluation uses leave-one-donor-out fitting and testing. OSDR remains fully held out. The comparison includes no correction, mean-direction projection, SVD PC1–1/2/3/5, paired additive residualization, and the already trained FE representation.

`library_auroc` retains the signed held-donor result. An AUROC near zero represents systematic inversion, not proof that library information vanished; `library_orientation_free_auroc` and `library_accuracy_chance_proximity` make that distinction explicit. FE Euclidean distances and original-versus-corrected vector cosines are not comparable across its 64-D coordinate system and the original 512-D space.""")
code("""simple=RESULTS/'task4_simple_correction_comparison'
trade=read_csv(simple/'correction_tradeoff_summary.csv')
show=['method','paired_cosine','paired_euclidean','pair_r1','library_auroc','library_orientation_free_auroc','library_accuracy_chance_proximity','RR1','RR3-39','RR3-40','response_matrix_preservation','median_response_preservation','mode_ARI','mode_silhouette','sample_cosine_preservation','top10_neighbor_overlap']
display(trade[[c for c in show if c in trade]].style.format(precision=3,na_rep='—'))
display(read_csv(simple/'svd_correction_curve.csv').style.format(precision=3))""")
code("""for name in ['figure_a_correction_curves.png','figure_b_pareto_tradeoff.png','figure_c_selected_response_matrices.png','figure_d_random_control.png']:
    img=plt.imread(simple/name);plt.figure(figsize=(14,6));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Correction–preservation conclusion

1. **Was SVD projection better?** Not generally. Mean-direction projection and SVD PC1 are nearly equivalent compromises. PC2+ is uniquely capable of making RR1 positive, but only alongside substantial loss of RR3-39 and broader response organization.
2. **Controlled pair removal:** SVD PC1–5 gives the smallest paired distance and perfect donor R@1, but held-donor protocol predictions remain orientation-decodable (orientation-free AUROC 0.713). Systematic prediction inversion for several methods must not be called protocol erasure.
3. **Best preservation:** Paired additive residualization preserves RR3 and all FLT−GC response geometry exactly because its study-constant protocol offset cancels algebraically. Consequently, it leaves RR1 unchanged. Among projections, mean-direction/PC1 preserves the most structure.
4. **RR1 without substantial damage:** No evaluated method does this. PC1 improves RR1 only from −0.804 to about −0.70. PC1–2 makes RR1 positive but response-matrix preservation falls to 0.619, median response preservation to 0.541, and ARI to 0.272.
5. **PC1 versus PC1–5:** PC1 is the better preservation compromise, but it is not an RR1 solution. PC1–5 is a stronger technical perturbation with unacceptable response damage.
6. **Pareto result:** Multiple methods are non-dominated because technical removal, RR1 movement, and response preservation conflict. There is no jointly successful operating point.
7. **Recommended use:** The controlled basis is most defensible for **diagnostic quantification**, not routine correction. It identifies RR1-sensitive directions, but current evidence does not demonstrate separability of technical and biological response components.

The residual spaces are not purified biology, and none of these results imply that BridgeRNA removes batch effects.""")

md("""## 9. Diagnostic decomposition of technical-replication discrepancies

This analysis does **not** correct or remove dimensions from any embedding. For each NASA technical replication it defines $\\delta=\\Delta z_{original}-\\Delta z_{remeasurement}$ and measures how much of that discrepancy lies in the uncentered SVD basis fitted only to 40 independent same-RNA T-cell Ribo-minus-PolyA displacements. Random 1/2/3/5-dimensional subspaces provide a 1,000-replicate calibration for each comparison and dimension.

`parallel` means *aligned with the independently characterized library-associated transformation*; it is not a pure technical effect. The orthogonal component is likewise not pure biology. For cross-context directional comparisons, NASA vectors are displayed in original-to-remeasurement orientation (`-delta`) to match the controlled PolyA-to-Ribo convention.""")
code("""diag=RESULTS/'task4_discrepancy_decomposition'
compact=read_csv(diag/'rr1_rr3_compact_comparison.csv')
display(compact.style.format(precision=4))
display(read_csv(diag/'random_subspace_calibration.csv').style.format(precision=4))""")
code("""for name in ['discrepancy_alignment.png','random_subspace_calibration.png']:
    img=plt.imread(diag/name);plt.figure(figsize=(13,5));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Additive-vector and preservation-interaction checks

The controlled donor displacements are internally coherent, but the only held-out pooled-blood and colon source vectors point in the opposite direction from the T-cell mean. Those two sources each represent only one biological source and cannot establish tissue specificity. Together with heterogeneous NASA alignment, this is inconsistent with treating one additive vector as universally transferable; a context-dependent transformation remains the more defensible working model.

The Lai Polo study explicitly reported that collection/preservation effects were exacerbated with polyA selection. Our cached OSDR design audit nevertheless does not contain the crossed, same-material combinations needed to independently estimate preservation, library selection, and their interaction. OSD-48 C13/C14 offers a preservation comparison in different animals; OSD-48 C14 to OSD-168 offers exact-material remeasurement across a broader library/sequencing transition; OSD-168 ERCC comparisons hold library and source material fixed. These pieces are informative but do not form a causal factorial design.""")
code("""display(read_csv(diag/'additive_vector_context_alignment.csv').query("source_class != 'T-cell donor'").style.format(precision=4))
display(read_csv(diag/'preservation_library_interaction_feasibility.csv').style.hide(axis='index'))""")
md("""### Diagnostic conclusion

RR1 has **53.9%** of squared discrepancy on controlled PC1 and **96.0%** within PC1–5. RR3-39 and RR3-40 have **22.6%** and **56.0%** within PC1–5, respectively. All three exceed 1,000 matched-dimensional random subspaces (one-sided empirical $p=0.001$ at PC1–5), so alignment is not exclusive to RR1. RR1 is nevertheless the strongest and most concentrated case, especially relative to RR3-39.

This establishes diagnostic alignment, not a causal fraction explained by library preparation. The simplest decisive next experiment is a crossed same-RNA design spanning multiple biological contexts: aliquot the same RNA under at least two preservation states and process every aliquot with both polyA selection and ribodepletion, holding sequencing workflow fixed and replicating the design across tissues/donors. That design can estimate preservation, library-associated, and interaction terms without relying on study-level confounding.""")

md("""## 10. Donor-resampling robustness of the Technical Alignment Score

This robustness analysis uses only cached embeddings. It repeats the exact uncentered-SVD convention under 1,000 paired-donor bootstraps, 250 random 32-train/8-held-out donor splits, and leave-one-donor-out validation. It does not remove PCs or modify any representation.

Subspace comparisons use principal angles and projection matrices, because secondary PCs may rotate or exchange order. Bootstrap uncertainty answers reference reproducibility; held-out alignment answers donor generalization; the earlier random-subspace test answers chance geometry. These are separate questions.""")
code("""robust_dir=RESULTS/'task4_technical_subspace_robustness'
prof=read_csv(robust_dir/'bootstrap_profiler_score_summary.csv')
sub=read_csv(robust_dir/'bootstrap_subspace_stability_summary.csv')
held=read_csv(robust_dir/'heldout_donor_alignment_summary.csv')
order=read_csv(robust_dir/'bootstrap_ordering_stability.csv')
display(prof.style.format(precision=4))
display(order.style.format(precision=4))""")
code("""display(sub.query("metric in ['largest_principal_angle_deg','mean_principal_angle_deg','projection_similarity']").style.format(precision=4))
display(held.style.format(precision=4))""")
code("""for name in ['figure_a_bootstrap_alignment.png','figure_b_subspace_stability.png','figure_c_heldout_donor_alignment.png','figure_d_rr1_cumulative_alignment.png']:
    img=plt.imread(robust_dir/name);plt.figure(figsize=(12,5));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Secondary components and final robustness interpretation

PC1 is exceptionally stable: median bootstrap projection similarity is **0.9995**, with a median principal angle of **1.24°**. PC1–2 is also stable (median projection similarity **0.9925**; largest angle **7.01°**). The complete PC1–5 span is only moderately stable: median projection similarity is **0.8584**, and its largest principal angle is **45.60°** (95% interval 24.13°–85.76°).

Despite instability in weaker dimensions, the NASA profiler scores are stable. RR1 PC1–5 has bootstrap median **0.9566**, mean **0.9557**, SD **0.0054**, and 95% interval **0.9430–0.9637**, close to the full-reference value 0.9597. The ordering RR1 > RR3-40 > RR3-39 holds in 100% of bootstraps at every tested k.

The increase in RR1 is specifically dominated by two stable dimensions: PC1 contributes **0.5390** and PC2 contributes **0.4141**; PCs 3–5 together contribute only about **0.0066** in the full reference. Thus, the high PC1–5 score is reproducible because RR1 aligns strongly with the stable PC1–2 core—not because each of PC3–5 is individually reproducible.

The learned reference generalizes strongly within this controlled experiment. Across repeated 32/8 splits, held-out donor median alignment is **0.9792** for k=1 and **0.9936** for k=5; leave-one-donor-out values are nearly identical. This supports the Technical Alignment Score as a reproducible diagnostic of similarity to **this T-cell PolyA→Ribo transformation**. It does not establish a universal tissue-independent technical direction, causal attribution, purified biology, or successful batch correction.""")

md("""## 11. BridgeRNA Technical Confounding Profiler

### Response context

**Biological system:** mouse liver  
**Perturbation:** spaceflight  
**Response:** Flight minus Ground Control  
**Comparison:** original measurement versus technical remeasurement

- **RR1:** OSD-48 versus OSD-168. PolyA, single-end 50-bp HiSeq 3000 sequencing transitions to ribodepletion, paired-end 150-bp HiSeq 4000 sequencing without ERCC; both were sequenced at UC Davis.
- **RR3-39 / RR3-40:** 39- and 40-day OSD-137 responses versus OSD-168 ERCC remeasurements. Both sides are ribodepleted, paired-end 150-bp HiSeq 4000 data from UC Davis.

The profiler keeps four questions separate: **Response Reproducibility** asks whether the response reproduced; **Technical Alignment Score** asks whether a discrepancy resembles the controlled transformation; **Biological Overlap** asks whether technical-associated structure intersects response geometry; and **Biological Impact** asks what molecular programs/features changed. They are not collapsed into one score.

$R=\\cos(\\Delta z_A,\\Delta z_B)$ and $T=\\|P_{PC1:2}\\delta\\|^2/\\|\\delta\\|^2$, where $\\delta=\\Delta z_A-\\Delta z_B$. PC1–2 is the operational reference learned from controlled same-RNA T-cell PolyA/Ribo pairs. Its cross-tissue universality has not been established. Category labels are descriptive, not inferential thresholds.""")
code("""profiler_dir=RESULTS/'task4_confounding_profiler'
profiler=read_csv(profiler_dir/'profiler_summary.csv')
img=plt.imread(profiler_dir/'biological_programs.png')
plt.figure(figsize=(12,7));plt.imshow(img);plt.axis('off');plt.show()
display(read_csv(profiler_dir/'biological_programs.csv').style.format({'p_value':'{:.2e}','minus_log10_adjusted_p':'{:.2f}'}).hide(axis='index'))""")
md("""### Primary diagnostics

Biological Preservation is a **global Task 3 response-geometry statistic**, not a separate RR1/RR3 value and not a percentage of biology. It measures Spearman preservation of the complete response-similarity matrix after removal. PC1–5 is highlighted because it substantially resolves RR1.""")
code("""for name in ['response_reproducibility_bars.png','technical_alignment_bars.png','biological_preservation_bars.png']:
    img=plt.imread(profiler_dir/name);plt.figure(figsize=(10,6));plt.imshow(img);plt.axis('off');plt.show()
primary=['Comparison','Response Reproducibility','Response Category','Technical Alignment PC1-2']
display(profiler[primary].style.format({'Response Reproducibility':'{:.3f}','Technical Alignment PC1-2':'{:.3f}'}).hide(axis='index'))""")
md("""### Compact experiment profiles""")
code("""for name in ['rr1_profile.png','rr3_39_profile.png','rr3_40_profile.png']:
    img=plt.imread(profiler_dir/'figures'/name)
    plt.figure(figsize=(10,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""The Technical Sensitivity Map remains saved as a secondary diagnostic, but it is no longer the primary display. Bar graphs keep Response Reproducibility, Technical Alignment, and Biological Preservation on separate scales.""")

md("""<details><summary><b>Technical Alignment Evidence — why trust T?</b></summary>

Bootstrap uncertainty asks whether donor resampling yields the same reference. Random-subspace calibration asks whether alignment exceeds chance geometry. Held-out-donor alignment asks whether a basis learned from some controlled donors captures the transformation in unseen T-cell donors.

PC1 has projection similarity 0.9995 and median principal angle 1.24°. PC1–2 has projection similarity 0.9925 and largest angle 7.01°. Repeated 32/8 held-out-donor median PC1–2 alignment is 0.9893. Full PC1–5 is only moderately stable and is not the operational profiler reference.
</details>""")
code("""evidence=['Comparison','Technical Alignment PC1-2','PC1 Contribution','PC2 Contribution','Bootstrap Median','Bootstrap 95% Low','Bootstrap 95% High','Random-Subspace Percentile','Random-Subspace P']
display(profiler[evidence].style.format(precision=4).hide(axis='index'))""")

md("""<details><summary><b>Biological Overlap evidence</b></summary>

At PC1 removal, RR1 changes from −0.804 to −0.699 while broader response-matrix preservation is 0.988 and mode ARI remains 1.0. At PC1–5 removal, RR1 becomes +0.195, but matrix preservation falls to 0.559 and mode ARI to 0.116. Removing enough technical-associated structure to resolve RR1 therefore reorganizes broader spaceflight-response geometry. This identifies neither pure technical dimensions nor a pure biological residual.
</details>""")
code("""overlap=read_csv(RESULTS/'task4_simple_correction_comparison/svd_correction_curve.csv')
display(overlap.query("method in ['none','svd_pc1_1','svd_pc1_5']")[['method','RR1','RR3-39','RR3-40','response_matrix_preservation','mode_ARI']].style.format(precision=3).hide(axis='index'))""")

md("""<details><summary><b>Biological Impact</b></summary>

Gene attribution measures which input genes BridgeRNA relied upon most strongly for the response. RR1's low signed attribution concordance means its gene-level basis was substantially reweighted. Because 96.2% of shared influential genes retain direction, this is not simple reversal of the same genes.

Existing enrichment implicates lipid metabolism, bile secretion, fatty-acid/peroxisomal metabolism, and small-molecule catabolism. These are biological programs implicated in the technically sensitive latent response—not pathways labeled as technical artifacts.
</details>""")
code("""impact=['Comparison','Attribution Spearman','Top100 Shared','Shared Top100 Sign Agreement','Measurement-Specific Top100 Union','Existing Pathway Summary']
display(profiler[impact].style.format({'Attribution Spearman':'{:.3f}','Shared Top100 Sign Agreement':'{:.1%}'}).hide(axis='index'))""")

md("""### Contextual Gene Reproducibility — validated metric audit

Gene attribution and contextual embeddings answer different questions. IG asks which inputs most influence the final sample-level response. Here, contextual analysis asks which genes' learned relationships to the broader transcriptomic state change across remeasurement.

For each gene, $\\Delta h_g=\\overline h_{g,FLT}-\\overline h_{g,GC}$ and Gene Context Reproducibility is $C_g=\\cos(\\Delta h_{g,A},\\Delta h_{g,B})$. Contextual discrepancy is $\\|\\Delta h_{g,A}-\\Delta h_{g,B}\\|$; ranked enrichment uses the normalized form $D_g/(\\|\\Delta h_{g,A}\\|+\\|\\Delta h_{g,B}\\|)$ so negligible response vectors are not prioritized only because their cosine is noisy.

Frozen inference used the exact 34 cached Task 3 `log1p(TPM)` inputs. The audit verifies all six response memberships, the FLT-minus-GC direction, 15,165 genes, 512 contextual dimensions, unique samples within each response, and no zero response vectors. No model inference was repeated for this validation.""")
code("""robust_dir=profiler_dir/'contextual_robustness'
display(read_csv(robust_dir/'sample_and_metric_audit.csv').style.hide(axis='index'))
metric_audit=read_csv(robust_dir/'metric_robustness_summary.csv')
display(metric_audit.style.format(precision=3).hide(axis='index'))
for name in ['median_context_reproducibility.png','context_reversal_fraction.png','median_normalized_discrepancy.png','magnitude_filtered_context_reproducibility.png']:
    img=plt.imread(robust_dir/'figures'/name);plt.figure(figsize=(8,5));plt.imshow(img);plt.axis('off');plt.show()""")
md("""#### Low-response-norm sensitivity

The symmetric magnitude statistic is $\sqrt{\|\Delta h_A\|\|\Delta h_B\|}$. RR1's reversed genes do have lower response magnitude than its reproducible genes, so raw cosine alone overstates some instability. That does **not** explain the overall result: after removing the lowest 10% by magnitude, RR1 median cosine remains **−0.089** and 57.8% of retained genes remain reversed, versus medians **0.793** and **0.878** for RR3-39/RR3-40. Even after removing 20%, RR1 remains negative (**−0.052**).

The complementary normalized discrepancy is exactly $\|\Delta h_A-\Delta h_B\|/(\|\Delta h_A\|+\|\Delta h_B\|+\epsilon)$. RR1 median is **0.766** (0.756 after the 10% filter), compared with **0.374** and **0.297** for RR3. Directional instability and normalized discrepancy are strongly rank-correlated (RR1 Spearman **0.987**).""")
code("""display(read_csv(robust_dir/'response_norm_by_reproducibility_group.csv').style.format(precision=3).hide(axis='index'))
display(read_csv(robust_dir/'magnitude_filter_sensitivity.csv').style.format(precision=3).hide(axis='index'))""")
md("""#### Ranked-GSEA audit and sensitivity

The original analysis ranked all valid BridgeRNA genes by normalized discrepancy in descending order; positive NES therefore means contextual instability and negative NES means relative contextual reproducibility. It used GO Biological Process, KEGG, and Reactome, gene sets of 10–500 tested genes, 250 permutations, and the 15,165-gene BridgeRNA universe. The repeated 0.004 values were the empirical resolution imposed by 250 permutations (approximately $1/(250+1)$; GSEApy reports its finite-permutation floor).

The validation reruns **only enrichment** with 1,000 permutations (observed nominal/FDR floor 0.001) and two rankings: normalized discrepancy, and negative cosine after excluding the bottom 10% of genes by contextual-response magnitude. Gene sets are intersected with each tested BridgeRNA universe. Enrichment therefore describes programs represented within BridgeRNA's vocabulary, not the complete mouse transcriptome. A pathway is called robust below only when it is FDR < 0.05 with the same NES direction under both rankings.""")
code("""display(pd.DataFrame([read_json(robust_dir/'gsea_audit.json')]).style.hide(axis='index'))
unstable=read_csv(robust_dir/'robust_unstable_pathways.csv')
stable=read_csv(robust_dir/'robust_stable_pathways.csv')
display(pd.DataFrame([
 {'comparison':c,'robust_unstable_terms':len(unstable.query('comparison == @c')),'robust_stable_terms':len(stable.query('comparison == @c'))}
 for c in ['RR1','RR3-39','RR3-40']]).style.hide(axis='index'))
for name in ['robust_rr1_unstable_pathways.png','robust_rr1_stable_pathways.png']:
    path=robust_dir/'figures'/name
    if path.exists():
        img=plt.imread(path);plt.figure(figsize=(11,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""RR1 RNA processing/splicing, chromatin organization/remodeling, and DNA repair/metabolism are **robust** across both defensible rankings. Their leading-edge genes have substantial contextual-response magnitudes (family medians approximately 0.93–1.00), non-negligible expression, and high normalized discrepancy; they are not dominated by zero vectors. Redundant GO terms share many leading-edge genes, so these are three broad program families rather than hundreds of independent discoveries.

Fatty-acid beta-oxidation, respiration, peroxisomal/lipid metabolism, and several translation programs consistently occupy the **relatively stable** end. “Stable” means their contextual FLT−GC response is comparatively reproducible across technical remeasurement—not unchanged biologically.""")
code("""families=read_csv(robust_dir/'rr1_major_pathway_families.csv')
display(families.groupby(['family','ranking']).agg(pathways=('pathway','nunique'),best_NES=('nes','max'),best_FDR=('fdr','min')).reset_index().style.format({'best_NES':'{:.3f}','best_FDR':'{:.3g}'}).hide(axis='index'))
leading=read_csv(robust_dir/'leading_edge_gene_audit.csv')
display(leading.query("comparison == 'RR1'").sort_values('normalized_context_discrepancy',ascending=False)[['gene_symbol','context_reproducibility','original_response_norm','remeasurement_response_norm','normalized_context_discrepancy','mean_log1p_tpm','in_IG_Top100_union','pathway']].drop_duplicates('gene_symbol').head(25).style.format(precision=3).hide(axis='index'))""")
md("""#### Why RR3-39 produced visual/epidermal terms

Most RR3-39 phototransduction, keratinization, cornified-envelope, and intermediate-filament findings are low-expression and ranking-dependent. Cornified/keratin terms have leading-edge median `log1p(TPM)` around 0.01–0.06 and do not recur under both rankings. Two nearly identical light-detection GO terms do recur; they are driven by the same small, annotation-redundant set of about 9–11 low-expression genes. Their response norms are not near zero, so this is not solely a cosine singularity, but there is no liver-biological support here. We classify the phototransduction result as a **low-expression, annotation-redundant unresolved signal**, and the epidermal terms as **metric-dependent exploratory findings**, not biology.""")
code("""display(read_csv(robust_dir/'rr3_39_suspicious_pathway_audit.csv').style.format(precision=3).hide(axis='index'))
display(read_csv(profiler_dir/'contextual_gene_overlap.csv').style.hide(axis='index'))""")
md("""#### Contextual audit decision

- **Contextual-gene result: ROBUST.** RR1 remains dramatically less reproducible than RR3 after magnitude filtering and under normalized discrepancy.
- **RR1 RNA processing/splicing: robust.**
- **RR1 chromatin organization/remodeling: robust.**
- **RR1 DNA repair/metabolism: robust.**
- **RR1 fatty-acid/peroxisomal metabolism: robust relative stability**, not instability.
- **RR3-39 phototransduction: unresolved low-expression/annotation-redundant signal.**
- **RR3-39 keratinization/cornified envelope: exploratory and metric-dependent.**

The unstable RR1 pathway leading edges overlap little with Top-100 IG genes. This is expected rather than a failed validation: IG asks which input genes most influence the final sample response, whereas contextual reproducibility asks which genes' learned contextual responses fail to reproduce. Here they identify mostly different genes and partly different programs.""")

md("""### Final controlled PolyA/Ribo contextual-gene validation

This final Task 4 analysis asks whether the RR1 programs above are independently sensitive when **only library selection is changed**. It uses Chen et al.'s 40 healthy-donor naive-CD4 T-cell RNA samples, each split into a PolyA-selected and rRNA-depleted library. Frozen BridgeRNA inference uses the established checkpoint, count-to-gene-length-TPM-to-natural-`log1p` preprocessing, canonical gene order, and zero filling for absent vocabulary genes. All 40 authoritative donor pairs and all 15,165 model genes are retained (15,120 observed in the source).

For donor $i$ and gene $g$, $d_{g,i}=h_{g,Ribo,i}-h_{g,PolyA,i}$. Directional consistency is the mean cosine between each donor displacement and the mean of the other 39 donors. Prevalence is the fraction of donors with positive leave-one-out cosine. The transparent primary sensitivity score is median displacement magnitude multiplied by the positive part of mean leave-one-out consistency. The sensitivity ranking uses $\|\bar d_g\|$ as an independent magnitude-aware alternative.""")
code("""controlled_dir=profiler_dir/'controlled_gene_context'
controlled_summary=read_json(controlled_dir/'final_summary.json')
display(pd.DataFrame([controlled_summary]).style.format(precision=4).hide(axis='index'))
for name in ['controlled_context_reproducibility.png','controlled_sensitive_pathways.png']:
    path=controlled_dir/'figures'/name
    if path.exists():
        img=plt.imread(path);plt.figure(figsize=(11,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""#### Is the controlled gene-context effect reproducible?

Yes, within this controlled T-cell experiment. Median gene-level leave-one-donor-out directional consistency is **0.959**; every gene has positive mean consistency and a majority of donors aligned with its consensus. Across 250 donor bootstraps, the median gene-ranking Spearman correlation is **0.993** and the median Top-500 overlap is **485/500**. A 1,000-replicate donor-direction sign-flip control gives empirical $p=0.001$ for the observed median consistency.

This establishes a reproducible contextual transformation in these donors. It does not establish that its size or direction is universal across cell types.""")
code("""display(read_csv(controlled_dir/'bootstrap_ranking_stability.csv').describe().style.format(precision=3))
controlled_genes=pd.read_parquet(controlled_dir/'controlled_gene_sensitivity.parquet')
display(controlled_genes.head(25).style.format(precision=3).hide(axis='index'))""")
md("""#### Controlled pathway sensitivity and RR1 concordance

Preranked GSEA uses the same GO BP, KEGG, Reactome, 10–500 represented-gene limits, 15,165-gene universe, and 1,000 permutations as the validated RR1 audit. Both rankings incorporate magnitude: the primary median-magnitude × reproducibility score and the norm of the mean donor displacement.

RNA processing/splicing, chromatin organization/remodeling, and DNA repair/metabolism are **SUPPORTED** by the controlled experiment because each appears significantly at the sensitive end under both rankings. Thus, the same three broad pathway families independently recur in RR1 and a controlled library-selection perturbation.

Direct gene identity is different. Representative leading edges share **0 genes** for all three families, and RR1-versus-controlled gene rankings correlate only **0.163**. Top-N overlap is not significant at 100, 250, or 500; it becomes modestly enriched only at Top-1000 (85 observed versus 65.9 expected; $p=0.0088$). The result is therefore pathway-level convergence involving different genes—not evidence that the same individual genes transmit the effect in T cells and liver.""")
code("""display(read_csv(controlled_dir/'pathway_family_decisions.csv').style.hide(axis='index'))
display(read_csv(controlled_dir/'rr1_controlled_pathway_concordance.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
display(read_csv(controlled_dir/'leading_edge_concordance.csv').style.format({'expected_overlap':'{:.2f}','hypergeom_p':'{:.3g}'}).hide(axis='index'))
display(read_csv(controlled_dir/'rr1_rr3_rank_correlations.csv').style.format(precision=3).hide(axis='index'))
display(read_csv(controlled_dir/'rr1_rr3_topn_overlap.csv').style.format({'expected_overlap':'{:.1f}','fold_enrichment':'{:.2f}','hypergeom_p':'{:.3g}'}).hide(axis='index'))
for name in ['rr1_controlled_pathway_concordance.png','rr1_controlled_gene_overlap.png']:
    img=plt.imread(controlled_dir/'figures'/name);plt.figure(figsize=(10,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""#### RR3 comparator and comparatively stable hepatic programs

Controlled sensitivity is positively associated with RR1 instability but negatively associated with RR3-39/40 instability (Spearman **−0.111/−0.044**), supporting preferential—but weak—RR1 correspondence. Threshold overlaps are not uniquely RR1, however; RR3-39 has stronger small-Top-N overlap. This prevents claiming a specific gene-level RR1 signature.

Fatty-acid/lipid metabolism is **NOT SUPPORTED as a controlled sensitive family** across both magnitude-aware rankings. RR1 places fatty-acid oxidation toward relative contextual stability, while the controlled T-cell result is mixed and sparse. Because the controlled biological context is naive T cells rather than liver, this cannot establish general library insensitivity of hepatic metabolism; it only says that strong controlled sensitivity was not reproduced here.""")
md("""#### Final Task 4 controlled-impact conclusion

1. Controlled PolyA→Ribo selection produces a highly reproducible contextual-gene transformation across these 40 same-RNA donors.
2. The strongest individual genes include histone/chromatin-related and other high-displacement features, but the ranking should be interpreted as T-cell-context sensitivity rather than a universal technical gene list.
3. RNA processing/splicing, chromatin organization/remodeling, and DNA repair/metabolism independently recur under controlled library selection.
4. Concordance with RR1 is primarily **pathway-level, not gene-level**: genome-wide rank correlation is weak and representative leading edges do not overlap.
5. Fatty-acid/hepatic metabolic stability cannot be generalized from T cells; the controlled result is mixed and does not robustly mark this family as sensitive.
6. Evidence is **PARTIAL CONCORDANCE**: strong sample-level alignment, robust controlled gene-context transformation, and shared pathway families, but weak individual-gene correspondence.
7. RR1 also changed library chemistry, read configuration/length, instrument, depth, and other workflow details. Controlled PolyA/Ribo sensitivity therefore strengthens the library-selection hypothesis without causally identifying it as the sole source of RR1 instability.

**Paper-safe statement:** Although the RR1 replication discrepancy strongly aligned with the controlled PolyA/Ribo reference at the sample level, concordance at gene-context and pathway levels was partial. The same broad RNA-processing, chromatin, and DNA-repair families were independently sensitive to controlled library selection, but different leading-edge genes drove the T-cell and liver results. This is consistent with context dependence and the multiple technical changes separating OSD-48 and OSD-168.

The planned Task 4 evidence is now sufficient to **freeze this benchmark for the paper**, provided the claims remain diagnostic and associative: no batch correction, purified biological space, universal T-cell reference, or sole PolyA/Ribo causality is claimed.""")

md("""### Conventional Expression Baseline

This baseline asks whether BridgeRNA's final contextual pathway result is already recoverable from standard raw-count differential expression. It changes no contextual analysis and reruns no model inference.

- **Controlled T cells:** edgeR quasi-likelihood paired model `~ donor + library_prep`, testing Ribo versus PolyA across 40 same-RNA donors.
- **RR1:** edgeR quasi-likelihood model on the exact nine animal-matched OSD-48/OSD-168 pairs, with animal blocking and a Measurement × FlightStatus interaction. The reported effect is `(FLT−GC)_OSD48 − (FLT−GC)_OSD168`; it is not the OSD-48 versus OSD-168 abundance difference.
- edgeR's standard `filterByExpr`, TMM normalization, robust dispersion estimation, and quasi-likelihood testing are used. The tested universes contain 11,373 T-cell and 11,916 RR1 genes from the same 15,165-gene input universe.
- T-cell ranked GSEA uses signed `sign(logFC) × sqrt(QLF)`; RR1 response-instability GSEA uses `sqrt(QLF)`. The same pathway files, 10–500 represented-gene limits, and 1,000 permutations are used.""")
code("""conv_dir=profiler_dir/'conventional_expression_baseline'
conv_summary=read_json(conv_dir/'final_summary.json')
display(pd.DataFrame([conv_summary]).style.format(precision=3).hide(axis='index'))
display(read_csv(conv_dir/'analysis_summary.csv').style.hide(axis='index'))
for name in ['pathway_family_comparison.png','gene_ranking_concordance.png','gene_topn_overlap.png']:
    img=plt.imread(conv_dir/'figures'/name);plt.figure(figsize=(11,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""#### Are context-sensitive genes simply the most differentially affected genes?

Not entirely. Controlled T-cell conventional and contextual rankings correlate substantially (**Spearman 0.581**) and share 51/100, 254/500, and 541/1000 leading genes, so BridgeRNA largely preserves the strong controlled expression effect. In RR1, genome-wide correlation is only **0.075**, despite significant concentration at the top (31/100, 124/500, and 248/1000). Thus RR1 contextualization selects many conventionally unstable genes but substantially reorganizes their broader ordering.

At a 10% rank threshold, T cells contain 522 genes with weak conventional but strong contextual change; RR1 contains **902**. The same qualitative result persists at 5% and 20%, so Category C is not created by one arbitrary cutoff.""")
code("""display(read_csv(conv_dir/'conventional_vs_contextual_rank_correlations.csv').style.format(precision=3).hide(axis='index'))
display(read_csv(conv_dir/'conventional_vs_contextual_topn_overlap.csv').style.format({'expected':'{:.1f}','fold_enrichment':'{:.2f}','hypergeom_p':'{:.2e}'}).hide(axis='index'))
categories=read_csv(conv_dir/'expression_context_gene_categories.csv')
display(categories.drop(columns='gene_symbols').style.hide(axis='index'))
for analysis in ['controlled_tcell','RR1']:
    genes_row=categories.query("analysis == @analysis and top_fraction == 0.10 and category == 'weak_expression_strong_context'").iloc[0]
    print(f"{analysis} — example weak-expression/strong-context genes:", ', '.join(str(genes_row.gene_symbols).split(';')[:25]))""")
md("""#### Pathway-family comparison

Conventional T-cell DE detects all three predefined families. Conventional RR1 interaction analysis detects RNA processing/splicing, but **not** chromatin or DNA repair at FDR < 0.05. Consequently, conventional expression reproduces only **1/3** cross-context pathway families and shares no exact significant pathway between T cells and RR1. BridgeRNA contextual analysis reproduces **3/3**, shares nine exact significant pathways, and has a positive cross-context absolute-NES correlation (0.166 versus −0.085 conventionally).

Leading-edge results sharpen this distinction. In controlled T cells, conventional and contextual leading edges strongly overlap for all three families. In RR1, RNA-processing leading edges overlap 40 genes, whereas chromatin and DNA repair have no conventional significant leading edge to compare. BridgeRNA therefore does not invent the controlled effect, but its RR1 contextual representation exposes coherent chromatin/DNA-repair organization not recovered by the standard interaction ranking.""")
code("""display(read_csv(conv_dir/'pathway_family_concordance.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
display(read_csv(conv_dir/'cross_context_pathway_agreement.csv').style.format(precision=3).hide(axis='index'))
le=read_csv(conv_dir/'representative_leading_edge_comparison.csv')
display(le.drop(columns='overlap_genes').style.format({'expected_overlap':'{:.2f}','hypergeom_p':'{:.2e}'}).hide(axis='index'))""")
md("""#### Conventional-baseline decision

1. **Are BridgeRNA genes simply the most DE genes?** Partly in the controlled experiment, but not in RR1. RR1 has weak global rank agreement and hundreds of strong-context genes outside the equivalently sized conventional top set.
2. **Does conventional T-cell PolyA/Ribo DE identify the three families?** Yes: RNA processing, chromatin, and DNA repair all appear.
3. **Does conventional RR1 response instability identify them?** Only RNA processing/splicing; chromatin and DNA repair do not reach pathway significance.
4. **Does conventional analysis reproduce T-cell→NASA convergence?** Partially—1/3 predefined families, compared with 3/3 contextually.
5. **Are there low-expression-change/high-context-change genes?** Yes, robustly across 5%, 10%, and 20% rank thresholds.
6. **Does BridgeRNA add information?** It adds organized contextual structure in RR1, but that structure is rooted in and overlaps conventional expression effects. It is not wholly independent information.
7. **Classification:** **AMPLIFICATION/REORGANIZATION OF CONVENTIONAL SIGNAL.** This is stronger than simple preservation, but the evidence does not justify claiming a completely novel latent biological signal unavailable to conventional analysis.

This baseline reinforces the paper-safe Task 4 conclusion: BridgeRNA contextualization organizes technical-associated expression changes into cross-context pathway structure, particularly chromatin and DNA-repair families in RR1, while retaining substantial conventional signal. It does not prove causal regulation, pathway activation, or exclusive PolyA/Ribo causality.""")

md("""### Expression-Adjusted Contextual Sensitivity

This final gene-level control asks whether context changes more than expected given conventional expression behavior. It does not remove Top-N DE genes. Each experiment is modeled independently with robust LOWESS (`frac=0.20`, three robustifying iterations):

- Predictor: `log1p(sqrt(edgeR QL F))`, a continuous conventional-effect magnitude.
- Controlled T-cell outcome: `log1p(contextual sensitivity score)`.
- RR1 outcome: normalized contextual discrepancy.
- Expression-adjusted contextual sensitivity: observed outcome minus LOWESS prediction. Positive values indicate more contextual change than expected from conventional behavior.
- Sensitivity analysis: LOWESS on within-experiment percentile ranks.

Only genes valid in both analyses are used: **11,373 controlled T-cell genes** and **11,916 RR1 genes**.""")
code("""adjusted_dir=profiler_dir/'expression_adjusted_context'
adjusted_summary=read_json(adjusted_dir/'final_summary.json')
display(pd.DataFrame([adjusted_summary]).style.format(precision=4).hide(axis='index'))
for name in ['expression_vs_contextual_sensitivity.png','residual_pathway_family_comparison.png','cross_context_residual_concordance.png']:
    img=plt.imread(adjusted_dir/'figures'/name);plt.figure(figsize=(12,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""#### Residual validity and context-excess genes

Residuals are effectively uncorrelated with their fitted expression statistic (T cells **0.011**, RR1 **−0.003**) and are not dominated by low-expression genes. Only 1–2% of the RR1 positive residual tail lies in the lowest expression decile; 0.4–5.1% lies in the lowest contextual-magnitude decile. Re-fitting after jointly excluding the lowest 5%, 10%, or 20% by expression and contextual magnitude preserves residual rankings strongly (Spearman 0.982–1.000).

The controlled residual remains correlated with contextual displacement magnitude (0.763), as expected for a response whose definition explicitly contains magnitude. RR1 residual is inversely correlated with contextual-response magnitude (−0.550), but the positive tail is not dominated by low-magnitude genes. These residuals isolate excess relative to conventional effects; they do not eliminate all dependence on the contextual metric's own construction.""")
code("""audit=read_csv(adjusted_dir/'residual_artifact_audit.csv')
display(audit.style.format(precision=3).hide(axis='index'))
cats=read_csv(adjusted_dir/'context_categories.csv')
display(cats.drop(columns='gene_symbols').style.hide(axis='index'))
for analysis in ['controlled_tcell','RR1']:
    q=read_csv(adjusted_dir/f'{analysis}_context_excess_ranking.csv')
    display(Markdown(f'**{analysis}: Top 25 context-excess genes**'))
    keep=[c for c in ['gene_symbol','expression_statistic','logFC','FDR','contextual_statistic','predicted_contextual_sensitivity','residual_contextual_sensitivity','standardized_residual','expression_abundance'] if c in q]
    display(q[keep].head(25).style.format(precision=3).hide(axis='index'))""")
md("""#### Residual pathway results

Residual GSEA uses the same 15,165-gene-compatible tested universes, GO BP/KEGG/Reactome files, 10–500 gene-set limits, and 1,000 permutations. It is run for both raw LOWESS residuals and rank residuals.

- **Controlled T cells:** RNA processing/splicing is partially retained. Chromatin and DNA repair no longer reach residual-GSEA significance and are classified as explained by conventional expression in this context.
- **RR1:** RNA processing, chromatin, and DNA repair are strongly retained under both residual definitions. Thus, RR1's chromatin/DNA-repair contextual organization is not a simple monotonic restatement of its edgeR interaction statistic.
- **Strict cross-context conclusion:** only RNA processing survives expression adjustment in both experiments. It is driven by different genes: the representative residual leading edges share zero genes.

The 1,000-shuffle competitive family control places the observed joint mean residual percentile above all permutations for each broad family (`p=0.001`). However, because T-cell chromatin and DNA-repair fail the stricter residual GSEA criterion, those two are not claimed as robust residual cross-context convergence.""")
code("""display(read_csv(adjusted_dir/'pathway_family_comparison.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
display(read_csv(adjusted_dir/'pathway_family_decisions.csv').style.hide(axis='index'))
display(read_csv(adjusted_dir/'permutation_pathway_control.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(adjusted_dir/'residual_leading_edge_comparison.csv').drop(columns='overlap_genes').style.format({'expected':'{:.2f}','hypergeom_p':'{:.3g}'}).hide(axis='index'))""")
md("""#### Cross-context residual gene test

Expression adjustment removes gene-level cross-context concordance: residual T-cell and RR1 rankings have Spearman **0.005**. Top-100 overlap is zero; Top-250/500/1000 overlaps are at or below random expectation and nonsignificant. Therefore, any residual shared functional organization is unequivocally **same broad program, different genes**, not a conserved individual-gene signature.""")
code("""display(read_csv(adjusted_dir/'cross_context_gene_overlap.csv').drop(columns='genes').style.format({'expected':'{:.2f}','fold_enrichment':'{:.2f}','hypergeom_p':'{:.3g}'}).hide(axis='index'))""")
md("""#### Final expression-adjusted decision

1. Conventional effects explain a substantial fraction of controlled T-cell contextual ranking structure (unadjusted Spearman 0.581) but little of RR1's genome-wide contextual ordering (0.075).
2. Large, stable residual contextual signals remain after nonlinear and rank-based adjustment, without low-expression or near-zero-context domination.
3. RR1 context-excess genes form coherent RNA-processing, chromatin, and DNA-repair programs.
4. Strict T-cell→liver residual convergence remains only for RNA processing/splicing.
5. Residual RNA-processing leading edges contain different genes in the two contexts.
6. Chromatin and DNA repair remain RR1-specific residual contextual organization, not independently reproduced residual families in controlled T cells.
7. **Classification: PARTIAL ADDITIONAL CONTEXTUAL ORGANIZATION.** BridgeRNA encodes functional gene-context organization beyond what the conventional per-gene statistic predicts, especially within RR1, but the strongest three-family cross-context result is partly explained by conventional expression. It is neither wholly novel nor merely a nonlinear restatement of DE.

This is the final planned Task 4 gene-level control. No causal gene, dedicated pathway dimension, pathway activation, purified biology, or universal technical reference is inferred.""")

md("""## 12. Independent biological replication of contextual programs

This section asks whether the contextual programs that were technically unstable in RR1 also recur during independent mouse-liver FLT−GC responses. It is **not** a correction analysis. BridgeRNA remains frozen and no technical subspace is removed.

The primary analysis contains **11 stratified contrasts and 84 samples** from six independent OSD datasets: RR1-CASIS (21/22 day), RR1-NASA (two preservation strata), RR3 (39/40/41 day), STS-135, RR9, and RR6 (ISS-T/LAR). OSD-168 is excluded from biological recurrence because it remeasures RR1/RR3 material, but is retained in the final technical/biological triangulation.

For each gene and contrast, contextual response is `Δh_g = mean(h_g,FLT) − mean(h_g,GC)` and its ranking statistic is `||Δh_g||₂`, exactly matching the response-magnitude construction used in the validated Task 4 contextual analysis. Ranked GSEA uses the same 15,165-gene universe, GO BP/KEGG/Reactome resources, 10–500 gene-set limits, deterministic seed, and 1,000 permutations. Contextual NES indicates concentration toward high versus low contextual-response magnitude; it is not an up/down expression direction.

The conventional baseline uses raw counts, TMM normalization, and a separate robust edgeR quasi-likelihood FLT-vs-GC model per contrast. The 1-FLT/1-GC RR1-CASIS 22-day stratum has no residual degrees of freedom and is explicitly retained only as a descriptive fixed-BCV edgeR ranking; it is not treated as equivalent inferential evidence.""")
code("""bio=profiler_dir/'independent_biological_replication'
display(read_csv(bio/'independent_contrast_summary.csv')[['contrast_id','OSD','mission','flight_duration','preservation','n_FLT','n_GC']].style.hide(axis='index'))
display(read_csv(bio/'program_recurrence_summary.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
display(read_csv(bio/'program_classification.csv').style.hide(axis='index'))""")
code("""for name in ['program_recurrence_by_method.png','gene_vs_pathway_recurrence.png']:
    path=bio/'figures'/name
    if path.exists():
        img=plt.imread(path);plt.figure(figsize=(12,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Recurrence, leading edges, and triangulation

Recurrence is summarized by both contrast and independent OSD so multiple strata from one mission are not counted as independent biological studies. The gene-level comparison uses Spearman rank agreement and cosine similarity between per-gene contextual-response magnitudes. Program-level agreement uses pathway NES ranks. Leading-edge overlap is reported directly; pathway recurrence does not require identical genes.

The four predefined families are summaries applied only after full unbiased enrichment. The triangulation table keeps three observations distinct: controlled PolyA/Ribo sensitivity, RR1 technical-replication instability, and independent FLT−GC recurrence.""")
code("""display(read_csv(bio/'technical_biological_triangulation.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
agreement=read_csv(bio/'gene_vs_pathway_agreement.csv')
display(agreement.describe().T[['mean','std','min','50%','max']].style.format(precision=3))
leading=read_csv(bio/'leading_edge_overlap.csv')
if len(leading): display(leading.drop(columns='overlap_genes').groupby(['analysis','family']).agg(comparisons=('jaccard','size'),median_leading_edge_jaccard=('jaccard','median'),max_leading_edge_jaccard=('jaccard','max')).reset_index().style.format(precision=3).hide(axis='index'))""")
md("""### Interpretation

The conclusion combines independent-study recurrence, the conventional edgeR comparator, and the existing controlled/technical results. Recurrence supports a spaceflight-associated biological interpretation only when it spans independent OSDs; it does not prove that a program is free of technical sensitivity. Conversely, technical sensitivity in RR1 does not make a recurrent program purely technical. The defensible interpretation distinguishes biological recurrence, measurement sensitivity, and biology × technical interaction.""")
code("""decision=read_json(bio/'decision_summary.json')
display(pd.DataFrame([decision]).T.rename(columns={0:'result'}))""")
md("""**Decision: B, with important limitations.** Chromatin organization/remodeling and DNA-damage programs are present in independent FLT−GC evidence and in conventional edgeR, so they are not adequately characterized as an RR1-only technical artifact. However, BridgeRNA high-context recurrence is limited to two independent OSDs for each family, and one DNA-supporting OSD is the unreplicated 1-vs-1 RR1-CASIS stratum. This is provisional evidence for recurrent spaceflight-associated biology that is technically vulnerable in RR1—not proof of causality or clean separation from technical effects.

RNA processing behaves differently: it is strongly measurement-sensitive in the controlled and RR1 analyses, but enrichment among the strongest independent contextual responders occurs in only one OSD. Conventional edgeR RNA-processing recurrence is broader. Across all contrast pairs, median gene-rank agreement is lower than median pathway-rank agreement, consistent with some program-level convergence despite changing gene rankings; this does not require or imply identical leading-edge genes.""")

md("""## 13. Exploratory biological–technical latent overlap

This analysis asks whether stronger conventional FLT−GC responses have larger components in the independently learned **controlled PolyA/Ribo-associated technical-reference subspace**. It does not call that reference pure batch space, perform correction, or establish technical causation.

The same 11 independent contrasts are used. The primary reference is the stable PC1–2 span of uncentered `Ribo − PolyA` differences from 40 cached same-RNA T-cell pairs. For each contrast, the full Bridge response `Δz` is decomposed orthogonally into aligned and residual components. Numerical Pythagorean closure is checked. Conventional response strength is RMS edgeR log2FC among that contrast's expressed/tested Bridge-vocabulary genes; it does not depend on significance or sample size.

Program-specific strength is RMS log2FC over the union of expressed Bridge genes assigned to each predefined family by the same GO BP, KEGG, and Reactome resources. Sensitivity analyses use PC1, PC1–5, and a nine-contrast set excluding RR3 41-day (1/2) and RR1-CASIS 22-day (1/1). A 1,000-member random orthonormal 2D-subspace null tests whether any PC1–2 association is specific rather than a generic consequence of projecting stronger latent responses.""")
code("""overlap_dir=profiler_dir/'biological_technical_overlap'
overlap_summary=read_json(overlap_dir/'summary.json')
display(pd.DataFrame([overlap_summary]).style.format(precision=4).hide(axis='index'))
overlap_table=read_csv(overlap_dir/'per_contrast_overlap_metrics.csv')
display(overlap_table[['OSD','mission','flight_duration','n_FLT','n_GC','rms_log2FC','bridge_total_magnitude','aligned_magnitude_PC1_2','orthogonal_magnitude_PC1_2','aligned_fraction_PC1_2']].style.format(precision=4).hide(axis='index'))""")
code("""display(read_csv(overlap_dir/'correlation_summary.csv').style.format(precision=4,na_rep='—').hide(axis='index'))
display(read_csv(overlap_dir/'program_specific_correlations.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(overlap_dir/'program_gene_universes.csv').style.hide(axis='index'))""")
code("""for name in ['de_vs_aligned_magnitude.png','de_vs_aligned_fraction.png','per_contrast_decomposition.png','program_specific_associations.png','random_subspace_null.png']:
    path=overlap_dir/'figures'/name
    if path.exists():
        img=plt.imread(path);plt.figure(figsize=(12,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Result and interpretation

The full 11-contrast result matches **Outcome 1 with an Outcome 5 qualification**:

- RMS log2FC correlates with total Bridge response magnitude and with absolute PC1–2-aligned magnitude (both Spearman `ρ = 0.745`, `p = 0.0085`).
- RMS log2FC does **not** correlate with the fraction occupying PC1–2 (`ρ = 0.264`, `p = 0.433`).
- After excluding both severely underpowered contrasts, aligned magnitude remains associated (`ρ = 0.667`, nominal `p = 0.0499`), while aligned fraction is effectively unrelated (`ρ = 0.067`, `p = 0.865`).
- The aligned-magnitude association is not specific relative to matched random 2D subspaces (empirical two-sided `p = 0.314`). The aligned-fraction result is likewise unexceptional (`p = 0.539`).

RNA-processing, chromatin, DNA-repair, and hepatic-metabolic RMS effects all correlate positively with aligned **magnitude** in the full cohort, but none predicts aligned **fraction**. These family effects largely track overall response strength; they do not establish program-specific occupation of the controlled reference.

The supported interpretation is therefore modest: stronger conventional responses produce larger Bridge responses and consequently larger absolute projections into many latent subspaces, including PC1–2. The relative composition does not become more PolyA/Ribo-reference-like, and the observed magnitude relationship is common under random 2D projections. These data do **not** provide specific evidence that stronger biological responses increasingly occupy the controlled technical reference. They also do not refute biological–technical overlap demonstrated by the earlier correction–preservation experiment; they show only that this particular scaling test cannot establish its specificity.""")

md("""### RR3-only 39–41 day follow-up

RR3 41-day was absent from the earlier OSD-137→OSD-168 technical-replication comparison for a concrete provenance reason. OSD-168 includes remeasurements of RR3 animals F1/F2 and G1/G2 (39 day) and F3/F4 and G3/G5 (40 day), but no corresponding F6, G6, or G7 material from the 41-day stratum. The original OSD-137 41-day biological response remains calculable from 1 FLT and 2 GC samples, but it has no valid OSD-168 technical counterpart and must not be forced into that replication analysis.""")
code("""rr3_dir=overlap_dir/'rr3_timecourse'
display(read_csv(rr3_dir/'rr3_timepoint_metrics.csv')[['flight_duration_days','n_FLT','n_GC','technical_replication_counterpart','bridge_total_magnitude','aligned_magnitude_PC1_2','aligned_fraction_PC1_2','orthogonal_magnitude_PC1_2','rms_log2FC']].style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_dir/'rr3_monotonic_trend_summary.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(rr3_dir/'rr3_timecourse_overlap.png');plt.figure(figsize=(12,5));plt.imshow(img);plt.axis('off');plt.show()""")
md("""The absolute response quantities are non-monotonic: total Bridge magnitude, aligned magnitude, orthogonal magnitude, and RMS log2FC all decline at day 40 and rise to their maximum at day 41. The aligned fraction increases numerically from **0.678 → 0.733 → 0.735**, but essentially plateaus between days 40 and 41. With only three time points—and only one FLT animal at day 41—the formally perfect rank ordering of the fraction is descriptive, not evidence for a time-dependent trend. The defensible conclusion is **no clear progressive change in absolute overlap**, with a possible early increase and plateau in proportional overlap that requires better-replicated time points.""")

md("""## 14. Full-transcriptome versus Bridge-vocabulary conventional expression

This conventional control asks whether restricting RNA-seq analysis to the exact 15,165 genes visible to BridgeRNA changes the biological interpretation. It uses the unchanged 11 independent mouse-liver FLT−GC contrasts and their original raw counts. No Bridge embeddings or contextual representations enter either ranking.

- **Full transcriptome:** each study's original Ensembl count rows are mapped to GENCODE mouse symbols, collapsed by symbol, filtered independently by edgeR, and analyzed with raw-count TMM/robust quasi-likelihood methods.
- **Bridge vocabulary:** the existing edgeR results restricted to the exact canonical vocabulary are reused.
- Both rankings use identical GO BP, KEGG, and Reactome collections, gene-set size limits, 1,000 permutations, and deterministic seed.
- Obvious redundant summary labels (for example, the three overlapping Reactome rRNA-processing terms) are collapsed only in the compact Top-3 display. Detailed pathway files retain every original term.
""")
code("""fv_dir=HERE/'results/task4_full_vs_bridge_vocab_expression'
fv_primary=read_csv(fv_dir/'primary_summary.csv')
display(fv_primary.style.format({'Coverage':'{:.1%}','NES_profile_Spearman':'{:.3f}'}).hide(axis='index'))
display(read_csv(fv_dir/'program_family_summary.csv').style.hide(axis='index'))
display(read_csv(fv_dir/'pathway_agreement_summary.csv').style.format({'nes_spearman':'{:.3f}','shared_fraction_of_union':'{:.1%}','shared_direction_agreement':'{:.1%}'},na_rep='—').hide(axis='index'))""")
md("""### Interpretation

The comparison is evaluated by continuous NES-profile correlation, overlap and direction of significant pathways, and preservation of the four predefined program families. This distinction matters in the smallest strata: pathway-level coordination can be detectable even when no individual gene survives edgeR FDR correction. RR3 41-day (1 FLT/2 GC) and RR1-CASIS 22-day (1 FLT/1 GC; descriptive fixed-BCV ranking) remain explicitly underpowered.

The detailed results below determine whether vocabulary restriction alone can explain differences between conventional expression and BridgeRNA contextual findings. A high full-versus-vocabulary NES correlation and retention of the central families would argue against that explanation; losses unique to the restricted arm identify the specific conventional programs unavailable within the model vocabulary.""")
md("""### Result

Restriction retains roughly **71–80% of the genes tested by edgeR** in each contrast (median **75.1%**) while preserving the global pathway ranking very strongly: per-contrast full-versus-vocabulary NES Spearman ranges from **0.943 to 0.987** (median **0.961**). Across significant pathways, the median shared fraction of the union is **59.9%**, and every exact pathway significant in both arms retains its enrichment direction.

The principal biological conclusions are therefore broadly robust, but restriction is not lossless. RNA processing is retained in RR3 39d, RR3 40d, RR3 41d, STS-135, RR6 ISS-T/LAR, and the better-powered RR1-NASA contrast. Full-only RNA-processing significance appears in RR9 and the small RR1-NASA preservation stratum. Central chromatin, DNA-response, and hepatic-metabolic programs are generally retained, although individual family calls or the strongest representative term can change where power is weak or related pathways occupy opposite ranked tails.

RR3 39d remains stronger than RR3 40d after restriction. The best RNA-processing NES changes from **2.545 versus 2.086** in the full analysis to **2.270 versus 2.004** in the Bridge-vocabulary analysis. Both use ribodepleted libraries, as do several contrasts with substantially different RNA-processing strength or direction. RNA-processing enrichment is therefore not explained simply by the PolyA-versus-ribodepletion label.

These results argue that vocabulary restriction alone cannot reasonably explain the larger differences between conventional and Bridge contextual analyses. It can weaken or remove particular pathway calls, especially in marginal contrasts, but it does not erase the dominant conventional pathway organization.""")

md("""## 15. RR3 39-day versus 40-day functional overlap with PC1–2

This targeted analysis asks why removing the independently learned controlled T-cell PolyA/Ribo-associated PC1–2 reference damages RR3-39 technical replication more than RR3-40. It does not redefine samples, responses, embeddings, or the reference and does not treat PC1–2 as technical-only.

Two source-value corrections are essential. The saved PC-removal curve gives **0.790→0.480** for RR3-39 and **0.917→0.876** for RR3-40 at PC1–2; the previously quoted 0.497/0.894 values correspond to PC1–5 removal. Also, the previously quoted RR3-40 aligned fraction of 0.733 uses the full biological 3-FLT/2-GC contrast. The exact OSD-137→OSD-168 technical comparison is animal-matched 2/2 and has an OSD-137 PC1–2 aligned fraction of **0.447**. Both definitions remain available and are not mixed.

Metadata support exact source-animal matching and describe OSD-168 as the same RR3 RNA material resequenced with ERCC. The conservative wording remains **same source animal and supported same RNA material**; the metadata do not independently prove that every library used the identical physical aliquot. Both time points retain ribodepletion, paired 150-bp sequencing, HiSeq 4000, UC Davis, and liquid-nitrogen preservation. OSD-168 adds ERCC and reports KAPA RNA HyperPrep; the OSD-137 kit is not reported, so a kit change cannot be asserted.""")
code("""rr3_overlap=profiler_dir/'rr3_functional_overlap'
display(read_csv(rr3_overlap/'protocol_variable_audit.csv').style.hide(axis='index'))
display(read_csv(rr3_overlap/'rr3_sample_correspondence_and_protocol.csv')[['timepoint','animal_id','condition','OSD137_sample','OSD168_sample','exact_animal_match','biological_material_status','identical_RNA_status','OSD168_ERCC_mix']].style.hide(axis='index'))
display(read_csv(rr3_overlap/'compact_rr3_39_vs_40_summary.csv').style.format(precision=3).hide(axis='index'))""")
md("""### Response decomposition

The component geometry provides a direct explanation for the point estimate. In both time points, the PC1–2-projected response is nearly identical across remeasurement. RR3-39, however, has a weakly reproducible orthogonal component, whereas RR3-40 retains a strongly reproducible orthogonal response. Removing PC1–2 therefore leaves RR3-39 dominated by its less reproducible remainder but leaves RR3-40 with substantial reproducible structure.""")
code("""display(read_csv(rr3_overlap/'response_component_metrics.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_overlap/'component_replication_similarity.csv').style.format(precision=4).hide(axis='index'))
for name in ['component_replication_similarity.png','component_programs.png','bootstrap_removal_effect.png']:
    img=plt.imread(rr3_overlap/'figures'/name);plt.figure(figsize=(12,5));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Gene attribution, pathway specificity, and conventional expression

Zero-baseline Integrated Gradients uses the same frozen encoder and validated Task 4 attribution definition. Separate original and remeasurement attributions are normalized and averaged with equal measurement weight for each timepoint/component. Component GSEA uses the exact 15,165-gene universe and the established 1,000-permutation GO BP/KEGG/Reactome workflow.

No predefined RNA/ribosome, chromatin, DNA-response, or hepatic-metabolic family reaches component-IG GSEA FDR < 0.05. A complementary competitive rank test finds a small but non-random RR3-39 preference for RNA-processing genes in the parallel rather than orthogonal ranking (mean percentile difference 0.0276; empirical `p=0.0007`); RR3-40 shows no such preference (0.0040; `p=0.555`). Chromatin and DNA-response families show similarly sized RR3-39 preferences, so RNA processing is not uniquely implicated. Ribosome/rRNA-processing genes do not preferentially occupy the parallel ranking.

Conventional expression and component attribution provide different evidence. Conventional edgeR GSEA establishes coordinated RNA-processing involvement (RR3-39 NES 2.270; RR3-40 NES 2.004). Component IG asks which inputs influence the PC1–2-associated latent response. Their genome-wide magnitude correlations are modest and Top-100 overlaps are small but above chance; pathway-level involvement does not require identical leading genes.""")
code("""display(read_csv(rr3_overlap/'top25_component_genes.csv').groupby(['timepoint','component'],as_index=False,group_keys=False).head(10).style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_overlap/'component_program_family_summary.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
display(read_csv(rr3_overlap/'component_family_competitive_permutation.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_overlap/'conventional_gene_rank_comparison.csv').style.format(precision=4).hide(axis='index'))""")
md("""### Random-subspace and sample-size controls

Against the existing 500 matched random 2D removals, both observed cosine decreases are more negative than every random realization (`p=0.001996` with the +1 correction). This establishes specificity to the controlled reference geometry, not biological causality. Functional specificity is assessed separately with 10,000 gene-set competitive permutations because reproducing Integrated Gradients for every random latent subspace would require a new gradient calculation for each null direction and would not reuse the established cached null.

The paired animal bootstrap is decisive for interpretation. RR3-39's median change is −0.186, but its 95% interval spans **−0.411 to +0.121**. RR3-40's smaller change similarly spans zero (**−0.117 to +0.019**). With only two matched FLT and two matched GC animals per technical comparison, the contrast between the point estimates is not stable enough for a mechanistic conclusion.""")
code("""display(read_csv(rr3_overlap/'random_subspace_loss_control.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_overlap/'paired_bootstrap_stability_summary.csv').style.format(precision=4).hide(axis='index'))
display(pd.DataFrame([read_json(rr3_overlap/'decision_summary.json')]).style.hide(axis='index'))""")
md("""### Decision: D — unstable / underpowered

At the observed group means, PC1–2 removal hurts RR3-39 because its exceptionally reproducible PC1–2 component (`cos=0.995`) is removed, exposing an orthogonal component with much lower replication (`cos=0.480`). RR3-40 retains a reproducible orthogonal component (`cos=0.876`). RR3-39 also shows a statistically unusual but small enrichment of RNA-processing genes toward the parallel attribution ranking. That preference is shared in magnitude by chromatin and DNA-response families, fails the primary component-GSEA FDR criterion, and is unstable at the response level under sample resampling.

The data therefore do **not** support claiming that RNA processing specifically causes the RR3-39 sensitivity. Sampling variability is a reasonable explanation for the magnitude of the 39d-versus-40d difference. The stronger paper-safe conclusion is: **the controlled PC1–2 reference contains highly reproducible RR3 response information, but the functional content and subtraction consequence are sample-dependent; with 2/2 matched animals, RR3-39 cannot establish a specific RNA-processing mechanism. Controlled technical references should be used diagnostically, not removed as if they were biologically empty batch dimensions.**""")

md("""## 16. Concise comparison and interpretation

### Concise comparison and interpretation

1. **Expected behavior is reproduced.** RR1 is opposing (`R = −0.804`) and strongly aligned with PC1–2 (`T = 0.953`); RR3-39 is moderately reproducible (`R = 0.790`) with lower alignment (`T = 0.141`); RR3-40 is highly reproducible (`R = 0.917`) despite appreciable alignment (`T = 0.466`).
2. **R and T are distinct.** RR3-40 demonstrates that technical-associated structure can be detectable in the comparatively small discrepancy while the primary FLT−GC response remains reproducible.
3. **RR1 is unusually aligned.** Its score is above all 1,000 random two-dimensional subspaces (`p = 0.001`) and stable across donor bootstraps (median 0.948; 95% interval 0.925–0.958).
4. **Biological overlap matters.** Stronger removal resolves RR1 only while disrupting broader response organization; technical-associated and biological structure are not cleanly separable.
5. **Attribution adds impact context.** RR1 shows gene reweighting within hepatic metabolic programs, while RR3 has greater attribution concordance and Top-100 overlap.
6. **Contextual relationships also change.** RR1 has median gene-context reproducibility −0.125 with 61.6% reversed gene-context responses, compared with medians 0.783/0.864 and reversal fractions 7.5%/6.6% for RR3-39/RR3-40. The most contextually unstable RR1 genes show RNA-processing, chromatin, and DNA-repair enrichment; they overlap little with Top-100 IG genes, indicating complementary information.
7. **The assumption remains explicit.** The reference is stable within controlled T cells, but its cross-tissue universality is unknown.

**RR1 overall:** The RR1 mouse-liver spaceflight response reverses across remeasurement (`R = −0.804`). Its discrepancy is strongly aligned with the controlled PolyA/Ribo reference (`T = 0.953`). Removing sufficient technical-associated structure to resolve this reversal preserves only 0.559 correlation with the broader response-similarity organization. Gene attribution shows response-basis reweighting, while contextual analysis shows widespread instability in genes' learned transcriptomic relationships. The associated latent structure is therefore technically sensitive and biologically entangled—not proven purely technical, causal, or safely correctable.""")

md("## 17. Final benchmark summary")
code("""if not summary.empty:
    final = summary.rename(columns={'representation':'Representation','auroc':'PolyA/Ribo AUROC','pair_cosine':'Pair cosine','pair_r1':'Pair R@1'})
    if not task3.empty:
        challenge = task3.pivot(index='representation',columns='comparison',values='cosine').reset_index().rename(columns={'representation':'Representation'})
        final = final.merge(challenge, on='Representation', how='left')
    final = final.rename(columns={'biology_metric_value':'Biology metric (source-ID MRR)'})
    keep = [c for c in ['Representation','PolyA/Ribo AUROC','Pair cosine','Pair R@1','Biology metric (source-ID MRR)','macro_f1','RR1','RR3-39','RR3-40'] if c in final]
    display(final[keep].style.format(precision=3,na_rep='—'))""")

md("""## 18. Conservative interpretation

The completed evidence should answer whether original Bridge encodes library information, whether displacement generalizes, whether FE reduces it without erasing biology, whether RE isolates it, whether the decomposition generalizes, and whether RR1 improves without damaging RR3.

Until several independent authoritative same-RNA studies are available, generalization and causal attribution remain provisional. The current external test contains only two biological source RNAs, so effect sizes and failures matter more than nominal significance.""")
code("""if not summary.empty and not task3.empty:
    s = summary.set_index('representation')
    t = task3.pivot(index='representation',columns='comparison',values='cosine')
    print(f\"Held-out Bridge AUROC: {s.loc['Bridge','auroc']:.3f}; FE AUROC: {s.loc['FE','auroc']:.3f}; RE AUROC: {s.loc['RE','auroc']:.3f}\")
    print(f\"Pair cosine Bridge→FE: {s.loc['Bridge','pair_cosine']:.3f} → {s.loc['FE','pair_cosine']:.3f}\")
    print(f\"Pair R@1 Bridge→FE: {s.loc['Bridge','pair_r1']:.3f} → {s.loc['FE','pair_r1']:.3f}\")
    print(f\"RR1 cosine Bridge→FE: {t.loc['Bridge','RR1']:.3f} → {t.loc['FE','RR1']:.3f}\")
    print('Conclusion: this exploratory FE/RE fit does not meet the predefined success criteria. It neither improves held-out pair correspondence nor remedies RR1. The result cannot specifically implicate PolyA/Ribo; the RR1 reversal remains attributable to a broader protocol transition.')""")

md("""## 19. RR1 preservation-context diagnostic

This section asks whether occupancy of the independently characterized T-cell PolyA/Ribo-associated PC1–2 reference differs between RR1 CASIS/on-orbit-dissection material (OSD-47) and RR1 NASA preservation strata (OSD-48). It reuses the validated Task 3 contrasts and cached frozen BridgeRNA inputs/embeddings; no samples are pooled, no response is redefined, and no correction is applied.

The source audit matters. OSD-47 reports flight livers dissected on orbit at 21 or 22 days and frozen in a Mini Cold Bag; ground controls followed its matched ground protocol. OSD-48 contains distinct sample-level `Upon euthanasia` and `Carcass` strata. Both RNA-seq datasets report PolyA selection, Illumina TruSeq stranded RNA preparation, single-end 50-bp sequencing, HiSeq 3000, and UC Davis. This is not a PolyA-versus-ribodepletion comparison. The studies also differ in strain, age, duration, animals, and collection context, so preservation is not experimentally isolated.""")
code("""rr1_context=profiler_dir/'rr1_preservation_context'
display(read_csv(rr1_context/'rr1_protocol_comparison.csv').style.hide(axis='index'))
display(read_csv(rr1_context/'rr1_sample_protocol_audit.csv').style.hide(axis='index'))""")

md("""### Response decomposition and controls

The better-powered OSD-47 21-day response has little PC1–2 occupancy (0.045), whereas the OSD-48 carcass response is highly aligned (0.938). OSD-48 upon-euthanasia is intermediate (0.562). OSD-47 22-day is also high (0.789), but it is one FLT versus one GC animal and cannot establish a preservation association. Aligned fraction is geometric occupancy, not percent technical artifact.""")
code("""display(read_csv(rr1_context/'rr1_response_decomposition.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr1_context/'random_2d_subspace_control.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr1_context/'sample_bootstrap_summary.csv').style.format(precision=4).hide(axis='index'))
for name in ['rr1_alignment_by_context.png','random_subspace_controls.png']:
    img=plt.imread(rr1_context/'figures'/name);plt.figure(figsize=(11,6));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Apples-to-apples technical-replication components

The unchanged PC1–2 basis was applied identically to RR1, RR3-39, and RR3-40. RR1's full responses reverse (`cos = −0.804`), and its responses within PC1–2 are even more strongly opposed (`−0.974`); the outside-PC1–2 responses are weakly concordant (`0.221`). Both RR3 projected components reproduce almost perfectly, while RR3-40 also retains strong outside-reference concordance.

The discrepancy statistic answers a distinct question from per-response occupancy: 0.953 of the squared magnitude of the RR1 replication discrepancy lies in PC1–2. It must not be read as “95.3% caused by library preparation.”""")
code("""display(read_csv(rr1_context/'rr1_rr3_component_replication_comparison.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr1_context/'rr1_technical_replication.csv').style.format(precision=4,na_rep='—').hide(axis='index'))""")

md("""#### Exact RR1 sample trace

The original `−0.804` result did **not** pool OSD-48 preservation strata. It used only the carcass stratum and only animals having an OSD-168 no-ERCC counterpart: four FLT animals (M25, M26, M28, M30) and five GC animals (M36–M40). Carcass FLT animal M27 was excluded because no corresponding OSD-168 profile exists. The upon-euthanasia animals M21/M22 and M31/M32 have no OSD-168 counterparts, so a stratum-specific technical-replication cosine cannot be calculated without manufacturing unmatched comparisons.""")
code("""display(read_csv(rr1_context/'rr1_all_osd48_samples_and_counterparts.csv').style.hide(axis='index'))
display(read_csv(rr1_context/'rr1_stratum_specific_replication_metrics.csv').style.format(precision=4,na_rep='—').hide(axis='index'))""")

md("""### Conventional and Bridge functional content

Conventional results are reused from validated raw-count edgeR/GSEA in the full expressed transcriptome and exact Bridge vocabulary. Bridge component results use the exact contextual-gene decomposition: because each sample embedding is the mean of contextual gene states, every gene's FLT−GC contextual vector can be projected into PC1–2 and its orthogonal complement. Rankings use contextual-vector magnitude and the established 1,000-permutation GO BP/KEGG/Reactome workflow over the 15,165-gene universe.

These results describe programs associated with each component; they do not make a component technical-only. Small strata, especially OSD-47 22-day, remain descriptive.""")
code("""display(read_csv(rr1_context/'conventional_expression_summary.csv').style.hide(axis='index'))
display(read_csv(rr1_context/'conventional_program_families.csv').style.hide(axis='index'))
display(read_csv(rr1_context/'component_program_family_summary.csv').style.format(precision=3,na_rep='—').hide(axis='index'))
display(read_csv(rr1_context/'top25_component_contextual_genes.csv').groupby(['contrast_id','component'],as_index=False,group_keys=False).head(10).style.format(precision=4).hide(axis='index'))""")

md("""### Decision: D — underpowered/confounded

OSD-48 carcass-derived RR1 shows much stronger PC1–2 occupancy than the better-powered OSD-47 21-day response, and the within-OSD-48 upon-euthanasia stratum is lower than the carcass stratum. This is consistent with a preservation-context association, but cannot isolate preservation: the high OSD-47 22-day 1-vs-1 estimate exposes animal sensitivity, and OSD-47/48 differ biologically and procedurally. Only OSD-48 carcass material has a valid OSD-168 remeasurement; no comparison was manufactured for OSD-47 or OSD-48 upon-euthanasia.

The defensible conclusion is that **the carcass-preserved OSD-48 response occupies a technically sensitive region of BridgeRNA space, but the RR1 design cannot isolate preservation, a library interaction, or biology as the cause**. PC1–2 remains a PolyA/Ribo-associated—not technical-only—reference, and implicated biological programs must not be labeled artifacts.""")
code("""display(pd.DataFrame([read_json(rr1_context/'decision_summary.json')]).style.hide(axis='index'))""")

md("""## 20. Direction within the controlled PolyA/Ribo-associated reference

Occupancy does not reveal which way a response points. This additive diagnostic therefore orients the unchanged PC1–2 plane using the controlled paired experiment's mean displacement, `mean(z_Ribo − z_PolyA)`, projected into PC1–2. It does not use arbitrary SVD signs or NASA results to orient the reference.

The directional cosine ranges from −1 (PolyA-directed) to +1 (Ribo-directed); the signed projection additionally reports magnitude. These are geometric descriptions relative to the controlled T-cell reference, not evidence that a biological response is artifact. The `weak/no direction` label is descriptive only and uses `|cosine| < 0.25`.""")
code("""direction_dir=profiler_dir/'polya_ribo_directionality'
direction=read_csv(direction_dir/'spaceflight_polya_ribo_directionality.csv')
display(direction.style.format(precision=4).hide(axis='index'))
display(read_csv(direction_dir/'matched_remeasurement_directionality.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(direction_dir/'figures/spaceflight_polya_ribo_directionality_plane.png')
plt.figure(figsize=(12,9));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Directional interpretation

1. **RR1 carcass switches direction.** Its matched OSD-48 response is PolyA-directed (`cos = −0.785`, signed projection `−0.707`), whereas the OSD-168 no-ERCC remeasurement is Ribo-directed (`+0.626`, `+0.268`).
2. **RR3 orientations reproduce.** RR3-39 remains PolyA-directed (`−0.693 → −0.619`), and RR3-40 remains Ribo-directed (`+0.820 → +0.780`).
3. **RR1 follows the independently oriented axis rather than merely reversing elsewhere in PC1–2.** Its parallel-component replication cosine is `−0.974`, and its signed projection crosses from strongly negative to positive relative to the controlled PolyA→Ribo arrow. The match is not perfectly collinear, and this remains association rather than causal attribution.
4. **Directionality refines occupancy.** High occupancy can contain a stable response, as in RR3, or an orientation switch, as in RR1. Occupancy alone therefore cannot diagnose instability or artifact.

The OSD-48 upon-euthanasia biological response is Ribo-directed despite being measured with a PolyA protocol. This is an especially useful caution: “Ribo-directed” describes a latent direction, not the library actually used or the cause of the biological response. Its OSDR technical remeasurement cannot be tested because OSD-168 lacks counterparts for M21/M22/M31/M32.""")

md("""## 21. BridgeRNA Biological Confounding Profiler

The final profiler integrates response geometry, controlled-reference directionality, matched remeasurement, conventional edgeR/GSEA, contextual response, expression-adjusted context excess, and independent biological recurrence. Its central principle is: **technical sensitivity identifies vulnerability, not artifact**.

The response table includes all 14 validated Task 3 FLT−GC contrasts. Reproducibility fields remain missing when no matched remeasurement exists; occupancy alone is never used to infer stability. Exact matched definitions are included separately for RR1 carcass, RR3-39, and RR3-40.""")
code("""final_profiler=profiler_dir/'final_profiler'
response_profiler=read_csv(final_profiler/'response_profiler.csv')
program_profiler=read_csv(final_profiler/'program_profiler.csv')
display(response_profiler.style.format(precision=4,na_rep='—').hide(axis='index'))""")
md("""### Validation cases

- **RR1 carcass — measurement-vulnerable:** original/remeasurement overlap is 0.933/0.780, whole-response reproducibility is −0.804, PC1–2 reproducibility is −0.974, outside-PC1–2 reproducibility is 0.221, and discrepancy localization is 0.953. Its oriented projection switches PolyA-directed to Ribo-directed.
- **RR3-39 — reproducible despite overlap:** overlap is 0.678/0.548, whole-response reproducibility 0.790, and PC1–2 reproducibility 0.995. It remains PolyA-directed.
- **RR3-40 — reproducible despite overlap:** overlap is 0.447/0.689, whole-response reproducibility 0.917, and PC1–2 reproducibility 0.998. It remains Ribo-directed.

RR3 demonstrates why globally subtracting PC1–2 is inappropriate: the same associated reference contains highly reproducible response structure.""")
code("""validation=response_profiler[(response_profiler.matched_remeasurement)&(response_profiler.measurement=='matched original')]
display(validation.style.format(precision=4,na_rep='—').hide(axis='index'))
img=plt.imread(final_profiler/'figures/bridge_biological_confounding_profiler.png')
plt.figure(figsize=(18,11));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Program-level vulnerability and contextual organization beyond DE

RNA processing has conventional biological support, Bridge contextual support, and strong controlled PolyA/Ribo contextual sensitivity. It is measurement-vulnerable in RR1 but cannot be dismissed as noise. Chromatin and DNA-response programs also show technical sensitivity, yet recur across independent spaceflight datasets; they are candidate biological organization, not proven biology. Lipid/metabolic programs recur across all six independent OSDs in both conventional and contextual analyses and were not supported by the controlled T-cell contextual-sensitivity analysis, although their program-specific matched-remeasurement behavior remains unresolved.

Bridge additionally identifies **596 context-excess / non-significant-DE genes**—the RR1 top 5% positive expression-adjusted contextual residuals with edgeR FDR ≥ 0.05. These show 23 significant RNA-processing, 5 chromatin, and 2 DNA-repair terms. They are not “genes missed by RNA-seq” or automatically novel; they represent contextual organization not explained by the per-gene conventional expression statistic.""")
code("""display(program_profiler.style.hide(axis='index'))
contextual=read_csv(final_profiler/'contextual_organization.csv')
display(contextual.head(25).style.format(precision=4,na_rep='—').hide(axis='index'))
print((final_profiler/'profiler_summary.md').read_text())""")

md("""## 22. Which RNA-processing genes underlie technical sensitivity and reproducible biology?

This gene-level analysis uses an exact **1,108-gene RNA-processing universe** assembled reproducibly from the benchmark's existing GO BP, KEGG, and Reactome definitions, restricted to the 15,165 Bridge genes. It includes RNA/mRNA/rRNA processing, splicing, RNA metabolism, and existing ribosome biogenesis/maturation definitions; the complete pathway membership is saved.

The controlled T-cell 40-donor displacement tensor was reused from cache. Exact cross-experiment contextual-vector direction was not present in earlier summary tables, so the six existing RR1/RR3 matched contextual response tensors were regenerated once from the unchanged 34 profiles and frozen model, then cached. No embeddings, response definitions, or model parameters changed.""")
code("""rna_dir=profiler_dir/'rna_processing_gene_analysis'
rna_profiles=read_csv(rna_dir/'rna_processing_gene_profiles.csv')
display(read_csv(rna_dir/'rna_processing_rank_correlations.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rna_dir/'sensitivity_set_overlaps.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rna_dir/'matched_rna_gene_permutations.csv').style.format(precision=4).hide(axis='index'))""")

md("""### Main gene-level findings

- Controlled sensitivity and RR1 instability have weak rank association (`Spearman = 0.134`). Their Top-5% sets overlap by 7 genes versus 2.83 expected (`2.47×`, empirical `p=0.019`), but the enrichment does not persist at Top 10% or 20%.
- Controlled sensitivity and reproducible RR3-39 show consistent set enrichment: Top-5% overlap 9 versus 2.83 expected (`3.18×`, `p=0.0012`), Top-10% 19 versus 11.1 (`p=0.010`), and Top-20% 56 versus 44.5 (`p=0.0247`). RR3-40 does not show significant controlled-set enrichment.
- RR3-39 and RR3-40 have weak overall reproducible-score correlation (`0.119`) but strongly enriched Top-5/10/20% overlap. Their Top-10% sets share 34 genes versus 11.1 expected (`3.06×`, empirical `p=0.0001`). These shared genes reproduce strongly within each timepoint (median original/remeasurement contextual cosine `0.978` and `0.988`).
- Across all RNA genes, the median direct RR3-39-versus-RR3-40 original-response cosine is `−0.0065`, with 50.6% opposite. Thus the timepoints share a reproducible subset but do not represent a simple global sign reversal of identical machinery.
- At the primary Top-10% definition, three genes—**CNOT3, RBM14, and ZFC3H1**—are simultaneously controlled-sensitive, RR1-unstable, and reproducible in at least one RR3 timepoint. They are especially difficult to interpret, but are not labeled artifacts.""")
code("""display(read_csv(rna_dir/'top20_informative_rna_processing_genes.csv')[['gene','tcell_sensitivity_rna_rank','rr1_instability_rna_rank','rr3_39_reproducible_rna_rank','rr3_40_reproducible_rna_rank']].style.hide(axis='index'))
display(read_csv(rna_dir/'evidence_categories.csv').groupby('category').size().rename('genes').reset_index().style.hide(axis='index'))
display(pd.DataFrame([read_json(rna_dir/'rr3_39_vs_40_state_test.json')]).style.format(precision=4).hide(axis='index'))""")
code("""for name in ['rna_contextual_response_heatmap.png','tcell_vs_rr1_gene_ranks.png','rr3_39_vs_40_gene_ranks.png','rna_gene_evidence_categories.png','top20_rna_processing_genes.png']:
    img=plt.imread(rna_dir/'figures'/name);plt.figure(figsize=(12,9));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Interpretation: mixed shared machinery and higher-order convergence

The data do not support a model in which one fixed set of RNA-processing genes uniformly drives controlled library sensitivity, RR1 instability, and both RR3 responses. They also do not support completely unrelated gene implementations: stringent overlaps—especially controlled T-cell with RR3-39 and RR3-39 with RR3-40—are substantially above matched RNA-gene expectations.

The most defensible interpretation is a **mixture**. A reproducible subset of shared molecular machinery participates across settings, while different contextual gene rankings and directions converge onto a higher-order RNA-processing-associated latent organization. RR3-39 and RR3-40 are best classified as **mixed**, not definitively two opposite biological states. RR3 comparisons contain only 2 FLT/2 GC animals per matched measurement, so rankings remain descriptive even where set-level permutation evidence is significant.

The exact master table retains contextual magnitudes and direction, conventional logFC/FDR, context excess, PC1–2 association, rank metrics, and pathway membership for every RNA-processing gene.""")
code("""print((rna_dir/'rna_processing_gene_summary.md').read_text())""")

md("""## 23. Component-level decomposition of the controlled PolyA/Ribo-associated space

This analysis asks whether the controlled 40-donor T-cell PolyA→Ribo difference space contains components that can be removed as technical nuisance without also removing coherent or independently overlapping biological-response structure. It uses the existing frozen sample embeddings and cached contextual-gene displacement tensor; BridgeRNA was not rerun.

The paired-difference matrix has rank at most 40, so all 40 uncentered difference components are characterized. Component signs are oriented to the mean paired Ribo-minus-PolyA displacement. The operational full correction uses the smallest prefix explaining at least 99.5% of controlled displacement energy. Candidate labels are deliberately conservative: pathway absence is not proof of technical purity, and NASA overlap contributes to the exploratory labels, so NASA correction results are descriptive rather than confirmatory.""")
code("""component_dir=HERE/'results/task4_technical_component_decomposition'
pc_metrics=read_csv(component_dir/'technical_pc_metrics.csv')
component_classes=read_csv(component_dir/'technical_pc_classification.csv')
correction_comparison=read_csv(component_dir/'correction_comparison.csv')
display(pc_metrics.head(10).style.format(precision=4).hide(axis='index'))
display(component_classes.style.format(precision=4).hide(axis='index'))""")
code("""for name in ['technical_difference_spectrum.png','cumulative_response_overlap.png','pc_pathway_enrichment.png']:
    img=plt.imread(component_dir/'figures'/name)
    plt.figure(figsize=(13,8));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Correction tradeoff

PC1 dominates the controlled displacement (97.47%), and PCs 1–5 jointly explain 99.52%. Nevertheless, all five prefix components show either significant pathway organization or appreciable overlap with an independent RR1/RR3 response. Consequently, **no component satisfies the prespecified exploratory technical-only candidate rule**. The selective strategy therefore correctly performs no subtraction rather than inventing a nuisance direction.

Removing PCs 1–5 reduces leave-one-donor-out PolyA/Ribo AUROC from 1.000 to 0.452 and raises same-RNA cross-library R@1 from 0.100 to 1.000. That apparent technical benefit has a marked biological-geometry cost: mean individual Task 3 response preservation is 0.541 and the response-cosine matrix correlation is 0.559. RR1 changes from −0.804 to +0.195, but RR3-39 falls from 0.790 to 0.497, while RR3-40 remains comparatively stable (0.917 to 0.894). Resolving RR1 by broad subtraction is therefore not evidence of clean biological correction.""")
code("""display(correction_comparison.style.format(precision=4,na_rep='—').hide(axis='index'))
img=plt.imread(component_dir/'figures/correction_tradeoff.png')
plt.figure(figsize=(11,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Interpretation

The controlled library-associated space cannot currently be divided into a well-supported technical-only portion and a separable biological portion. Several components have coherent chromatin, DNA-response, mitochondrial RNA-processing, or rRNA/tRNA-processing enrichment, while PC1 and PC2 account for most of the RR1 discrepancy and overlap strongly with RR1/RR3 responses.

This supports the hypothesis that protocol-associated latent variation can overlap meaningful molecular organization. It does **not** establish that the enriched pathways caused the protocol effect, that any component is pure biology, or that the T-cell reference is universal across tissues. Selective correction offers no demonstrated advantage here because no component met the conservative technical-only criteria; a new controlled dataset would be required for confirmatory component selection and held-out biological evaluation.""")

md("""## 24. RR3 39/40/41-day cohort audit

This audit uses the exact Task 3 memberships, authoritative cached OSDR metadata, frozen embeddings, and existing expression/contextual results. No contrasts were redefined. Cohorts contain 2 FLT/2 GC at 39 days, 3 FLT/2 GC at 40 days, and 1 FLT/2 GC at 41 days. OSD-168 remeasures F1/F2/G1/G2 at 39 days and F3/F4/G3/G5 at 40 days; F5 and the 41-day F6/G6/G7 cohort lack counterparts.""")
code("""rr3_audit=HERE/'results/task4_confounding_profiler/rr3_cohort_audit'
rr3_samples=read_csv(rr3_audit/'rr3_sample_metadata_audit.csv')
rr3_meta=read_csv(rr3_audit/'rr3_cohort_metadata_summary.csv')
rr3_loo=read_csv(rr3_audit/'rr3_leave_one_out_sensitivity.csv')
rr3_decomp=read_csv(rr3_audit/'rr3_flt_gc_decomposition.csv')
display(rr3_samples.style.hide(axis='index'))
display(rr3_meta.style.hide(axis='index'))""")
md("""### Individual samples and FLT/GC decomposition

Every leave-one-animal-out 39-day response remains negative (`−0.723` to `−0.672`), while every 40-day response remains positive (`0.569` to `0.870`). This is sign-stable but not proof of robustness: 39 days has only 2/2 animals.

Within the oriented PC1–2 reference, the FLT centroid moves from `−9.351` to `−9.136`, whereas GC changes from `−9.217` to `−9.242`. The full latent FLT cohort shift is larger than the GC shift (norm `0.358` versus `0.186`) and is more concentrated in PC1–2 (0.620 versus 0.087). The reversal is therefore primarily flight-animal/cohort driven in this reference, although both arms and animal identities differ.""")
code("""display(rr3_decomp.style.format(precision=4).hide(axis='index'))
display(rr3_loo.style.format(precision=4).hide(axis='index'))
for name in ['rr3_individual_pc12.png','rr3_flt_gc_centroids.png','rr3_leave_one_out.png','rr3_flt_gc_decomposition.png','rr3_conventional_expression_pca.png']:
    img=plt.imread(rr3_audit/'figures'/name)
    plt.figure(figsize=(11,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Conventional expression and gene-level context

The conventional log1p(TPM) 39-day and 40-day FLT−GC response vectors are nearly orthogonal (`cosine = 0.081`), whereas BridgeRNA makes them more strongly opposed (`−0.567`). Bridge therefore reorganizes/amplifies an existing cohort distinction. Existing full-transcriptome GSEA shows stronger RNA-processing enrichment at 39 days than 40 days (best NES 2.545 versus 2.086; vocabulary-restricted control 2.270 versus 2.004).

The 34 shared Top-10% reproducible RNA-processing genes form a reproducible core but do not imply identical response direction: median direct 39d-versus-40d contextual cosine is near zero. CNOT3 and RBM14 are preferentially reproducible at 39 days, whereas ZFC3H1 is preferentially reproducible at 40 days; none is in the shared 34-gene Top-10% core.""")
code("""rr3_genes=read_csv(rr3_audit/'rr3_39_vs_40_gene_programs.csv')
display(rr3_genes.style.format(precision=4,na_rep='—').hide(axis='index'))""")
md("""### Conservative classification: G — mixed / unresolved

The cohorts share strain, sex, age, tissue, diet, habitat, euthanasia method, ribodepletion, stranded paired-end 150-bp sequencing, facility, and liquid-nitrogen preservation. They differ necessarily in animal identity, duration/exposure and collection cohort, FLT radiation dose, carcass weight, RIN, read depth, rRNA contamination, and library index. Exact euthanasia/dissection times and order, extraction batch, sequencing lane, cage position, food/water consumption, and several health variables are unavailable.

Therefore 39 versus 40 days is not separable from collection/euthanasia cohort and animal identity. The paper-safe conclusion is: *OSD-137 contains two technically reproducible but biologically unresolved FLT−GC response configurations. Their orientation is stable to leave-one-animal-out checks and is driven more strongly by the FLT cohort in the fixed PC1–2 reference, but the design cannot distinguish duration-associated biology from collection-cohort or unmeasured animal-level effects.* They should not be called two biological modes.""")

md("""## 25. Are disappearing RR1↔RR3 similarities carcass, RNA-quality, or protocol effects?

This diagnostic examines why strong cross-experiment similarities disappear after removing the controlled T-cell PolyA/Ribo-associated PC1–2. It uses authoritative OSDR metadata, existing frozen response vectors, existing component-level contextual rankings, and the controlled T-cell gene signature. No model inference or correction optimization was performed.""")
code("""state_dir=HERE/'results/task4_confounding_profiler/rr1_rr3_sample_state'
state_sim=read_csv(state_dir/'rr1_rr3_similarity_before_after.csv')
state_meta=read_csv(state_dir/'sample_processing_metadata.csv')
state_quality=read_csv(state_dir/'rna_quality_individuals.csv')
state_quality_corr=read_csv(state_dir/'rna_quality_correlations.csv')
state_overlap=read_csv(state_dir/'shared_pc12_gene_overlap.csv')
display(state_sim.style.format(precision=4).hide(axis='index'))
display(state_meta.style.hide(axis='index'))""")
md("""### Carcass and RNA-quality evidence

Carcass status is not sufficient to explain the crossed matching pattern. RR1 carcass matches RR3-39 (`cosine 0.813`) but strongly opposes RR3-40 (`−0.830`); RR1 upon-euthanasia instead matches RR3-40 by cosine (`0.620`) and opposes RR3-39 (`−0.697`). Both RR3 cohorts have the same reported liquid-nitrogen preservation, ribodepletion, paired 150-bp reads, HiSeq 4000 platform, and facility. The authoritative metadata do not provide a postmortem interval or dissection order that makes RR3-39 uniquely similar to RR1 carcass.

RIN does not provide a simple explanation. Mean RIN is approximately 8.4 for RR1 carcass, 9.55 for RR1 upon-euthanasia, 7.25 for RR3-39, and 7.22 for RR3-40. Across the 23 samples, descriptive RIN correlations are modest for PC1 (`ρ=0.364`) and PC2 (`ρ=0.315`) and negative for the explicitly labeled mean RNA-processing-expression proxy (`ρ=−0.514`). These values are cohort-confounded and are not causal tests.""")
code("""display(state_quality.groupby(['collection_group','condition']).RIN.agg(['count','mean','min','max']).style.format(precision=3))
display(state_quality_corr.style.format(precision=4).hide(axis='index'))
for name in ['rr1_rr3_2x2_similarity.png','rin_pc1.png']:
    img=plt.imread(state_dir/'figures'/name);plt.figure(figsize=(11,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Gene and pathway content inside PC1–2

High PC1–2 gene-contribution sets overlap more than expected between every RR1/RR3 pairing (Top-500 overlaps 25–48 genes versus 16.5 expected). However, only 0–2 of those shared genes also enter the controlled T-cell Top-500 signature. This favors **similar broad programs implemented by different genes**, rather than the same genes driving both controlled PolyA/Ribo displacement and NASA response similarity.

The shared high-contribution genes are enriched chiefly for fatty-acid oxidation/metabolism, peroxisomal lipid metabolism, bile/cholesterol transport, PPARα-related regulation, and small-molecule metabolism. RNA-processing terms are not the dominant recovered shared signal. Because the existing full RR1 contextual output stores component magnitudes rather than signed vectors, this is a high-contribution overlap analysis—not a signed concordance claim.""")
code("""display(state_overlap.drop(columns=['shared_genes','shared_controlled_genes']).style.format(precision=4).hide(axis='index'))
state_enrich=read_csv(state_dir/'shared_pc12_pathway_enrichment.csv')
display(state_enrich[state_enrich.fdr.lt(.05)].sort_values('fdr').head(30).style.format(precision=4).hide(axis='index'))
img=plt.imread(state_dir/'figures/shared_pc12_pathways.png');plt.figure(figsize=(12,8));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Interpretation

The evidence is mixed rather than exclusive:

- **Library-associated measurement structure:** strongly supported geometrically by the independent same-RNA experiment and the RR1 technical discrepancy. It is not supported as the identical gene mechanism: controlled Top-500 overlap with the shared RR1/RR3 sets is minimal.
- **Postmortem/sample-state effect:** plausible but unproven. RR1 preservation strata differ, yet RR3 lacks the postmortem timing needed to establish correspondence, and RIN does not reproduce the crossed pairing.
- **Experimental biology:** supported at a broad-program level by coherent hepatic lipid/fatty-acid/peroxisomal/bile programs shared inside PC1–2, but the small, confounded cohorts prevent a causal biological label.

Removing PC1–2 improves RR1 technical replication because the OSD-48/OSD-168 discrepancy lies strongly in that space. It simultaneously destroys RR1↔RR3 similarity because the same space also carries coherent cross-experiment hepatic response organization. The controlled reference is therefore **PolyA/Ribo-associated but not biologically empty**. Neither the residual space nor the removed space can be labeled pure biology or pure artifact.""")

md("""## 26. T-cell-signature-selective filtering inside PolyA/Ribo-sensitive PC1–2

This analysis tests whether the independently defined controlled T-cell PolyA/Ribo gene contribution can be removed without globally subtracting PC1–2. Technical genes are selected **only from the 40 paired T-cell donors**: contextual Top-100/250/500/1000 sets, paired-expression-supported genes, and the RNA-processing leading edge. RR1 and RR3 never enter signature selection.

The response-level implementation follows the model's exact mean-pooling decomposition. For a signature \(S\), it subtracts

\[|G|^{-1}\sum_{g\in S}P_{1:2}\Delta h_g\]

from the 512-D response while retaining every non-signature gene contribution, including contributions occupying PC1–2. This is a diagnostic response-vector experiment, not a deployable sample correction. The selected and non-selected vector sums can interfere, so their norm ratios are descriptive and are not causal percentages.""")
code("""selective_dir=HERE/'results/task4_selective_tcell_signature_filter'
selective_summary=read_csv(selective_dir/'selective_filter_summary.csv')
three_way=read_csv(selective_dir/'three_way_benchmark.csv')
display(selective_summary.style.format(precision=4,na_rep='—').hide(axis='index'))
display(three_way[three_way.signature.eq('top_500')].pivot(index='comparison',columns='method',values=['cosine','pearson','spearman']).style.format(precision=4))""")
md("""### Result: selective localization is specific but too small to resolve RR1

The T-cell Top-500 filter changes RR1 technical replication only from **−0.8042 to −0.7937** (gain 0.0106). The Top-1000 filter reaches **−0.7819** (gain 0.0224). Both improvements exceed their 200 expression/context-score-matched random-set nulls (`p=0.00498`), but their absolute magnitude is biologically insufficient: the reversal remains.

In contrast, whole-PC1–2 subtraction changes RR1 to **+0.2208**, while collapsing the four independent RR1/RR3 relationships (for example, carcass/RR3-39 **0.8139→0.1566**). Selective Top-500 filtering largely preserves them (**0.8037**, **−0.8216**, **−0.6833**, and **0.6046**), mainly because it removes very little of the response. The paired-DE definition contains 10,329 genes and improves RR1 to −0.1885, but it is not a selective signature and substantially erases the biological relationships.

This is outcome **B** from the prespecified possibilities: the technical effect appears distributed and cannot be localized adequately with the controlled T-cell high-ranking genes alone.""")
code("""decomp=read_csv(selective_dir/'response_signature_decomposition.csv')
display(decomp[decomp.signature.isin(['top_500','top_1000'])][['response','signature','signature_genes','inside_PC1_2_norm','signature_inside_norm','non_signature_inside_norm','signature_to_inside_norm_ratio','signature_to_inside_squared_norm_ratio']].style.format(precision=4).hide(axis='index'))
overlap=read_csv(selective_dir/'tcell_rr_gene_mechanism_overlap.csv')
display(overlap[overlap.top_n.eq(500)].style.format(precision=4).hide(axis='index'))""")
md("""### Mechanism, RNA processing, and preserved biology

For Top-500 selection, the signature contribution is only **2.8–3.9% of the PC1–2 vector norm** across these responses (roughly 0.07–0.15% by squared-norm ratio). T-cell/RR response score correlations are weak (absolute Spearman at most about 0.13), and most Top-500 overlaps are at or below random expectation; RR3-39 is the exception (28 genes versus 16.5 expected, `p=0.0046`) but direction agreement is only 35.7%. Thus shared occupation of PC1–2 generally arises through **different genes**, not one common T-cell-like mechanism.

The controlled signatures themselves are coherently enriched for mitochondrial RNA/tRNA/rRNA processing. Yet only **0–2** of the 37–57 shared high-contribution RR1/RR3 genes occur in the controlled Top-500 signature, leaving 35–55 genes retained. Some retained sets include RNA-processing genes, demonstrating that an RNA-related label alone cannot identify technical contribution. Existing independent analyses place the retained shared response primarily in hepatic fatty-acid/peroxisomal, PPARα, bile/cholesterol, and small-molecule programs and show leave-one-animal-out stability of the RR3 response orientation, while also establishing that the small cohorts limit mechanistic inference.""")
code("""display(read_csv(selective_dir/'preserved_biology_summary.csv').style.hide(axis='index'))
sig_enrich=read_csv(selective_dir/'controlled_signature_pathway_enrichment.csv')
display(sig_enrich[(sig_enrich.signature.isin(['top_500','RNA_leading_edge'])) & sig_enrich.fdr.lt(.05)].sort_values(['signature','fdr']).groupby('signature').head(10).style.format({'fdr':'{:.2e}'}).hide(axis='index'))
for name in ['three_way_benchmark.png','top500_random_control.png']:
    img=plt.imread(selective_dir/'figures'/name);plt.figure(figsize=(12,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Final interpretation

1. The controlled T-cell signature explains a statistically specific but quantitatively small part of the RR1 discrepancy when restricted to its Top-500/1000 genes.
2. RR liver responses and controlled T-cell PolyA/Ribo displacement occupy the same sensitive directions largely through different genes; pathway sharing is partial and does not establish an identical mechanism.
3. Selective filtering preserves RR1/RR3 relationships but does **not** meaningfully restore RR1 technical reproducibility.
4. Whole-PC1–2 removal is overcorrection for these biological comparisons: it improves the technical pair by removing a direction that also contains independently recurring hepatic organization.
5. Candidate information lost by whole removal includes fatty-acid oxidation, peroxisomal/lipid metabolism, PPARα-related regulation, bile/cholesterol transport, and small-molecule metabolism.
6. RNA processing is a strong controlled T-cell signature, but RR RNA-processing contributions are not reducible to the same T-cell genes.
7. A latent direction can be sensitive to a controlled technical perturbation without being specific to it. Technical perturbations should therefore be treated as diagnostic references rather than biologically empty axes that can safely be subtracted wholesale.""")

md("""## 27. Same PC1–2 geometry, same contextual mechanism?

This final diagnostic asks whether controlled T-cell PolyA→Ribo and RR liver responses reach the fixed PolyA/Ribo-sensitive PC1–2 directions through the same contextual gene programs. It performs **no correction**. Every response retains all 15,165 per-gene PC1 and PC2 contribution coordinates from the cached contextual tensors. Pairwise attribution similarity is calculated on the flattened, fixed-orientation `15,165 × 2` profiles; pathway scores aggregate gene contributions oriented by the independently defined mean T-cell PolyA→Ribo direction.

This distinction prevents arbitrary response-specific sign choices from making opposing responses appear mechanistically similar. Raw `log1p(TPM)` response comparisons provide the negative control.""")
code("""mechanism_dir=HERE/'results/task4_pc12_attribution_mechanisms'
mechanism_pairs=read_csv(mechanism_dir/'pairwise_attribution_similarity.csv')
pathway_pairs=read_csv(mechanism_dir/'pairwise_pathway_similarity.csv')
mechanism_pairs['pair_key']=mechanism_pairs.apply(lambda r:' | '.join(sorted([r.response_A,r.response_B])),axis=1)
pathway_pairs['pair_key']=pathway_pairs.apply(lambda r:' | '.join(sorted([r.response_A,r.response_B])),axis=1)
mechanism_compare=mechanism_pairs.merge(pathway_pairs.drop(columns=['response_A','response_B']),on='pair_key')
important=(mechanism_compare.response_A.str.contains('T-cell') |
           (mechanism_compare.response_A.str.contains('RR1') & mechanism_compare.response_B.str.contains('RR3')) |
           (mechanism_compare.response_A.str.contains('OSD-48 carcass') & mechanism_compare.response_B.str.contains('OSD-168')))
display(mechanism_compare.loc[important,['response_A','response_B','latent_PC1_2_cosine','attribution_cosine','attribution_spearman','top500_overlap','top500_hypergeom_p','top500_direction_agreement','pathway_profile_pearson','pathway_profile_spearman','expression_cosine','expression_spearman']].style.format(precision=4).hide(axis='index'))""")
md("""### Main diagnostic results

The answer is qualified **outcome A: the same latent direction can arise from substantially different contextual mechanisms**, although some RR pairs retain moderate gene-profile similarity.

- **Controlled T cell versus liver:** T-cell/RR latent cosines have substantial magnitude (`|cos|=0.63–0.78`), but whole-profile attribution Spearman correlations are only `−0.255` to `+0.173`. Top-500 overlaps are 8–27 genes, and pathway-profile correlations are much weaker than the latent geometry. OSD-168 has the clearest T-cell-like component (Top-500 overlap 27, `p=0.0084`, direction agreement 92.6%), but its pathway-profile Pearson is still only 0.138. The relationship is therefore partly mechanistic, not an identity of programs.
- **RR1 technical reversal:** OSD-48 carcass and OSD-168 are almost opposite within PC1–2 (`−0.978`) and oppose at the complete gene-coordinate level (`−0.654`). Yet their rank correlation is positive (0.202), 33 Top-500 genes overlap (`p=1.25×10⁻⁴`), and only 51.5% retain signed direction. This is a broad reversal/reweighting involving overlapping genes plus distributed contributions—not one compact RNA-processing Top-N set. That explains why whole-axis removal changes the reversal while Top-500/1000 filtering barely does.
- **RR1 carcass versus RR3-39:** latent cosine is 0.993 and attribution cosine is moderately concordant at 0.665; 37 Top-500 genes overlap (`p=3.91×10⁻⁶`) with 83.8% direction agreement. However, genome-wide attribution Spearman is only 0.175 and the signed pathway-profile Pearson is 0.020. This supports some shared gene-level organization but not a common global pathway-attribution profile.
- **RR1 euthanasia versus RR3-40:** latent cosine is 0.991, but attribution cosine is only 0.334, attribution Spearman is −0.076, and pathway-profile Pearson is −0.054. Although 46 Top-500 genes overlap, only 19.6% share direction. Its biological-mechanism interpretation is therefore much weaker than its latent similarity alone suggests.
- **Raw expression control:** the corresponding expression cosines are small (`−0.17` to `+0.32`) and do not reproduce the contextual geometry. Contextual attribution supplies information beyond ordinary expression similarity, but it is not itself proof of biological equivalence.""")
code("""for name in ['central_mechanism_comparison.png','latent_vs_attribution_similarity.png','pathway_attribution_heatmap.png']:
    img=plt.imread(mechanism_dir/'figures'/name);plt.figure(figsize=(15,8));plt.imshow(img);plt.axis('off');plt.show()
display(read_csv(mechanism_dir/'reference_positioning.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(mechanism_dir/'program_family_attribution.csv').style.format(precision=3).hide(axis='index'))""")
md("""### RR1 reversal and biological interpretation

The OSD-48 carcass response carries negative RNA-processing and hepatic lipid/peroxisomal attribution along the T-cell-oriented direction. OSD-168 instead carries negative RNA/ribosome attribution but positive cholesterol/lipid attribution. Thus the reversal is **not simply the same RNA-processing genes changing sign**: it combines broad gene reweighting with different pathway mixtures. OSD-168 is somewhat closer to the controlled perturbation than OSD-48, but neither is mechanistically identical to it.

Likewise, RR liver responses use PC1–2 for coherent hepatic programs that are weak in the controlled T-cell reference. Whole-axis subtraction helps RR1 technical agreement because most of the discrepancy vector occupies these directions; it overcorrects because those directions also carry distinct liver biology implemented through other genes. Top-N filtering fails because the technical-associated displacement is distributed beyond the most sensitive T-cell genes and because latent aggregation allows many weak contextual contributions to accumulate.

### Final answers

1. **Yes:** responses can occupy the same PC1–2 direction while using different genes; latent cosine is consistently stronger than rank-level mechanistic agreement.
2. **Yes:** signed pathway profiles can differ sharply even when latent directions nearly coincide.
3. Controlled T-cell PolyA/Ribo and RR liver responses resemble one another more geometrically than mechanistically. OSD-168 contains the strongest partial T-cell-like gene signal.
4. RR1 reversal reflects opposing distributed PC1/PC2 gene-coordinate profiles, with both overlapping genes and broad reweighting; it is not localized to the T-cell Top-N RNA-processing signature.
5. Whole-PC removal suppresses the distributed discrepancy because it removes the entire shared aggregation direction.
6. Top-N filtering fails because the effect is distributed and the selected genes account for only a small fraction of the response vector.
7. RR1 carcass/RR3-39 has moderate contextual support; RR1 euthanasia/RR3-40 is primarily a latent-geometry relationship and has weak mechanistic support.
8. Contextual attribution reveals organization not present in raw expression similarity, but its pathway and rank instability limits strong mechanistic claims.

The results support **same latent space, partially different mechanisms**, not a clean separation of technical and biological dimensions. PC1–2 remains PolyA/Ribo-sensitive, not technical-only.""")

md("""## 28. Does the RR1↔RR3 relationship survive matched ribodepletion?

This analysis directly replaces the OSD-48 PolyA RR1 measurement with its OSD-168 ribodepleted technical remeasurement before comparison with ribodepleted RR3. No correction is applied, PC1–2 is diagnostic rather than the primary endpoint, and RR3-39 and RR3-40 remain separate predefined cohorts.

The sample audit distinguishes profiles from animals. OSD-168 RR1 contains 10 FLT and 10 GC profiles because five animals per condition each have no-ERCC and ERCC measurements. Technical profiles are averaged within animal before FLT/GC means and animal-level bootstrap resampling. This equals the all-profile point estimate without treating technical replicates as independent animals. OSD-168 cannot be partitioned into carcass and upon-euthanasia states equivalent to OSD-48.""")
code("""ribo_dir=HERE/'results/task4_matched_ribo_rr1_rr3'
ribo_audit=read_csv(ribo_dir/'comparison_matrix_sample_audit.csv')
display(ribo_audit.style.hide(axis='index'))
ribo_results=read_csv(ribo_dir/'matched_ribo_comparison_summary.csv')
display(ribo_results[['comparison_type','response_A','response_B','full_latent_cosine','bootstrap_cosine_low','bootstrap_cosine_high','PC1_2_cosine','outside_PC1_2_cosine','attribution_cosine','attribution_spearman','pathway_pearson','pathway_spearman']].style.format(precision=4).hide(axis='index'))""")
md("""### Sanity checks and critical matched-Ribo result

The original OSD-48 comparisons reproduce: carcass/RR3-39 **0.813**, carcass/RR3-40 **−0.830**, euthanasia/RR3-39 **−0.697**, and euthanasia/RR3-40 **0.620**. Their PC1–2 cosines are approximately +0.992, −1.000, −0.999, and +0.991.

Replacing PolyA RR1 with the full ribodepleted OSD-168 RR1 response does **not** preserve those relationships:

- OSD-168 RR1 ↔ RR3-39: full cosine **−0.852**, PC1–2 **−1.000**, attribution cosine **−0.676**, pathway Pearson **−0.409**.
- OSD-168 RR1 ↔ RR3-40: full cosine **+0.790**, PC1–2 **+0.992**, attribution cosine **+0.557**, pathway Pearson **+0.032**.
- OSD-168 RR1 ↔ OSD-168 RR3-40: full cosine **+0.750**, PC1–2 **+0.987**, attribution cosine **+0.518**, pathway Pearson **+0.072**.

Thus the RR3 preference flips from RR3-39 to RR3-40 when RR1 is remeasured. Animal-bootstrap intervals are wide in these small cohorts—especially for RR3-40—so sign alone is not treated as replication. Nevertheless, the point estimates and contextual profiles agree that the original crossed relationships are measurement-sensitive.""")
code("""img=plt.imread(ribo_dir/'figures/central_matched_ribo_comparison.png')
plt.figure(figsize=(15,10));plt.imshow(img);plt.axis('off');plt.show()
display(ribo_results[['comparison_type','response_A','response_B','full_latent_cosine','PC1_2_cosine','attribution_cosine','attribution_pearson','attribution_spearman','pathway_pearson','top100_overlap','top250_overlap','top500_overlap','top1000_overlap']].style.format(precision=4).hide(axis='index'))""")
md("""### Three levels of reproducibility and technical controls

The three levels do not move together:

1. **Geometric reproducibility:** matched-Ribo OSD-168 RR1 aligns with RR3-40 rather than RR3-39.
2. **Contextual reproducibility:** gene-attribution similarity follows that switch, becoming negative for RR3-39 and moderately positive for RR3-40.
3. **Mechanistic reproducibility:** pathway agreement remains weak for RR3-40 (`Pearson=0.032`; same-OSD `0.072`) and negative for RR3-39. Matching ribodepletion therefore does not establish a shared pathway mechanism.

RR3-40 OSD-137↔OSD-168 is stable at all levels: full cosine **0.919**, attribution cosine **0.829**, and pathway Pearson **0.930**. In contrast, full-cohort RR1 OSD-48 carcass↔OSD-168 is opposing: full cosine **−0.901**, attribution cosine **−0.773**, and pathway Pearson **0.098**. The earlier −0.804 RR1 result used only exact matched no-ERCC animals; −0.901 here uses the full OSD-168 5+5-animal cohort and is intentionally labeled separately.

RR3's greater stability is consistent with its same-library comparison, but cannot be attributed exclusively to library selection because ERCC, library kit, sequencing configuration, and other processing variables also differ.""")
md("""### Direct answers

1. **No:** the original RR1↔RR3 relationship does not survive replacement by matched-ribodepleted RR1; the preferred RR3 state reverses.
2. Matching library preparation does not improve agreement with the original RR3-39 state; it instead makes RR1 resemble RR3-40.
3. Contextual attribution follows that switch but does not provide stable agreement with both states.
4. Pathway agreement does not broadly improve and remains weak for positive RR1/RR3-40 comparisons.
5. The original OSD-48 PolyA↔RR3 pattern is not robust to replacing RR1 with its OSD-168 remeasurement.
6. RR3-40 same-library technical replication is substantially more stable than RR1 cross-library replication at all three levels.
7. The evidence supports a **measurement-sensitive latent relationship**, not a securely shared RR1/RR3 biological mechanism.

This most closely matches **outcome B**, with an element of outcome C: latent and gene-attribution geometry switch coherently, but pathway equivalence remains weak. High cosine alone is not shared biology, and these data do not isolate PolyA/ribodepletion as the sole cause.""")

md("""## 29. Sample-first audit of RR1/RR3 in the controlled T-cell PC1–2 reference

This section deliberately steps back from aggregate RR1↔RR3 cosines. It audits all **54 profiles** from OSD-48 RR1, OSD-168 RR1, OSD-137 RR3, and the relevant OSD-168 RR3 remeasurements. Every sample is projected onto the unchanged controlled T-cell PolyA/Ribo-sensitive PC1–2 basis; RR samples are never used to fit the basis.

The master parquet retains local and authoritative OSDR identifiers, biological metadata, sequencing/QC fields, explicit unavailable batch variables, embedding identifiers/norms, and latent coordinates. OSD-168 ERCC/no-ERCC profiles are separate rows but are averaged within their five source animals per condition for cohort responses and animal bootstrap. The fraction of centered sample magnitude uses the independent controlled T-cell grand mean as its reference and is descriptive—not technical occupancy or causality.""")
code("""audit_dir=HERE/'results/task4_rr1_rr3_sample_pc12_audit'
sample_pc=read_csv(audit_dir/'sample_PC1_PC2_table.csv')
cohort_audit=read_csv(audit_dir/'cohort_level_audit.csv')
display(cohort_audit[['OSD','mission','cohort','sample_state','library_prep','n_FLT','n_GC','mean_FLT_RIN','mean_GC_RIN','GC_mean_PC1','GC_mean_PC2','FLT_mean_PC1','FLT_mean_PC2','delta_PC1','delta_PC2','full_response_norm','PC1_2_response_magnitude','response_energy_fraction_PC1_2','outside_PC1_2_response_magnitude','occupancy_bootstrap_low','occupancy_bootstrap_high']].style.format(precision=4,na_rep='NA').hide(axis='index'))
display(sample_pc.style.format(precision=4,na_rep='NA').hide(axis='index'))""")
md("""### Baseline position, flight position, and response are distinct

OSD-48 RR1 carcass lies at a high absolute baseline (GC PC1/PC2 **11.118/7.075**) and flight shifts still higher (**11.850/7.645**), yielding ΔPC1/ΔPC2 **+0.733/+0.571**. OSD-168 RR1 is globally lower (GC **8.943/5.657**), but flight shifts lower still (**8.503/5.190**), yielding **−0.440/−0.467**.

Both GC and FLT therefore undergo a large dataset/protocol-associated baseline shift from OSD-48 to OSD-168. Crucially, the FLT shift is larger: relative to the GC shift, the additional FLT displacement is approximately **−1.173 PC1** and **−1.038 PC2**. This is a condition-by-measurement-context interaction, not merely a global batch translation. It directly produces the response reversal.

RR3-40 behaves differently. OSD-137 and OSD-168 GC centroids differ only modestly, as do their FLT centroids, and their response components remain nearly identical: **−0.109/−0.089** versus **−0.118/−0.090**. That is why RR3-40 is technically reproducible while RR1 is not.""")
code("""display(read_csv(audit_dir/'technical_pair_position_audit_wide.csv').style.hide(axis='index'))
for name in ['rr1_response_arrows.png','rr3_40_response_arrows.png','individual_samples_by_panel.png']:
    img=plt.imread(audit_dir/'figures'/name);plt.figure(figsize=(14,9));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### How much of each response occupies PC1–2?

The squared response-energy fractions in the controlled reference are:

- RR1 carcass: **0.938**
- RR1 upon-euthanasia: **0.562**
- RR1 OSD-168 remeasurement: **0.886**
- RR3-39: **0.678**
- RR3-40: **0.733**
- RR3-41: **0.735** (1 FLT/2 GC; descriptive)
- RR3-39 remeasurement: **0.548**
- RR3-40 remeasurement: **0.689**

These are geometric energy fractions, not percentages caused by library preparation. Animal-bootstrap intervals are broad in the smallest cohorts, emphasizing that point estimates should not be treated as precise population parameters.""")
code("""associations=read_csv(audit_dir/'metadata_latent_associations.csv')
display(associations.sort_values('effect_size',key=abs,ascending=False).head(30).style.format(precision=4).hide(axis='index'))
display(read_csv(audit_dir/'condition_technical_interactions.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(audit_dir/'what_is_confounded.csv').style.hide(axis='index'))""")
md("""### Metadata associations and structural confounding

Cohort is the strongest descriptive correlate of PC1, PC2, and PC1–2 magnitude (`η²≈0.80–0.86`). Sample-state grouping, OSD, library kit, library selection, platform/layout, and preservation are also strongly associated. Those effects are not independently identifiable: in RR1, OSD-48 is PolyA/SE50/HiSeq3000 and contains carcass or immediate-dissection strata, whereas OSD-168 is ribodepleted/PE150/HiSeq4000 with ERCC/no-ERCC remeasurement. OSD and these protocol variables are structurally confounded.

The RR1 condition×dataset interaction is large for both coordinates (approximately −1.17 and −1.04). A descriptive FLT/GC×RIN interaction is near zero, but RIN is cohort-confounded and cannot exclude RNA-quality effects. Facility is reported as UC Davis throughout and cannot explain the observed contrast.

Globally extreme samples are predominantly OSD-48 carcass profiles because that entire cohort occupies a shifted region; this is not evidence that one animal drives the response. The output separately flags within-cohort/condition distance outliers. The carcass response is coherent across its five animals, while two-animal strata and RR3-41 remain intrinsically fragile.""")
code("""for name in ['positions_by_library_prep.png','positions_by_RIN.png','positions_by_sample_state.png','positions_by_cohort.png']:
    img=plt.imread(audit_dir/'figures'/name);plt.figure(figsize=(12,8));plt.imshow(img);plt.axis('off');plt.show()
display(sample_pc[sample_pc.within_cohort_condition_extreme].style.format(precision=4).hide(axis='index'))""")
md("""### Conservative conclusions

1. OSD-48 and OSD-168 RR1 differ in baseline location, library/sequence protocol, sample-state representation, and FLT-versus-GC displacement.
2. The reversal is not mainly a baseline shift: both groups shift globally, but FLT moves farther than GC, reversing ΔPC1 and ΔPC2. This is consistent with a measurement-context×condition interaction.
3. Most RR1 carcass and OSD-168 response energy lies in PC1–2; other RR responses also occupy those directions substantially.
4. Cohort, sample state, OSD, library preparation, platform/layout, preservation, and library kit track the coordinates most strongly.
5. In RR1 these variables change together and cannot be statistically separated.
6. No single animal provides a sufficient explanation for the full RR1 reversal; the main limitation is small and differently composed cohorts rather than one obvious outlier.
7. RR3-40 retains nearly identical FLT−GC PC1/PC2 components across remeasurement, whereas RR1 changes FLT and GC unequally.
8. The data reject a simple global-shift-only explanation. Measurement context changes the apparent condition response.
9. Metadata establish association and structural confounding, not which specific protocol component is causal.
10. The most conservative explanation is that the RR1 response is unstable across a compound protocol/measurement transition involving library selection, sequencing configuration, ERCC/resequencing context, and non-equivalent sample-state composition. The design cannot assign the reversal to PolyA/ribodepletion alone.

The controlled reference remains a useful diagnostic coordinate system, but it is not a technical-only space and no corrected or pure biological response is claimed.""")

md("""## 30. RR3-39 versus RR3-40: four underlying states

This analysis decomposes the opposing RR3 responses into **GC39, FLT39, GC40, and FLT40** rather than treating Δ39 versus Δ40 as a self-explanatory time course. Source-audited animals are F1/F2 versus G1/G2 for RR3-39 and F3/F4/F5 versus G3/G5 for RR3-40. The exact duration, RIN, weight, habitat, preservation, sequencing, feeding schedule, and all available timing fields are retained; missing collection/euthanasia/dissection timestamps remain `NA`.

The authoritative API contains multiple assays per animal, so the join is explicitly restricted to RNA-seq. One source inconsistency is retained rather than hidden: the local/audited G5 profile maps to an API RNA extract labeled G7. This is flagged in the metadata table and does not change the established contrast membership.""")
code("""rr3_four=HERE/'results/task4_rr3_39_40_four_state'
rr3_animals=read_csv(rr3_four/'animal_metadata_audit.csv')
display(rr3_animals.style.format(precision=4,na_rep='NA').hide(axis='index'))
display(read_csv(rr3_four/'four_state_PC_summary.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_four/'four_centroid_pairwise.csv').style.format(precision=4).hide(axis='index'))""")
md("""### Which centroids moved?

The 39/40 control centroids differ, but mostly outside PC1–2: GC39→GC40 has full-space distance **0.186**, PC1–2 magnitude **0.055**, and only **8.7%** of its squared energy in PC1–2. The flight centroids differ more: FLT39→FLT40 distance is **0.358**, PC1–2 magnitude **0.282**, with **62.0%** of its energy in PC1–2.

Thus both baseline and flight cohorts differ (**scenario C**), but the PC1–2 reversal emerges primarily from the larger FLT-cohort shift (**scenario B component**). The cohort-transition vectors are only weakly aligned overall (`cos=0.278`): inside PC1–2 they oppose (`−0.929`), whereas outside PC1–2 they agree (`+0.837`).

The actual FLT−GC responses show the same decomposition. Δ39 and Δ40 have full cosine **−0.567**, PC1–2 cosine **−0.995**, but outside-PC1–2 cosine **+0.457**. Their opposition is therefore concentrated in the PolyA/Ribo-sensitive directions rather than distributed uniformly across the 512-D representation. This is a geometric description, not evidence of a technical cause.""")
code("""display(read_csv(rr3_four/'four_state_response_geometry.csv').style.format(precision=4,na_rep='—').hide(axis='index'))
for name in ['four_state_pc12.png','four_state_pathways.png']:
    img=plt.imread(rr3_four/'figures'/name);plt.figure(figsize=(14,10));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Contextual genes and pathways

The GC39→GC40 and FLT39→FLT40 contextual shifts share substantial structure: attribution cosine **0.482**, Spearman **0.607**, and 254 overlapping Top-500 genes with **96.1%** coordinate-sign agreement. This indicates a broad cohort shift already present in controls and amplified/reoriented among flight animals.

By contrast, Δ39 versus Δ40 attribution cosine is **−0.369**. Their Top-500 sets overlap by 69 genes, but only **34.8%** retain coordinate direction. The responses therefore reuse some influential genes while changing their contextual direction and weighting.

Both responses implicate overlapping hepatic metabolic families, but not identically. Lipid/cholesterol/PPAR/peroxisomal programs are strongly represented and tend to oppose between Δ39 and Δ40. Mitochondrial/small-molecule programs are strong in both but have a more mixed directional relationship. This is best described as **partly opposing regulation of shared metabolic programs plus different weighting**, not entirely unrelated programs and not a proven physiological-state transition.""")
code("""display(read_csv(rr3_four/'attribution_comparison.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_four/'pathway_comparison.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_four/'pathway_family_diagnostic.csv').style.format(precision=3).hide(axis='index'))""")
md("""### Circadian, feeding, stress, and technical diagnostics

Circadian annotations are moderate for Δ39 and weak for Δ40; glucose/insulin and stress-related annotations are also present. These are transcriptional signatures only. Both cohorts are reported ad libitum under the same light/dark schedule, while actual food intake, collection phase, euthanasia time, and dissection order are unavailable. Circadian phase, feeding state, and stress exposure therefore cannot be assigned.

Reported preservation, ribodepletion, platform, facility, habitat, sex, strain, and broad protocol are shared. Duration, animal identity, collection cohort, library index, radiation exposure, RIN/read-depth distributions, and potentially unreported collection variables differ together. Duration cannot be isolated as the cause.""")
code("""display(read_csv(rr3_four/'technical_confounding_table.csv').style.hide(axis='index'))
display(read_csv(rr3_four/'animal_bootstrap_summary.csv').style.format(precision=4).hide(axis='index'))
loo=read_csv(rr3_four/'leave_one_animal_out.csv')
display(loo.style.format(precision=4).hide(axis='index'))
display(read_csv(rr3_four/'decision_table.csv').style.hide(axis='index'))""")
md("""### Animal-level robustness and final interpretation

The observed Δ39/Δ40 cosine is −0.567, but leave-one-animal-out values range from **−0.747 to +0.157**. The animal bootstrap median is −0.380 with a 95% interval of **−0.822 to +0.697**. Δ39 PC1 remains positive in the bootstrap interval, whereas Δ40 PC1 spans zero. The claim that the two population responses are genuinely opposing is therefore underpowered, despite the stability of the observed centroids and independent technical reproducibility of the matched RR3-40 subset.

Direct answers:

1. The mathematical reversal arises from **both** control and flight cohort movement, but the larger FLT shift—especially inside PC1–2—dominates.
2. GC39 and GC40 are descriptively different, predominantly outside PC1–2; this argues against a flight-only explanation.
3. FLT39 and FLT40 are about twice as far apart as their controls and differ strongly within PC1–2.
4. The two responses partly reuse hepatic metabolic programs with different/opposing contextual direction, rather than engaging wholly separate pathways.
5. Different physiological/metabolic states are a coherent hypothesis, not an established result.
6. Circadian, feeding, stress, collection timing, library index, and other cohort variables remain possible and mostly unmeasured or confounded.
7. Duration, animal identity, and collection cohort cannot be distinguished by this design.
8. RR3-40's OSD-168 reproduction supports that its observed response is technically measurable, but does not explain its biological origin or make the Δ39/Δ40 difference animal-robust.
9. BridgeRNA legitimately identifies a reproducible RR3-40 latent response and localizes the observed opposition to contextual metabolic organization within PC1–2. It cannot identify the causal physiological state.
10. Resolution requires an independent or factorial cohort with adequate animals at both durations, matched collection/euthanasia timing, measured food intake/circadian phase and metabolic phenotypes, and identical library/sequencing processing.

It would be incorrect to say that one additional day of spaceflight reverses the liver response. The paper-safe conclusion is that OSD-137 contains two small, technically coherent but biologically unresolved cohort responses whose observed opposition is concentrated in the controlled T-cell PolyA/Ribo-sensitive directions.""")

md("""## 31. Strict same-animal RR1/RR3 technical replication

This final audit replaces cohort-level approximations with the strict animal mappings requested for the three reconstructable technical replications. RR1 uses the four matched FLT animals M25/M26/M28/M30 and five matched GC animals M36–M40; M27 and M29 are excluded because they lack reciprocal measurements. RR3-39 uses F1/F2 and G1/G2. The strict RR3-40 analysis uses F3/F4 and G3/G5, with F5 retained only as a secondary full-stratum sensitivity analysis.

The original and remeasurement responses are always defined as `mean(FLT) - mean(GC)`. OSD-168 is treated as a technical remeasurement, never an independent biological experiment. The controlled T-cell PC1–2 reference is unchanged and no correction, model fitting, or new embedding inference is performed.""")
code("""paired_dir=HERE/'results/task4_rr1_rr3_paired_technical_replication'
display(read_csv(paired_dir/'animal_mapping.csv').style.hide(axis='index'))
display(read_csv(paired_dir/'mapping_discrepancies.csv').style.hide(axis='index'))
display(read_csv(paired_dir/'metadata_audit.csv').style.hide(axis='index'))""")

md("""### Response-vector replication

Strict same-animal comparison reproduces the established response cosines: **RR1 −0.804**, **RR3-39 +0.790**, and **RR3-40 +0.917**. RR1 therefore reverses across measurement contexts, whereas both RR3 subgroup responses retain their direction. The RR3-39 and RR3-40 responses remain mutually opposed in both measurements (OSD-137: **−0.456**; OSD-168: **−0.493**), showing that the observed subgroup distinction precedes remeasurement. This does not establish a duration effect because duration, animals, and collection cohort remain inseparable.

Including unmatched F5 in the original RR3-40 stratum changes the strict response appreciably (strict-versus-full cosine **0.831**) but does not erase its broad state. The strict two-animal definition is the primary apples-to-apples result.""")
code("""rep=read_csv(paired_dir/'response_replication_metrics.csv')
display(rep.style.format(precision=4,na_rep='—').hide(axis='index'))
img=plt.imread(paired_dir/'figures/response_replication.png');plt.figure(figsize=(13,8));plt.imshow(img);plt.axis('off');plt.show()
img=plt.imread(paired_dir/'figures/rr3_four_state_replication.png');plt.figure(figsize=(13,9));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### What drives the RR1 reversal?

For every technical replication, the identity

`Δz_remeasurement − Δz_original = mean(D_FLT) − mean(D_GC)`

holds numerically, where each animal displacement is `D = z_remeasurement − z_original`.

RR1 FLT and GC displacements point in almost the **same** direction (cosine **0.996**), so the reversal is not caused by opposite global technical shifts. Instead, their magnitudes differ substantially: `||D_FLT||=3.984`, `||D_GC||=2.666`, and `||D_FLT−D_GC||=1.353`. The measurement transition therefore moves FLT farther than GC along a broadly shared direction, changing the biological contrast.

The corresponding differential displacement is much smaller for RR3-39 (**0.144**) and RR3-40 (**0.073**). This supports a condition-dependent measurement effect in RR1 rather than a single additive batch translation.""")
code("""display(read_csv(paired_dir/'displacement_summary.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(paired_dir/'paired_animal_displacements.csv').style.format(precision=4).hide(axis='index'))
for name in ['paired_displacements.png','rr1_matched_animals.png']:
    img=plt.imread(paired_dir/'figures'/name);plt.figure(figsize=(13,8));plt.imshow(img);plt.axis('off');plt.show()""")

md("""### Resampling, contextual attribution, and pathway replication

Animal resampling reinforces the contrast between RR1 and RR3, while exposing the small-N uncertainty. Every RR1 leave-one-animal-out response remains negative (approximately **−0.880 to −0.581**), but its bootstrap interval is wide and crosses zero. RR3-39 and RR3-40 remain positive in both leave-one-out and bootstrap summaries.

The same pattern appears mechanistically. RR1 attribution is weak/opposing (cosine **−0.635**; only **7** shared Top-100 genes), and its pathway profile is weakly reproduced (Pearson **0.138**). RR3-39 and RR3-40 show substantially stronger attribution cosines (**0.775**, **0.858**), Top-100 overlaps (**62**, **85**), and pathway-profile Pearson correlations (**0.852**, **0.965**). Thus RR3 reproduces not only response direction but much of the learned gene/pathway organization; RR1 does not.""")
code("""display(read_csv(paired_dir/'bootstrap_summary.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(paired_dir/'leave_one_animal_out.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(paired_dir/'attribution_replication.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(paired_dir/'pathway_replication.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(paired_dir/'decision_table.csv').style.hide(axis='index'))""")

md("""### Final interpretation

- **RR1:** the strict same-animal FLT−GC response is not technically reproducible. FLT and GC experience a similarly directed but unequally sized measurement displacement, and latent, attribution, and pathway results all change markedly.
- **RR3-39 and RR3-40:** both strict responses reproduce across remeasurement, including their mutually opposed subgroup geometry. Their biological cause remains unresolved because each subgroup contains only two FLT and two GC animals and collection cohort is confounded with duration.
- **Supported combined conclusion:** within a fixed model, approximately global technical displacement can be distinguished from a measurement context whose displacement differs by biological condition. RR3 subgroup response geometry and contextual mechanisms reproduce under same-animal remeasurement; RR1's do not.

These results do **not** isolate PolyA selection as the sole cause of RR1 instability, establish pure biological axes, or prove a 39-versus-40-day time effect. The RR1 transition combines library selection, library kit, read setup, ERCC/resequencing context, handling, and OSD. The strict paired design diagnoses response instability under that compound transition.""")

md("""## 32. Empirical biological-contrast null for fixed T-cell PC1–2 occupancy

This section asks whether biological response vectors generally travel through the unchanged T-cell PolyA/Ribo-sensitive PC1–2 plane. It uses **20 independently curated contrasts with at least two samples per arm**: eight Task 2 skeletal-muscle exercise responses and twelve Task 3 mouse-liver FLT−GC responses. Two additional OSDR contrasts with one sample in an arm are retained as underpowered secondary observations but excluded from the primary empirical null.

For each response `Δz`, occupancy is `E_PC12 = ||Proj_PC12(Δz)||² / ||Δz||²`. This is a geometric energy fraction. It does not establish that PC1–2 is technical-specific, biologically meaningful, RNA-processing-specific, or causally responsible for a response.""")
code("""null_dir=HERE/'results/task4_pc12_empirical_biological_null'
occ=read_csv(null_dir/'biological_contrast_occupancy.csv')
refs=read_csv(null_dir/'reference_contrast_percentiles.csv')
display(occ.sort_values('E_PC12',ascending=False).style.format({'E_PC12':'{:.1%}','permutation_p_ge':'{:.4f}','bootstrap_low':'{:.1%}','bootstrap_high':'{:.1%}'}).hide(axis='index'))
display(refs.style.format({'E_PC12':'{:.1%}','empirical_percentile':'{:.1f}','empirical_p_ge':'{:.4f}'}).hide(axis='index'))
img=plt.imread(null_dir/'figures/empirical_occupancy_distribution.png');plt.figure(figsize=(15,8));plt.imshow(img);plt.axis('off');plt.show()
img=plt.imread(null_dir/'figures/contrast_occupancy.png');plt.figure(figsize=(14,10));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Empirical-null result

PC1–2 occupancy is common and often large in these curated responses: the primary-null median is **72.9%** (range **1.0–93.8%**). Consequently:

- RR3 GC39→GC40 (**8.7%**) is low but not exceptional (10th percentile).
- RR3 FLT39→FLT40 (**62.0%**) is ordinary (35th percentile).
- OSD-48 RR1 carcass (**93.8%**) is the maximum and therefore extreme relative to this small null (empirical `p=1/21=0.0476`).
- OSD-168 RR1 (**88.6%**) is high but not uniquely so (85th percentile).
- RR3-40 original/remeasurement (**73.3%/68.9%**) are near the middle of the biological distribution.

The empirical null is deliberately curated rather than randomly split, but it spans only exercise and spaceflight in skeletal muscle and liver. It is not a transcriptome-wide catalogue of biological perturbations.""")
code("""display(read_csv(null_dir/'random_subspace_null_summary.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(null_dir/'global_subspace_alignment.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(null_dir/'global_variance_spectrum.csv').head(20).style.format(precision=4).hide(axis='index'))
img=plt.imread(null_dir/'figures/global_variance_alignment.png');plt.figure(figsize=(14,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Random geometry and global BridgeRNA variance

Every observed contrast strongly exceeds isotropic random 2-D planes, which capture about **0.4%** of a fixed vector in 512 dimensions. That null is insufficient by itself because BridgeRNA is highly anisotropic. A second control samples 2-D planes from global BridgeRNA PCs with probability proportional to their variance; results are therefore reported separately and interpreted as an approximate variance-matched sensitivity analysis, not a unique null.

The fixed T-cell plane captures **55.3% of total global centered BridgeRNA variance** in 40,000 diverse ARCHS4 embeddings. Against the top two global PCs, one principal angle is only **12.9°**, while the other is **79.6°**. Mean squared overlap is **0.491**. Thus one T-cell direction is closely related to globally dominant latent variance, while the second is substantially more specific. This is **partial overlap**, not a wholly generic plane and not a wholly perturbation-specific plane.""")
code("""display(read_csv(null_dir/'contrast_robustness_summary.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(null_dir/'coordination_correlations.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(null_dir/'occupancy_metadata_characterization.csv').sort_values('E_PC12',ascending=False).style.format(precision=4).hide(axis='index'))
img=plt.imread(null_dir/'figures/coordination_hypothesis.png');plt.figure(figsize=(14,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Labels, coordination, and robustness

Within-study permutation tests are mostly nonsignificant even for high point estimates, and bootstrap intervals are often wide. Only the adequately replicated RR6 ISS-T contrast reaches the nominal matched-permutation threshold (`p≈0.049`). High occupancy occurs in human and mouse exercise as well as spaceflight, across PolyA, ribodepleted, and unharmonized library metadata; it does not identify one study, species, tissue, or protocol.

The prespecified expression-coordination diagnostics do not support a simple “few concentrated genes” explanation. Occupancy is negatively associated with the fraction of absolute expression response in the top 1% of genes (`ρ=−0.547`) and positively associated with expression effective gene count (`ρ=+0.597`). This suggests more distributed expression responses can occupy PC1–2 more strongly, but with only 20 heterogeneous contrasts it does not establish a coordinated-program axis. Response RMS is not significantly associated (`ρ=0.331`, `p=0.154`).

Cached contextual attribution is not uniformly available across all 20 contrasts, so no selective gene/pathway mechanism comparison was added; doing so only for OSDR would confound occupancy with dataset and tissue.""")
md("""### Conclusions from the empirical null

The evidence best supports **partial generic high-variance structure with context-dependent reuse**:

1. RR3 FLT39→FLT40's 62% occupancy is not unusual; GC39→GC40's 8.7% is low but within the observed range.
2. RR1's 89–94% values are high, but only OSD-48 carcass is extreme in this 20-contrast reference.
3. RR3-40's 69–73% occupancy is typical despite its strong technical reproducibility, confirming that occupancy and reproducibility answer different questions.
4. PC1–2 overlaps substantially with global BridgeRNA variance, principally through one direction; it is not merely an arbitrary 2-D plane.
5. High-occupancy contrasts do not share one obvious metadata class, and current data do not establish shared genes or pathways.
6. The coordinated-program hypothesis is not demonstrated. The observed correlations favor distributed rather than sparse expression change, but the contrast set is too small and heterogeneous for a mechanistic label.
7. PC1–2 can be treated as a **reused latent reference plane**, but not assigned a fixed technical or biological meaning. The controlled experiment establishes its PolyA/Ribo sensitivity; it does not establish specificity.

The primary limitation is breadth: this empirical null contains two perturbation families and two tissues. Broader curated perturbation datasets would be needed before generalizing the occupancy distribution.""")

md("""## 33. PC1 versus PC2: geometry and molecular realization

The combined T-cell PC1–2 result is decomposed here without changing either fixed direction. Geometry asks where a response moves; contextual attribution asks which input-gene representations locally generate that movement. Neither level alone assigns biological or technical meaning.

Across the same 20 primary biological contrasts, median energy is **42.3% in PC1** and **31.6% in PC2**. Both therefore contribute materially to the high combined occupancy.""")
code("""pc_dir=HERE/'results/task4_pc1_pc2_specific_geometry'
pcgeom=read_csv(pc_dir/'pc_specific_biological_contrasts.csv')
pcref=read_csv(pc_dir/'pc_specific_reference_contrasts.csv')
display(pcgeom.sort_values('E_PC1_plus_E_PC2',ascending=False).style.format({'E_PC1':'{:.1%}','E_PC2':'{:.1%}','E_PC1_plus_E_PC2':'{:.1%}','E_PC1_empirical_percentile':'{:.1f}','E_PC2_empirical_percentile':'{:.1f}'}).hide(axis='index'))
display(pcref.style.format({'E_PC1':'{:.1%}','E_PC2':'{:.1%}','E_PC1_plus_E_PC2':'{:.1%}','E_PC1_empirical_percentile':'{:.1f}','E_PC2_empirical_percentile':'{:.1f}'}).hide(axis='index'))
for name in ['pc_specific_geometry.png','reference_pc_decomposition.png']:
    img=plt.imread(pc_dir/'figures'/name);plt.figure(figsize=(15,8));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Which PC drives the reference observations?

- **RR1 carcass OSD-48:** PC1 carries **58.4%** and PC2 **35.4%** of response energy. OSD-168 carries **41.7%** and **46.9%**, respectively. Both signed coordinates reverse—from `+0.733/+0.571` to `−0.440/−0.467`. RR1 measurement sensitivity is therefore **distributed across both**, with PC1 somewhat larger originally and PC2 somewhat larger after remeasurement.
- **RR1 upon-euthanasia:** PC1/PC2 carry **26.1%/30.2%**, with both coordinates negative. Its geometry differs from the carcass stratum even within OSD-48.
- **RR3-40:** PC1/PC2 contributions are **43.7%/29.6%** in OSD-137 and **43.4%/25.6%** in OSD-168. Both coordinates preserve sign and similar magnitude. Its reproducibility is supported by both PCs, predominantly PC1.
- **GC39→GC40:** combined occupancy is low because PC1 contributes only **2.1%** and PC2 **6.6%**; PC2 is the larger of two small components.
- **FLT39→FLT40:** PC1 contributes **37.3%** and PC2 **24.7%**. Both coordinates change, with PC1 larger.

The observed 39/40 differences remain confounded with animal identity, collection cohort, and duration and are not interpreted as a causal one-day effect.""")
code("""display(read_csv(pc_dir/'global_pc_specific_geometry.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(pc_dir/'figures/global_variance_alignment.png') if (pc_dir/'figures/global_variance_alignment.png').exists() else plt.imread(HERE/'results/task4_pc12_empirical_biological_null/figures/global_variance_alignment.png')
plt.figure(figsize=(14,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Global ARCHS4 geometry

PC1 captures **30.4%** and PC2 **24.9%** of total centered variance across 40,000 ARCHS4 embeddings. Their strongest individual match is global ARCHS4 PC1 (`|cos|=0.723` and `0.653`). PC1 has 54.2% of its direction inside the top-two global-PC space; PC2 has 44.0%. Both exceed all sampled isotropic directions in variance.

Thus PC1 is the more globally dominant T-cell direction, but only modestly: PC2 is also a major global direction. The earlier combined-plane principal-angle result should not be paraphrased as PC2 being globally distinct or technical-specific. Orthogonal T-cell directions can both project strongly onto different components of the same global high-variance subspace.""")
code("""pcgenes=pd.read_parquet(pc_dir/'pc_specific_gene_attribution.parquet')
display(pcgenes[pcgenes['rank']<=25].sort_values(['PC','context','rank']).style.format(precision=5).hide(axis='index'))
display(read_csv(pc_dir/'pc_specific_attribution_similarity.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(pc_dir/'pc_specific_pathway_similarity.csv').style.format(precision=4).hide(axis='index'))
for name in ['pc_attribution_similarity.png','pc_specific_pathways.png']:
    img=plt.imread(pc_dir/'figures'/name);plt.figure(figsize=(16,10));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### PC-specific attribution reuse

Molecular realization is context dependent on both PCs. Across all context pairs, median attribution cosine is **0.368 for PC1** but only **0.013 for PC2**; median rank correlations are **0.107** and **0.029**. The strongest positive control is RR3-40 technical replication: attribution cosine is **0.877 on PC1** and **0.730 on PC2**, with pathway-profile Pearson **0.930** and **0.852**. This demonstrates that the method detects molecular reuse when it occurs.

The controlled T-cell response and liver responses generally share little Top-100 gene identity even when their latent coordinates align. For example, T-cell versus OSD-168 RR1 PC1 attribution cosine is moderately positive (**0.606**) but has **zero** shared Top-100 genes; similarity is distributed outside the most extreme genes. PC2 sometimes has greater pathway-profile similarity than gene-level similarity, but this is inconsistent across contexts.

Accordingly, neither PC has a stable universal functional label. PC1 shows somewhat greater gene-level reuse, especially within technical replication. PC2 is more molecularly heterogeneous, despite carrying almost one-quarter of global variance. Shared movement along a PC can arise from different gene combinations and cannot be treated as a shared mechanism.""")
md("""### Integrated interpretation

1. PC1 accounts for slightly more of the 55.3% global variance capture (**30.4% versus 24.9%**), but both are globally prominent.
2. Both contribute to the high biological-null median; PC1 is larger on average.
3. RR1's unusually high occupancy and its measurement reversal involve **both PCs**.
4. RR3-40 reproducibility also involves both PCs, predominantly PC1.
5. Low GC39→GC40 occupancy reflects small engagement of both directions; FLT39→FLT40 engages both, especially PC1.
6. The more weakly global PC2 does **not** show greater technical specificity or consistently shared molecular content.
7. Contextual and pathway reuse is strong for true RR3 technical replication but heterogeneous across unrelated contexts.

The best-supported description is therefore **globally prominent, reusable geometry with context-dependent molecular realization**. PC1 is somewhat more globally dominant and molecularly reusable; PC2 is somewhat more heterogeneous. Neither is technical-only, biologically pure, or entitled to a semantic pathway label. Conventional expression and independent replication remain necessary for biological validation.""")

md("""## 34. Multi-layer response reproducibility benchmark

This benchmark asks whether response geometry alone identifies a known replication relationship, or whether response-specific gene attribution and pathway profiles add information. The strict comparable core contains six mouse-liver FLT−GC responses and all 15 pairings among them:

- three designed positives: RR1, RR3-39, and RR3-40 same-animal technical remeasurements;
- twelve same-species, same-tissue cross-response negatives.

All attribution vectors are existing full-response Integrated Gradients computed by the same Task 4 implementation. No embeddings, IG, or model inference are recomputed. Importantly, these positives establish **measurement correspondence**, not independent biological-cohort generalization.""")
code("""multi_dir=HERE/'results/task4_multilayer_reproducibility'
display(read_csv(multi_dir/'response_metadata.csv').style.hide(axis='index'))
bench=read_csv(multi_dir/'pairwise_benchmark.csv')
display(bench.sort_values(['is_replication','latent_cosine'],ascending=[False,False]).style.format(precision=4).hide(axis='index'))
for name in ['layer_distributions.png','layer_relationships.png']:
    img=plt.imread(multi_dir/'figures'/name);plt.figure(figsize=(16,8));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Geometry is not sufficient

Latent cosine separates the designed pairs only moderately. RR3-40 and RR3-39 reproduce geometrically (**0.917**, **0.790**), whereas RR1 reverses (**−0.807** in the strict vectors used here). Meanwhile, one unrelated pair—OSD-48 RR1 versus original RR3-39—is a clear geometric false friend: latent cosine **0.811**, but attribution cosine only **0.258**, pathway Pearson **0.220**, and conventional expression Pearson **0.046**.

RR1 also illustrates why a replication design label and a reproduced response are different. Although its latent direction reverses, full-response attribution retains substantial absolute-gene structure: attribution cosine **0.727**, rank correlation **0.194**, 53 shared Top-100 genes, and 96.2% sign agreement among those shared genes. Its pathway-profile Pearson is **0.734**. This indicates partial molecular continuity alongside a non-reproducible aggregate latent response; it does not make the response technically robust.""")
code("""scores=read_csv(multi_dir/'composite_score_performance.csv')
display(scores.style.format({'ROC_AUC':'{:.3f}','PR_AUC':'{:.3f}','bootstrap_low':'{:.3f}','bootstrap_high':'{:.3f}','leave_one_positive_out_min':'{:.3f}','leave_one_positive_out_max':'{:.3f}'}).hide(axis='index'))
display(read_csv(multi_dir/'geometric_false_friends.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(multi_dir/'figures/composite_performance.png');plt.figure(figsize=(14,8));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Descriptive discrimination and conventional baseline

With only three positives, no classifier is fitted. Each model is an unweighted mean of within-dataset percentile ranks:

- latent geometry only: ROC AUC **0.694**, PR AUC **0.632**;
- attribution only: **1.000/1.000**;
- pathway only: **1.000/1.000**;
- geometry + attribution: **0.972/0.917**;
- geometry + attribution + pathway: **0.972/0.917**;
- conventional expression Pearson/Spearman: **1.000/1.000**.

Thus molecular layers add information beyond latent cosine in this small controlled set, but pathway agreement adds no measurable discrimination beyond attribution, and including geometry slightly hurts because the known RR1 pair is geometrically reversed. Conventional expression performs equally well. These values are descriptive and strongly optimistic: the same 15 pairs define percentile scaling and evaluation, positives share studies with negatives, and the three positives are not statistically independent examples of biological generalization.""")
code("""display(read_csv(multi_dir/'latent_bootstrap_summary_reused.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(multi_dir/'latent_leave_one_out_reused.csv').style.format(precision=4).hide(axis='index'))""")
md("""### Conclusions and limits

1. **No:** high latent cosine is not sufficient evidence of replicated biology. One high-cosine unrelated pair is molecularly discordant, and one known same-material pair reverses geometrically.
2. In this compact null, **1/12** unrelated pairs exceeds latent cosine 0.75. This frequency is dataset-specific, not a universal false-friend rate.
3. Attribution and pathway profiles distinguish the observed geometric false friend and the designed pairs, but only three positives are available.
4. Pathway agreement does not add discrimination beyond attribution in this sample.
5. The combined score does not outperform attribution alone and is worse than conventional expression; there is no evidence here that BridgeRNA is superior to log-expression response comparison.
6. Existing animal bootstrap and leave-one-out results support RR3 stability and RR1 instability at the geometry level. Attribution/pathway uncertainty cannot be propagated without rerunning IG and is not claimed.
7. No category-C independent-cohort replication with fully comparable response-specific IG was available. Therefore this is a **technical measurement-reproducibility benchmark**, not yet a general biological reproducibility benchmark.

The defensible framework is layered rather than scalar: geometry asks whether aggregate latent responses align; attribution asks whether similar molecular inputs support them; pathways summarize whether those inputs imply related programs; experimental replication determines whether any of those agreements generalize. None substitutes for the others, and high scores do not establish causality.""")

md("""## 35. Does BridgeRNA attribution add information beyond expression change?

This analysis directly compares conventional gene-expression change with the existing full-response Integrated Gradients attribution for nine contrasts, each over the exact 15,165-gene vocabulary. Seven attribution profiles are reused unchanged; the two missing four-state profiles (GC39→GC40 and FLT39→FLT40) use the identical frozen model, zero-input baseline, response-direction target, and 16-step IG implementation.

For each contrast, a simple five-fold out-of-fold linear model predicts signed attribution from signed expression change, absolute expression change, mean expression, and log-transformed expression variance. Residual attribution is observed minus out-of-fold prediction. Genes are folds—not independent biological samples—so cross-validation measures descriptive generalization across genes, not external biological validation. Residuals are model-derived quantities and are not automatically biological signal.""")
code("""ae_dir=HERE/'results/task4_attribution_vs_expression'
corr=read_csv(ae_dir/'expression_attribution_correspondence.csv')
display(corr.style.format(precision=4).hide(axis='index'))
img=plt.imread(ae_dir/'figures/expression_vs_attribution.png');plt.figure(figsize=(16,20));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Direct correspondence and predictability

Attribution **magnitude ranks are strongly coupled** to conventional expression magnitude: Spearman correlations range from **0.663 to 0.770** (median **0.683**). A simple model predicts magnitude rank with Spearman **0.638–0.816**. Thus highly changing genes are much more likely to receive high attribution.

Signed relationships are context dependent, ranging from **−0.613 to +0.509**. Despite strong magnitude-rank association, the linear expression model explains little signed attribution variance out of fold: median cross-validated R² is only **0.031**. Magnitude R² is also low (**0.005–0.112**), indicating a strongly rank-monotonic but poorly calibrated/nonlinear relationship.

This rules out both extremes: attribution is not merely identical to log-expression change, but neither is it independent of expression magnitude.""")
code("""genes=pd.read_parquet(ae_dir/'contrast_gene_tables.parquet')
catc=genes[genes.category.eq('C_low_moderate_expression_high_attribution')]
display(catc.groupby('contrast').size().rename('category_C_genes').reset_index().style.hide(axis='index'))
display(catc.sort_values(['contrast','residual_rank']).groupby('contrast').head(25)[['contrast','gene_symbol','expression_change','attribution','predicted_attribution_oof','residual_attribution','standardized_residual','expression_rank','attribution_rank','residual_rank']].style.format(precision=5).hide(axis='index'))
display(read_csv(ae_dir/'discordant_category_reproducibility.csv').query("category=='C_low_moderate_expression_high_attribution'").style.format(precision=4).hide(axis='index'))""")
md("""### Genes disproportionate to expression change

Category C is defined consistently as genes in the top 10% of absolute attribution but at or below the median absolute expression change. It contains **138–258 genes per contrast**. Examples include hepatic transport/metabolic and regulatory genes such as `TTR`, `CYP27A1`, `SLC27A2`, `GCKR`, `HNF4A`, `APOE`, `PIGR`, and `GPX4`, depending on contrast.

Category-C overlap across remeasurement is **22 genes for RR1**, **80 for RR3-39**, and **75 for RR3-40**, well above the roughly 2–4 genes expected from random sets of these sizes. This is evidence that disproportionate attribution is not entirely random. It remains technical-remeasurement evidence and does not establish causal biological importance.""")
code("""resrep=read_csv(ae_dir/'technical_replication_residual_metrics.csv')
display(resrep.style.format(precision=4).hide(axis='index'))
gsea=pd.read_parquet(ae_dir/'residual_attribution_gsea.parquet')
display(gsea[gsea.fdr.lt(.05)].sort_values(['contrast','fdr']).style.format(precision=4).hide(axis='index'))
display(read_csv(ae_dir/'geometric_false_friend_expression_adjustment.csv').style.format(precision=4).hide(axis='index'))""")
md("""### Does residual attribution reproduce?

Residual-attribution cosine remains **0.734 for RR1**, **0.869 for RR3-39**, and **0.694 for RR3-40**. Corresponding rank correlations are **0.255**, **0.549**, and **0.468**. These profiles are therefore not erased by the tested expression covariates.

However, residual pathway evidence is weak. Across 27 contrast/source combinations, only one pathway reaches FDR < 0.05: alanine/aspartate/glutamate metabolism in RR3-40 OSD-168. The residual profiles reproduce geometrically without yielding broadly significant, reproducible pathway enrichment. This limits their biological interpretability.

For the RR1↔RR3-39 geometric false friend, expression cosine is **0.121**, attribution cosine **0.258**, and residual cosine **0.249**. Residualization does not create additional separation: attribution distinguishes the pair primarily because it follows already discordant molecular responses, not because an expression-independent residual becomes uniquely divergent.""")
code("""auc=read_csv(ae_dir/'added_information_auc.csv')
display(auc.style.format(precision=3).hide(axis='index'))
img=plt.imread(ae_dir/'figures/summary.png');plt.figure(figsize=(16,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Added-information decision

Expression, raw attribution, residual attribution, and expression-plus-residual all achieve descriptive AUC **1.0** for separating the three same-material pairs from twelve cross-response pairs. Because expression alone is already perfect, residual attribution provides **no measurable incremental discrimination** in this tiny benchmark. Bootstrap intervals are degenerate because all sampled pairwise orderings remain separated; this does not imply population certainty.

The most defensible classification is **mixed / insufficient evidence**:

1. Much of attribution ranking tracks expression magnitude (`median ρ≈0.683`).
2. Signed attribution is only weakly explained by the tested linear expression features (`median CV R²≈0.031`).
3. Expression-adjusted residual profiles reproduce across all three technical remeasurements, and category-C genes recur above chance.
4. Residual pathway organization is almost entirely nonsignificant, so the reproduced residual cannot yet be assigned coherent biology.
5. Residual similarity does not improve discrimination beyond conventional expression.

BridgeRNA attribution currently has **descriptive interpretive value** for showing how the model weights genes and how that weighting changes, but this experiment does not demonstrate superior or independently validated biological information beyond conventional expression analysis. Independent biological replications—not only technical remeasurements—are needed to establish a model-specific contextual signal.""")

md("""## 36. Conventional expression PCA versus frozen BridgeRNA

This audit asks what the frozen 512-dimensional representation adds beyond the linear covariance structure in the exact 15,165-gene `log1p(TPM)` input. The primary reference is **PCA-15165**. PCA was fitted on 32,029 study-disjoint ARCHS4 training samples and evaluated on the same 40,000 samples used for the global BridgeRNA audit. A same-sample full-vocabulary ARCHS4 matrix was not available, so **global PCA-FULL is reported as unavailable rather than approximated**.

Two qualifications matter. First, only the first 512 exact PCA-15165 eigenvalues were computed; statistics requiring the complete spectrum use a documented uniform residual-tail approximation. Second, reused OSDR full-vocabulary PCA and joint human/mouse tissue PCA are transductive descriptive controls, not substitutes for the study-disjoint global PCA reference.""")
code("""pca_dir=HERE/'results/task4_pca_vs_bridgerna'
display(read_csv(pca_dir/'global_variance_summary.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(pca_dir/'figures/global_scree.png');plt.figure(figsize=(14,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Global variance and geometry

BridgeRNA is substantially more anisotropic than its input: PC1+PC2 explain **65.2%** of latent variance versus **23.7%** in PCA-15165. The participation ratio is **2.98** for BridgeRNA versus approximately **22.60** for expression PCA (the latter uses the residual-tail estimate). BridgeRNA needs 19 and 30 PCs for 90% and 95% variance, respectively.

The representations are related but not equivalent. Across 299,891 deterministic sample pairs, Euclidean-distance Spearman correlation is **0.747**. Mean neighborhood overlap is **43.5%, 47.3%, 53.6%, and 52.9%** at k=5, 10, 50, and 100. Normalized orthogonal Procrustes R² is **0.582**. High canonical correlations show that shared linear directions exist, while the neighborhood and Procrustes results show material geometric reorganization.""")
code("""display(read_csv(pca_dir/'pairwise_geometry.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(pca_dir/'neighborhood_preservation.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(pca_dir/'linear_alignment.csv').style.format(precision=4).hide(axis='index'))""")
md("""### Controlled T-cell PolyA/Ribo displacement

The paired transformation is already exceptionally low-dimensional in conventional expression: PC1+PC2 explain **94.8%** of displacement variance. BridgeRNA raises this to **99.1%**. Donor Gram-matrix CKA is **0.979**, although donor-distance Spearman is **0.442**. Thus BridgeRNA modestly concentrates and reorganizes a transformation whose dominant low-dimensional structure is already present in expression; it did not create that structure de novo.""")
code("""display(read_csv(pca_dir/'tcell_displacement_summary.csv').style.format(precision=4).hide(axis='index'))""")
md("""### Response geometry and the geometric false friend

The globally fitted PCA-15165 baseline essentially reproduces BridgeRNA's strong RR3 technical-remeasurement concordance, but not the severe RR1 reversal. RR1 cosine is **−0.207 in PCA**, **+0.369 in raw expression**, and **−0.807 in BridgeRNA**. RR3-39 and RR3-40 are nearly identical between PCA and BridgeRNA (**0.790** and approximately **0.915**, respectively).

Most importantly, the RR1 OSD-48 ↔ original RR3-39 false-friend cosine is only **0.361 in PCA** and **0.121 in raw expression**, versus **0.811 in BridgeRNA**. The learned representation therefore amplifies this misleading similarity.

The secondary full-vocabulary PCA values below were fitted on all 112 OSDR samples and are transductive. With the maximum 111 PCs, cosine is mathematically equivalent to centered full-expression geometry; sample-rank limitation prevents a true 512-PC baseline.""")
code("""display(read_csv(pca_dir/'response_focus_comparisons.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(pca_dir/'pca_full_osdr_technical_metrics_reused.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(pca_dir/'figures/response_comparison.png');plt.figure(figsize=(14,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Reused tissue readouts

In the balanced cross-species tissue benchmark, raw expression and joint PCA outperform BridgeRNA for centroid cosine and 1-NN on average. BridgeRNA outperforms joint PCA for the linear probe, while raw expression remains competitive. Because joint PCA was fitted to the combined human/mouse matrix, these results are a transductive downstream reference rather than a strict source-only PCA generalization test.""")
code("""tissue=read_csv(pca_dir/'downstream_tissue_balanced_reused.csv')
display(tissue.groupby(['representation','readout'],as_index=False).accuracy_mean.mean().style.format({'accuracy_mean':'{:.1%}'}).hide(axis='index'))""")
md("""### Masked-expression reconstruction

The PCA basis was learned without TCGA. Test-sample scores were estimated by least squares using only observed genes; masked values never entered score estimation. Results below use 1,000 TCGA samples and seed 0 for PCA, whereas the reused BridgeRNA values summarize 10 seeds, so this is not a variance-matched significance comparison.

At 50% masking, the best predefined PCA result (512 PCs) gives Pearson **0.9440**, slightly above BridgeRNA (**0.9392**). At 90% masking, 256-PC PCA gives **0.9244**, far above BridgeRNA (**0.6780**). This provides strong evidence that extreme-mask reconstruction in this dataset is largely supported by linear transcriptomic covariance. The non-monotonic 512-PC result at 90% masking is consistent with less stable missing-data score estimation at excessive dimensionality.""")
code("""display(read_csv(pca_dir/'pca_imputation.csv').style.format(precision=5).hide(axis='index'))
display(read_csv(pca_dir/'bridgerna_imputation_reused.csv').style.format(precision=5).hide(axis='index'))
img=plt.imread(pca_dir/'figures/pca_imputation.png');plt.figure(figsize=(14,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Audit decision table""")
code("""display(read_csv(pca_dir/'summary_decision_table.csv').style.hide(axis='index'))""")
md("""### Interpretation

For the endpoints evaluated here, the result most closely supports **Outcome C: BridgeRNA amplifies low-dimensionality in ways that can be destructive**, with an important mixed qualification.

1. BridgeRNA clearly adds nonlinear concentration/reorganization: its variance is much more concentrated and only about half of local PCA neighbors are preserved.
2. That reorganization is not demonstrably beneficial in the tested response and reconstruction endpoints. PCA matches RR3 reproducibility, avoids much of the RR1 reversal and false-friend amplification, and equals or exceeds masked reconstruction.
3. The controlled T-cell response is already nearly one-dimensional in expression, so its concentration cannot be attributed solely to the model.
4. Existing tissue results are mixed: BridgeRNA can outperform joint PCA under a linear probe, but raw/PCA often perform better under centroid and neighborhood readouts.
5. Global PCA-FULL remains unavailable on the exact same 40,000 samples, and the PCA imputation baseline currently has one seed. These constraints preclude a universal claim that PCA is superior.

What BridgeRNA demonstrably adds is a compact, highly anisotropic nonlinear reparameterization. This audit does **not** show that the added reorganization improves the biological questions tested here; in RR1 and the false-friend example it makes geometry less faithful to conventional molecular agreement.

The single most informative next experiment is a preregistered, study-disjoint external benchmark comparing source-only PCA-15165, PCA-FULL, raw expression, and BridgeRNA across several biological tasks with identical splits and repeated seeds. That would determine whether the mixed tissue-probe advantage generalizes beyond these diagnostic examples.""")

md("""## 37. Biological generalization beyond expression PCA

This follow-up uses identical samples, labels, study splits, and linear classifiers for raw 15,165-gene `log1p(TPM)`, fold-fitted PCA-15165, and frozen BridgeRNA-512. Tissue labels are conservative exact matches to ARCHS4 `source_name_ch1`; original text and mapping rules remain in the manifest.

The cohort contains **3,272 human samples, 1,678 GSE studies, and 14 tissues**. Every tissue has at least 40 samples and 20 studies. The upstream 40,000-sample reference capped every GSE at two samples, making five-fold GroupKFold the primary stable study-disjoint estimate. Individual LOSO estimates and study-ID prediction would have only one or two observations per held-out study/class; they are marked unavailable rather than presented as meaningful benchmarks.

PCA and hyperparameters are fitted or selected strictly inside each training fold. PCA dimensions are selected from the predefined 10, 25, 50, 100, 256, and 512 grid using an inner GSE-disjoint validation split. No BridgeRNA weights or embeddings were recomputed.""")
code("""gen_dir=HERE/'results/task4_pca_vs_bridgerna_generalization'
display(read_csv(gen_dir/'cohort_summary.csv').style.hide(axis='index'))
display(read_csv(gen_dir/'metric_summary_with_bootstrap_ci.csv').query("scheme=='study_disjoint_group5'").style.format(precision=4).hide(axis='index'))
img=plt.imread(gen_dir/'figures/study_disjoint.png');plt.figure(figsize=(13,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Primary study-disjoint result

BridgeRNA does not improve tissue generalization. Mean macro F1 is **0.780 ± 0.028** for BridgeRNA, versus **0.888 ± 0.016** for validation-selected PCA and **0.872 ± 0.021** for raw expression. The BridgeRNA-minus-PCA difference is **−0.108 macro-F1 points**. Accuracy is 86.9%, 93.1%, and 92.1%, respectively.

BridgeRNA trails selected PCA in 13 of 14 tissues. Skeletal muscle is the only exception: BridgeRNA and raw expression both reach approximately 0.967 F1 versus 0.945 for PCA.""")
code("""folds=read_csv(gen_dir/'fold_metrics.csv')
display(folds.style.format(precision=4).hide(axis='index'))
pc=read_csv(gen_dir/'per_class_metrics.csv')
display(pc.groupby(['representation','tissue'],as_index=False).f1.mean().pivot(index='tissue',columns='representation',values='f1').style.format('{:.3f}'))""")
md("""### Random split and dimensionality

The easier random split has the same ordering: raw **0.951**, PCA-512 **0.942**, and BridgeRNA **0.857** macro F1. It is a sanity check, not evidence of cross-study generalization.

Training-only validation favors high-dimensional PCA: four outer folds select 512 PCs and one selects 256. Mean inner-validation macro F1 rises from about 0.46 at 10 PCs to 0.87 at 512 PCs, so the comparison does not cherry-pick an artificially weak low-dimensional PCA.""")
code("""display(read_csv(gen_dir/'summary_metrics.csv').style.format(precision=4).hide(axis='index'))
curve=read_csv(gen_dir/'pca_dimension_validation_curve.csv')
curve_summary=curve.groupby('dimension',as_index=False).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'))
display(curve_summary.style.format(precision=4).hide(axis='index'))
plt.figure(figsize=(8,4));plt.errorbar(curve_summary.dimension,curve_summary.macro_f1_mean,yerr=curve_summary.macro_f1_sd,marker='o',capsize=3);plt.xlabel('PCA dimensions');plt.ylabel('Inner-validation macro F1');plt.title('Training-only PCA dimension curve');plt.show()""")
md("""### Tissue and study neighborhoods

At k=10, tissue purity is **0.832 raw**, **0.818 PCA**, and **0.774 BridgeRNA**. Study purity is low for all representations (**0.076, 0.076, and 0.066**) because every GSE contributes at most two samples. BridgeRNA does not improve tissue locality, but it also does not amplify study locality in this capped cohort.""")
code("""display(read_csv(gen_dir/'neighborhood_results.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(gen_dir/'figures/neighborhood_purity.png');plt.figure(figsize=(14,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Low-label and dominant-PC sensitivity

The learning curve uses one fixed held-out group fold and one deterministic training-study subsample per fraction, so it is exploratory. BridgeRNA is lower at every fraction. At 10%, macro F1 is **0.485 BridgeRNA**, **0.539 PCA**, and **0.563 raw**; at 100% it is **0.769**, **0.872**, and **0.857**.

Removing dominant BridgeRNA PCs does not rescue performance. Macro F1 is 0.769 with no removal, 0.769 after PC1, 0.764 after PC1–2, and 0.765 after PC1–5. These exploratory removals are not assigned a technical or biological mechanism.""")
code("""display(read_csv(gen_dir/'learning_curves.csv').style.format(precision=4).hide(axis='index'))
display(read_csv(gen_dir/'bridge_pc_removal_sensitivity.csv').style.format(precision=4).hide(axis='index'))
img=plt.imread(gen_dir/'figures/learning_and_pc_sensitivity.png');plt.figure(figsize=(14,6));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Secondary external evidence and unavailable endpoints

The existing balanced GTEx-human ↔ ENCODE-mouse experiment is the compatible external result already available. It jointly changes source and species and uses transductive joint PCA, so it is secondary—not a clean ARCHS4→GTEx source-only classifier. Raw and PCA generally lead centroid/1-NN readouts; BridgeRNA is stronger than joint PCA for the linear probe but not consistently stronger than raw.

PCA-FULL is unavailable for the same ARCHS4 reference. Study prediction and per-study LOSO are not responsibly estimable from a reference capped at two samples/GSE. These are recorded in `limitations.json`, not interpreted as negative results.""")
code("""display(read_csv(HERE/'results/task4_pca_vs_bridgerna/downstream_tissue_balanced_reused.csv').style.format(precision=4).hide(axis='index'))
display(pd.DataFrame([json.loads((gen_dir/'limitations.json').read_text())]).T.rename(columns={0:'status'}).style)""")
md("""### Decision

The primary summary is:""")
code("""display(read_csv(gen_dir/'primary_summary_table.csv').style.format(precision=4).hide(axis='index'))""")
md("""The best-supported decision is **D: destructive compression for this biological-identity endpoint**.

- BridgeRNA trails PCA and raw expression on random and study-disjoint tissue classification.
- Its study-disjoint deficit versus selected PCA is **−0.108 macro F1**.
- It shows no low-label advantage and lower tissue-neighborhood purity.
- Dominant-PC removal does not restore performance.
- External source/species results are mixed but establish no consistent advantage.

The narrow answer to **“What does BridgeRNA demonstrably provide beyond conventional expression PCA?”** is that it supplies compact, nonlinear, highly anisotropic reorganization—but this audit finds **no tissue-generalization advantage**. In this ARCHS4 study-disjoint setting, compression obscures tissue information retained by regularized raw expression and PCA.

This does not show that Transformers cannot help transcriptomics or that BridgeRNA lacks value for every task. The most informative follow-up is an uncapped, curated multi-sample-per-study cohort supporting genuine LOSO and tissue-controlled study prediction, plus a clean source-only ARCHS4→GTEx test with PCA fitted only on ARCHS4.""")

md("""## 38. Where does representation collapse occur?

The frozen pipeline is 15,165 `log1p(TPM)` values → learned gene plus 512-D rotary expression embeddings → 12 pre-norm residual attention/FFN blocks → 15,165×512 contextual tokens → unweighted mean → 512-D sample embedding. There is no CLS token, species embedding, or post-pooling projection.

For the same 3,272 samples and five GSE-disjoint folds, six fixed summaries were cached at input and every layer: mean, median, max, SD, expression-weighted mean, and mean+SD. All layer comparisons use the same class-balanced LSQR ridge probe with training-only regularization selection.""")
code("""collapse=HERE/'results/task4_information_collapse'
display(read_csv(collapse/'final_information_loss_table.csv').style.format(precision=4,na_rep='—').hide(axis='index'))
for image_name in ['participation_ratio.png','pc1_pc2.png','tissue_f1.png']:
    img=plt.imread(collapse/'figures'/image_name);plt.figure(figsize=(15,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Localization

Mean pooling is nearly rank-one at the input embedding (**PC1+2 99.6%; participation ratio 1.28**). The Transformer then increases mean-pooled dimensionality and tissue decodability: macro F1 rises from 0.083 at input to 0.741 after layer 1, peaks at **0.810 at layer 7**, and ends at 0.800. Mean-pooled tissue 10-NN purity peaks near **0.820 at layers 5–7**, then falls to **0.774 at layer 12**, demonstrating modest late-layer degradation rather than monotonic collapse.

Alternative readouts recover information. Final-layer mean+SD is best at **0.836 ± 0.026 macro F1**, versus 0.800 for mean; SD alone gives 0.816 and participation ratio 5.10. No layer/readout reaches raw expression (**0.872**) or PCA (**0.888**). High dimensionality alone is insufficient: max pooling has higher participation ratio and neighborhood purity but weaker F1.""")
code("""loss=read_csv(collapse/'information_loss_map.csv')
display(loss.nlargest(20,'study_disjoint_macro_f1')[['stage','pool','dimensions','study_disjoint_macro_f1','macro_f1_sd','participation_ratio','PC1_2','tissue_10nn_purity','study_10nn_purity']].style.format(precision=4).hide(axis='index'))
display(loss.query("stage=='layer_12'").sort_values('study_disjoint_macro_f1',ascending=False).style.format(precision=4).hide(axis='index'))""")
md("""### Decision

The supported localization is **E: mixed readout loss and late-layer degradation**. The Transformer initially adds accessible tissue information, middle layers preserve the strongest neighborhoods, later layers lose some locality, and mean pooling discards second-order token information. Mean+SD partially repairs the readout but remains 0.052 F1 below PCA, so pooling alone does not explain the deficit.

The smallest justified model change is a mean+SD readout with a small controlled projection, evaluated at both the final layer and layers 6–7 before retraining the backbone.

This section completes architecture mapping, all-layer pooling, effective dimensionality, tissue decoding, and neighborhood localization. Layer-wise OSDR response geometry, masked reconstruction, and random-PC controls require separate response-cohort inference and are not inferred from these results.""")

md("""## 39. Frozen readout selection and response-geometry safety

We next compared fixed, label-free summaries of frozen contextual gene embeddings at layers 4–9 and 12. Candidate readouts included mean, SD, mean+SD, mean+variance, mean+max, expression-weighted mean, and their prespecified combinations. Every tissue result uses the exact same five GSE-disjoint folds and a class-balanced ridge probe with regularization selected using training studies only.

The best fixed readout is final-layer **mean+SD (1,024 dimensions)**. It improves study-disjoint tissue macro F1 from 0.800 for the standard final mean to **0.833 ± 0.025**, but remains below raw expression (0.872) and selected PCA (0.888). Unsupervised fold-local PCA compression does not improve it. This supports a mixed conclusion: the standard mean readout discards useful token-distribution information, but readout loss alone does not explain the full deficit.""")
code("""readout_dir=HERE/'results/task4_frozen_readout_selection'
ranking=read_csv(readout_dir/'final_readout_ranking.csv')
display(ranking.head(15).style.format(precision=4).hide(axis='index'))

fig,axes=plt.subplots(1,2,figsize=(15,6))
top=ranking.head(12).copy()
top['label']='L'+top.layer.astype(str)+' '+top.readout.str.replace('_',' ')
axes[0].barh(top.label[::-1],top.macro_f1[::-1],xerr=top.macro_f1_sd[::-1],color='#3973ac',alpha=.9)
axes[0].axvline(.8716,color='#555',ls='--',label='Raw expression')
axes[0].axvline(.8883,color='#b34d4d',ls='--',label='Selected PCA')
axes[0].set(xlabel='Study-disjoint macro F1',title='Best fixed frozen readouts')
axes[0].legend(frameon=False)
axes[1].scatter(ranking.participation_ratio,ranking.macro_f1,c=ranking.layer,cmap='viridis',s=45)
axes[1].set(xlabel='Participation ratio',ylabel='Study-disjoint macro F1',title='Dimension alone does not determine performance')
fig.tight_layout();plt.show()""")
md("""### OSDR safety check

The selected mean+SD readout was then evaluated on the exact strict RR1 and RR3 technical-remeasurement contrasts used throughout Task 4. This is a safety check, not a new selection endpoint. The frozen backbone and contrast memberships were unchanged, and the standard mean readout first reproduced the established response cosines within tolerance.

Mean+SD does **not** erase or conceal the known RR1 failure: RR1 remains opposing (−0.788 versus −0.807). It also preserves the qualitative RR3 controls, with RR3-39 at 0.767 and RR3-40 at 0.914. Thus the improved tissue readout does not manufacture technical robustness, and it does not materially damage the already strong RR3-40 replication.""")
code("""safety=read_csv(readout_dir/'osdr_response_safety.csv')
display(safety.style.format({'cosine':'{:.4f}','spearman':'{:.4f}','norm_original':'{:.4f}','norm_remeasurement':'{:.4f}'}).hide(axis='index'))
plot=safety.pivot(index='comparison',columns='readout',values='cosine').loc[['RR1','RR3-39','RR3-40']]
ax=plot.plot.bar(figsize=(10,5),color=['#8c8c8c','#3973ac'])
ax.axhline(0,color='black',lw=.8);ax.set(ylabel='Technical-replication response cosine',xlabel='',title='Selected readout preserves the established OSDR diagnosis')
ax.legend(['Current mean','Selected mean + SD'],frameon=False);plt.xticks(rotation=0);plt.tight_layout();plt.show()""")
md("""### Final readout decision

Use **layer-12 mean+SD** as the best frozen general-purpose readout found in this controlled search when retaining a 1,024-D representation is acceptable. Keep the original 512-D mean as the canonical checkpoint output for compatibility. The selected readout is a partial repair rather than evidence that the backbone surpasses conventional PCA: it improves tissue decoding, preserves the major OSDR replication conclusions, but still trails raw/PCA tissue generalization. No supervised projection is promoted as a universal embedding because such a projection would be label- and task-specific.""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3.11'}}
nbf.write(nb, HERE/'library_prep_disentanglement_benchmark.ipynb')
print(HERE/'library_prep_disentanglement_benchmark.ipynb')
