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

md("""### Concise comparison and interpretation

### Concise comparison and interpretation

1. **Expected behavior is reproduced.** RR1 is opposing (`R = −0.804`) and strongly aligned with PC1–2 (`T = 0.953`); RR3-39 is moderately reproducible (`R = 0.790`) with lower alignment (`T = 0.141`); RR3-40 is highly reproducible (`R = 0.917`) despite appreciable alignment (`T = 0.466`).
2. **R and T are distinct.** RR3-40 demonstrates that technical-associated structure can be detectable in the comparatively small discrepancy while the primary FLT−GC response remains reproducible.
3. **RR1 is unusually aligned.** Its score is above all 1,000 random two-dimensional subspaces (`p = 0.001`) and stable across donor bootstraps (median 0.948; 95% interval 0.925–0.958).
4. **Biological overlap matters.** Stronger removal resolves RR1 only while disrupting broader response organization; technical-associated and biological structure are not cleanly separable.
5. **Attribution adds impact context.** RR1 shows gene reweighting within hepatic metabolic programs, while RR3 has greater attribution concordance and Top-100 overlap.
6. **Contextual relationships also change.** RR1 has median gene-context reproducibility −0.125 with 61.6% reversed gene-context responses, compared with medians 0.783/0.864 and reversal fractions 7.5%/6.6% for RR3-39/RR3-40. The most contextually unstable RR1 genes show RNA-processing, chromatin, and DNA-repair enrichment; they overlap little with Top-100 IG genes, indicating complementary information.
7. **The assumption remains explicit.** The reference is stable within controlled T cells, but its cross-tissue universality is unknown.

**RR1 overall:** The RR1 mouse-liver spaceflight response reverses across remeasurement (`R = −0.804`). Its discrepancy is strongly aligned with the controlled PolyA/Ribo reference (`T = 0.953`). Removing sufficient technical-associated structure to resolve this reversal preserves only 0.559 correlation with the broader response-similarity organization. Gene attribution shows response-basis reweighting, while contextual analysis shows widespread instability in genes' learned transcriptomic relationships. The associated latent structure is therefore technically sensitive and biologically entangled—not proven purely technical, causal, or safely correctable.""")

md("## 12. Final benchmark summary")
code("""if not summary.empty:
    final = summary.rename(columns={'representation':'Representation','auroc':'PolyA/Ribo AUROC','pair_cosine':'Pair cosine','pair_r1':'Pair R@1'})
    if not task3.empty:
        challenge = task3.pivot(index='representation',columns='comparison',values='cosine').reset_index().rename(columns={'representation':'Representation'})
        final = final.merge(challenge, on='Representation', how='left')
    final = final.rename(columns={'biology_metric_value':'Biology metric (source-ID MRR)'})
    keep = [c for c in ['Representation','PolyA/Ribo AUROC','Pair cosine','Pair R@1','Biology metric (source-ID MRR)','macro_f1','RR1','RR3-39','RR3-40'] if c in final]
    display(final[keep].style.format(precision=3,na_rep='—'))""")

md("""## 13. Conservative interpretation

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

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3.11'}}
nbf.write(nb, HERE/'library_prep_disentanglement_benchmark.ipynb')
print(HERE/'library_prep_disentanglement_benchmark.ipynb')
