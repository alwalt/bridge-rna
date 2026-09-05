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
for name in ['rr1_profile.png','rr3_39_profile.png','rr3_40_profile.png']:
    img=plt.imread(profiler_dir/'figures'/name)
    plt.figure(figsize=(10,7));plt.imshow(img);plt.axis('off');plt.show()""")
md("""### Technical Sensitivity Map

Dashed lines at `T=0.5` and `R=0` are visual guides only. RR3-40 is the key counterexample to treating technical alignment as automatic biological invalidation: technical-associated structure is detectable in its discrepancy while the primary FLT−GC response remains highly reproducible.""")
code("""img=plt.imread(profiler_dir/'technical_sensitivity_map.png')
plt.figure(figsize=(11,8));plt.imshow(img);plt.axis('off');plt.show()
primary=['Comparison','Response Reproducibility','Response Category','Technical Alignment PC1-2','Biological Overlap','Biological Overlap Evidence']
display(profiler[primary].style.format({'Response Reproducibility':'{:.3f}','Technical Alignment PC1-2':'{:.3f}'}).hide(axis='index'))""")

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

md("""### Contextual gene-embedding feasibility

No cached per-sample `15,165 × 512` contextual gene-embedding tensors were found for the RR1/RR3 Task 3 samples. Available contextual-gene artifacts belong to the separate exercise-response benchmark and cannot answer this remeasurement question. Under the no-recomputation constraint, this exploratory analysis was not run. IG asks which genes influence a sample-level response; contextual embeddings would instead ask which genes' learned contexts change across remeasurement.

### Concise comparison and interpretation

1. **Expected behavior is reproduced.** RR1 is opposing (`R = −0.804`) and strongly aligned with PC1–2 (`T = 0.953`); RR3-39 is moderately reproducible (`R = 0.790`) with lower alignment (`T = 0.141`); RR3-40 is highly reproducible (`R = 0.917`) despite appreciable alignment (`T = 0.466`).
2. **R and T are distinct.** RR3-40 demonstrates that technical-associated structure can be detectable in the comparatively small discrepancy while the primary FLT−GC response remains reproducible.
3. **RR1 is unusually aligned.** Its score is above all 1,000 random two-dimensional subspaces (`p = 0.001`) and stable across donor bootstraps (median 0.948; 95% interval 0.925–0.958).
4. **Biological overlap matters.** Stronger removal resolves RR1 only while disrupting broader response organization; technical-associated and biological structure are not cleanly separable.
5. **Attribution adds impact context.** RR1 shows gene reweighting within hepatic metabolic programs, while RR3 has greater attribution concordance and Top-100 overlap.
6. **The assumption remains explicit.** The reference is stable within controlled T cells, but its cross-tissue universality is unknown.

**RR1 overall:** The mouse-liver spaceflight response is technically unstable. Its discrepancy strongly resembles the independently characterized PolyA/Ribo transformation, but associated latent dimensions overlap biological-response structure and cannot be cleanly removed without changing broader organization. This is diagnostic alignment, not causal attribution or batch correction.""")

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
