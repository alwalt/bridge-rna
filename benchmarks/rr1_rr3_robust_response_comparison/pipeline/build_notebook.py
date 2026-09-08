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
M(nbf.v4.new_markdown_cell("""## 6. Robust and reversing pathways

An existing signed pathway profile is called protocol-robust when it is mutually Top-250 and directionally consistent. Direction-reversing pathways have opposite sign and are Top-250 in at least one measurement. These are contextual/pathway-profile scores, not conventional GSEA NES."""))
M(nbf.v4.new_code_cell("""counts = pd.read_csv(R/'summary/pathway_counts.csv')
display(counts.style.hide(axis='index'))
display(Image(filename=str(R/'figures/pathway_profiles.png'), width=1050))"""))
M(nbf.v4.new_markdown_cell("""RR1 has only 58 protocol-robust and 169 direction-reversing pathway profiles. RR3-39 has 158 robust and 6 reversing; RR3-40 has 211 robust and none reversing. This is the clearest pathway-scale distinction between the cohorts."""))
M(nbf.v4.new_markdown_cell("""## 7. Shared and cohort-specific biology"""))
M(nbf.v4.new_code_cell("""bio = pd.read_csv(R/'summary/biological_summary.csv')
shared = json.loads((R/'summary/shared_unique_counts.json').read_text())
display(pd.Series(shared, name='N').to_frame().style)
display(bio.query("feature_type == 'gene' and interpretation == 'shared_all_three'").style.hide(axis='index'))
display(bio.query("feature_type == 'pathway' and interpretation == 'shared_all_three'").head(25).style.hide(axis='index'))"""))
M(nbf.v4.new_markdown_cell("""Only five ≥2-level genes are shared by all three comparisons: **GADD45G, IGFBP1, NRN1, SLC41A2, and TCIM**. The shared pathway core is broader (29 profiles), emphasizing respiration/mitochondrial gene expression, translation, and amino-acid metabolism. RR3-39 and RR3-40 share 24 robust genes and 53 robust pathway profiles, consistent with a larger RR3 core despite distinct duration-dependent states. Cohort-specific calls remain candidates rather than universal spaceflight markers."""))
M(nbf.v4.new_markdown_cell("""## 8. Direct answers

1. **How much RR1 survives?** A restricted core survives: 179 expression genes, 246 IG genes, 476 positive local graph neighborhoods, 114 genes supported by ≥2 levels, and 26 by all three. Its global and Hallmark response directions do not reproduce.
2. **Compared with RR3?** RR3-40 is strongest (381 expression genes; 172 ≥2-level; 31 all-three), followed by RR3-39 at the global/pathway scales. RR1 is markedly less reproducible.
3. **Is RR1 uniquely splicing-sensitive?** It is the clearest cohort for measurement-sensitive mRNA processing/splicing enrichment. Related terms occur elsewhere, so “unique” should not be interpreted as exclusive biology.
4. **What is robust in all three?** Five multiscale genes and 29 pathway profiles, chiefly translation, respiration/mitochondrial expression, and amino-acid metabolism.
5. **What is cohort-specific?** 90 RR1-only, 75 RR3-39-only, and 134 RR3-40-only ≥2-level genes under these definitions; these may reflect duration, cohort, or protocol.
6. **Does attribution identify a more stable subset than expression alone?** It supplies orthogonal model evidence and narrows candidates through intersections, but RR1 attribution ranks themselves are weakly reproducible; it does not simply outperform expression.
7. **Does the graph add confirmation?** Yes, at the gene-neighborhood level, but median local agreement remains much lower in RR1 (0.138) than RR3-39/40 (0.340/0.386).
8. **Why does RR3-40 reproduce better?** Numerically it is more concordant at expression, global latent, Hallmark, graph, and pathway levels. The benchmark establishes this association, not a causal mechanism.
9. **Is RR1 failure technical or biological?** Mixed. It coincides with a major protocol transition and strong measurement-sensitive programs, while a smaller concordant biological core remains.
10. **What remains defensible?** A technically reproducible core involving mitochondrial/respiratory, translational, amino-acid metabolic, and a small set of shared genes. Claims should remain at the level of concordant response."""))
M(nbf.v4.new_markdown_cell("""## 9. Limitations and provenance

- OSD-168 is a technical remeasurement, not an independent biological replicate.
- “Protocol-robust” only applies to the compared measurement pair.
- The integrated core counts unweighted agreement across expression, signed input-gene IG, and contextual graph evidence.
- Expression-interaction ORA is restricted to the 15,165-gene universe; it is distinct from full-transcriptome raw-count edgeR/GSEA.
- Pathway-profile scores are reused outputs and should not be described as conventional NES.
- No causal assignment of RR1 instability to one protocol variable is possible from this comparison."""))

nb["cells"] = cells
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
