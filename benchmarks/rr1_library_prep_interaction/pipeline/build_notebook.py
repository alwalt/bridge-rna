#!/usr/bin/env python3
"""Build the RR1 interaction benchmark notebook from machine-readable results."""
from pathlib import Path
import nbformat as nbf
HERE=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook();nb["metadata"]["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"};nb["metadata"]["language_info"]={"name":"python","version":"3"}
def md(s):nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbf.v4.new_code_cell(s))
md("""# RR1 library-preparation interaction benchmark

## Primary question

Is the RR1 OSD-48→OSD-168 reversal explained by a common PolyA→ribodepletion displacement, or does the protocol transition modify the estimated FLT−GC response itself?

This benchmark reuses exact animal mappings and frozen Task 3/4 assets. In RR1, OSD, PolyA/ribodepletion, read layout/length/depth, preservation, and processing change together. We therefore test a **protocol-associated prep×flight interaction** and do not assign causality to library selection alone.""")
code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd()
if not (ROOT/'results').exists():ROOT=ROOT/'benchmarks/rr1_library_prep_interaction'
R=ROOT/'results';pd.set_option('display.max_colwidth',120)
summary=pd.read_csv(R/'summary/multiscale_interaction_summary.csv')
decision=json.loads((R/'summary/decision.json').read_text())""")
md("""## 1. Exact sample matching

Only the OSD-48 carcass stratum has an OSD-168 counterpart. The analysis uses four matched FLT animals (M25, M26, M28, M30) and five matched GC animals (M36–M40). M27 is excluded because it lacks an OSD-168 counterpart. The OSD-48 upon-euthanasia animals have no OSD-168 remeasurement and are analyzed only as a handling sensitivity, not pooled into the primary comparison. Same animal/liver is established; identical RNA aliquot across OSD-48 and OSD-168 is not.""")
code("""mapping=pd.read_csv(R/'manifest/exact_animal_mapping.csv')
display(mapping[mapping.cohort.eq('RR1')][['measurement','OSD','condition','animal_id','sample_id','library_preparation','sequencing_parameters','preservation']])""")
md("""## 2. Four-effect decomposition

For every vector representation:

`Flight_PolyA = FLT_PolyA − GC_PolyA`

`Flight_Ribo = FLT_Ribo − GC_Ribo`

`Prep_FLT = FLT_Ribo − FLT_PolyA`

`Prep_GC = GC_Ribo − GC_PolyA`

`Interaction = Prep_FLT − Prep_GC = Flight_Ribo − Flight_PolyA`

The matched-animal exact permutation test reassigns the nine observed animal-level prep displacements to groups of four versus five. It tests whether the FLT/GC difference in prep displacement is unusually large; it cannot identify which co-changing protocol variable caused it.""")
code("""cols=['representation','flight_response_agreement','prep_FLT_vs_GC_cosine','interaction_norm','interaction_to_mean_prep_ratio','interaction_exact_permutation_p']
display(summary[cols].style.format({c:'{:.3f}' for c in cols if c!='representation'},na_rep='—'))""")
md("""### Primary finding

The FLT and GC prep displacements point in nearly the same direction for expression, global, L12 mean+SD, and Hallmark representations (cosine 0.962–0.996): there is a large common protocol displacement. However, their difference is not negligible. The interaction is 35–41% of the mean prep-displacement magnitude in the conventional/global/program vector spaces and is tied only by the observed allocation and its complement among 126 exhaustive condition-label permutations (`p = 2/126 = 0.0159`). Thus, a common shift exists **and** the protocol transition modifies the estimated FLT−GC response.

The modification is consequential: global BridgeRNA FLT−GC agreement reverses (−0.804), L12 mean+SD is −0.785, and Hallmark module response is −0.416. Signed input-gene IG attribution retains positive cosine (0.727) despite low genome-wide rank reproducibility, while expression and graph agreement remain only partially positive. IG should not be confused with the separate contextual-token PC1–2 projection metric.""")
code("display(Image(filename=str(R/'figures/critical_decomposition.png')))")
md("""## 3. Independent T-cell PolyA→Ribo reference

The controlled same-RNA T-cell experiment defines an external technical direction separately in each mathematically compatible space. Projection is diagnostic, not causal attribution. PCA is fit once on all Task 3 samples and then used to transform T-cell profiles; no RR1 outcome is used to choose the direction.""")
code("""cols=['representation','prep_FLT_tcell_alignment_cosine','prep_GC_tcell_alignment_cosine','interaction_tcell_alignment_cosine','interaction_projected_energy_fraction','flight_response_agreement','flight_response_agreement_after_removal']
display(summary[cols].style.format({c:'{:.3f}' for c in cols if c!='representation'},na_rep='—'))""")
md("""The global RR1 prep displacements align strongly with the T-cell direction (FLT 0.782; GC 0.803), and 51.9% of the global interaction energy lies along that single mean direction. Removing it improves global response cosine only from −0.804 to −0.709; it does not restore reproducibility. Expression improves from 0.369 to 0.378 and PCA to 0.402. The graph-level T-cell displacement is nearly orthogonal to the RR1 graph interaction and removal has essentially no effect.

Therefore, a generic controlled prep direction explains part of the interaction, but not the RR1 reversal. This is consistent with the earlier PC1–2 analysis: removing broader technical-associated structure can improve RR1 further, but at the cost of reorganizing biological response geometry.""")
code("display(Image(filename=str(R/'figures/before_after_residualization.png')))")
md("""## 4. Genes and programs localizing the interaction

The gene table uses the exact 15,165-gene log1p(TPM) model input—not manufactured counts. It reports the expression interaction, sign flips, existing response IG values, and graph-interaction neighborhood magnitude. These are effect-size diagnostics rather than a new DE test.""")
code("""genes=pd.read_parquet(R/'interaction/per_gene_interaction.parquet')
display(genes.head(25).style.format({c:'{:.3f}' for c in genes.select_dtypes('number').columns}))""")
code("""enrich=pd.read_csv(R/'interaction/top250_interaction_enrichment.csv')
display(enrich.head(20).style.format({'p_value':'{:.2e}','fdr':'{:.2e}'}))""")
md("""The strongest interaction genes include SNURF, MTCP1, FUBP1, ZNF169, PLIN4, SLC34A2, PFKFB2, and OGT. The post hoc Top-250 enrichment is dominated by mRNA processing, splice-site recognition, spliceosomal regulation, and related RNA-metabolic programs. This supports protocol sensitivity of RNA-processing measurements, but it does not establish that these pathways are artifacts or that library selection is the only cause.""")
md("""## 5. Handling and RNA quality

The upon-euthanasia stratum cannot be technically replicated because OSD-168 lacks corresponding animals. It therefore cannot adjudicate the primary matched comparison. Existing RIN interaction models are descriptive and cohort-confounded; neither PC1 nor PC2 shows evidence of an FLT/GC-by-RIN interaction.""")
code("""strata=pd.read_csv(R/'handling/stratum_replication.csv');display(strata)
rin=pd.read_csv(R/'handling/rin_and_condition_diagnostics.csv');display(rin[rin.analysis.str.contains('RIN')])""")
md("""## 6. RR3 positive controls

RR3 was never used to define the T-cell direction. Its role is to show that technical remeasurement can remain reproducible in the same overall project. RR3-39 and RR3-40 retain global response cosines of 0.790 and 0.917; their program and attribution agreement are also much stronger than RR1. RR3-40 is the cleanest control because it remains concordant across every tested scale.""")
code("""controls=pd.read_csv(R/'summary/rr1_rr3_multiscale_controls.csv')
display(controls.pivot(index='comparison',columns='representation',values='similarity').style.format('{:.3f}'))""")
md("""# Protocol-robust core RR1 response

The complete RR1 vectors are unstable, but that does not imply every gene or program is unstable. This section identifies evidence that survives both measurements without forcing the complete vectors to agree.

The primary gene-level definition is transparent and prespecified:

- **Robust expression:** mutual Top-500 absolute response and the same direction.
- **Robust attribution:** mutual Top-500 absolute signed IG and the same direction.
- **Robust contextual neighborhood:** Top-500 per-gene local graph similarity with positive concordance.
- **Multilevel candidate:** support from at least two of these three independent readouts.

Top-100 and Top-250 sensitivity results are retained separately. The evidence count is not learned or weighted.""")
code("""CORE=R/'robust_core'
sizes=pd.read_csv(CORE/'rr3_controls/robust_core_size.csv')
metrics=pd.read_csv(CORE/'rr3_controls/multilevel_reproducibility.csv')
overlap=pd.read_csv(CORE/'rr3_controls/topn_overlap.csv')
display(sizes)
display(metrics.style.format({c:'{:.3f}' for c in metrics.select_dtypes('number').columns},na_rep='—'))
display(overlap.style.format({'direction_agreement':'{:.1%}'}))""")
md("""## RR1 robust-core size

RR1 contains **179** robust-expression genes, **246** robust-IG genes, and **476** locally stable graph genes under the Top-500 definitions. Their intersections yield **114 genes supported by at least two levels**, including **26 supported by all three**.

At Top-100, expression shares 46 genes with 100% sign agreement, while IG shares 53 genes with 96.2% sign agreement. Genome-wide expression Spearman is 0.425; IG Spearman is only 0.194 despite cosine 0.727, showing that a concentrated common direction coexists with extensive rank reweighting.""")
code("""candidates=pd.read_csv(CORE/'integrated/protocol_robust_candidates.csv')
cols=['gene_symbol','evidence_count','expression_original','expression_remeasurement','attribution_original','attribution_remeasurement','graph_local_cosine','absolute_interaction_rank','handling_sensitive','tcell_top500','interpretation']
display(candidates[cols].head(40).style.format({c:'{:.3f}' for c in cols if c not in ['gene_symbol','evidence_count','handling_sensitive','tcell_top500','interpretation']}))""")
md("""The 26 three-level candidates include **PER3, TNS2, DBP, CHKA, PPP1R3B, MFSD2A, SLC45A3, FMO3, GPCPD1, SLC22A7, THRSP, PLA2G12A, POR, CYP8B1, APOA5, FABP5, BNIP3, CDKN1A, HAO2, and GADD45G**. These genes span circadian regulation, hepatic transport, lipid handling, oxidation/peroxisomal biology, stress response, and intermediary metabolism.

Most are not among the independent T-cell Top-500 expression shifts. This supports—but does not prove—that the robust set differs from the generic T-cell library signature.""")
code("display(Image(filename=str(CORE/'figures/expression_robust_vs_unstable.png')))")
code("display(Image(filename=str(CORE/'figures/attribution_concordance.png')))")
md("""## Robust programs versus interaction programs

The multilevel robust set and the Top-500 protocol×flight interaction set were enriched separately against the exact 15,165-gene universe. The separation is biologically coherent:

- **Robust core:** lipid metabolism, circadian clock, small-molecule transport, cholesterol homeostasis, long-chain fatty-acid transport, PPARA/PPARα regulation, complement, and triglyceride metabolism.
- **Protocol-sensitive interaction:** mRNA processing, alternative splicing, splice-site recognition, and processing of capped intron-containing pre-mRNA.

Thus, the robust biology is not simply the same RNA-processing signal that dominates the disagreement, although the sets need not be perfectly separable.""")
code("""enr=pd.read_csv(CORE/'programs/robust_vs_interaction_enrichment.csv')
display(enr.groupby('gene_set',group_keys=False).head(15).style.format({'p_value':'{:.2e}','fdr':'{:.2e}'}))
programs=pd.read_csv(CORE/'programs/rr1_program_classification.csv')
display(programs[programs.classification.eq('protocol_robust')].sort_values(['rank_polyA','rank_ribo']).head(25).style.format({'polyA_score':'{:.3f}','ribo_score':'{:.3f}'}))""")
md("""Fifty-eight existing pathway profiles are directionally concordant and rank in the Top-250 in both measurements. Leading examples include protein localization/translation, fatty-acid β-oxidation, peroxisomal protein import and lipid metabolism, respiratory electron transport, mitochondrial gene expression, branched-chain amino-acid degradation, and the TCA cycle. Conversely, 169 prominent terms reverse direction; the whole pathway profile itself remains weakly reproducible, so the robust program subset should not be mistaken for global pathway agreement.""")
code("display(Image(filename=str(CORE/'figures/biological_core_heatmap.png')))")
md("""## Handling and positive-control context

Several candidates are flagged as handling-sensitive because the small OSD-48 carcass and upon-euthanasia responses differ in sign or fall within the Top-500 subgroup differences. This is a warning flag, not an exclusion: the euthanasia samples lack matched OSD-168 remeasurements.

RR3-40 provides the strongest positive-control expectation, with 172 genes supported by at least two levels and 31 by all three, versus RR1's 114 and 26. RR3-39 has 104 and 15. Core size alone is not a quality metric, but RR3-40 also has markedly stronger genome-wide expression, attribution, program, and graph reproducibility.""")
md("""## What RR1 biology remains defensible?

Yes, a restricted reproducible RR1 response exists beneath the technical instability. The most defensible evidence concerns hepatic lipid/fatty-acid and peroxisomal metabolism, PPARα-associated regulation, cholesterol/triglyceride biology, transport, circadian regulation, mitochondrial respiration, and selected stress-response genes. These conclusions are supported by same-direction effects across both measurements and by multiple frozen BridgeRNA readouts.

They should be called **protocol-robust within these two RR1 measurements**, not protocol-independent or the uniquely true spaceflight response. RNA-processing/splicing biology is much more prominent among the protocol×flight interaction genes and therefore remains especially measurement-sensitive in RR1.""")
md("""## Final interpretation

1. **Technical displacement:** very large and nearly parallel in FLT and GC, indicating a strong common protocol shift.
2. **Interaction:** nevertheless large relative to that shift and maximal in the exact condition-label permutation test. The estimated flight response changes with measurement context.
3. **T-cell similarity:** strong in global embedding space, but weaker in expression and absent in the contextual graph interaction. Cross-tissue universality is not established.
4. **Residual response:** removing only the independently learned mean T-cell direction does not restore RR1; global cosine remains negative.
5. **Biology:** RNA processing/splicing dominates the largest expression-interaction genes, with additional metabolic candidates. These are programs whose measurement/representation is sensitive—not automatically technical artifacts.
6. **Handling/RIN:** handling cannot be tested with a matched OSD-168 euthanasia counterpart; available RIN results do not explain the interaction.
7. **Decision:** **mixed**—a substantial generic protocol displacement, a condition-dependent protocol×flight interaction, and deeper RR1 instability remain. The data cannot isolate PolyA/ribodepletion causally from the broader OSD-48→OSD-168 transition.

The strongest justified conclusion is: *RR1 contains a large common PolyA/ribodepletion-associated displacement, but the FLT and GC measurements are not shifted identically enough to preserve the inferred spaceflight response. Removing the independently measured mean technical direction explains only part of the reversal.*""")
nbf.write(nb,HERE/'rr1_library_prep_interaction_benchmark.ipynb')
