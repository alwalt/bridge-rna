#!/usr/bin/env python3
"""Build the primary multiscale-response notebook from saved outputs."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}
def md(s): nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s): nb.cells.append(nbf.v4.new_code_cell(s))

md("""# Multiscale response similarity

## Scientific question

When two transcriptomic responses appear similar, **at what biological resolution are they actually similar?**

This benchmark integrates five existing, frozen views of a treatment-minus-control response: conventional expression, the global BridgeRNA sample embedding, Hallmark/program response, gene attribution, and the contextual-gene graph. It preserves every component separately and does not retrain BridgeRNA or invent missing measurements.""")
code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / 'results').exists(): ROOT = ROOT / 'benchmarks/multiscale_response_similarity'
R = ROOT / 'results'
pd.set_option('display.max_colwidth', 100)
profiles = pd.read_csv(R / 'pairwise_profiles/multiscale_pairwise_profiles.csv')
primary = pd.read_csv(R / 'summary/primary_table.csv')
inventory = pd.read_csv(R / 'summary/compatibility_inventory.csv')
provenance = json.loads((R / 'summary/provenance.json').read_text())""")
md("""## Data compatibility audit

Only pairs of treatment-control responses can populate the five-scale profile. Controlled T-cell PolyA/Ribo is one controlled technical displacement, while ARCHS4/recount3 comprises same-sample processing pairs; neither is silently converted into a response-pair comparison. Other Task 3 contrasts remain incomplete because matched Hallmark, IG, and graph-response components were not previously computed.""")
code("display(inventory)")
md("""## Metric definitions

- **Expression:** cosine between established conventional response vectors.
- **Global BridgeRNA:** cosine between frozen sample-embedding response vectors.
- **Hallmark/program:** cosine between established Hallmark module responses.
- **Attribution:** signed IG-response cosine. Exercise uses the saved signed Top-250 IG vectors and is explicitly labeled as truncated.
- **Contextual graph:** signed weighted-edge response similarity from the validated frozen layer-12, union-kNN graph (`k=10`).

Raw values are retained. Within-domain percentile ranks are used only to place unlike metric scales on common visual axes.""")
md("## Primary RR1/RR3 stress cases")
code("display(primary.style.format({c: '{:.3f}' for c in ['expression','global_bridgerna','hallmark_program','attribution','contextual_graph']}))")
code("display(Image(filename=str(R / 'figures/rr1_rr3_profiles.png')))")
md("""### What the stress cases show

- **RR3-40** is concordant at every scale and is the strongest true technical remeasurement.
- **RR3-39** is also positive at every scale, although agreement is weaker than RR3-40.
- **RR1 remeasurement** is not merely a global-geometry failure: global, program, and attribution responses reverse or disagree, while expression and graph retain only partial positive agreement. This is deeper multiscale technical sensitivity.
- **RR1↔RR3-39 false friend** has high global cosine (0.811), but weak expression (0.121), modest program/attribution agreement, and near-zero graph agreement (0.012). Global similarity alone therefore overstates molecular agreement.""")
md("## Global versus contextual-relational agreement")
code("display(Image(filename=str(R / 'figures/global_vs_graph.png')))")
md("""## Exercise response profiles

The eight established exercise contrasts contribute 28 prespecified pairwise comparisons. Within the previously frozen A/B groupings, same-pattern pairs are treated as biologically related; cross-pattern pairs are controls; comparisons involving the intermediate study remain ambiguous. These labels are inherited and were not fit here.""")
code("""exercise = profiles[profiles.domain.eq('exercise')]
summary = (exercise.groupby('relationship_type')[['expression','global_bridgerna','hallmark_program','attribution','contextual_graph']]
           .agg(['mean','median','count']))
display(summary.style.format('{:.3f}'))""")
code("display(Image(filename=str(R / 'figures/multiscale_heatmap.png')))")
md("""## Descriptive discrimination

The following AUROC/AUPRC values combine the two true RR technical remeasurements with prespecified same-pattern exercise pairs as positives. The set is small and pairwise observations share contrasts, so these are **descriptive ranking diagnostics**, not inferential performance estimates or a trained classifier.""")
code("""disc = pd.read_csv(R / 'discrimination/descriptive_discrimination.csv')
display(disc.style.format({'auroc':'{:.3f}','auprc':'{:.3f}'}))""")
md("""## Conclusions

1. True RR3 remeasurements agree across all five scales; RR3-40 is consistently strongest.
2. Global BridgeRNA similarity is highly discriminative across the prespecified exercise groupings, but it can create a specific false friend.
3. Graph and attribution/program layers diagnose *why* a high global similarity should not automatically be interpreted as molecular equivalence.
4. RR1 shows disagreement at multiple biological resolutions, not just a defect in global cosine geometry.
5. The controlled T-cell and ARCHS4/recount3 experiments remain important technical controls, but their units do not support the same five-response-vector calculation.
6. Combining all five components does not automatically improve descriptive discrimination. The value of the framework is resolution and falsification of overbroad global interpretations, not a guaranteed composite-score gain.

**Claim boundary:** multiscale agreement does not establish a shared mechanism, contextual graph edges are not regulatory networks, and none of these analyses demonstrate batch removal.""")
nbf.write(nb, HERE / 'multiscale_response_similarity_benchmark.ipynb')
