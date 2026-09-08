#!/usr/bin/env python3
"""Build the human-readable RR1/RR3 case-study notebook from saved results."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# RR1/RR3 contextual-graph case study

## Why this case study exists

This notebook asks whether retaining **gene-resolved contextual organization** can distinguish true technical remeasurements from a misleading similarity created by globally averaging all 15,165 gene tokens.

The method was fixed independently of RR1/RR3: frozen layer-12 contextual embeddings, directed cosine kNN per gene, `k=10`, then union symmetrization. No BridgeRNA inference, graph tuning, or model training occurs here. RR1/RR3 are held-out stress cases.""")

code("""from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'results').exists():
    ROOT = ROOT / 'benchmarks/rr1_rr3_contextual_graph_case_study'
RESULTS = ROOT / 'results'
pd.set_option('display.max_colwidth', 100)
summary = json.loads((RESULTS / 'case_study_summary.json').read_text())""")

md("""## Cohort and comparison definitions

- **RR1:** OSD-48 RR1-NASA 37-day response versus its OSD-168 technical remeasurement.
- **RR3-39:** OSD-137 RR3 39-day response versus its OSD-168 technical remeasurement.
- **RR3-40:** OSD-137 RR3 40-day response versus its OSD-168 technical remeasurement.
- **False friend:** RR1 original response versus the biologically unrelated RR3-39 original response.

Every response is `mean(FLT) - mean(GC)`. Technical remeasurements are repeated material, not independent biological replication.""")
code("""manifest = pd.read_csv(RESULTS / 'sample_manifest.csv')
cohort = (manifest.groupby(['measurement', 'contrast_id', 'OSD', 'condition'])
          .agg(n=('sample_id', 'size'), sample_ids=('sample_id', lambda x: ', '.join(x)))
          .reset_index())
display(cohort)""")

md("""## The pooling failure and the graph result

Global mean pooling collapses 15,165 contextual gene embeddings into a single 512-D vector. It can therefore make two responses similar while discarding *which genes have which contextual neighborhoods*.

The graph readout keeps every canonical gene as a fixed node. For each sample, an edge connects a gene to its most similar contextual-gene neighbors. Study responses are signed changes in these weighted edges (`FLT - GC`). We compare true replicate response graphs with the unrelated false-friend pair.

Because vector cosine and graph similarity have different scales, the valid comparison is **ordering/discrimination within each representation**, not raw score magnitude across representations.""")
code("""scores = pd.read_csv(RESULTS / 'global_mean_vs_graph.csv')
wide = scores.pivot(index='comparison', columns='representation', values='similarity')
display(wide.style.format('{:.3f}'))
display(Image(filename=str(RESULTS / 'figures/global_mean_vs_graph.png')))""")

md("""### Main result

With global BridgeRNA mean pooling, the unrelated RR1↔RR3-39 pair scores **0.811**, slightly higher than the true RR3-39 remeasurement (**0.790**). This is the global-embedding false friend.

With the prespecified contextual graph, RR3-39 (**0.319**) and RR3-40 (**0.399**) both rank well above the false friend (**0.012**). The graph therefore recovers the correct relationship ordering. RR1 remains poorly reproducible (**0.183**) and is not presented as rescued.

This is evidence that gene-resolved contextual organization contains useful structure lost by a single global mean. It is not evidence that graph edges are causal or regulatory interactions.""")

md("""## Which gene neighborhoods distinguish true replication?

For each fixed gene, we compare its signed neighborhood-change profile between the two RR3 measurements. The diagnostic score is the mean true-replicate local cosine minus the RR1↔RR3-39 false-friend local cosine. This localization is descriptive; it did not define or tune the graph.""")
code("""genes = pd.read_csv(RESULTS / 'top50_discriminative_gene_neighborhoods.csv')
display(genes.head(20).style.format({c: '{:.3f}' for c in genes.columns if 'cosine' in c or 'discrimination' in c}))
display(Image(filename=str(RESULTS / 'figures/top_gene_neighborhoods.png')))""")

md("""## Biological annotation of the discriminative neighborhoods

The Top-250 diagnostic genes were annotated after ranking, using the exact 15,165-gene vocabulary as the enrichment background. Enrichment is a hypothesis-generating annotation of the genes whose neighborhoods discriminate the cases; it does not turn graph edges into regulatory claims.""")
code("""enrichment = pd.read_csv(RESULTS / 'top250_neighborhood_enrichment.csv')
display(enrichment.head(15).style.format({'p_value': '{:.2e}', 'fdr': '{:.3g}'}))""")

md("""## What BridgeRNA contributes—and what it does not

**Showcase result.** A single global embedding produces a false-friend ordering, whereas the frozen contextual graph distinguishes both true RR3 technical remeasurements from that unrelated pair. Gene-level localization highlights neighborhoods involving lipogenic/cholesterol, bile-acid/fatty-acid, nuclear-receptor, and translation-related programs.

**Interpretation.** BridgeRNA's contextual tokens preserve information about *how genes are organized relative to one another* that global averaging can erase. This offers a scientist-facing route from a sample-level similarity anomaly to specific gene neighborhoods and testable biological programs.

**Limits.** Raw expression, PCA, and Hallmark module pooling also avoid this particular false ordering, so this is not a claim that graphs uniquely outperform conventional representations. RR1 remains a cross-protocol reproducibility failure. The cohort is small, graph scores are representation-specific, and contextual proximity does not establish physical interaction, regulation, or causality.""")

nbf.write(nb, HERE / 'rr1_rr3_contextual_graph_case_study.ipynb')
