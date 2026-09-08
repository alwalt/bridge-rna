#!/usr/bin/env python3
"""Build and execute the contextual-network benchmark notebook."""
from pathlib import Path
import subprocess
import sys
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
NB = HERE / "contextual_gene_network_utility_benchmark.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md("""# BridgeRNA contextual gene-network utility

This is the scientist-facing record of a frozen layer-12 contextual-gene graph
benchmark. Edges denote contextual relationships—not causal regulation. Graph
parameters were fixed independently of RR1/RR3, and no encoder inference or
training occurs in this benchmark."""),
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = Path('results')
prov = json.loads((ROOT/'summary/provenance.json').read_text())
display(pd.Series(prov, name='value').to_frame())"""),
    md("""## 1. Known functional relationships

Positive pairs share Hallmark, GO Biological Process, or Reactome membership.
Negatives are matched on expression mean and graph degree. BridgeRNA is scored
by occurrence-weighted contextual Top-10 edge strength over a study-diverse
cohort; raw expression correlation is the conventional comparator. Non-neighbor
pairs score zero because the cache intentionally retains only Top-20 neighbors."""),
    code("""relationship = pd.read_csv(ROOT/'relationship_recovery/relationship_recovery_summary.csv')
display(relationship.style.format({'AUROC':'{:.3f}','AUPRC':'{:.3f}','top_1pct_enrichment':'{:.2f}×','top_5pct_enrichment':'{:.2f}×','top_10pct_enrichment':'{:.2f}×'}).hide(axis='index'))
display(Image(filename=str(ROOT/'figures/relationship_recovery.png')))"""),
    md("""## 2. Perturbation network rewiring

Each response is the mean treatment graph minus the mean control graph. Tables
expose gained/strengthened and lost/weakened contextual relationships, plus the
genes with greatest incident edge-weight turnover. Annotation follows—not
defines—the network calculation."""),
    code("""manifest = pd.read_csv(ROOT/'perturbation_rewiring/contrast_manifest.csv')
display(manifest)
for cohort in ['exercise','task3','tcell']:
    print(f'\\n{cohort}: representative k=10 events')
    display(pd.read_csv(ROOT/f'perturbation_rewiring/{cohort}_top_edges_k10.csv').groupby('contrast_id').head(3))"""),
    md("""## 3. Cross-study response conservation

Signed network-response cosine is calculated on the union of changed edges.
Established study groupings are retained and never selected from graph results."""),
    code("""cross_study = pd.read_csv(ROOT/'cross_study/all_pair_graph_response_similarity.csv')
display(cross_study.sort_values('weighted_edge_response_cosine', ascending=False).head(20))
display(Image(filename=str(ROOT/'figures/exercise_response_similarity.png')))"""),
    md("""## 4. Cross-species conservation

Human and mouse exercise graphs use the fixed one-to-one ortholog node order.
Every prespecified human×mouse study comparison is shown."""),
    code("""cross_species = pd.read_csv(ROOT/'cross_species/human_mouse_exercise_network_conservation.csv')
display(cross_species.sort_values('weighted_edge_response_cosine', ascending=False).style.format({'weighted_edge_response_cosine':'{:.3f}','top1000_edge_jaccard':'{:.3f}'}).hide(axis='index'))"""),
    md("""## 5. Existing attribution and DE agreement

Graph-important genes are compared with saved IG and edgeR results. Neither is
recomputed. Older deletion panels use different rankings and are not presented
as graph-guided validation."""),
    code("""overlap = pd.read_csv(ROOT/'attribution_validation/graph_gene_ig_de_overlap.csv')
display(overlap.groupby('contrast_id').agg(top_graph_genes=('gene','size'), overlap_IG=('in_existing_IG_results','sum'), overlap_DE=('in_top100_DE_any_exercise_study','sum')).reset_index())
display(pd.read_csv(ROOT/'attribution_validation/deletion_status.csv'))"""),
    md("""## 6. Decision and interpretation

The decision is derived from saved relationship-recovery results. Cross-study
and cross-species response comparisons remain exploratory given eight curated
exercise contrasts. Louvain communities are not promoted because prior
technical-pair community agreement was not discriminative."""),
    code("""decision = pd.read_csv(ROOT/'summary/decision_summary.csv')
display(decision)
display(relationship.pivot(index='source', columns='method', values=['AUROC','AUPRC']).style.format('{:.3f}'))"""),
    md("""### Scientist-facing conclusion

This benchmark asks a question raw expression and PCA do not directly expose:
**which contextual gene neighborhoods change under a perturbation, and do those
changes recur across studies or species?** Evidence is classified as strong,
task-specific, coexpression recapitulation, or mixed/insufficient. A useful
contextual relationship is not automatically a regulatory interaction, and
graph-guided functional leverage is not claimed without a dedicated deletion
experiment."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
nbf.write(nb, NB)
subprocess.run([sys.executable,"-m","nbconvert","--to","notebook","--execute","--inplace","--ExecutePreprocessor.timeout=600",str(NB)], check=True, cwd=HERE)
print(NB)
