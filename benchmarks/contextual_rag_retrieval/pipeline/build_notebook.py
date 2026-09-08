#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
NB = HERE / "contextual_rag_retrieval_benchmark.ipynb"

sections = [
    ("markdown", """# Contextual RAG retrieval benchmark

This notebook tests whether cached BridgeRNA program and contextual-gene
features improve scientist-facing sample and response retrieval. Retrieved
transcriptomic analogues do not establish a shared mechanism. BridgeRNA remains
frozen and no expensive representation is regenerated."""),
    ("code", """from pathlib import Path
import json, pandas as pd
from IPython.display import display, Image
R = Path('results')
decision = json.loads((R/'summary/decision.json').read_text())
display(pd.Series(decision).to_frame('value'))"""),
    ("markdown", """## 1. Study-excluded sample retrieval

Every ARCHS4 query excludes candidates from its own GSE. A hit means the same
curated tissue is found within Top-k. Contextual *response* fingerprints are not
manufactured for individual samples lacking a treatment/control reference."""),
    ("code", """sample = pd.read_csv(R/'sample_retrieval/sample_retrieval_summary.csv')
display(sample.style.format({c:'{:.1%}' for c in ['R@1','R@5','R@10','MRR']}).hide(axis='index'))"""),
    ("markdown", """## 2. Exercise-response retrieval

The contextual fingerprint is a 15,165-gene neighborhood-turnover vector, not
a giant token concatenation. Previous Axis A/B labels are not independent
ground truth because BridgeRNA contributed to their discovery; results here are
diagnostic rather than confirmatory."""),
    ("code", """response = pd.read_csv(R/'response_retrieval/exercise_response_retrieval_summary.csv')
display(response.style.format({c:'{:.1%}' for c in ['R@1','R@5','R@10','MRR']}).hide(axis='index'))
display(pd.read_csv(R/'reranking/candidate_depth_audit.csv'))
display(Image(filename=str(R/'figures/retrieval_summary.png')))"""),
    ("markdown", """## 3. Explainable contextual reranking

The transparent composite retains Hallmark and contextual similarities as
separate fields. Pathway or IG explanations are not generated unless directly
supported by saved evidence."""),
    ("code", """display(pd.read_csv(R/'reranking/transparent_contextual_reranking_top5.csv').head(25))"""),
    ("markdown", """## 4. RR1/RR3 false-friend safety

Genuine RR3 technical replications should rank above the known RR1↔RR3-39 false
friend. Graph parameters were frozen before these cases were evaluated."""),
    ("code", """display(pd.read_csv(R/'false_friend/false_friend_safety.csv'))
display(pd.read_csv(R/'false_friend/contextual_graph_stress.csv'))"""),
    ("markdown", """## 5. ARCHS4/recount3 and cross-species references

The established recount3 and balanced human↔mouse results are reused unchanged.
Contextual recount3 reranking is explicitly unavailable because those tokens
were not cached."""),
    ("code", """display(pd.read_csv(R/'technical_pairs/paired_recount3_existing_summary.csv'))
display(pd.read_csv(R/'technical_pairs/contextual_reranking_status.csv'))
display(pd.read_csv(R/'cross_species/existing_balanced_tissue_retrieval.csv').head(30))"""),
    ("markdown", """## 6. Decision

Candidate K=25/50/100 is not a distinct test because the curated response corpus
contains fewer than 25 eligible independent candidates. A contextual RAG layer
is supported only if reranking improves over PCA/global baselines while reducing
the false friend and preserving true RR3 matches."""),
    ("code", """display(pd.read_csv(R/'summary/decision.csv').T)"""),
]

cells = [nbf.v4.new_markdown_cell(x) if kind == "markdown" else nbf.v4.new_code_cell(x) for kind, x in sections]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
nbf.write(nb, NB)
subprocess.run([sys.executable,"-m","nbconvert","--to","notebook","--execute","--inplace","--ExecutePreprocessor.timeout=600",str(NB)], cwd=HERE, check=True)
print(NB)
