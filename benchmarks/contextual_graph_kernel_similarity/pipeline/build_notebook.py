#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]
NB = HERE / "contextual_graph_kernel_similarity_benchmark.ipynb"

content = [
    ("m", "# Contextual graph-kernel similarity\n\nThis benchmark evaluates higher-order similarity on frozen BridgeRNA layer-12 contextual graphs. It tests graph similarity, not GRN inference. RR1/RR3 did not determine any graph or kernel parameter."),
    ("c", "from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display\nR=Path('results')\ndecision=json.loads((R/'summary/decision.json').read_text())\ndisplay(pd.Series(decision).to_frame('value'))"),
    ("m", "## Methods and tractability\n\nWL uses two subtree-refinement iterations with identity-aware and structural labels. Shortest-path uses eight fixed landmarks, graphlet uses 1,000 deterministic 3-node and 4-node samples, and spectral similarity uses 16 leading normalized weighted-adjacency eigenvalues. Higher-order tissue kernels use a prespecified study-diverse subset of 140 samples."),
    ("c", "display(pd.read_csv(R/'tissue/subset_manifest.csv').groupby('tissue').agg(samples=('gsm','size'),studies=('gse','nunique')).reset_index())\ndisplay(pd.read_csv(R/'summary/kernel_summary.csv').style.format({'R@1':'{:.1%}','R@5':'{:.1%}','R@10':'{:.1%}','MRR':'{:.3f}','runtime_seconds':'{:.1f}'}).hide(axis='index'))"),
    ("m", "## Technical-pair retrieval\n\nForty controlled same-RNA T-cell PolyA/Ribo pairs provide the independent technical-retrieval sanity check."),
    ("c", "display(pd.read_csv(R/'technical/kernel_retrieval.csv').style.format({'R@1':'{:.1%}','R@5':'{:.1%}','R@10':'{:.1%}','MRR':'{:.3f}'}).hide(axis='index'))"),
    ("m", "## Study-disjoint tissue retrieval\n\nCandidates from the query GSE are excluded. Full-cohort non-graph neighborhood purity is shown separately because it is not numerically equivalent to subset retrieval MRR."),
    ("c", "display(pd.read_csv(R/'tissue/kernel_retrieval.csv').style.format({'R@1':'{:.1%}','R@5':'{:.1%}','MRR':'{:.3f}'}).hide(axis='index'))\ndisplay(pd.read_csv(R/'tissue/non_graph_full_cohort_reference.csv'))"),
    ("m", "## Perturbation endpoint and RR1/RR3 stress test\n\nStandard kernels compare complete graphs. A treatment-minus-control graph feature difference is signed and is not itself a validated positive-semidefinite graph kernel. The benchmark does not invent one merely to fill the primary endpoint. Existing signed-edge response similarities are retained as secondary stress references."),
    ("c", "display(pd.read_csv(R/'response/status.csv'))\ndisplay(pd.read_csv(R/'stress/simple_graph_stress.csv').query(\"k == 10 and graph == 'union'\"))"),
    ("m", "## Null control\n\nThe technical feature-label permutation checks retrieval specificity. Structural-kernel invariance to node relabeling is expected; prior degree-matched random-graph results remain the relevant structural control."),
    ("c", "display(pd.read_csv(R/'nulls/technical_null.csv').style.format({'observed_MRR':'{:.3f}','label_shuffled_MRR':'{:.3f}','degradation':'{:.3f}'}).hide(axis='index'))"),
    ("m", "## Final decision\n\nThe higher-order kernels do not establish a better perturbation-response retrieval method. They may serve as secondary sample-graph similarities, but their computational cost and modest tissue retrieval do not justify replacing the validated simple graph metrics."),
    ("c", "display(pd.read_csv(R/'summary/important_output_table.csv').style.format(na_rep='—').hide(axis='index'))"),
]

cells = [nbf.v4.new_markdown_cell(x) if k == "m" else nbf.v4.new_code_cell(x) for k, x in content]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
nbf.write(nb, NB)
subprocess.run([sys.executable,"-m","nbconvert","--to","notebook","--execute","--inplace","--ExecutePreprocessor.timeout=600",str(NB)], cwd=HERE, check=True)
print(NB)
