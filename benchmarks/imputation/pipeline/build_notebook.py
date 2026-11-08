#!/usr/bin/env python3
"""Build and execute the report from saved results, without model inference."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md('''# TCGA expression imputation

Do frozen BridgeRNA representations reconstruct masked expression more accurately than BulkFormer and a training-derived gene-expression baseline?

This is a fresh inference run on the previously frozen cohort. Execution of this notebook reads saved results only; the pipeline performs inference.'''),
code('''from pathlib import Path
import json
import pandas as pd
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if not (ROOT / 'config.json').exists():
    ROOT = ROOT / 'benchmarks' / 'imputation'
RESULTS = ROOT / 'results'
config = json.loads((ROOT / 'config.json').read_text())
summary_path = RESULTS / 'summary_results.csv'
print('Run complete' if summary_path.exists() else 'Fresh inference is pending or running; final scores are not available yet.')'''),
md('''## Data and methods

- The frozen cohort contains 1,000 randomly selected TCGA primary-tumor samples. Cancer types are not balanced. The saved cohort manifest defines sample identity and order.
- BridgeRNA uses 15,165 canonical genes, of which 15,105 have measured TCGA expression. BulkFormer uses 20,010 genes, of which 18,949 are measured. HGNC mapping gives 15,101 unambiguous shared targets.
- Inputs use each checkpoint's intended natural `log1p(TPM)` preprocessing. Gene lengths and TPM denominators differ between model target spaces; cross-model MSE therefore does not measure error against identical numerical targets.
- This analysis masks each model's own measured genes. Samples, masking ratios, and seeds are matched, but evaluated gene sets and exact masks differ between vocabularies. This is not a matched-target comparison.
- Mask ratios are 15%, 30%, 50%, 70%, 90%, and 100%, with the same 3 mask seeds for every model. All backbones and checkpoints remain frozen; a seed repeats masking, not training.
- Training-gene mean and median are frozen profiles derived from 5,000 ARCHS4 human training samples, without using TCGA evaluation expression. Mean and median in both target spaces are retained numerically; figures display only the training mean for BRIDGE genes to avoid overlapping baseline curves.
- Pearson, Spearman, and MSE are calculated per sample on masked positions only, averaged across samples within each seed, then summarized across seeds. Error bars show mask-seed SD, not confidence intervals across patients. Undefined correlations are counted explicitly and are not replaced with zero.'''),
code('''display(pd.read_csv(RESULTS / 'cohort_manifest.csv').groupby('cancer_type').size().rename('samples').to_frame())
if (RESULTS / 'run_provenance.json').exists():
    display(json.loads((RESULTS / 'run_provenance.json').read_text()))'''),
md('''## Native-gene reconstruction

Models are evaluated on their own measured native gene panels. Colors identify methods. Training-gene mean is calculated from ARCHS4 training samples and scored on the BRIDGE gene panel with its preprocessing; it is not a BRIDGE model prediction. Model error bars and the thin shaded baseline band show SD across three mask seeds. At 100% masking, no measured expression is available. Different evaluated gene sets limit direct cross-model comparisons.'''),
code('''if summary_path.exists():
    import sys
    sys.path.insert(0, str(ROOT / 'pipeline'))
    from plot_figures import render, save_style_provenance
    summary = pd.read_csv(summary_path)
    fig = render(summary, ['pearson', 'spearman'], RESULTS, 'native_imputation_correlations')
    plt.show()
    save_style_provenance(RESULTS)'''),
md('''## Reconstruction error

MSE is shown separately because models use different gene panels, gene lengths, and TPM denominators. Values describe error within each native target space; they are not a comparison against identical numerical targets. Error bars show SD across three mask seeds.'''),
code('''if summary_path.exists():
    fig = render(summary, ['mse'], RESULTS, 'native_imputation_mse')
    plt.show()
    display(summary.round(6))'''),
md('''## Quality control and cancer-specific performance

Correlation can be undefined when predictions are constant. The QC table reports how many of the 1,000 samples have defined correlations for each condition. Per-cancer tables and the unweighted cancer-type mean provide a secondary view without changing the frozen cohort.'''),
code('''if summary_path.exists():
    qc = pd.read_csv(RESULTS / 'correlation_qc.csv')
    display(qc[(qc.pearson_defined < qc.samples) | (qc.spearman_defined < qc.samples)])
    macro = pd.read_csv(RESULTS / 'cancer_macro_per_seed.csv')
    display(macro.groupby(['benchmark', 'method', 'target_space', 'mask_ratio'])[['pearson', 'spearman', 'mse']].mean().round(6))'''),
md('''## Interpretation and limitations

Interpretation must follow the fresh results shown above. Higher correlation under this protocol supports more accurate reconstruction of masked expression patterns, rather than a claim of universal model superiority. Each model is evaluated on a different gene panel; score differences cannot be attributed to model quality alone. Cross-model MSE remains sensitive to model-specific preprocessing. Random gene masking does not reproduce every experimental missingness mechanism, and TCGA sample or study exposure during model pretraining has not been established here. Cancer-specific summaries do not make the cohort balanced.

## Reproduce the analysis

See `README.md` for preparation, pilot, inference, aggregation, and report commands. Expensive inference is performed only by the pipeline, never automatically by this notebook.''')])
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
NotebookClient(nb,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(nb,ROOT/'imputation.ipynb')
print(ROOT/'imputation.ipynb')
