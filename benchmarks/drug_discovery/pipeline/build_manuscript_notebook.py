#!/usr/bin/env python3
"""Build the cached-evidence scientific notebook for the approved benchmark.

This helper copies only compact, frozen aggregate evidence. It does not run
Bridge, differential expression, enrichment, or any source-data preparation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / "benchmarks" / "drug_discovery"
PACKAGE = REPO / "notebooks" / "drug_discovery"
CACHE = PACKAGE / "cached_evidence"

SOURCES = {
    "disease_gene_recovery.csv": REPO / "benchmarks/deweerd_replication/results/evaluation/disease_gene_recovery.csv",
    "locked_disease_cohorts.csv": REPO / "benchmarks/deweerd_replication/results/locked_cohorts.csv",
    "dataset_matrix.csv": RESEARCH / "results/dataset_matrix.csv",
    "partition_overall_summary.csv": RESEARCH / "results/expanded_chembl_sensitivity/partition_overall_summary.csv",
    "partition_condition_summary.csv": RESEARCH / "results/expanded_chembl_sensitivity/partition_condition_summary.csv",
    "partition_deletion_summary.csv": RESEARCH / "results/expanded_chembl_sensitivity/partition_deletion_summary.csv",
    "expanded_vs_restrictive.csv": RESEARCH / "results/expanded_chembl_sensitivity/expanded_vs_restrictive.csv",
    "radiation_stability.csv": RESEARCH / "results/radiation_context_sensitivity/bridge_vs_de_stability.csv",
    "radiation_core_genes.csv": RESEARCH / "results/radiation_context_sensitivity/radiation_core_genes.csv",
    "fibroblast_comparison.csv": RESEARCH / "results/radiation_context_sensitivity/fibroblast_comparison.csv",
    "muscle_target_summary.csv": RESEARCH / "results/muscle_target_decomposition/bridge_target_drug_dataset_summary.csv",
    "muscle_target_removal.csv": RESEARCH / "results/muscle_target_decomposition/bridge_target_removal_sensitivity.csv",
    "space_validation.json": RESEARCH / "results/expanded_chembl_sensitivity/validation_summary.json",
    "disease_validation.json": REPO / "benchmarks/deweerd_replication/results/validation_summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def copy_evidence() -> list[dict[str, object]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    records = []
    for name, source in SOURCES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination = CACHE / name
        shutil.copy2(source, destination)
        records.append(
            {
                "packaged_path": destination.relative_to(REPO).as_posix(),
                "source_path": source.relative_to(REPO).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    return records


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> None:
    cells = [
        markdown(
            """
# BridgeRNA biological prioritization and cross-study generalization

## Scientific questions

This benchmark separates two questions that require different evidence.

1. **Single-disease biological validation:** in multiple sclerosis (MS), Crohn's disease, and systemic lupus erythematosus (SLE), do frozen Bridge top-500 modules recover documented disease genes compared with equally sized differential-expression (DE) modules from the same samples?
2. **Cross-study biological generalization:** in muscle atrophy, bone loss, and radiation injury, do genes prioritized only by Bridge recur across independent terrestrial studies and transfer to spaceflight/OSDR experiments more often than genes prioritized only by DE?

The hypothesis was that Bridge captures learned, context-dependent programs complementary to strong marginal changes detected by DE. Pharmacological mapping was evaluated only after biological reproducibility and is interpreted as hypothesis generation, not therapeutic efficacy.
"""
        ),
        code(
            """
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

PACKAGE = Path.cwd()
if not (PACKAGE / 'cached_evidence').exists():
    candidates = [Path.cwd() / 'notebooks/drug_discovery', Path.cwd().parent / 'drug_discovery']
    PACKAGE = next((p.resolve() for p in candidates if (p / 'cached_evidence').exists()), PACKAGE)
CACHE = PACKAGE / 'cached_evidence'
FIGURES = PACKAGE / 'figures'
FIGURES.mkdir(exist_ok=True)

mpl.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white'
})
BRIDGE = '#2A6F97'
DE = '#D17A22'

print(f'Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | matplotlib {mpl.__version__}')
print(f'Cached evidence: {CACHE}')
"""
        ),
        markdown(
            """
## Data and frozen protocol

Task 1 used one case-control cohort per disease. Task 2 used terrestrial discovery, terrestrial replication, and OSDR validation datasets for each space-relevant condition. Expression was aligned to Bridge's canonical 15,165-gene order and transformed as natural `log1p(TPM)`. The frozen `r7hnr92k` encoder and signed Integrated Gradients pipeline produced top-500 Bridge modules; matched DE used identical samples, contrasts, pairing/covariates, expressed-gene universes, and module sizes.

ARCHS4 exposure is an interpretation variable: several terrestrial replication cohorts were seen during pretraining, whereas the OSDR cohorts and most discovery cohorts were unseen. Results involving seen studies establish reproducibility but not strict out-of-pretraining generalization.

Frozen cohort definitions are recorded in the [disease cohort table](../../benchmarks/deweerd_replication/results/locked_cohorts.csv), [space-condition dataset matrix](../../benchmarks/drug_discovery/results/dataset_matrix.csv), and [space-condition sample manifest](../../benchmarks/drug_discovery/results/manifests/sample_manifest.csv). Detailed scientific interpretation is retained in the [disease benchmark summary](../../benchmarks/deweerd_replication/results/integrated_benchmark_summary.md), [space-condition benchmark summary](../../benchmarks/drug_discovery/results/phase2_summary.md), and underlying condition-specific reports.
"""
        ),
        code(
            """
disease_cohorts = pd.read_csv(CACHE / 'locked_disease_cohorts.csv')
space_matrix = pd.read_csv(CACHE / 'dataset_matrix.csv')

display(disease_cohorts[[
    'disease_contrast', 'accession', 'tissue', 'case_samples', 'control_samples',
    'mapped_canonical_genes', 'archs4_train', 'archs4_validation', 'archs4_absent'
]].rename(columns={'disease_contrast': 'disease'}))
display(space_matrix)

assert len(disease_cohorts) == 3
assert len(space_matrix) == 9
assert set(space_matrix['condition']) == {'Radiation', 'Bone loss', 'Muscle atrophy'}
"""
        ),
        markdown(
            """
## Methods and metrics

For Task 1, exact-CUI DisGeNET v7 overlap was tested with right-sided Fisher tests and BH correction. The all-association reference and top-500 module were prespecified as primary. Because the exact filtered reference sets used in the motivating publication were unavailable, this is a documented methodological analysis rather than an exact numerical replication.

For Task 2, each top-500 pair was partitioned into **Bridge-only**, **shared**, and **DE-only** genes. Primary descriptive metrics were recurrence in another independent same-condition dataset and terrestrial-to-OSDR Jaccard overlap. External condition-gene support, pathway coherence, mask-token deletion, and ChEMBL targetability were additional evidence layers. Overall partition differences do not have a saved paired null and are therefore reported as descriptive effect sizes. Radiation also had a prespecified matched-context analysis with paired tests and matched random gene sets.

Expanded ChEMBL 37 mapping included 4,389 clinical drugs, 2,046 human targets, and 16,986 drug-target edges, with a prespecified >10-target eligibility rule. Drug overlap can identify pharmacologically connected hypotheses but cannot establish beneficial direction or efficacy.
"""
        ),
        markdown("## Results: Task 1 — single-disease biological validation"),
        code(
            """
recovery = pd.read_csv(CACHE / 'disease_gene_recovery.csv')
primary = recovery.query("module_size == 500 and reference_definition == 'all_v7_associations'").copy()
primary['disease_short'] = primary['disease'].map({
    'Multiple sclerosis — active lesion': 'MS',
    "Crohn's disease — ileum": "Crohn's",
    'Systemic lupus erythematosus — blood': 'SLE',
})
primary = primary.sort_values(['disease_short', 'method'])
assert len(primary) == 6
assert (primary['fdr'] < 0.05).all()
display(primary[['disease_short', 'method', 'known_genes_recovered', 'odds_ratio', 'pvalue', 'fdr', 'universe_genes']])

order = ['MS', "Crohn's", 'SLE']
pivot = primary.pivot(index='disease_short', columns='method', values='known_genes_recovered').loc[order]
x = np.arange(len(order)); width = 0.34
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for offset, method, color in [(-width/2, 'Bridge', BRIDGE), (width/2, 'DE', DE)]:
    vals = pivot[method].to_numpy()
    bars = ax.bar(x + offset, vals, width, label=method, color=color)
    ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xticks(x, order)
ax.set_ylabel('Documented disease genes in top 500')
ax.set_title('Disease-gene recovery depends on condition')
ax.legend(frameon=False)
ax.set_ylim(0, pivot.to_numpy().max() * 1.2)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIGURES / f'task1_disease_gene_recovery.{ext}', bbox_inches='tight')
plt.show()

pivot.reset_index().to_csv(FIGURES / 'task1_disease_gene_recovery.csv', index=False)
"""
        ),
        markdown(
            """
All six top-500 modules were significantly enriched for documented disease genes. DE recovered more documented genes in MS (87 versus 72) and Crohn's disease (79 versus 70), whereas Bridge recovered more in SLE (106 versus 72). These observed differences were not tested as direct Bridge-versus-DE significance contrasts and therefore support disease-dependent complementarity, not a universal ordering.
"""
        ),
        markdown("## Results: Task 2 — cross-study and terrestrial-to-spaceflight generalization"),
        code(
            """
overall = pd.read_csv(CACHE / 'partition_overall_summary.csv')
condition = pd.read_csv(CACHE / 'partition_condition_summary.csv')
focus = condition[condition['partition'].isin(['Bridge-only', 'DE-only'])].copy()

display(overall)
display(focus[['condition', 'partition', 'recurrence_fraction', 'osdr_jaccard', 'external_overlap', 'external_min_q', 'targetable_fraction']])

assert np.isclose(overall.set_index('partition').loc['Bridge-only', 'recurrence_fraction'], 0.435307, atol=5e-7)
assert np.isclose(overall.set_index('partition').loc['DE-only', 'osdr_jaccard'], 0.016042, atol=5e-7)

conditions = ['Overall', 'Muscle atrophy', 'Bone loss', 'Radiation']
rows = []
for partition in ['Bridge-only', 'DE-only']:
    o = overall.set_index('partition').loc[partition]
    rows.append({'condition': 'Overall', 'partition': partition,
                 'recurrence_fraction': o['recurrence_fraction'], 'osdr_jaccard': o['osdr_jaccard']})
rows.extend(focus[['condition', 'partition', 'recurrence_fraction', 'osdr_jaccard']].to_dict('records'))
plot_df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
for ax, metric, title in [
    (axes[0], 'recurrence_fraction', 'Recurrence in another condition-matched study'),
    (axes[1], 'osdr_jaccard', 'Terrestrial-to-OSDR gene overlap'),
]:
    for offset, partition, color in [(-width/2, 'Bridge-only', BRIDGE), (width/2, 'DE-only', DE)]:
        vals = (plot_df[plot_df.partition == partition].set_index('condition').loc[conditions, metric] * 100).to_numpy()
        bars = ax.bar(np.arange(len(conditions)) + offset, vals, width, label=partition, color=color)
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in vals], padding=3, fontsize=8, rotation=90)
    ax.set_xticks(np.arange(len(conditions)), ['Overall', 'Muscle', 'Bone', 'Radiation'])
    ax.set_ylabel('Percent')
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(55, ax.get_ylim()[1] * 1.15))
axes[0].legend(frameon=False, loc='upper right')
fig.suptitle('Bridge-only genes show stronger cross-study generalization', fontweight='bold')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIGURES / f'task2_cross_study_generalization.{ext}', bbox_inches='tight')
plt.show()
plot_df.to_csv(FIGURES / 'task2_cross_study_generalization.csv', index=False)
"""
        ),
        markdown(
            """
Across partitions, 43.53% of Bridge-only genes recurred in another same-condition dataset versus 11.73% of DE-only genes. Mean terrestrial-to-OSDR Jaccard was 0.15897 versus 0.01604. Muscle atrophy was the clearest individual condition (50.26% versus 19.66% recurrence; Jaccard 0.17472 versus 0.01998), and its Bridge-only module had significant external muscle-atrophy support (59 overlaps, q=5.24×10⁻⁵) whereas the DE-only partition did not pass FDR (33 overlaps, q=0.0708). Bone loss showed a similar transfer pattern and coherent extracellular-matrix/ossification biology. These aggregate partition comparisons are descriptive because a matched overall null was not saved.
"""
        ),
        markdown("### Representation deletion and targetability"),
        code(
            """
deletion = pd.read_csv(CACHE / 'partition_deletion_summary.csv')
observed = deletion.query("panel_type == 'observed'")
del_summary = observed.groupby('partition', as_index=False).agg(
    mean_absolute_change=('absolute_score_change', 'mean'),
    sd_absolute_change=('absolute_score_change', 'std'),
    n=('absolute_score_change', 'size'),
    mean_per_100=('absolute_change_per_100_genes', 'mean'),
)
display(del_summary)
display(overall[['partition', 'targetable_fraction', 'deletion_change', 'deletion_per100']])
"""
        ),
        markdown(
            """
Masking Bridge-only genes altered the frozen Bridge axis more than masking DE-only genes in the saved descriptive deletion analysis (overall absolute change 0.0635 versus 0.0246; per 100 genes 0.0143 versus 0.00549). Targetable fractions were similar (16.33% versus 14.61%), so greater Bridge-only recurrence is not explained by substantially greater representation in ChEMBL. Deletion is a representation diagnostic, not a causal intervention.
"""
        ),
        markdown("## Radiation requires context matching"),
        code(
            """
stability = pd.read_csv(CACHE / 'radiation_stability.csv')
core = pd.read_csv(CACHE / 'radiation_core_genes.csv')
fibro = pd.read_csv(CACHE / 'fibroblast_comparison.csv')

all_pairs = stability.query("context_distance == 'all'").iloc[0]
core_counts = (core.assign(core=lambda d: d['classification'].eq('majority_cross_study_core') |
                                           d['classification'].eq('strict_universal'))
                 .query('core').groupby('method')['gene'].nunique())
strict_counts = core.query("classification == 'strict_universal'").groupby('method')['gene'].nunique()
core_table = pd.DataFrame({
    'cross_study_core': core_counts.reindex(['Bridge', 'DE'], fill_value=0),
    'all_seven_modules': strict_counts.reindex(['Bridge', 'DE'], fill_value=0),
})
display(stability)
display(core_table)
display(fibro[['osdr_modality', 'method', 'overlap_count', 'jaccard', 'matched_random_empirical_p']])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
core_table.plot.bar(ax=axes[0], color=[BRIDGE, '#7BA6C2'], rot=0)
axes[0].set_ylabel('Unique genes')
axes[0].set_title('Radiation core size')
axes[0].legend(['Cross-study core', 'All seven modules'], frameon=False)
fib_pivot = fibro.pivot(index='osdr_modality', columns='method', values='jaccard').loc[['gamma','proton','iron','silicon']]
fib_pivot.plot.bar(ax=axes[1], color=[BRIDGE, DE], rot=0)
axes[1].set_xticklabels(['Gamma', 'Proton', 'Iron', 'Silicon'])
axes[1].set_ylabel('Jaccard')
axes[1].set_xlabel('Radiation modality')
axes[1].set_title('Terrestrial NHDF ↔ OSDR fibroblasts')
axes[1].legend(frameon=False)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIGURES / f'radiation_context_results.{ext}', bbox_inches='tight')
plt.show()
core_table.reset_index(names='method').to_csv(FIGURES / 'radiation_core_counts.csv', index=False)
fib_pivot.reset_index().to_csv(FIGURES / 'radiation_fibroblast_jaccard.csv', index=False)
"""
        ),
        markdown(
            """
Bridge exceeded DE in 18 of 21 matched radiation comparisons; the mean paired Bridge-minus-DE Jaccard difference was +0.1052 (Wilcoxon p=1.83×10⁻⁵). The Bridge cross-study core contained 237 genes versus 35 for DE; 15 Bridge genes and no DE genes occurred in all seven modules. In the biologically matched NHDF-to-OSDR-fibroblast analysis, Bridge Jaccard was 0.186–0.208 with matched-random empirical p=0.0005 for every modality, versus 0.017–0.022 for DE. Radiation therefore supports stable prioritization after context matching, but not necessarily stable signed response across every tissue and exposure.
"""
        ),
        markdown("## Pharmacologically connected hypotheses"),
        code(
            """
drug_sensitivity = pd.read_csv(CACHE / 'expanded_vs_restrictive.csv')
targets = pd.read_csv(CACHE / 'muscle_target_summary.csv')
removal = pd.read_csv(CACHE / 'muscle_target_removal.csv')
display(drug_sensitivity)
display(targets.head(7))
display(removal)
"""
        ),
        markdown(
            """
In muscle, recurrent target mapping connected ERBB2 to afatinib, dacomitinib, and neratinib, and PDE5A/PDE10A to ibudilast. These are candidates for condition-relevant experiments, not established treatments. The restrictive drug graph produced Bridge versus DE convergence Jaccard of 0.714 versus 0.182, but expanded ChEMBL produced 0.164 versus 0.0068. Removing ERBB2 lowered the restrictive Bridge estimate to 0.500, and removing PDE5A lowered it to 0.619; individual recurrent drug enrichments did not pass FDR. The target universe and connected hub genes therefore materially affect compound-level convergence even when the qualitative Bridge-greater-than-DE ordering remains.
"""
        ),
        markdown(
            """
## Negative results and scope boundary

Bridge did not uniformly exceed DE. DE had stronger documented disease-gene recovery in MS and Crohn's disease, while Bridge was stronger in SLE. The expanded-ChEMBL muscle effect was much smaller than the restrictive estimate, and the exact method-label permutation test was not significant. No drug candidate is validated by this benchmark.

Exploratory generic perturbation checks using LINCS/L1000 and GSE264130 were conducted elsewhere in the research record, but their assay and cell contexts were not sufficiently matched to the six conditions for manuscript-level inference. They are excluded from this notebook's evidence base and do not alter the frozen biological-prioritization results.
"""
        ),
        markdown(
            """
## Limitations

- Only three independent studies were available per space-relevant condition, and tissues, exposures, and model systems were heterogeneous.
- Some terrestrial cohorts were represented in ARCHS4 pretraining. GSE184119 and GSE113165 are reproducibility cohorts, not strict held-out generalization tests; GSE211204 had partial exposure. The OSDR cohorts were held out.
- Exact DisGeNET reference filtering from the motivating study and an authorized DrugBank export were unavailable; documented DisGeNET v7 and ChEMBL 37 universes were used instead.
- Overall Bridge-only versus DE-only recurrence differences are descriptive and lack a saved paired null. Condition-specific external enrichment and radiation matched-random tests provide stronger inferential support.
- ChEMBL connectivity is influenced by target annotation density and hub genes and does not determine beneficial direction, dose, safety, or efficacy.
- Radiation prioritization is context dependent; recurrence of rank does not imply consistent direction in every tissue or radiation modality.
"""
        ),
        markdown(
            """
## Conclusion

Bridge recovered recognizable disease biology and, most importantly, identified complementary gene programs that transferred across independent terrestrial and spaceflight studies more strongly than genes prioritized only by conventional DE. Evidence was strongest for muscle atrophy and bone loss and became clear for radiation when biological context was matched. Portions of these reproducible programs map to pharmacologically actionable targets, but the drug lists are resource- and hub-sensitive hypotheses. The benchmark supports Bridge as a complement to DE for discovering reproducible space-biology programs; it does not establish robust therapeutic countermeasures.
"""
        ),
        code(
            """
# Final integrity checks for the displayed evidence.
assert set(primary['known_genes_recovered']) == {72, 87, 70, 79, 106}
assert int(all_pairs['bridge_higher_pairs']) == 18 and int(all_pairs['pairs']) == 21
assert core_table.loc['Bridge', 'cross_study_core'] == 237
assert core_table.loc['DE', 'cross_study_core'] == 35
assert core_table.loc['Bridge', 'all_seven_modules'] == 15
assert core_table.loc['DE', 'all_seven_modules'] == 0
assert np.allclose(fibro.query("method == 'Bridge'")['matched_random_empirical_p'], 1 / 2001, atol=1e-9)
print('All notebook evidence checks passed.')
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
    )
    nbf.write(notebook, PACKAGE / "drug_discovery.ipynb")


def write_readme() -> None:
    content = """# BridgeRNA biological prioritization benchmark

This directory contains an executed scientific notebook that reads only compact cached aggregate evidence. It does not require raw expression, model checkpoints, embeddings, or network access.

## Contents

- `drug_discovery.ipynb`: executed report for the approved two-task benchmark.
- `cached_evidence/`: frozen compact source tables, hashes, provenance, dependencies, and validation log.
- `figures/`: PNG/PDF figures and the exact CSV data plotted by the notebook.

The research source of truth remains `benchmarks/drug_discovery/` and `benchmarks/deweerd_replication/`. Existing detailed reports and original research notebooks are preserved there.

## Lightweight execution

From the repository root:

```bash
python3 benchmarks/drug_discovery/pipeline/build_manuscript_notebook.py
(cd notebooks/drug_discovery && jupyter nbconvert --to notebook --execute drug_discovery.ipynb \\
  --output drug_discovery.ipynb --ExecutePreprocessor.timeout=300)
python3 benchmarks/drug_discovery/pipeline/validate_manuscript_notebook.py
```

Tested report dependencies are recorded in `cached_evidence/dependencies.json`. A minimal environment can be installed with:

```bash
python3 -m pip install pandas numpy matplotlib nbformat nbconvert jupyter ipykernel
```

## Full benchmark commands (not run by this report)

The expensive source analyses require the Bridge checkpoint `model/r7hnr92k/best_model.pt`, canonical gene assets, downloaded GEO/OSDR expression sources, ARCHS4 split metadata, R with edgeR/limma, and the frozen ChEMBL 37 archive described by `benchmarks/drug_discovery/references/drug_target_resource.json`.

The exact source-analysis entrypoints, from the repository root, are:

```bash
python3 benchmarks/drug_discovery/pipeline/prepare_expression.py
python3 benchmarks/drug_discovery/pipeline/run_bridge.py --device cuda:0 --ig-steps 16
Rscript benchmarks/drug_discovery/pipeline/run_differential_expression.R
python3 benchmarks/drug_discovery/pipeline/prepare_chembl.py
python3 benchmarks/drug_discovery/pipeline/enrich_and_evaluate.py
python3 benchmarks/drug_discovery/pipeline/run_expanded_chembl_sensitivity.py
python3 benchmarks/drug_discovery/pipeline/run_partition_deletion.py
python3 benchmarks/drug_discovery/pipeline/run_partition_pathways.py
python3 benchmarks/drug_discovery/pipeline/run_radiation_context_sensitivity.py

python3 benchmarks/deweerd_replication/pipeline/lock_and_prepare_cohorts.py
python3 benchmarks/deweerd_replication/pipeline/run_bridge_attribution.py --accession GSE138614 --device cuda:0 --ig-steps 16
python3 benchmarks/deweerd_replication/pipeline/run_bridge_attribution.py --accession GSE101794 --device cuda:0 --ig-steps 16
python3 benchmarks/deweerd_replication/pipeline/run_bridge_attribution.py --accession GSE72509 --device cuda:0 --ig-steps 16
Rscript benchmarks/deweerd_replication/pipeline/run_differential_expression.R
python3 benchmarks/deweerd_replication/pipeline/evaluate_recovery.py
```

The space-condition pipeline is documented in `benchmarks/drug_discovery/README.md`; the disease pipeline is documented under `benchmarks/deweerd_replication/`. These commands can require GPU inference, substantial local source data, R/Bioconductor dependencies, and hours of compute. Monitor their run logs and free storage before execution; do not run them merely to refresh this notebook.

## Scope

The notebook reports Task 1 single-disease biological validation and Task 2 cross-study terrestrial-to-OSDR generalization. Exploratory LINCS/L1000 and GSE264130 directionality checks are outside its evidence base. ChEMBL mappings are interpreted as experiment-generating pharmacological connections, not therapeutic efficacy.
"""
    (PACKAGE / "README.md").write_text(content)


def write_provenance(records: list[dict[str, object]]) -> None:
    import matplotlib
    import nbconvert
    import numpy
    import pandas

    provenance = {
        "report": "BridgeRNA biological prioritization and cross-study generalization",
        "source_git_commit": git("rev-parse", "HEAD"),
        "source_branch": git("branch", "--show-current"),
        "approval_scope": {
            "task_1": "MS, Crohn's disease, and SLE single-cohort Bridge-versus-DE disease-gene recovery",
            "task_2": "muscle atrophy, bone loss, and radiation cross-study terrestrial-to-OSDR generalization",
            "pharmacology": "ChEMBL-connected experimental hypotheses only",
            "excluded": "LINCS/L1000 and GSE264130 exploratory directionality analyses",
        },
        "frozen_model": {
            "checkpoint": "model/r7hnr92k/best_model.pt",
            "canonical_genes": 15165,
            "input_transform": "natural log1p(TPM)",
            "attribution": "signed Integrated Gradients; all-zero baseline; 16 midpoint-Riemann steps",
            "primary_module_size": 500,
        },
        "cohort_references": [
            "benchmarks/deweerd_replication/results/locked_cohorts.csv",
            "benchmarks/drug_discovery/results/dataset_matrix.csv",
            "benchmarks/drug_discovery/results/manifests/sample_manifest.csv",
        ],
        "metric_definitions": {
            "recurrence_fraction": "fraction of partition genes recurring in another independent same-condition dataset",
            "osdr_jaccard": "mean Jaccard overlap between terrestrial and OSDR gene partitions",
            "disease_gene_recovery": "top-500 module overlap with exact-CUI DisGeNET v7 genes; right-sided Fisher test with BH correction",
            "radiation_stability": "paired top-500 Jaccard comparisons with Wilcoxon signed-rank test and matched random-gene nulls",
        },
        "random_seeds": {"radiation_matched_random": "recorded in source provenance; 2,000 replicates"},
        "verification": {
            "level": "aggregate-table recalculation and invariant checking; prediction-level arrays not packaged",
            "tolerance": "5e-7 for selected stored summary values",
            "status": "pending notebook execution and package validation",
        },
        "dependencies": {
            "python": subprocess.check_output(["python3", "--version"], text=True).strip(),
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__,
            "nbformat": nbf.__version__,
            "nbconvert": nbconvert.__version__,
        },
        "limitations": [
            "Three independent studies per space condition.",
            "Heterogeneous tissues, exposures, and model systems.",
            "ARCHS4 exposure in some terrestrial cohorts.",
            "Documented DisGeNET v7 differs from unavailable motivating-study filters.",
            "ChEMBL used because no authorized DrugBank export was available.",
            "Overall partition effects are descriptive without a saved paired null.",
            "Target connectivity does not establish beneficial direction or efficacy.",
        ],
        "artifacts": records,
    }
    (CACHE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (CACHE / "dependencies.json").write_text(json.dumps(provenance["dependencies"], indent=2) + "\n")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "figures").mkdir(exist_ok=True)
    records = copy_evidence()
    build_notebook()
    write_readme()
    write_provenance(records)
    print(f"Built {PACKAGE.relative_to(REPO)} from {len(records)} compact source artifacts.")


if __name__ == "__main__":
    main()
