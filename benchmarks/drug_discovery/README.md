# BridgeRNA biological prioritization benchmark

This directory contains an executed scientific notebook that reads only compact cached aggregate evidence. It does not require raw expression, model checkpoints, embeddings, or network access.

## Contents

- `drug_discovery.ipynb`: executed report for the approved two-task benchmark.
- `data/`: frozen compact source tables, hashes, provenance, dependencies, and validation log.
- `figures/`: PNG/PDF figures and the exact CSV data plotted by the notebook.

The research source of truth remains `research/drug_discovery/` and `research/deweerd_replication/`. Existing detailed reports and original research notebooks are preserved there.

## Lightweight execution

From the repository root:

```bash
python3 research/drug_discovery/pipeline/build_manuscript_notebook.py
(cd benchmarks/drug_discovery && jupyter nbconvert --to notebook --execute drug_discovery.ipynb \
  --output drug_discovery.ipynb --ExecutePreprocessor.timeout=300)
python3 research/drug_discovery/pipeline/validate_manuscript_notebook.py
```

Tested report dependencies are recorded in `data/dependencies.json`. A minimal environment can be installed with:

```bash
python3 -m pip install pandas numpy matplotlib nbformat nbconvert jupyter ipykernel
```

## Full benchmark commands (not run by this report)

The expensive source analyses require the Bridge checkpoint `model/r7hnr92k/best_model.pt`, canonical gene assets, downloaded GEO/OSDR expression sources, ARCHS4 split metadata, R with edgeR/limma, and the frozen ChEMBL 37 archive described by `research/drug_discovery/references/drug_target_resource.json`.

The exact source-analysis entrypoints, from the repository root, are:

```bash
python3 research/drug_discovery/pipeline/prepare_expression.py
python3 research/drug_discovery/pipeline/run_bridge.py --device cuda:0 --ig-steps 16
Rscript research/drug_discovery/pipeline/run_differential_expression.R
python3 research/drug_discovery/pipeline/prepare_chembl.py
python3 research/drug_discovery/pipeline/enrich_and_evaluate.py
python3 research/drug_discovery/pipeline/run_expanded_chembl_sensitivity.py
python3 research/drug_discovery/pipeline/run_partition_deletion.py
python3 research/drug_discovery/pipeline/run_partition_pathways.py
python3 research/drug_discovery/pipeline/run_radiation_context_sensitivity.py

python3 research/deweerd_replication/pipeline/lock_and_prepare_cohorts.py
python3 research/deweerd_replication/pipeline/run_bridge_attribution.py --accession GSE138614 --device cuda:0 --ig-steps 16
python3 research/deweerd_replication/pipeline/run_bridge_attribution.py --accession GSE101794 --device cuda:0 --ig-steps 16
python3 research/deweerd_replication/pipeline/run_bridge_attribution.py --accession GSE72509 --device cuda:0 --ig-steps 16
Rscript research/deweerd_replication/pipeline/run_differential_expression.R
python3 research/deweerd_replication/pipeline/evaluate_recovery.py
```

The space-condition pipeline is documented in `research/drug_discovery/README.md`; the disease pipeline is documented under `research/deweerd_replication/`. These commands can require GPU inference, substantial local source data, R/Bioconductor dependencies, and hours of compute. Monitor their run logs and free storage before execution; do not run them merely to refresh this notebook.

## Scope

The notebook reports Task 1 single-disease biological validation and Task 2 cross-study terrestrial-to-OSDR generalization. Exploratory LINCS/L1000 and GSE264130 directionality checks are outside its evidence base. ChEMBL mappings are interpreted as experiment-generating pharmacological connections, not therapeutic efficacy.

## Directory layout

- `data/`: cached evidence, provenance, validation records, and dependency files.
- `figures/`: report figures and associated plot tables.
- `research_archive/`: local, Git-ignored historical research copies; not required to execute this notebook.
- `drug_discovery.ipynb`: scientific report.

Run the notebook from this directory with its documented Python dependencies. Historical source paths in provenance describe the original runs and are retained unchanged. Research scripts now live under the local `research/` directory; archive copies preserve their original contents.
