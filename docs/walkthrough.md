# BRIDGE Walkthrough

> **Status:** Draft — to be expanded

## Overview

BRIDGE (Bulk RNA Inter-species Deep Gene Embeddings) is a transformer-based model that learns joint representations of bulk RNA-seq expression data across human and mouse. This document provides a step-by-step walkthrough for setting up the environment, preprocessing data, training the model, and evaluating results.

---

## 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/alwalt/bridge-rna.git
cd bridge-rna

# Create and activate conda environment
conda env create -f environment.yml
conda activate bridge-rna
```

---

## 2. Data Preprocessing

Run the preprocessing script to prepare raw RNA-seq data:

```bash
python preprocessing.py
```

For large datasets, shard the parquet files:

```bash
python shard_parquet.py
```

---

## 3. Training

### Single-run training

```bash
bash sweep_train_single.sh
```

### Sweep training (hyperparameter search)

```bash
bash sweep_train.sh
```

---

## 4. Evaluation

<!-- TODO: Add evaluation steps and metrics once finalized -->

---

## 5. Examples

See the `examples/` directory for sample notebooks and scripts demonstrating model usage.

---

## Notes

- Checkpoints are saved under `checkpoints_performer/`
- WandB logs are stored in `wandb/`
- Scratch work and experiments live in `scratch/`

<!-- TODO: Expand each section with detailed parameter descriptions and expected outputs -->
