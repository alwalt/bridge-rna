# Task 4 — Library-prep disentanglement

This standalone benchmark asks whether library-associated variation can be
separated from biological variation in **frozen** BridgeRNA sample embeddings.
It was motivated by Task 3, but it neither modifies nor trains on Task 3.

Start with [`library_prep_disentanglement_benchmark.ipynb`](library_prep_disentanglement_benchmark.ipynb).
Pipeline scripts are the reproducible source of truth; the notebook only reads
saved results.

## Controlled evidence and current scope

The ARCHS4 audit is conservative. `molecule_ch1 = total RNA` is not called
rRNA-depleted unless another metadata field explicitly describes rRNA removal.
ARCHS4 is classified **OBSERVATIONAL** because it lacks authoritative same-RNA
pair identifiers, so it is not used to supervise the primary model.

Controlled resources verified from their deposited metadata:

| Dataset | Design | Role |
|---|---|---|
| Chen et al. 2020, DOI 10.1038/s41597-020-00719-4 | 40 donors; the same naïve CD4 T-cell RNA processed by PolyA selection and Ribo-Zero | Train |
| Zhao et al. 2018, SRP127360 | pooled blood and colon source RNA; four technical libraries per protocol | Completely held-out test |

GSE150097 is retained only as a validation *candidate*. Its public metadata
contains both protocols but does not provide a defensible cross-protocol
same-RNA mapping for every sample. It is not silently promoted to paired data.
Consequently, the exploratory run uses a fixed, predeclared epoch count and no
validation-driven model selection. This is more conservative than splitting a
single study and claiming study-disjoint validation, but the external test has
only two biological source RNAs and cannot support definitive generalization.

## Reproduce

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/audit_archs4_library_prep.py

Rscript benchmarks/library_prep_disentanglement/pipeline/download_srp127360_recount3.R

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/prepare_controlled_data.py \
  --include-srp127360 --device cuda:0 --batch-size 4

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/characterize_bridge.py \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/chen_2020_tcells \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/zhao_2018_srp127360

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/run_task4.py \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/chen_2020_tcells \
  --dataset benchmarks/library_prep_disentanglement/work/datasets/zhao_2018_srp127360 \
  --device cuda:0 2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_disentanglement/run.log

.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/evaluate_task3_challenge.py \
  --device cuda:0 2>&1 | tee benchmarks/library_prep_disentanglement/results/task4g_task3_challenge/run.log
```

## Data and output policy

- Official downloads, TPM matrices, and embeddings live under ignored `work/`.
- Compact audit tables, metrics, logs, figures, and provenance live in `results/`.
- Input is natural `log1p(TPM)` in the canonical 15,165-gene order.
- No NASA/OSDR sample is used for training, model selection, or tuning.
- The held-out test is reported as exploratory because its biological N is two.
- Use “library-associated” unless controlled evidence supports a causal claim.

The neural decomposition is compared with original Bridge, linear removal,
no-pair-loss, no-adversarial-loss, shuffled-label, and shuffled-pair controls.
Success requires held-out library suppression in FE, library retention in RE,
improved same-RNA retrieval, and preservation of Task 3 RR3 controls—not merely
changing the sign of the RR1 cosine.

## Controlled-subspace follow-up

The follow-up diagnostic tests whether the RR1 protocol transition aligns with
the controlled T-cell PolyA→Ribo displacement before attempting another neural
correction. It uses cached frozen embeddings and does not retrain anything:

```bash
.venv/bin/python benchmarks/library_prep_disentanglement/pipeline/analyze_controlled_library_subspace.py \
  2>&1 | tee benchmarks/library_prep_disentanglement/results/task4_followup_controlled_subspace_run.log
```

Compact outputs and figures are under
`results/task4_followup_controlled_subspace/`. The SVD removal is explicitly a
diagnostic; it is not presented as a production correction.
