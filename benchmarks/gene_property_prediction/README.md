# Gene-property prediction benchmark

This benchmark is currently at the **published-protocol audit gate**. No
BridgeRNA inference, fine-tuning, or property prediction has been run.

## Audit conclusion

The two requested benchmarks are not the same kind of task.

| Benchmark | Published task | Labels recovered? | BridgeRNA overlap | Status |
|---|---|---:|---:|---|
| BulkFormer essentiality | Sample-conditioned continuous cancer-cell-line gene-dependency score prediction; reported as mean Pearson correlation | Yes; both exact pickles downloaded and checksum-validated | 14,456/17,910 target genes; 14,761/18,757 expression genes | Protocol partial |
| Geneformer dosage sensitivity | Binary dosage-sensitive versus dosage-insensitive transcription-factor token classification | Yes: exact 490-gene pickle | 245/490 (119/122 sensitive; 126/368 insensitive) | Labels and core CV recovered |
| GeneCompass dosage sensitivity | Paper says it follows Geneformer on 10,000 random cells | No distinct label file found; Geneformer labels are recoverable | 245/490 if truly identical | Protocol partial |

The limited dosage overlap is expected from BridgeRNA's one-to-one
human–mouse ortholog vocabulary, but it changes the negative class much more
than the positive class. Any evaluation must report that shift and use identical
folds for every representation.

## Exact evidence and unresolved fields

### BulkFormer gene essentiality

- The [official paper](https://doi.org/10.1016/j.cels.2026.101657) and
  [official README](https://github.com/KangBoming/BulkFormer/blob/main/README.md)
  report **mean PCC**, 0.186 for BulkFormer. This rules out silently treating it
  as a binary essential/nonessential-gene classifier.
- The [official Zenodo record](https://doi.org/10.5281/zenodo.15744294)
  distributes `gene_essentiality_expr_data.pkl` (166.6 MB; MD5
  `46414f0652f51aa0e77c08eed1b27744`) and
  `gene_essentiality_score.pkl` (159.1 MB; MD5
  `1e454ef311f288f1665f43c7b3092f8c`).
- Both files were downloaded and exactly match the published MD5 checksums.
  Inspection with NumPy 2.2.6/pandas 2.3.3 found:
  - expression: **1,108 cell lines × 18,757 Ensembl genes**, complete;
  - dependency scores: **1,108 × 17,910 genes**, 0.724% missing values;
  - identical ACH cell-line indices and ordering;
  - 17,465 genes shared between input and target matrices;
  - 14,761 input genes and 14,456 target genes map to BridgeRNA.
- The expression values range from 0 to 17.10 and resemble DepMap's
  log-transformed expression matrix, but the exact transformation/version is
  not encoded in the pickle and remains **UNKNOWN** rather than inferred.
- The [public preprint](https://doi.org/10.1101/2025.06.11.659222) describes
  final-layer contextual gene embeddings fed to an MLP, evaluated with ten-fold
  cross-validation using PCC and SCC. Its earlier data dimensions (1,103 cell
  lines and 17,862 genes) and reported scores differ from the final release, so
  those details are supporting evidence, not proof of the final folds.
- The current official repository contains model extraction code but no
  downstream essentiality implementation, split file, or label manifest.
- The publicly accessible five-page supplemental PDF contains supplementary
  figures, not the missing essentiality methods.
- Therefore the exact final fold assignments, MLP architecture, loss, missing-
  target handling, and preprocessing remain **UNKNOWN**. Crucially, this task
  evaluates sample-conditioned contextual gene tokens. A single static or
  context-averaged embedding per gene cannot reproduce it.

### Geneformer dosage sensitivity

- The [Geneformer paper](https://doi.org/10.1038/s41586-023-06139-9) defines the
  task as dosage-sensitive versus dosage-insensitive **transcription factors**,
  trained using 10,000 random single-cell transcriptomes, and reports ROC AUC
  0.91.
- The official Geneformer example
  [`examples/gene_classification.ipynb`](https://github.com/jkobject/geneformer/blob/main/examples/gene_classification.ipynb)
  links the exact label pickle and specifies `StratifiedKFold(n_splits=5,
  random_state=0, shuffle=True)`, token classification, a two-logit head,
  cross-entropy via `BertForTokenClassification`, 10,000 cells, one epoch,
  learning rate 5e-5, batch size 12, warmup 500, weight decay 0.001, and four
  frozen encoder layers for the V1 example.
- Exact labels: 122 sensitive and 368 insensitive Ensembl IDs. The source is
  [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification/dosage_sensitive_tfs).
- The paper attributes the gene-set construction to prior references, but the
  distributed pickle has no per-gene provenance or fixed fold assignment.
  Per-gene source attribution and a canonical published fold file are therefore
  **UNKNOWN**.
- Geneformer is fine-tuned and its gene representations are contextual token
  states. A frozen BridgeRNA embedding plus conventional classifier is a useful
  transfer benchmark, but is not literally the same optimization protocol.

### GeneCompass dosage sensitivity

- The [GeneCompass paper](https://doi.org/10.1038/s41422-024-01034-y) states
  that it follows Geneformer's protocol and fine-tunes on 10,000 random cells,
  reporting AUC about 0.95.
- The current [official repository](https://github.com/xCompass-AI/GeneCompass)
  does not distribute dosage-sensitivity code or a distinct label file.
- Its exact folds, optimizer settings, class balancing, and whether its labels
  are byte-identical to the Geneformer pickle are **UNKNOWN**. We will not claim
  exact GeneCompass reproduction without those artifacts.

## BridgeRNA representations available

### Static/reference embedding

The frozen checkpoint contains
`model_state_dict['gene_embedding.weight']`, shape **15,165 × 512**, float32.
This is the model-intrinsic gene identity embedding before sample-specific
Transformer contextualization. Extraction is seconds and produces about
31.1 MB (29.6 MiB). It is the defensible primary BridgeRNA representation.

### Context-averaged layer-12 embedding

No compatible study-diverse, per-gene layer-12 cache exists. Existing contextual
caches are narrow experiment-specific tensors or graph summaries and must not
be repurposed as a general reference.

A suitable expression reference already exists: 40,000 study-diverse ARCHS4
samples over the canonical vocabulary in
`benchmarks/cross_species_exercise_response/work/hallmark_readout/`. Its saved
512-D arrays are **sample** embeddings/PCA and cannot substitute for contextual
gene embeddings.

The safe implementation is streaming:

1. freeze the checkpoint and process a prespecified study-diverse subset;
2. accumulate `sum_s h[s,g,:]` directly;
3. save only the 15,165 × 512 mean plus sample/GSE manifest;
4. never save the full sample × 15,165 × 512 tensor.

At observed project inference rates (roughly 7.5–8.7 samples/s for related
forward workloads), 1,000 samples is approximately 2–10 minutes and 5,000 is
approximately 10–50 minutes on one RTX 3090 after allowing for contextual-token
transfer overhead. The final mean is 31.1 MB float32; a full float16 contextual
cache would cost **15.5 MB per sample** (15.5 GB per 1,000), so it is prohibited.
The exact rate must be measured with a small dry run before approval.

## Proposed next stage (not run)

1. Ask the BulkFormer authors/full methods for the final folds, MLP definition,
   loss, preprocessing version, and missing-target policy.
2. Implement BulkFormer matching as a distinct sample-conditioned task: map
   each DepMap cell line into BridgeRNA input space, extract contextual tokens,
   and score only the 14,456 mapped dependency targets. This requires new
   inference and is not the same as testing a fixed gene embedding.
3. Ask the BulkFormer authors/full methods if the split/readout remains absent;
   otherwise label those fields UNKNOWN and reproduce only recoverable choices.
4. Freeze deterministic Geneformer gene folds after mapping the 245 overlapping
   TFs. Use the exact same folds for static, contextual-average, expression
   statistics, and random controls.
5. Extract the static checkpoint embedding first. Compute mean/variance and
   detection fraction from the existing 40,000-sample ARCHS4 memmap.
6. Only after approval, benchmark a 1,000-sample streaming contextual average;
   expand to 5,000 only if estimates stabilize.

## Commands

Run/reproduce the audit only:

```bash
.venv/bin/python benchmarks/gene_property_prediction/pipeline/audit_published_benchmarks.py
```

Outputs:

- `results/audit/published_benchmark_audit.csv`
- `results/audit/audit_details.json`
- `data/processed/geneformer_dosage_sensitivity_labels.csv`

Placeholder evaluation scripts intentionally stop with an audit-gate message.

## Limitations

- BulkFormer's full downstream protocol is not present in its public GitHub
  repository and the article is not open access; only resource names and the
  reported continuous metric are currently verified.
- Geneformer/GeneCompass fine-tune contextual token classifiers across cells,
  whereas this benchmark ultimately aims to evaluate frozen fixed BridgeRNA
  gene representations. Results must be described as label/task matched, not
  architecture-identical.
- HGNC mapping is used only to translate the published Ensembl identifiers to
  canonical BridgeRNA symbols; ambiguous/non-ortholog genes are not rescued by
  changing the model vocabulary.
