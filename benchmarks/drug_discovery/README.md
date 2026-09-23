# BridgeRNA drug-discovery benchmark: Phase 1 audit and frozen matrix

Audit/freeze date: 2026-09-20. This document is the pre-execution design record for the
radiation, bone-loss, and skeletal-muscle-atrophy benchmark. No embeddings,
attributions, differential-expression tests, drug enrichment, or drug rankings
were run in Phase 1.

## Decision summary

The benchmark is feasible and its nine dataset slots are now frozen. The two
issues found by the initial audit were resolved without running the benchmark:

1. **OSD-867/GSE273868 replaces OSD-546/GSE100930.** OSD-867 is bulk poly(A)
   RNA-seq from human bone-marrow mesenchymal stem/stromal cells cultured on the
   ISS or in synchronous ground controls for one or two weeks. Each time stratum
   has three flight and three ground samples. Its STAR count matrix maps 15,096
   of 15,165 Bridge genes (99.5%). It is a relevant bone-lineage MSC spaceflight
   model, but not differentiated bone tissue and not a direct clinical bone-loss
   measurement; that limitation must accompany every cross-domain conclusion.
2. **ChEMBL 37 replaces unavailable DrugBank data.** No authorized DrugBank
   export is present locally, and DrugBank 5.1.22 academic downloads were
   temporarily paused at freeze time. The benchmark will use the openly
   downloadable, versioned ChEMBL 37 mechanism table for approved small
   molecules and direct human protein targets under the frozen rules in
   `references/drug_target_resource.json`. Results must be called ChEMBL
   mechanism-target enrichment, not a DrugBank replication.

**Pretraining leakage remains a design attribute, not an unresolved dataset
issue.** All 28 GSE184119 samples, all 56 GSE113165
samples, and all 6 GSE126865 samples occur in the reconstructed Bridge
train/validation split. Seven of 31 GSE211204 samples also occur (4 train,
3 validation). These datasets remain useful for reproducibility but are not
strict held-out generalization tests. Conversely, all 12 GSE234465/OSD-684
samples were in the archived `unused` split and are confirmed held out.

The final interpretation must separate confirmed-unseen cohorts from
leakage-labeled reproducibility cohorts. Performance on GSE184119 and GSE113165
cannot be reported as out-of-pretraining generalization; GSE211204 requires the
prespecified exposed-subject sensitivity analysis.

## Reference method: de Weerd et al. (2024)

The published method used a VAE trained on 16,819 STRING-covered GTEx genes.
For each case-control dataset it:

1. encoded cases and controls separately using the VAE mean nodes;
2. formed the disease vector as mean(case) minus mean(control);
3. multiplied that vector by `eta = 3` and decoded it;
4. decoded 1,000 standard-normal latent samples as a reference background;
5. ranked each gene by the number of reference decodes below the disease-vector
   decode and retained the top 500;
6. for network-module analyses only, used STRING confidence >=700 and retained
   the largest connected component;
7. compared the module with an equally sized list of Expression Atlas DEGs
   after restricting both comparisons to a common STRING/DisGeNET universe;
8. tested every DrugBank compound having more than 10 mapped targets with a
   right-sided Fisher exact test, using the intersection of the 16,819 model
   genes and DrugBank proteins (16,600 proteins) as background; and
9. ranked drugs by FDR-corrected p-value.

The paper does not establish therapeutic reversal: it tests target enrichment,
and explicitly notes that direction of gene dysregulation versus drug action is
not considered.

Directly reusable elements are the case-control difference, equal-sized
Bridge/DEG modules, right-sided Fisher tests, explicit mapped background,
multiple-testing correction, and module-size sensitivity. The amplified decode
and normal-latent exceedance rank cannot be reproduced because Bridge has no
decoder and its latent space was not trained as a continuous generative prior.
STRING largest-component filtering should be reported as a sensitivity analysis,
not allowed to give only the Bridge arm a topology advantage.

## BridgeRNA infrastructure audit

Reusable local assets in the shared repository are:

- frozen checkpoint `model/r7hnr92k/best_model.pt` and configuration
  `model/r7hnr92k/config.json`;
- the canonical 15,165-gene order in `data/ensembl/canonical_genes.csv`;
- model loading in `src/fm_embed/model.py` and the standard natural
  `log1p(TPM)` transform/alignment utilities in `src/fm_embed/`;
- human/mouse gene lengths and ortholog maps under `data/gencode/` and
  `data/ensembl/`;
- GEO and OSDR source adapters in `src/fm_embed/sources/geo.py` and
  `src/fm_embed/sources/osdr.py`;
- ARCHS4 v2.5 metadata for 940,455 samples and an exact reconstructed split
  under `data/manifests/` and `data/archs4/training/sample_split/`;
- a validated signed Integrated Gradients implementation targeting the dot
  product between mean-pooled Bridge embeddings and a fixed response direction
  in `benchmarks/cross_species_exercise_response/pipeline/attribute_latent_axes.py`;
- native mask-token deletion controls in that same pipeline;
- edgeR/limma implementations and design patterns in existing benchmark
  pipelines; and
- downloaded OSDR count resources for older studies, though not the three
  proposed studies in the current shared cache.

No licensed DrugBank export or reusable local drug-target table was found.
DrugBank's current academic distribution requires authorized access and its
5.1.22 downloads were temporarily paused on the freeze date, so neither scraping
nor an unaudited derivative is acceptable. ChEMBL 37 (release 2026-05-01,
CC BY-SA 3.0) is selected because it has a stable bulk database, schema, release
identifier, and published SHA-256 checksum. Primary edges are approved
small-molecule parent drugs to direct human mechanism-of-action protein targets;
complexes are retained only when ChEMBL identifies the binding component. The
complete source, checksum, mapping, and filtering contract is frozen in
`references/drug_target_resource.json`.

No standalone local STRING interaction resource appropriate for this benchmark
was found. A versioned STRING human links file and protein-to-gene map are still
required only if the optional network sensitivity analysis is retained; this
does not block the frozen dataset matrix or primary target enrichment.

## Candidate dataset audit

### Radiation

**GSE297090 — recommended terrestrial discovery.** Forty poly(A), paired-end
RNA-seq profiles from five human jejunal organoid donors (J2, J3, J6, J8, J11).
Each donor contributes crypt-like and villus-like organoids under four arms:
1 Gy gamma, gamma mock, 1 Gy low-LET proton, and proton mock. Three technical
culture replicates were pooled before RNA extraction, so they are not independent
replicates. Analyze gamma and proton separately, with donor and organoid state
blocked: 10 exposed versus 10 modality-matched controls per modality. Do not
pair gamma with the proton mock or pool crypt/villus without a state term.
FeatureCounts provides Ensembl IDs/symbols and covers 14,954 genes (98.6%). All
40 samples are absent from ARCHS4 v2.5 and its reconstructed splits: confirmed
unseen.

**GSE184119 — recommended terrestrial replication, with leakage label.** Twenty-
eight RNA-seq profiles span ADSC, HMEC, LEC, NHDF, NHEK, and pericytes. Every
cell type has two flask replicates at 0 Gy and 10 Gy; ADSC and LEC additionally
have two replicates after 5 x 2 Gy over 48 hours. The primary analysis should
estimate 10 Gy versus 0 Gy within cell type and combine effects with a cell-type
block/meta-analysis, not treat the 24 libraries as exchangeable replicates.
Fractionated radiation is a separate 8-sample contrast. Files contain Entrez
gene IDs and gene length/read counts; HGNC mapping covers 12,713 Bridge genes
(83.8%). All samples are in pretraining (23 train, 5 validation): confirmed seen.

**OSD-993 / GLDS-805 / GSE297560 — recommended OSDR validation with replicate
caveat.** Twenty-one bulk paired-end RNA-seq cultures of primary human dermal
fibroblasts comprise charged-particle shipped controls (n=3), gamma shipped
controls (n=3), low-passage unshipped controls (n=3), and 4 Gy silicon, iron,
proton/hydrogen, and Cs-137 gamma radiation (n=3 each). Cells were assayed after
approximately 14 days of senescence induction. Run four separate modality
contrasts: each charged particle versus the charged-particle shipped control,
and gamma versus its gamma shipped control. Exclude low-passage unshipped cells
from the primary contrast. The bulk dataset contains no senotherapeutic arm;
the treatment concern belongs to related experiments. These are culture
replicates from one primary-cell source, not independent donors. STAR Ensembl
counts cover 15,096 Bridge genes (99.5%). The study is absent from ARCHS4 v2.5:
confirmed unseen.

### Bone loss

**GSE189524 — discovery candidate, but qualified.** Fourteen RNA-seq libraries
represent three osteoporosis MSC donors and four non-osteoporosis MSC donors,
with two libraries per donor (labels FO1/FO1D, etc.). The independent sample
size is seven donors, not 14. GEO supplies a full transcript-level FPKM workbook
but withholds raw reads/counts for patient confidentiality. Bridge input can be
formed only by summing transcript FPKM to genes and renormalizing each sample
to TPM; that conversion and duplicate relationship require validation before
use. Available metadata do not resolve all age/sex/surgical-indication covariates,
which may confound a 3-versus-4 donor comparison. The study is absent from
ARCHS4 v2.5: confirmed unseen. Recommended contrast is donor-blocked osteoporosis
versus control after collapsing technical duplicates; treat conclusions as
exploratory.

**GSE276529 — recommended qualified terrestrial replication.** This targeted
search result has cortical femur RNA-seq from five women with glucocorticoid-
induced osteoporosis and four age-matched non-GIOP women (ages 51–65), with
RSEM Ensembl counts covering 15,089 Bridge genes (99.5%). It is unpaired and
absent from ARCHS4 v2.5: confirmed unseen. Use GIOP versus non-GIOP with age as
a covariate if identifiable. Its strengths are clinical bone tissue, raw counts,
and an independent etiology; its limitation is inseparable glucocorticoid
exposure and different tissue/cell composition from GSE189524. Cross-study
agreement should therefore be interpreted as convergence across osteoporosis
etiologies, not a literal replication of primary osteoporosis MSC biology.

**OSD-867 / GLDS-719 / GSE273868 — selected OSDR validation.** Twelve human
bone-marrow MSC bulk poly(A), paired-end RNA-seq profiles compare actual ISS
spaceflight with synchronous ground controls. The frozen primary analysis has
two separate contrasts: one-week flight versus one-week ground (3 vs 3), and
two-week flight versus two-week ground (3 vs 3). Timepoints must not be pooled.
The exact groups are one-week flight GSM8438331/GSM8438333/GSM8438335 versus
ground GSM8438330/GSM8438332/GSM8438334, and two-week flight
GSM8438337/GSM8438339/GSM8438341 versus ground
GSM8438336/GSM8438338/GSM8438340.
NASA processed the data with its bulk RNA-seq workflow against GRCh38/Ensembl
112; the STAR raw-count matrix covers 15,096 Bridge genes (99.5%). All twelve
GSM8438330–GSM8438341 accessions are absent from every reconstructed ARCHS4 v2.5
split and are confirmed unseen. This is actual spaceflight and directly samples
bone-marrow-derived MSCs, but the cells were expanded in standard medium rather
than osteogenically differentiated. Treat it as a spaceflight bone-lineage/progenitor
model, not as direct evidence of osteoporosis or loss of mineralized bone.

**OSD-546 / GSE100930 — superseded.** The original osteogenic study remains
biologically relevant but is Affymetrix microarray data and therefore outside
Bridge's natural RNA-seq input domain. It is not part of the frozen nine-dataset
matrix and will not be used as a substitute input.

### Skeletal muscle atrophy

**GSE211204 — recommended terrestrial discovery, with partial leakage.** Thirty-
one total-RNA paired-end profiles cover 11 young healthy men before and after
10-day unilateral lower-limb suspension; nine also have 21-day active recovery.
The primary contrast is the 11 within-subject ULLS-minus-baseline differences.
Recovery-minus-ULLS and recovery-minus-baseline are prespecified secondary
directions. RSEM identifiers/symbols cover 14,603 Bridge genes (96.3%). Seven
samples occur in the reconstructed pretraining split (4 train, 3 validation),
while 24 are absent: confirmed partial exposure. Sensitivity analysis must drop
subjects having either endpoint exposed during pretraining.

**GSE113165 — select as terrestrial replication, but not held out.** Fifty-six
vastus-lateralis RNA-seq profiles form 28 complete pre/post pairs after five days
of bed rest: 9 young and 19 old participants (30 female, 26 male libraries).
Analyze within subject and preserve age, sex, and insulin-resistance
susceptibility strata; the prespecified primary effect is the overall paired
post-minus-pre response with age-stratified estimates. Ensembl counts cover
15,076 Bridge genes (99.4%). All samples occur in pretraining (48 train,
8 validation): confirmed seen. It is stronger than GSE126865 on power and
covariate annotation, but cannot support a held-out claim.

**GSE126865 — do not select.** Six vastus-lateralis RNA-seq profiles are three
older subjects before/after ten days of bed rest. Pairing is valid, but n=3 is
too fragile for the primary replication and all samples were in pretraining
(4 train, 2 validation). The processed workbook contains transcript/read and
RPKM columns rather than a clean gene-count matrix.

**OSD-684 / GLDS-615 / GSE234465 — recommended held-out OSDR validation.** Twelve
RNA-seq profiles are 3D myobundles made from young-active and old-sedentary
CD56+ myoblast sources, with three flight and three ground chip replicates per
age stratum after ten days. Analyze flight versus ground separately in young
and old strata; only estimate a common effect with an age-stratified model.
Replicate chips should not be presented as independent human donors. Ensembl
counts cover 15,096 Bridge genes (99.5%). All 12 samples were catalogued by
ARCHS4 but assigned to the reconstructed `unused` split: confirmed unseen by
the trained checkpoint.

## Frozen dataset matrix

| Condition | Role | Dataset | Exact primary contrast | N used | Bridge coverage | Pretraining | Recommendation |
|---|---|---|---|---:|---:|---|---|
| Radiation | Terrestrial discovery | GSE297090 | 1 Gy gamma vs gamma mock and 1 Gy proton vs proton mock, separately; donor + crypt/villus blocked | 20 per modality | 14,954 (98.6%) | confirmed unseen | Use |
| Radiation | Terrestrial replication | GSE184119 | 10 Gy vs 0 Gy within each of 6 cell types | 24 | 12,713 (83.8%) | confirmed seen | Use, leakage-labeled |
| Radiation | OSDR validation | OSD-993 / GSE297560 | each 4 Gy particle vs shipped particle control; gamma vs shipped gamma control | 6 per modality | 15,096 (99.5%) | confirmed unseen | Use; culture-replicate caveat |
| Bone loss | Terrestrial discovery | GSE189524 | osteoporosis vs non-osteoporosis MSCs after transcript summation, per-library TPM renormalization, and donor averaging | 7 donors / 14 libraries | 14,665 (96.7%) | confirmed unseen | Use; exploratory |
| Bone loss | Terrestrial replication | GSE276529 | GIOP cortical femur vs age-matched non-GIOP | 9 | 15,089 (99.5%) | confirmed unseen | Use, etiology-qualified |
| Bone loss | OSDR validation | OSD-867 / GSE273868 | ISS flight vs ground separately at 1 week and 2 weeks | 6 per time stratum | 15,096 (99.5%) | confirmed unseen | Use; bone-marrow MSC spaceflight proxy |
| Muscle atrophy | Terrestrial discovery | GSE211204 | paired day-10 ULLS vs baseline | 22 (11 pairs) | 14,603 (96.3%) | confirmed partial exposure | Use with leakage sensitivity |
| Muscle atrophy | Terrestrial replication | GSE113165 | paired day-5 bed rest vs pre, age/sex-stratified | 56 (28 pairs) | 15,076 (99.4%) | confirmed seen | Select, leakage-labeled |
| Muscle atrophy | OSDR validation | OSD-684 / GSE234465 | flight vs ground separately for young and old myobundles | 6 per age stratum | 15,096 (99.5%) | confirmed unseen (`unused`) | Use |

## Proposed implementation

### Bridge condition vectors and gene ranking

For each frozen contrast, produce sample embeddings with the frozen r7hnr92k
encoder and define `delta_z` from paired subject-level differences when pairing
exists, otherwise from covariate-adjusted group centroids. Never pool radiation
modalities, organoid states, age strata, media, or treatments merely to increase
N.

The primary Bridge gene score should reuse the validated signed Integrated
Gradients approach already in the repository:

- target score for a sample is the dot product of its mean-pooled frozen
  embedding and the unit `delta_z` for that dataset/contrast;
- compute IG from the all-zero natural-log1p-TPM baseline for condition and
  control profiles with the existing midpoint-Riemann implementation;
- for paired designs, calculate condition-minus-control IG within subject and
  aggregate robustly across subjects; for unpaired designs, compare group-level
  IG with the same covariate structure used for `delta_z`;
- rank genes by absolute signed attribution change for the primary undirected
  target-enrichment module while retaining signed scores and separate positive
  and negative lists as secondary outputs; and
- validate top genes with native `-10` mask-token deletion against deterministic
  size-matched random panels. Attribution completeness and deletion effect are
  mandatory QA, not optional illustrations.

This is a Bridge analogue, not a recreation of the VAE decoder. To avoid target
leakage, estimate each final study direction within that study but use leave-one-
subject-out directions for subject-level attribution where sample size permits.
Freeze top 500 as primary, with 100, 250, 750, and 1,000 sensitivity modules.

### Differential-expression baseline

Use raw gene counts and edgeR quasi-likelihood (or DESeq2 with an equivalent
frozen design) for count datasets. Include donor blocking for paired studies,
cell type/state/age strata as specified above, and do not count technical or chip
replicates as human biological replication. Rank the eligible Bridge-universe
genes by a prespecified two-sided statistic (`abs(t)`/`abs(Wald)`, with signed
log-fold change retained), then take exactly the same module sizes as Bridge.
For GSE189524, use limma with donor blocking on log2(TPM+offset) only after the
FPKM-to-TPM and duplicate audit. OSD-546 is outside the frozen matrix and must
not be fed to Bridge.

### Drug-target enrichment and convergence

Use the frozen ChEMBL 37 rules in `references/drug_target_resource.json`, freeze
the resulting canonical drug-gene edge table, and use the identical mapped
universe and exact Fisher implementation for Bridge and DEG modules. Phase 2 QC
showed that the originally borrowed threshold of at least 10 targets retained
only one drug in the deliberately narrow direct-mechanism graph. The validity
amendment therefore uses at least 3 eligible targets as primary (63 drugs) and
at least 2 and 5 as sensitivity thresholds. Save the full 2x2
table and overlapping genes, apply Benjamini-Hochberg within each dataset,
method, and module size, and rank by adjusted p-value with odds ratio and raw
p-value as deterministic tie breakers. Do not call enrichment therapeutic
reversal.

For each condition and method, report pairwise and three-way top-10/25/50
overlap, Jaccard and overlap coefficient, rank-biased overlap, Spearman on the
intersection plus a censored full-universe rank correlation, reciprocal
top-list enrichment by Fisher test, shared targets, and class/mechanism overlap
only where a versioned annotation supports it. Compare Bridge-minus-DEG
convergence using paired label permutations at the study level. Random-gene
modules must be sampled from each dataset's eligible universe and matched for
module size; expression/detectability bins should be added because drug targets
are not uniformly represented across expression levels. Label permutations must
swap within subject for paired designs and within blocks/strata otherwise.

### Planned benchmark layout

```text
benchmarks/drug_discovery/
├── README.md
├── drug_discovery_benchmark.ipynb
├── config.json
├── references/                 # frozen resource manifests only
├── pipeline/
│   ├── freeze_cohorts.py
│   ├── prepare_expression.py
│   ├── compute_bridge_modules.py
│   ├── run_differential_expression.R
│   ├── enrich_drug_targets.py
│   ├── evaluate_reproducibility.py
│   └── build_notebook.py
├── work/                       # ignored expression/embedding caches
└── results/
    ├── manifests/
    ├── modules/
    ├── drug_enrichment/
    ├── convergence/
    ├── nulls/
    ├── figures/
    └── provenance.json
```

## Phase 2 pre-execution gates

The two audit blockers and all four pre-execution gates are resolved:

1. GSE189524's 182,434 transcript rows were mapped through HGNC UCSC identifiers,
   summed by gene, renormalized to TPM per library, and averaged within the seven
   donors. The two libraries per donor have log1p-TPM Pearson correlations from
   0.897 to 0.964; 14,665 canonical genes are represented.
2. The official ChEMBL 37 SQLite archive is acquired and checked against the
   frozen EMBL-EBI SHA-256 before enrichment. The filtered API snapshot is also
   hashed and contains 1,344 unique approved-drug/direct-human-gene edges.
3. Sample and contrast manifests are frozen. The GSE211204 strict sensitivity
   retains seven subjects for whom neither baseline nor ULLS is in ARCHS4
   train/validation.
4. The optional STRING topology analysis was not retained, so it is not an
   execution dependency and cannot asymmetrically filter the Bridge arm.

The only frozen-design amendment is the target-count threshold above; no dataset
or contrast changed in Phase 2. Primary convergence is mean pairwise top-25
Jaccard across discovery, terrestrial replication, and OSDR rankings at module
size 500 and target threshold 3. Exact study-level Bridge/DE label swaps and
1,000 expression-decile-matched random modules are the prespecified nulls.
Results are stratified into strict-unseen cohort pairs and comparisons involving
ARCHS4 train/validation exposure. No direction-aware perturbation/reversal
analysis is included.
