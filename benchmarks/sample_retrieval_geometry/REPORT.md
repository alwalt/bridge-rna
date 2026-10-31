# Frozen Bridge sample-retrieval geometry: Stage 0–1 report

## Scope and provenance

This new benchmark evaluates the same 562 unseen-study recount3 queries against
510,709 human ARCHS4 references used by the read-only upstream
`/home/walt/bridge-rna/benchmarks/paired_recount3` benchmark. It directly reads
the shared ARCHS4 float16 embedding memmap, sample-location manifest, sample
manifest, ARCHS4 metadata, upstream recount3 embeddings, and upstream result
tables from the absolute paths recorded in `config.json` and `results/provenance.json`.
No embedding was regenerated and no large input was copied.

The upstream exact-pair cosine values and ranks are retained at their saved
precision. This matters: in this unusually dense space, recomputing a paired
cosine after float32 normalization can change a value by less than 1e-6 yet
move it by one or two ranks. All 562 baseline ranks are asserted against the
upstream table.

## recount3 discovery audit and study-balanced sensitivity analysis

### Direct answer

The original discovery was **exhaustive over the predefined mapped-single
Bridge evaluation candidates and the recount3 project metadata that it could
read, but strict exhaustiveness cannot be established from the preserved
provenance**. The workflow retained all 84,896 eligible candidates—82,588
`strict_unseen_single_gse` and 2,308 `unseen_sample_seen_study`—without a query,
GSE, or random-sampling limit. It enumerated all 8,803 human recount3 SRA
projects in the contemporaneous catalog and read 8,199 (93.14%), but recorded
604 `download_or_read_failed` projects (6.86%). It did not retry or otherwise
resolve those failures. Thus the defensible classification is **cannot
determine from available provenance**, with strong evidence of comprehensive
candidate coverage and catalog traversal but incomplete project-metadata
coverage.

The current upstream `config.json` lists only `strict_unseen_single_gse`, but
the older frozen candidate, match, final-pair, expression, and embedding
artifacts all contain both cohorts. This is configuration drift after the run;
the preserved tables are the source of truth for what was actually searched.

### Exact discovery procedure and funnel

The workflow started from the frozen Bridge master sample manifest, not from a
convenient recount3 subset. It selected human `split == unseen` samples with an
unambiguous single GSE, separated them by the repository's existing exposure
labels, then attached ARCHS4 SRA relations from the human ARCHS4 H5. The
recount3 matcher called `available_samples(organism="human")`, restricted to
`data_sources/sra`, and used the full project list (`max_projects=0`). It first
looked up explicit run accessions (SRR/ERR/DRR) and then searched every
character column in each readable project metadata table for exact GSMs. There
was no pagination, manual GSE/GSM list, random subsample, or project cap.

| Funnel step | Samples | Studies/GSE labels | Removed | Reason |
|---|---:|---:|---:|---|
| Human Bridge manifest | 510,709 | 26,262 | — | Frozen manifest |
| Predefined human unseen split | 110,709 | 5,678 | 400,000 | `split == unseen` |
| Single-GSE eligible | 84,896 | 4,375 | 25,813 | `mapping_status == mapped_single` |
| Frozen candidates searched | 84,896 | 4,375 | 0 | All eligible candidates retained |
| Any recount3 metadata match | 575 | 43 | 84,321 | 8,199/8,803 project files readable |
| Unambiguous project with run(s) | 574 | 42 | 1 | One GSM mapped to multiple projects |
| Expression/vocabulary processed | 574 | 42 | 0 | All selected pairs reached 15,165 genes |
| Published strict unseen-study result | 562 | 40 | 12 | Headline notebook excluded the other exposure cohort |

There were no `matched_no_run` rows and no sample-level QC removals after final
selection. The 574 processed pairs comprise 562 `unseen_study` pairs from 40
studies and 12 `seen_study` pairs from two studies. All have `split == unseen`;
the latter are unseen samples from studies exposed during training and can
support only a small, separately labeled sensitivity analysis—not an enlarged
unseen-study headline result.

### Composition check

The biological-validation findings reproduce exactly: GSE234297 contributes
144/562 (25.62%) queries, all normalized as `peripheral blood`; the five
largest GSEs contribute 287/562 (51.07%). It also contributes 122/187 (65.24%)
dense queries. The deterministic five-per-study subset contains 154 queries
from all 40 studies and limits every GSE to 3.25%; its density composition is
48.7% sparse, 35.1% intermediate, and 16.2% dense. The ten-per-study subset
contains 248 queries, caps every GSE at 4.03%, and is 45.6%, 36.3%, and 18.1%
sparse/intermediate/dense. Neither subset was selected using density or
retrieval performance.

### Study-balanced exact retrieval

Equal-weight study-macro results (uncertainty is a deterministic bootstrap over
40 studies; full intervals and across-study SDs are saved) are:

| Method | Macro R@1 | Macro R@5 | Macro R@10 | Macro R@100 | Macro MRR |
|---|---:|---:|---:|---:|---:|
| Cosine | 32.97% | 65.97% | 75.92% | 91.33% | 0.4886 |
| CSLS k=10 | 22.50% | 60.59% | 66.01% | 83.83% | 0.4066 |
| Local scaling k=10 | 34.27% | **72.08%** | 78.66% | **93.22%** | 0.5155 |
| NICDM k=10 | **34.68%** | 71.66% | **78.96%** | 92.82% | **0.5163** |

Study heterogeneity is large: for cosine macro R@1 the across-study SD is 33.9
percentage points and the bootstrap 95% interval is 22.9–43.6%. Therefore the
small local-scaling/NICDM differences are directional, not precise method
separations. CSLS's favorable micro-average does not survive equal study
weighting.

| Cap | Queries | Method | R@1 | R@5 | R@10 | R@100 | MRR |
|---:|---:|---|---:|---:|---:|---:|---:|
| 5 | 154 | Cosine | 31.17% | 66.23% | 72.08% | 89.61% | 0.4730 |
| 5 | 154 | CSLS k=10 | 27.27% | 64.94% | 69.48% | 88.31% | 0.4498 |
| 5 | 154 | Local scaling k=10 | **32.47%** | **72.08%** | **76.62%** | **90.91%** | **0.5068** |
| 5 | 154 | NICDM k=10 | 31.82% | 71.43% | **76.62%** | 90.26% | 0.4984 |
| 10 | 248 | Cosine | 31.85% | 69.76% | 76.61% | 90.73% | 0.4932 |
| 10 | 248 | CSLS k=10 | 29.44% | 70.16% | 75.00% | 91.53% | 0.4864 |
| 10 | 248 | Local scaling k=10 | **33.06%** | **74.19%** | 79.44% | **92.74%** | **0.5209** |
| 10 | 248 | NICDM k=10 | 32.26% | 73.79% | **79.84%** | 92.34% | 0.5142 |

Balancing therefore preserves a modest advantage for local scaling/NICDM at
R@5/R@10, makes their R@1 gains small and interchangeable, and weakens or
reverses the CSLS conclusion. The main claim should be that local normalization
helps technical identity modestly—not that one correction decisively wins.

### Revised density conclusion

Density remains a real geometric property, but its relationship to failure is
not robustly general. The query-micro dense-minus-sparse failure gap is +6.8
points; it is +6.8 points under study-macro weighting, +6.7 with cap 5, and
+3.4 with cap 10. Yet removing GSE234297 reverses it to −1.7 points. The
study/source-controlled models from the biological validation are also
imprecise and compatible with no independent effect. Study caps reduce the
dominant cohort but retain some of its observations and therefore cannot
resolve this contradiction. The sound conclusion is: dense regions exist;
GSE234297 couples density to failure in this benchmark; evidence that density
generally causes retrieval difficulty is weak and cohort-sensitive.

### Existing cohorts and expansion inventory

The frozen human unseen split contains 105,417 `unseen_study` samples and 5,292
`seen_study` samples. Under the original single-GSE criterion these become
82,588 and 2,308, respectively. The preserved discovery already searched every
one: it found the published 562 unseen-study pairs plus 12 processed
unseen-sample/seen-study pairs, with **zero additional mapped-single candidates
in the readable metadata snapshot**.

A new metadata-only intersection against the 8,199 cached project tables found
258 unprocessed, single-recount-project candidates: 257 `unseen_study` and one
`seen_study`. Crucially, **all 258 are `mapped_multiple`**, so all were outside
the documented single-GSE eligibility protocol. They must not be described as
missing members of the original cohort. The 257 unseen-study candidates span
23 composite GSE labels and 44 component GSEs, 43 component GSEs absent from
the published 562. They are biologically different—frontal/motor cortex,
neuronal material, and brain organoids are prominent—but not balanced: one
composite label contributes 110/257 (42.8%). They could add useful neural
diversity to a separately defined ambiguous-study technical-identity cohort,
after explicit adjudication of GSE identity and leakage; no expression was
downloaded and no embedding was generated here.

**Decision.** Keep the 562 as the strict single-GSE unseen-study benchmark and
make study-macro results primary alongside micro results. Cohort expansion is
worth a targeted follow-up only for the 257 mapped-multiple unseen-study
candidates, as a separately named/adjudicated cohort; the original criteria
yield no unprocessed accessible candidates. The 12 already-processed
unseen-sample/seen-study pairs are too small (two studies) for a headline arm.
No representation correction is warranted by this audit.

## Absolute representation similarity versus retrieval uniqueness

This extension separates two questions that the earlier retrieval tables could
conflate. **Bridge cosine is absolute representation agreement** between two
independently processed versions of the same transcriptome. **Exact rank is
retrieval uniqueness**: how many of 510,709 ARCHS4 references are still closer.
All values below reuse the frozen Stage 0/1 scores and ranks.

### Exact-pair cosine by cosine retrieval rank

Across all 562 pairs, mean exact-pair cosine is **0.997806** and median is
**0.999218** (SD 0.003510; Q1–Q3 0.998496–0.999460; range
0.974058–0.999822).

| Exact cosine-rank group | N | Mean | Median | SD | Q1 | Q3 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rank 1 | 160 | 0.999237 | 0.999385 | 0.000481 | 0.998961 | 0.999564 | 0.996436 | 0.999822 |
| Rank 2–5 | 231 | 0.998683 | 0.999317 | 0.001863 | 0.998697 | 0.999525 | 0.981911 | 0.999822 |
| Rank 6–10 | 43 | 0.997244 | 0.998624 | 0.002742 | 0.994999 | 0.999381 | 0.989748 | 0.999444 |
| Rank 11–100 | 82 | 0.995938 | 0.996405 | 0.003545 | 0.993314 | 0.999110 | 0.984819 | 0.999493 |
| Rank 101–1,000 | 32 | 0.994699 | 0.995907 | 0.004281 | 0.991411 | 0.998770 | 0.985970 | 0.999223 |
| Rank >1,000 | 14 | 0.986743 | 0.985951 | 0.010182 | 0.977758 | 0.997191 | 0.974058 | 0.997903 |

Cosine declines with rank, but not enough to make poor rank synonymous with
poor representation agreement. Among 46 rank >100 pairs, 33 still have cosine
at least 0.99, 25 at least 0.995, and four at least 0.999. Among 14 rank >1,000
pairs, seven remain at least 0.99 and six at least 0.995; none reaches 0.999.

### Paired cosine across studies

The ordinary sample-weighted mean/median are 0.997806/0.999218. After computing
within each of 40 GSEs and weighting every study equally, the macro mean of
study means is **0.996569** and the macro mean of study medians is **0.996740**.
The median study mean is 0.998803, study means range from 0.977107 to 0.999689,
and their SD is 0.004667. Very high cross-pipeline agreement is therefore
broadly observed rather than created by GSE234297, although equal weighting
exposes a few small, lower-agreement studies.

### Expanded retrieval cutoffs

| Metric | Cosine | Local scaling k=10 | NICDM k=10 | CSLS k=10 |
|---|---:|---:|---:|---:|
| R@1 | 28.47% | 30.25% | **30.43%** | 29.36% |
| R@5 | 69.57% | 75.27% | **75.80%** | 74.02% |
| R@10 | 77.22% | 81.67% | **82.56%** | 80.60% |
| R@100 | 91.81% | **94.31%** | **94.31%** | 94.13% |
| R@1,000 | 97.51% | **98.58%** | **98.58%** | 98.04% |
| MRR | 0.4764 | 0.5119 | **0.5125** | 0.5039 |
| Median rank | 2 | 2 | 2 | 2 |
| Mean rank | 119.46 | 66.02 | **59.49** | 69.78 |
| Study-macro R@10 | 75.92% | 78.66% | **78.96%** | 66.01% |
| Study-macro R@100 | 91.33% | **93.22%** | 92.82% | 83.83% |
| Study-macro R@1,000 | 96.98% | **97.47%** | **97.47%** | 94.95% |

R@100 covers 100/510,709 references (0.0196% of the corpus); R@1,000 covers
1,000/510,709 (0.1958%). R@1,000 completes the rank-distribution description
but is not a primary retrieval target. Micro metrics weight 562 samples equally;
study-macro metrics first score each GSE and then weight all 40 studies equally.

### Random-pair context and crowded neighborhoods

The existing deterministic two-million random-ARCHS4-pair sample has mean
cosine 0.878748, median 0.912954, and SD 0.118144, versus 0.997806, 0.999218,
and 0.003510 for exact cross-pipeline pairs. The exact-pair minimum is already
at the empirical 97.141st random-pair percentile. Exact Q1 is at the 99.9726th
percentile and the exact median at the 99.9936th percentile. These are empirical
placements in the saved random sample, not parametric tail extrapolations.

Across all queries, median Top1-minus-exact cosine is only **5.36e-7** (Q1 0;
Q3 4.06e-4). The margin grows with rank: median 0.002486 for ranks 101–1,000
and 0.010424 for ranks >1,000. Sub-micro negative differences caused by saved
float precision are clamped to zero when reporting the margin.

Neighborhood crowding grows much faster. Counts use the established definition
`reference cosine >= exact-pair cosine − delta`, so they include closer samples
and the exact pair:

| Rank group | Median within 0.0001 | Median within 0.001 | Median within 0.005 |
|---|---:|---:|---:|
| Rank 1 | 1 | 6 | 341 |
| Rank 2–5 | 2 | 10 | 463 |
| Rank 6–10 | 11 | 88 | 508 |
| Rank 11–100 | 31.5 | 93.5 | 476 |
| Rank 101–1,000 | 244.5 | 504 | 1,422.5 |
| Rank >1,000 | 3,518.5 | 5,944.5 | 14,889 |

The objective highest-cosine rank >100 case is GSM7461825 (GSE234297): cosine
0.999223 at rank 107, Top-1 margin 0.000290, with 2,927 references above
exact−0.001 and 13,615 above exact−0.005. The highest-cosine rank >1,000 case
is GSM7461712: cosine 0.997903 at rank 1,825, margin 0.001156, with 6,053 and
14,974 such references. Both are peripheral blood from GSE234297, reinforcing
the earlier composition/density finding. The diagnostic table uses a fixed
rule—five highest exact-pair cosines above each rank threshold—not manual
selection.

### Interpretation

An exact pair near cosine 0.999 demonstrates that Bridge maps the two processing
pipelines to almost the same direction in representation space. It does not
guarantee rank 1: rank asks whether any of hundreds or thousands of nearly
indistinguishable references are even closer. The high-cosine/poor-rank cases
are therefore chiefly failures of **uniqueness in crowded neighborhoods**, not
simple failures of cross-pipeline representation agreement. Native CSLS,
local-scaling, and NICDM scores remain ranking criteria and are reported under
their own names; they are not reinterpreted as Bridge cosine.

## Stage 1A–1H biological validation: primary answer

**The 562 queries are not biologically balanced.** They span 40 studies and 92
exact normalized source labels, but GSE234297 contributes 144 queries (25.6%)
and the five largest studies contribute 51.1%. The effective number of studies
is 19.2 by Shannon entropy and 10.6 by inverse Simpson concentration. The same
144 samples are labeled `peripheral blood`, making it the largest source
(25.6%); the top five sources contribute 44.8%, with effective source counts of
34.3 (Shannon) and 12.2 (inverse Simpson). Furthermore, 122/187 dense queries
(65.2%) are from GSE234297/peripheral blood. The cohort is varied outside this
block, but one study/source disproportionately defines the dense stratum.

All queries are human RNA-seq with transcriptomic library source. Platforms are
321 GPL11154, 160 GPL24676, 50 GPL18573, 28 GPL15433, and 3 GPL16791; the
corresponding instruments are chiefly HiSeq 2000 (321) and NovaSeq 6000 (160).
Molecule labels are 396 total RNA, 148 polyA RNA, 12 nuclear RNA, and 6
cytoplasmic RNA. No corpus-wide curated cell-type, cell-line, disease,
treatment, or primary-versus-cultured fields exist. Those attributes were not
parsed from titles or characteristics.

### Does density independently predict exact-retrieval failure?

Not convincingly in this cohort. Failure rates are 75.5%, 56.7%, and 82.4% in
the sparse, intermediate, and dense strata, respectively. The dense-minus-
sparse difference is +6.8 percentage points overall, but becomes −1.7 points
after excluding GSE234297; excluding the largest source gives the identical
result because that study supplies all 144 `peripheral blood` queries.

A one-SD continuous density increase has unadjusted failure OR 0.86 (95% CI
0.70–1.06), study-fixed-effect OR 0.98 (0.69–1.40), and source-fixed-effect OR
0.89 (0.63–1.27). Direct dense-versus-sparse OR is 1.51 (0.92–2.50), but is
extremely uncertain after study control (3.24, 0.49–21.63) and source control
(0.71, 0.12–4.18). Individual large contexts are heterogeneous: high-minus-low
density failure ranges from −37.5 points in GSE214017 to +80 points in
GSE244209. Therefore the previous density-stratified association is primarily
a composition warning, not evidence for a stable density-only causal effect.

### Defensible biological labels

Two exploratory labels pass prespecified support of at least 50 references in
at least three studies:

- Exact normalized `source_name_ch1` is structured and complete, but
  submitter-defined and heterogeneous. Of 562 queries, 286 have labels meeting
  support (23 query classes); 731 corpus source classes meet support among
  228,556 references. Peripheral blood is 50.3% of eligible queries.
- A pre-existing conservative anchored source-to-tissue mapping labels 73,162
  references in 16 tissues, but only 162 queries in two tissues: 144 blood and
  18 brain. Its 71.2% query missingness and 88.9% blood dominance make it a
  limited sensitivity analysis, not a pan-biological benchmark.

This metadata cannot support a reliable pan-corpus cell-type, disease, or cell-
state retrieval benchmark. Exact source matching is especially brittle across
studies because independent submitters use different names for similar biology.

### Biological and cross-study retrieval

The exact paired GSM is excluded from all biological metrics. Cross-study
metrics additionally remove every reference in the query GSE.

| Method | Exact R@1 | Exact R@10 | Source P@1 | Source P@10 | Cross-study source P@1 | Cross-study source P@10 | Max N100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosine | 28.47% | 77.22% | **66.43%** | **49.51%** | **3.15%** | 1.85% | 74 |
| CSLS k=10 | 29.36% | 80.60% | 53.85% | 37.24% | 2.80% | **1.89%** | 54 |
| Local scaling k=10 | 30.25% | 81.67% | 57.34% | 38.53% | 2.10% | 1.68% | 44 |
| NICDM k=10 | **30.43%** | **82.56%** | 56.29% | 36.82% | 2.10% | 1.71% | **43** |

Cosine is clearly strongest for exact normalized source retrieval within the
full corpus after removing the paired GSM. All methods perform poorly on
cross-study exact-source matching, which mainly exposes label incompatibility
rather than absence of biological information. Median P@10 enrichment remains
large within the full corpus (cosine 62.1-fold), but is zero across studies for
most queries because exact label strings rarely transfer.

| Method | Tissue P@1 | Tissue P@10 | Cross-study tissue P@1 | Cross-study tissue P@10 |
|---|---:|---:|---:|---:|
| Cosine | **82.10%** | **68.83%** | 34.57% | 30.80% |
| CSLS k=10 | 75.31% | 62.84% | **42.59%** | **39.01%** |
| Local scaling k=10 | 74.69% | 62.16% | 40.74% | 38.27% |
| NICDM k=10 | 76.54% | 60.37% | 41.36% | 37.16% |

The limited tissue task suggests a real tradeoff: cosine best preserves local
same-tissue neighborhoods, whereas CSLS/local scaling/NICDM improve retrieval
of blood/brain labels from independent studies. Because 89% of these queries
are blood from one study, this is suggestive rather than a general tissue result.

Mutual-kNN k=10 returns 7.5 samples/query on average for eligible source queries
and has 41.8% within-corpus set precision; after cross-study exclusion it
returns 4.5 and has 2.7% source precision. SNN maximum-score sets are too broad
(about 151 references/query for source-labeled queries) and have 11.1% source
precision; it is not a selective search result. Tissue set results follow the
same limited blood/brain pattern and are retained in machine-readable tables.

### Density-stratified biology and retrieval diversity

Cosine source P@10 is 46.5%, 50.1%, and 50.9% in sparse, intermediate, and dense
queries: dense cosine neighborhoods remain source-coherent even where exact GSM
retrieval struggles. NICDM source P@10 is only 37.1%, 41.6%, and 33.6%. Thus
NICDM's exact-identity gain in dense queries disrupts the available biological
source label rather than improving it. Cross-study exact-source precision is
near zero in every stratum and cannot discriminate methods reliably.

Ordinary cosine is also the most study-redundant: after excluding the exact GSM,
its Top-10 contains 3.93 unique GSEs on average, and the largest GSE occupies
59.5%. NICDM increases this to 4.44 GSEs and lowers the largest-GSE share to
52.0%; CSLS gives 4.42 and 52.8%. Once the query GSE is excluded, all methods
retrieve about 5.2–5.3 unique GSEs and the largest study occupies 40–41%.
Cosine therefore does return more redundant samples from its local dense study,
while normalization modestly diversifies results without an explicit reranker.

## Stage 0: geometry

The established cosine baseline is R@1 28.47%, R@5 69.57%, R@10 77.22%,
R@100 91.81%, MRR 0.4764, median rank 2, median percentile rank 0.999998,
same-GSE R@1 76.16%, and same-GSE R@5 85.59%.

Cosines are not confined near one. Across all 287,018,458 query/reference
comparisons, the exact minimum is -0.5681; quantiles at 0.01%, 0.1%, 1%, 5%,
25%, 50%, 75%, 95%, 99%, and 99.9% are -0.3569, -0.2258, 0.1456, 0.7040,
0.8652, 0.9154, 0.9445, 0.9695, 0.9887, and 0.9974. Query Top-1 cosines have
median 0.99933, while exact cross-pipeline pairs have median 0.99922. Thus
ordinary pairs are mostly positive and high, but values near zero and negative
values genuinely occur; nearest-neighbor and exact-pair similarities occupy a
very narrow extreme tail.

For 2,000,000 deterministic random distinct ARCHS4 pairs (seed 20260922), the
estimated minimum is -0.4431 and the median is 0.9130. The 1%, 5%, 95%, 99%,
and 99.9% estimates are 0.3548, 0.6734, 0.9678, 0.9844, and 0.9968. This
sampled minimum is **not** the exact global minimum among roughly 130 billion
possible pairs. The lowest sampled pair contrasts purified platelet mRNA
(GSM4810017, GSE158765) with metastatic ovarian gastric-cancer material
(GSM7655906, GSE237876); the complete objective list and metadata are in
`results/lowest_sampled_pairs.csv`.

Hubness is substantial under cosine. At k=1, only 396 references receive the
562 assignments; the largest hub receives 16, and the top ten receive 14.1%
of assignments. At k=100, 13,179 references receive 56,200 assignments, the
maximum N100 is 74, Gini is 0.9893, and the top ten receive 1.24%. The strongest
hubs are dominated by GSE234297 peripheral-blood samples, with a whole-blood
sample from GSE176260/GSE176262 also recurring. Local corrections reduce the
largest N100 to 43–58 and the top-ten share to about 0.73–0.93%, depending on
method and k.

Reference density was estimated by exact full-corpus neighbor searches for a
deterministic 1,024-reference probe (not a full 510k-by-510k matrix). Median
10th-, 50th-, and 100th-neighbor cosines are 0.99850, 0.99692, and 0.99600.
Query density strata show that cosine R@1 is lowest in the dense stratum
(17.65%), although density and exact rank are not monotonically associated
overall (Spearman rho 0.004, p=0.917). Density is therefore a useful failure
context, not a sufficient scalar explanation of every miss.

## Stage 1: ranked methods

| Method | R@1 | R@5 | R@10 | R@100 | MRR | Median rank | Same-GSE R@1 | Source R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosine | 28.47% | 69.57% | 77.22% | 91.81% | 0.4764 | 2 | 76.16% | 72.24% |
| CSLS (k=5) | 29.89% | 74.20% | 81.49% | 93.42% | 0.5081 | 2 | 72.42% | 69.75% |
| CSLS (k=10) | 29.36% | 74.02% | 80.60% | 94.13% | 0.5039 | 2 | 71.89% | 69.93% |
| Local scaling (k=10) | 30.25% | 75.27% | 81.67% | 94.31% | 0.5119 | 2 | 74.73% | 71.71% |
| NICDM (k=10) | **30.43%** | **75.80%** | **82.56%** | 94.31% | **0.5125** | 2 | 74.20% | 71.53% |
| Global empirical-null cosine | 28.47% | 69.57% | 77.22% | 91.81% | 0.4764 | 2 | 76.16% | 72.24% |

CSLS k=50 degrades R@1 to 20.46%, showing that the correction is sensitive to
overly broad neighborhoods. The global empirical P-value is a monotone cosine
transform and correctly leaves ranking unchanged. After Bonferroni correction
for 510,709 searches, no exact pair is significant at 0.05. Raw exact-pair
median P-values differ strongly by density stratum (dense 3.5e-5,
intermediate 4.3e-5, sparse 1.47e-3), demonstrating that one global null is
differently calibrated across local geometry even though it cannot alter rank.

Reciprocal-kNN and SNN are set methods, so conventional R@K/MRR are not
reported. Their exact-set and same-GSE-set recall, returned-set size, and empty
fraction are in `results/set_retrieval_metrics.csv`. SNN produces very large
maximum-score ties (about 280 references/query on average), limiting its value
as a selective retriever in this geometry.

The upstream rank table and upstream Top-10 table are numerically inconsistent
for some sub-micro-cosine ties (for example, the Top-1 table labels more exact
items than the saved exact-rank table assigns rank 1). Ranked metrics retain the
published rank table as source of truth; reciprocal-set construction retains
the published neighbor identities. Consequently mutual-kNN k=1 set recall is
not numerically interchangeable with cosine R@1. This is a useful warning about
precision-sensitive evaluation in this embedding space, not evidence that a
one-item set can outperform its originating total ranking.

## Density-stratified behavior and biological preservation

For dense queries, R@1 rises from 17.65% with cosine to 24.60% with CSLS k=10,
22.46% with local scaling, and 21.93% with NICDM; their dense R@10 values rise
from 74.33% to 79.68%, 80.75%, and 82.89%. The improvement is not uniform:
CSLS k=10 reduces intermediate-stratum R@1 from 43.32% to 39.57%, while local
scaling preserves 43.32% and NICDM reaches 43.85%. In sparse queries, changes
are small.

Exact-GSM gains trade against biological-neighborhood measures. NICDM gains
1.96 percentage points at R@1 and 5.34 points at R@10, but same-GSE R@1 falls
1.96 points and exact normalized source agreement falls 0.71 points. Local
scaling offers a slightly gentler tradeoff. No disease, condition, treatment,
or replicate label was inferred from free text; “source” means exact normalized
`source_name_ch1` agreement only.

## Equations and calibration

For cosine `s(x,y)`, CSLS is `2s(x,y) - r_T(x) - r_S(y)`, where each `r` is
the mean similarity to the k nearest samples in the opposite domain. Local
scaling follows Zelnik-Manor and Perona: with cosine distance `d=1-s` and each
endpoint's kth-neighbor distance `sigma`, rank by
`exp(-d(x,y)^2/(sigma_x sigma_y))`. NICDM follows Jégou et al.: rank by
`d(x,y)/sqrt(mu_x mu_y)`, where each `mu` is mean distance to k cross-domain
neighbors. Reciprocal kNN retains query/reference edges present in both
cross-domain kNN sets. Cross-domain SNN uses Jaccard overlap between a query's
k nearest query contexts (including itself) and a reference's k nearest query
contexts.

The empirical null is the deterministic two-million-pair ARCHS4 cosine sample.
P-values use `(number(null >= observed)+1)/(N+1)`; calibration is global, and
Bonferroni correction covers 510,709 reference tests. This is
CellAtlasSearch-inspired significance calibration, not an exact reproduction.

## Runtime, scaling, diagnostics, and conclusion

The complete cached run takes about two minutes on one RTX 3090. Exact
query-to-corpus cosine takes about 8 seconds; method scoring takes roughly 1–13
seconds/method; measured peak RSS is about 20 GB. The float32 query/corpus
cache is ~1.15 GB. Costs scale linearly in queries and references for the
ranked methods; the reference-density probe avoids quadratic corpus storage.
The full timings are in `results/runtime.csv`.

Rank changes are selected objectively (ten largest gains, ten largest losses,
unchanged cases, and dense failures per method) in
`results/rank_change_diagnostics.csv`, including GSM/GSE, both ranks, density,
both Top-1 samples, source labels, same-GSE flags, and cosine hubness. No cases
were hand-picked.

**Conclusion by retrieval objective.** For technical identity, NICDM is the
best evaluated primary metric (R@1 30.43%, R@10 82.56%), with local scaling a
close alternative. For within-corpus biological-source similarity, cosine is
clearly preferable: it has substantially higher source P@1/P@10 and tissue
P@1/P@10 after the paired GSM is removed. For cross-study scientific search,
no single winner is established: exact source labels do not transfer across
submitters, while the limited blood/brain tissue sensitivity analysis favors
CSLS, then local scaling/NICDM, over cosine. A practical interface should expose
cosine biological neighbors and an optional locally normalized, study-diverse
view rather than collapse these objectives into one score.

There is no remaining evidence here that representation correction is
warranted. The apparent dense-query failure excess largely disappears when the
dominant peripheral-blood study is removed, and existing metadata are too weak
to define a trustworthy pan-biological training objective. Better curated,
study-spanning tissue/cell-state labels are needed before reconsidering an
adapter or autoencoder. Stage 2–4 work remains unimplemented pending explicit
approval.

## ARCHS4 anisotropy and common-component diagnostic

This extension fits the mean and PCA basis on all 510,709 frozen human ARCHS4
reference embeddings only, then applies that unchanged transformation to the
562 recount3 queries. It reuses the exact deterministic two-million-pair Stage
0 sample; the original seed reproduces the retained raw cosines within
5.4e-7, and the reconstructed pair indices are saved under ignored `work/`.
Two million pairs provide about 200 observations at P99.99, so a ten-million
expansion was not needed. Every distribution below is a sampled estimate, not
an exhaustive calculation over roughly 130 billion corpus pairs.

### Why raw cosines are so high

The frozen space is strongly anisotropic. The raw global mean vector has norm
9.9909, while the mean/median sample norms are 10.6307/10.1919; the mean vector
is therefore 94.0% as long as a typical embedding. The mean of unit sample
directions has norm **0.93743**, whose square is **0.878774**. That squared norm
is the expected cosine of two independently sampled unit vectors and nearly
exactly equals the observed random-pair mean **0.878748**. Thus the high center
is quantitatively explained by most samples sharing a large common direction.

The median sample cosine to the unit global-mean direction is 0.95945 (mean
0.93743; SD 0.07973). Although rare outliers point elsewhere, most of the
510,709 embeddings lie in a narrow cone. This is not a numerical accident or
evidence that a random pair is biologically equivalent.

The retained raw random-pair distribution is:

| Statistic | Raw cosine |
|---|---:|
| N | 2,000,000 |
| Mean / SD | 0.878748 / 0.118144 |
| Minimum / maximum | −0.443058 / 0.999929 |
| P0.01 / P0.1 / P1 | −0.273698 / −0.106106 / 0.354765 |
| P5 / P10 / P25 | 0.673433 / 0.769935 / 0.862537 |
| Median / P75 | 0.912954 / 0.941536 |
| P90 / P95 / P99 | 0.958964 / 0.967828 / 0.984401 |
| P99.9 / P99.99 | 0.996779 / 0.999057 |
| IQR | 0.078999 |

For interpretation, cosine 0.90 is only the 41.42nd random-pair percentile;
0.95 is the 82.87th; 0.99, 0.995, and 0.999 are the 99.508th, 99.813th, and
99.988th percentiles.

### Centering and dominant components

Subtracting the ARCHS4 mean and renormalizing moves the random-pair mean from
0.87875 to **0.00645** and median from 0.91295 to **−0.00247**. The common mean
direction is therefore the primary explanation for the raw cosine center.
Centering exposes a highly structured residual space rather than an isotropic
512-dimensional cloud: centered PC1 explains 57.66% of variance, PC2 7.78%,
and the first 5/10/20/50 PCs explain 74.96%/82.96%/91.12%/98.20%. The covariance
participation ratio is only **2.91 effective dimensions**. PC1 and the raw mean
direction have absolute cosine 0.723; they overlap substantially but are not
the same object.

| Representation | Random mean | Median | SD | P5 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.878748 | 0.912954 | 0.118144 | 0.673433 | 0.967828 | 0.984401 |
| Mean-centered | 0.006452 | −0.002471 | 0.406919 | −0.655725 | 0.692437 | 0.863638 |
| Centered − PC1 | 0.004760 | −0.025625 | 0.246451 | −0.338807 | 0.466083 | 0.788729 |
| Centered − PCs 1–2 | 0.002777 | −0.021669 | 0.208268 | −0.277116 | 0.375084 | 0.761399 |
| Centered − PCs 1–5 | 0.001718 | −0.019016 | 0.187970 | −0.253360 | 0.322849 | 0.685160 |
| Centered − PCs 1–10 | 0.001168 | −0.011327 | 0.180508 | −0.258152 | 0.289961 | 0.644133 |

Removing PC1 chiefly reduces residual dispersion; subsequent PCs make smaller,
non-monotone changes to the tails. “More components removed” is not inherently
better, so retrieval and biological preservation remain necessary checks.

### Exact-pair separation

| Representation | Exact mean | Median | SD | Q1 | Q3 | Exact-median random percentile | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.997806 | 0.999218 | 0.003510 | 0.998496 | 0.999460 | 99.9936% | 0.99910 |
| Mean-centered | 0.985264 | 0.990891 | 0.024900 | 0.986773 | 0.994546 | 99.9862% | 0.99941 |
| Centered − PC1 | 0.966366 | 0.985575 | 0.059921 | 0.978018 | 0.988811 | 99.9716% | 0.99865 |
| Centered − PCs 1–2 | 0.963784 | 0.984695 | 0.061349 | 0.973656 | 0.989805 | 99.9703% | 0.99872 |
| Centered − PCs 1–5 | 0.952363 | 0.980446 | 0.073389 | 0.951959 | 0.991121 | 99.9730% | 0.99882 |
| Centered − PCs 1–10 | 0.951167 | 0.979425 | 0.069584 | 0.945899 | 0.989798 | 99.9799% | 0.99906 |

Absolute exact-pair cosines decrease after removing shared components, as they
must, but separation from ordinary pairs remains extreme. Mean-centering gives
the highest AUROC, while PC removal increases pooled standardized effect size
by narrowing the random distribution. These statistics answer different
questions and do not by themselves select a retrieval representation.

### Exact retrieval and comparison with local corrections

| Method / representation | R@1 | R@10 | R@100 | R@1,000 | MRR | Macro R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw cosine | 28.47% | 77.22% | 91.81% | 97.51% | 0.4764 | 75.92% |
| Mean-centered | 24.73% | 74.56% | 92.35% | 97.33% | 0.4467 | 66.55% |
| Centered − PC1 | 28.11% | 79.00% | 92.35% | 97.86% | 0.4769 | 79.22% |
| Centered − PCs 1–2 | 27.05% | 76.87% | 90.93% | 97.69% | 0.4733 | 80.37% |
| Centered − PCs 1–5 | 34.16% | 79.36% | 91.10% | 97.69% | 0.5356 | 77.15% |
| **Centered − PCs 1–10** | **37.19%** | **85.41%** | **96.26%** | **98.93%** | **0.5859** | **84.46%** |
| Local scaling k=10 | 30.25% | 81.67% | 94.31% | 98.58% | 0.5119 | 78.66% |
| NICDM k=10 | 30.43% | 82.56% | 94.31% | 98.58% | 0.5125 | 78.96% |
| CSLS k=10 | 29.36% | 80.60% | 94.13% | 98.04% | 0.5039 | 66.01% |

Mean-centering alone degrades R@1, R@10, MRR, and macro performance. Removing
ten centered PCs, however, substantially outperforms raw cosine and every local
correction tested. The benefit is therefore not equivalent to merely removing
the mean, and NICDM/local scaling only partly address the same neighborhood
geometry. Removing one, two, or five PCs is inconsistent; PC10 is an empirical
diagnostic result, not yet a universal hyperparameter.

### Biological neighborhoods, density, and hubness

| Representation | Same-GSE Top-1 | Same-GSE Top-10 | Source P@1 | Source P@10 | Tissue P@1 | Tissue P@10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 76.16% | 91.10% | 66.43% | 49.51% | 82.10% | 68.83% |
| Mean-centered | 75.27% | 88.97% | 67.13% | 48.67% | 83.33% | 68.70% |
| Centered − PC1 | 79.00% | 90.04% | 73.43% | 54.20% | 90.74% | 76.54% |
| Centered − PCs 1–2 | 79.18% | 89.15% | 72.03% | 51.57% | 85.19% | 70.49% |
| Centered − PCs 1–5 | 79.18% | 88.97% | 70.28% | 52.90% | 88.27% | 78.02% |
| Centered − PCs 1–10 | **84.34%** | **92.35%** | **73.78%** | **56.57%** | 88.89% | **80.86%** |

The PC10 representation does not obtain its identity gain by destroying the
available biological neighborhoods: all reported same-study, source, and
tissue metrics improve over raw cosine. Tissue remains the previously noted
blood/brain-heavy sensitivity task, and exact source names remain heterogeneous;
these results do not establish general cell-state preservation.

Hubness changes are mixed. At Top-1, PC10 selects 436 distinct references
versus 396 for raw, lowers maximum N1 from 16 to 12, and lowers the ten-largest-
hub share from 14.06% to 10.85%. At Top-10, however, it selects fewer distinct
references (1,732 versus 1,904), raises maximum N10 from 35 to 59, and raises
the top-ten share from 4.88% to 6.48%. It improves Top-1 concentration but does
not uniformly remove hubness.

Raw query Top-10 mean cosine has SD 0.00220; centering and PC removal expand
rather than shrink that numerical variation because the compressed cosine
baseline is removed. GSE234297 remains denser than other studies through PC2
(raw difference +0.0020; centered +0.0141; PC1 +0.0265), but becomes less dense
after PC5/PC10 (−0.0088/−0.0192). Thus common components partly encode the
peripheral-blood density anomaly, but density is redistributed rather than
uniformly erased.

### Recommendation for the next biological-retrieval benchmark

For absolute cross-pipeline representation agreement, continue to report raw
Bridge cosine: it retains the most interpretable connection to the frozen model.
For exact technical-identity retrieval, advance **reference-fitted centered
cosine with PCs 1–10 removed** as the leading diagnostic representation,
alongside raw cosine and NICDM/local scaling baselines. For biological search,
do not replace raw cosine universally yet: PC10 improved the available source
and limited tissue labels, but it was selected on this paired cohort and needs
independent validation.

Accordingly, a future GTEx→ARCHS4 benchmark should prespecify and freeze three
views before evaluation: raw cosine; centered-minus-PC10 fitted on ARCHS4 only;
and the best retrieval-time local correction (NICDM or local scaling). It should
compare biological and cross-study objectives rather than tune component count
on GTEx. No adapter, autoencoder, or Bridge fine-tuning is justified or was run.

## Literature-grounded post-processing comparison

This extension treats the previously selected centered-minus-PC10 transform as
frozen **ABTT D=10**; no additional component counts were tried. The
implementation follows [Mu and Viswanath (ICLR 2018)](https://openreview.net/forum?id=HkuGJ3kCb):
subtract the ARCHS4 reference mean and remove projections onto the top ten
ARCHS4-derived centered principal components. Full-dimensional whitening
follows [Su et al. (2021)](https://arxiv.org/abs/2103.15316):
`(x − μ) U diag(λ^-1/2)`, where `μ` and the maximum-likelihood covariance
`Σ=(1/N)Σ(x−μ)(x−μ)^T=U diag(λ)U^T` use the 510,709 ARCHS4 references only.
All 512 dimensions are retained and transformed vectors are L2-normalized.
The prespecified numerical floor is `λmax × 10^-12`; all eigenvalues exceeded
it, so no dimension was regularized or discarded.

Alignment and uniformity use the definitions from
[Wang and Isola (ICML 2020)](https://proceedings.mlr.press/v119/wang20k.html),
used diagnostically in [SimCSE](https://arxiv.org/abs/2104.08821): alignment is
`E||f(x)−f(x+)||²` for the 562 exact pairs, and uniformity is
`log E exp(−2||f(x)−f(y)||²)` over the fixed two-million ARCHS4 random pairs.
Lower alignment means closer exact pairs; more-negative uniformity means more
uniform sphere coverage. Neither alone defines useful biological retrieval.

### Geometry, alignment, and uniformity

| Representation | Random mean | Median | SD | Exact mean | Exact median | Alignment | Uniformity | Exact-pair AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw cosine | 0.878748 | 0.912954 | 0.118144 | 0.997806 | 0.999218 | **0.004388** | −0.413519 | 0.999096 |
| Mean-centered | 0.006452 | −0.002471 | 0.406919 | 0.985264 | 0.990891 | 0.029471 | −2.773730 | 0.999411 |
| ABTT D=10 | 0.001168 | −0.011327 | 0.180508 | 0.951167 | 0.979425 | 0.097665 | −3.606502 | 0.999055 |
| Whitening | 0.000232 | −0.002371 | **0.047161** | 0.791504 | 0.852378 | 0.416992 | **−3.972797** | **0.999781** |

Whitening nearly realizes the expected isotropic 512-dimensional sphere:
normalized-reference participation ratio rises to 446.87 dimensions, versus
5.30 raw, 6.03 centered, and 30.67 after ABTT. Before L2 normalization, its
covariance eigenvalues span only 0.9953–1.0062 and participation ratio is
511.999, versus a pre-whitening condition number of 2.97 million and
participation ratio 2.91. Whitening substantially lowers absolute exact-pair
cosines and worsens alignment, but ordinary-pair cosines contract much more;
consequently its exact-pair AUROC and empirical tail separation are strongest.
This is a preservation–uniformity tradeoff, not evidence that lower exact
cosine is intrinsically desirable.

### Exact retrieval: micro and equal-study macro

| Method | Type | R@1 | R@10 | R@100 | R@1,000 | MRR | Macro R@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw cosine | Representation | 28.47% | 77.22% | 91.81% | 97.51% | 0.4764 | 75.92% |
| Mean-centered | Representation | 24.73% | 74.56% | 92.35% | 97.33% | 0.4467 | 66.55% |
| ABTT D=10 | Representation | 37.19% | 85.41% | 96.26% | 98.93% | 0.5859 | 84.46% |
| **Whitening** | Representation | **46.62%** | **94.48%** | **99.11%** | **100.00%** | **0.6919** | **94.12%** |
| NICDM | Retrieval score | 30.43% | 82.56% | 94.31% | 98.58% | 0.5125 | 78.96% |
| Local Scaling | Retrieval score | 30.25% | 81.67% | 94.31% | 98.58% | 0.5119 | 78.66% |
| CSLS | Retrieval score | 29.36% | 80.60% | 94.13% | 98.04% | 0.5039 | 66.01% |

Study-macro results agree with the micro ranking. Whitening macro R@1/R@5/
R@10/R@100/R@1,000/MRR are 37.66%/92.43%/94.12%/97.92%/100%/0.6457,
compared with 35.41%/80.24%/84.46%/92.94%/97.50%/0.5746 for ABTT D=10.
Thus the improvement is not produced solely by the largest studies. ABTT
reproduces the previously frozen PC1–10 result exactly, while whitening clearly
outperforms it for this technical-identity objective without recount3 tuning.

### Biological-neighborhood preservation

| Method | Same-GSE Top-1 | Same-GSE Top-10 | Source P@1 | Source P@10 | Tissue P@1 | Tissue P@10 |
|---|---:|---:|---:|---:|---:|---:|
| Raw cosine | 76.16% | 91.10% | 66.43% | 49.51% | 82.10% | 68.83% |
| ABTT D=10 | 84.34% | 92.35% | **73.78%** | **56.57%** | **88.89%** | 80.86% |
| Whitening | **90.93%** | **97.51%** | 66.78% | 54.09% | 88.27% | **84.14%** |
| NICDM | 74.20% | 89.50% | 56.29% | 36.82% | 76.54% | 60.37% |
| Local Scaling | 74.73% | 89.32% | 57.34% | 38.53% | 74.69% | 62.16% |
| CSLS | 71.89% | 87.72% | 53.85% | 37.24% | 75.31% | 62.84% |

Whitening strongly preserves same-study neighborhoods and improves limited
tissue P@10, but it returns source P@1 essentially to the raw baseline and is
below ABTT on source P@1/P@10. ABTT offers the most consistent improvement
across the conservative source task. The tissue task remains heavily
blood/brain dominated and is sensitivity evidence, not a pan-tissue result.
Retrieval-score corrections improve exact identity modestly while reducing all
three available biological-neighborhood summaries relative to ABTT/whitening.

### Hubness

Whitening gives the least concentrated Top-1 assignments: 506 distinct
references, maximum N1=8, and a 7.47% share for the ten largest hubs, versus
396/16/14.06% raw and 436/12/10.85% ABTT. NICDM and local scaling also reduce
Top-1 concentration (440/8/9.07% and 426/9/9.96%). At Top-10, NICDM and local
scaling are strongest: their ten-hub shares are 2.86% and 2.79%, below raw
4.86%. ABTT and whitening instead raise Top-10 concentration to 6.48% and
6.39%, with maximum N10=59 and 71. Whitening therefore spreads first choices
widely but creates repeated references deeper in the neighborhood; no method
uniformly minimizes every hubness diagnostic.

### Interpretation and configurations frozen before GTEx

The high raw cosine center is primarily anisotropic geometry, not failed
discrimination: removing the common direction centers ordinary-pair cosine,
and whitening produces nearly isotropic covariance while retaining AUROC
0.99978 for exact versus random pairs. Rescaling all directions performs better
than discarding the top ten for cross-pipeline identity, while ABTT retains the
best conservative source-label results. Local corrections remain useful when
low Top-10 hub concentration matters, but do not explain whitening's larger
identity gain.

The exact candidates frozen before any GTEx inspection are:

- **Raw cosine:** L2-normalize original Bridge embeddings and rank by cosine.
- **ABTT D=10:** subtract the ARCHS4 mean, remove ARCHS4 centered PCs 1–10,
  L2-normalize, and rank by cosine. D=10 is frozen; no further D is allowed.
- **Whitening:** the full 512-dimensional ARCHS4-only `1/N` covariance transform
  specified above, relative eigenvalue floor `10^-12` (inactive here), followed
  by L2 normalization and cosine.
- **NICDM:** the already established k=10 configuration.
- **Local Scaling:** the already established k=10 configuration.

Mean-centering remains a diagnostic and CSLS k=10 remains supplemental. For
GTEx biological search, advance raw cosine for continuity and interpretability,
ABTT as the source-preserving post-processing candidate, whitening as the
strongest technical-identity and isotropy candidate, and NICDM/local scaling as
retrieval-time comparators. GTEx was not inspected, no post-processing choice
was tuned on it, and no Bridge parameters were changed.

## Independent GTEx → ARCHS4 biological retrieval

### Assets, cohort, annotation, and leakage

This stage reused the existing GTEx v11 cache from the mouse-ENCODE benchmark:
6,314 samples, 937 donors, 17 analysis tissues, and 18 authoritative detailed
tissue/site labels. The cache used GENCODE v49 counts converted to TPM, the
canonical 15,165 Bridge genes, natural `log1p`, and the frozen
`model/r7hnr92k/best_model.pt` checkpoint. Sample order matches its preserved
manifest; no inference or download was needed. Heart is the largest query
tissue (823/6,314, 13.03%), so no single tissue dominates.

Labels were harmonized before scoring. Four brain regions map to `brain`;
mammary gland to `breast`, subcutaneous adipose to `adipose`, sigmoid colon to
`colon`, and skeletal muscle to `skeletal_muscle`. Eligibility required at
least 100 labeled references from at least five GSEs. Adrenal failed (19
references, four GSEs), leaving **6,064 queries across 13 tissues**.

The ARCHS4 audit found 73,338 confidently labeled references among 510,376
post-exclusion references (**14.37% coverage**); 437,038 remain unknown. Blood
is 65.21% of labeled references. Labels represented by GTEx account for only
19,526 references, or **3.83% of the full corpus**. Annotation coverage—not
query composition—is the main limitation.

Metadata identified 333 references with explicit `GTEX-*`, “GTEx,” or
Genotype-Tissue Expression provenance. All were excluded uniformly. No direct
GSM/sample-ID overlap exists; the exclusion is broader than exact matching.
Unmarked derivatives cannot be ruled out.

### Micro and tissue-macro retrieval

Annotated precision conditions on a retrieved neighbor having a confident
tissue label. Hit@K counts confirmed same-tissue hits among all K neighbors;
Unknown@K reports the unlabeled fraction.

| Method | Annotated P@10 | Hit@10 | Unknown@10 | Tissue-macro P@10 | Macro enrichment |
|---|---:|---:|---:|---:|---:|
| Raw cosine | 95.15% | **26.38%** | 72.27% | 86.73% | 136.60× |
| ABTT D=10 | **95.39%** | 22.86% | 76.03% | 90.60% | 141.13× |
| Whitening | 94.97% | 25.16% | 73.50% | **91.21%** | **147.41×** |
| NICDM k=10 | 92.29% | 23.72% | 74.30% | 80.61% | 119.52× |
| Local Scaling k=10 | 92.46% | 24.01% | 74.03% | 81.20% | 121.68× |

Matching tissues are retrieved very reliably **when neighbors are annotated**,
with enrichment far above prevalence. Only 22.9–26.4% of all Top-10 positions
are confirmed hits because roughly three quarters are unknown. ABTT narrowly
leads micro annotated precision, raw leads absolute hit yield, and whitening
leads the primary imbalance-resistant tissue macro and enrichment metrics.

### Per-tissue behavior and confusions

Brain and liver attain 100% annotated P@10 for all methods. Raw cosine is also
strong for skeletal muscle (99.88%), lung (99.77%), heart (99.13%), testis
(99.41%), and ovary (96.45%). Its weakest tissues are stomach (52.00%), adipose
(55.63%), spleen (73.33%), and breast (73.80%).

Whitening most improves stomach (+24.32 points), adipose (+19.61), and pancreas
(+14.87), while reducing breast (−7.62), spleen (−2.01), and skeletal muscle
(−0.94). ABTT most improves adipose (+36.86), pancreas (+14.87), and stomach
(+13.67), but harms breast (−13.47) and spleen (−10.36). NICDM/local scaling
materially reduce pancreas, spleen, and ovary precision.

Frequent unlabeled Top-10 strings include `RNA from fusiform gyrus`, `Human
normal breast tissue`, `vastus lateralis muscle`, `Colon wall`, and
`subcutaneous adipose tissue (SAT)`. They are retained as future annotation
opportunities and were not added post hoc.

Dominant labeled confusions are adipose↔breast, pancreas↔stomach under raw, and
occasional stomach↔colon. A spleen→skeletal-muscle confusion emerges after
ABTT/whitening and is not assumed to be biologically meaningful. Whitening
reduces most other off-diagonal counts.

### Local density and cosine context

Raw mean Top-10 cosine ranges from 0.9873 for testis to 0.9993 for liver. Brain
is dense (0.99795) but heterogeneous across merged regions (SD 0.00158); testis
has the largest raw within-tissue SD (0.00219). Whitening does not reduce
between-tissue density differences: mean Top-10 score ranges from 0.650 for
spleen to 0.868 for pancreas. It removes the common baseline while exposing
tissue-dependent geometry. The cached cohort lacks Whole Blood, so this stage
cannot independently retest blood density.

Among raw Top-100 labeled neighbors, same-tissue cosine has median 0.99754 and
different-tissue median 0.99558; unknown-tissue median is 0.99613. These fall
near the 99.94th, 99.84th, and 99.87th percentiles of fixed random ARCHS4 pairs.
Correct and incorrect neighbors both occupy the dense tail, so arbitrary
thresholds such as 0.9 remain uninformative.

### GTEx → GTEx control

After excluding self and every sample from the same donor, tissue-macro P@10 is
90.61% raw, 93.35% ABTT, and **94.86% whitening**. Micro P@10 is 92.28%, 94.39%,
and **95.65%**. The standardized-dataset control agrees with the independent
ARCHS4 macro ordering and demonstrates the effect of ARCHS4 label missingness.

### Technical identity versus biology

| Method | recount3 exact R@10 | recount3 macro R@10 | GTEx annotated P@10 | GTEx macro P@10 | GTEx Hit@10 |
|---|---:|---:|---:|---:|---:|
| Raw | 77.22% | 75.92% | 95.15% | 86.73% | **26.38%** |
| ABTT D=10 | 85.41% | 84.46% | **95.39%** | 90.60% | 22.86% |
| Whitening | **94.48%** | **94.12%** | 94.97% | **91.21%** | 25.16% |
| NICDM | 82.56% | 78.96% | 92.29% | 80.61% | 23.72% |
| Local Scaling | 81.67% | 78.66% | 92.46% | 81.20% | 24.01% |

Whitening's identity advantage generalizes to tissue-macro retrieval and the
GTEx control, but it is not universally best. ABTT does not consistently
preserve broad tissue biology better than whitening. NICDM and local scaling
do not help this tissue task.

### Recommendation before cross-species work

- **Exact identity:** full 512-D ARCHS4-fitted whitening remains strongest.
- **Within-species tissue similarity:** use whitening as the primary
  tissue-macro representation, raw cosine for absolute-score continuity and
  hit coverage, and ABTT D=10 as a prespecified sensitivity analysis.
- **Future cross-species retrieval:** advance raw, ABTT D=10, and whitening as
  distinct frozen geometries. Do not select only whitening from human-only
  results. NICDM/local scaling may remain baselines but are not primary.

No GTEx parameter was fitted, no method was tuned, and no cross-species
retrieval was started.

## Retrospective representation × metric controls

This control factors representation, metric, and covariance normalization on
the unchanged 562 recount3 queries and 510,709 human ARCHS4 references. It did
not rerun or alter GTEx. The full expression matrix was reconstructed from the
existing ARCHS4 count HDF5 using the upstream exon-union-length TPM and natural
`log1p` procedure in the canonical 15,165-gene order. The reconstructed exact
counterpart rows reproduce the upstream paired matrix exactly (maximum absolute
difference 0). Four damaged HDF5 source columns were encountered in one block;
none was in the fixed reference manifest, so that block was read using only
eligible reference columns. No benchmark reference was dropped or imputed.

PCA was fit on the 510,709 ARCHS4 references only using a deterministic
randomized 512-dimensional range finder (32 oversamples, one power iteration)
and an exact Rayleigh–Ritz diagonalization inside the retained subspace. The
512 PCs explain 87.76% of total reference expression variance. Standard PCA
whitening retains all dimensions; no eigenvalue meets the fixed
`lambda_max × 1e-12` floor. Its verified covariance has unit diagonal and
maximum absolute off-diagonal (7.02\times10^{-14}). Recount3 was used only
after these transformations were frozen.

### Full factorial retrieval result

| Representation | Metric | Dim | R@1 | R@10 | R@100 | R@1,000 | Macro R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| log1p(TPM) | Cosine | 15,165 | 57.65% | 95.37% | 96.80% | 97.69% | 89.15% | 0.7625 |
| log1p(TPM) | Pearson | 15,165 | 51.78% | 96.09% | 97.51% | 98.75% | 91.33% | 0.7374 |
| log1p(TPM) | Euclidean | 15,165 | 46.62% | 93.42% | 97.86% | 98.93% | 90.23% | 0.6980 |
| log1p(TPM) | Manhattan | 15,165 | 77.94% | 95.02% | 97.33% | 98.93% | 93.79% | 0.8627 |
| log1p(TPM) | Canberra | 15,165 | **78.29%** | **98.04%** | **98.93%** | 98.93% | **95.72%** | **0.8795** |
| PCA-512 | Cosine | 512 | 49.11% | 91.64% | 95.91% | 98.93% | 85.46% | 0.6891 |
| PCA-512 | Euclidean | 512 | 38.79% | 91.64% | 94.84% | 98.93% | 84.95% | 0.6330 |
| PCA-512 | Manhattan | 512 | 71.53% | 92.35% | 97.33% | 98.75% | 92.73% | 0.8122 |
| PCA-512 | Canberra | 512 | 66.01% | 92.70% | 98.58% | 99.82% | 93.77% | 0.7811 |
| PCA-512 whitened | Cosine | 512 | 41.99% | 92.17% | 98.40% | 100.00% | 93.62% | 0.6595 |
| PCA-512 whitened | Euclidean | 512 | 41.81% | 91.64% | 96.26% | 98.04% | 93.57% | 0.6558 |
| PCA-512 whitened | Manhattan | 512 | 39.50% | 91.99% | 96.44% | 98.04% | 93.60% | 0.6474 |
| PCA-512 whitened | Canberra | 512 | 40.93% | 92.70% | 98.58% | 99.82% | 93.77% | 0.6499 |
| Bridge | Cosine | 512 | 28.47% | 77.22% | 91.81% | 97.51% | 75.92% | 0.4764 |
| Bridge | Euclidean | 512 | 23.84% | 77.05% | 92.53% | 97.69% | 69.29% | 0.4497 |
| Bridge | Manhattan | 512 | 40.57% | 75.98% | 90.39% | 97.69% | 66.66% | 0.5362 |
| Bridge | Canberra | 512 | 37.19% | 71.00% | 86.83% | 96.80% | 65.55% | 0.4973 |
| Bridge whitened | Cosine | 512 | 46.62% | **94.48%** | **99.11%** | 100.00% | **94.12%** | 0.6919 |
| Bridge whitened | Euclidean | 512 | 51.78% | 93.24% | 96.80% | 97.51% | 91.59% | 0.7100 |
| Bridge whitened | Manhattan | 512 | **65.84%** | 93.77% | 97.69% | 98.40% | 91.64% | **0.7846** |
| Bridge whitened | Canberra | 512 | 65.30% | 93.06% | 98.75% | 100.00% | 92.57% | 0.7767 |
| ABTT D=10 | Cosine | 502 | 37.19% | 85.41% | 96.26% | 98.93% | 84.46% | 0.5859 |
| NICDM k=10 | Native | 512 | 30.43% | 82.56% | 94.31% | 98.58% | 78.96% | 0.5125 |
| Local Scaling k=10 | Native | 512 | 30.25% | 81.67% | 94.31% | 98.58% | 78.66% | 0.5119 |
| CSLS k=10 | Native | 512 | 29.36% | 80.60% | 94.13% | 98.04% | 66.01% | 0.5039 |

Micro metrics weight all 562 samples equally. Macro metrics first calculate
within each of 40 GSEs and then weight every study equally. The full macro
R@1/R@5/R@10/R@100/R@1,000/MRR table and per-query ranks are saved alongside
this report.

### What drives performance?

- **Raw expression metric:** Canberra is best for identity retrieval (R@1
  78.29%, R@10 98.04%, macro R@10 95.72%). Pearson is second at R@10
  96.09%, while Manhattan is nearly tied at R@10 and has R@1 77.94%.
- **Raw Bridge metric:** cosine remains best at R@10 (77.22%). Euclidean is
  essentially tied (77.05%); Manhattan and Canberra reduce R@10. Metric choice
  alone therefore does not explain raw Bridge's limitation.
- **PCA compression:** no PCA-512 metric matches the best full expression
  control. Canberra leads PCA at R@10 92.70%, 5.34 points below full-expression
  Canberra. Compression nevertheless cuts dimensionality 29.62-fold.
- **PCA whitening:** it raises PCA cosine R@10 by only 0.53 point and does not
  improve the best PCA R@10 (Canberra remains 92.70%). It changes equal-study
  behavior more than micro performance: macro cosine R@10 rises from 85.46%
  to 93.62%.
- **Bridge whitening:** cosine R@10 rises 17.26 points over raw Bridge cosine
  (77.22% to 94.48%). No alternate raw-Bridge metric approaches this. In the
  whitened space, cosine has the best R@10 and macro R@10, whereas Manhattan
  has better R@1 and MRR; metric objectives are not interchangeable.
- **Matched 512-D comparison:** Bridge-whitened cosine exceeds PCA-whitened
  cosine by 2.31 R@10 points and exceeds the best PCA-whitened metric by 1.78
  points. This is evidence for useful Bridge information beyond a matched-size
  linear PCA representation, although it is not evidence that Bridge beats the
  uncompressed input for exact identity.

Exact-versus-random AUROC is high for multiple controls and is not synonymous
with retrieval rank. For example, PCA-whitened cosine AUROC is 0.99983 and
Bridge-whitened cosine AUROC is 0.99978, yet their R@10 differs. This reiterates
that discrimination against ordinary random pairs is easier than uniqueness
inside the extreme nearest-neighbor tail.

### Compute and storage

Float32 expression requires 60,660 bytes/sample and a 30.98 GB full-reference
cache; PCA-512 and analyzed whitened Bridge require 2,048 bytes/sample and
about 1.05 GB. Stored float16 Bridge requires 1,024 bytes/sample. Thus Bridge
and PCA use 29.62-fold fewer dimensions than expression (59.24-fold fewer
bytes for stored float16 Bridge versus float32 expression). On the available
GPU, all five expression sweeps took 451 seconds, dominated by Canberra at 348
seconds; four-metric sweeps for each 512-D representation took 15–20 seconds.
The reference-only PCA preparation peaked at 32.5 GB RSS after the one-time
expression cache had been constructed.

### Direct attribution and biological context

For this exact cross-pipeline identity objective, **all of Bridge-whitened
R@10 = 94.48% is reproducible—and exceeded—by conventional uncompressed
expression geometry**: raw-expression Pearson reaches 96.09% and Canberra
98.04%. Therefore exact-identity performance alone must not be presented as
Bridge-specific representation learning. At matched 512-dimensional storage,
however, Bridge whitening retains a real 1.78–2.31 point R@10 advantage over
the corresponding PCA controls, while providing the same ~29.6-fold dimension
reduction relative to expression.

This retrospective result does not replace the completed biological benchmark.
The already-saved GTEx analyses showed that Bridge whitening generalized to
tissue-macro retrieval; raw expression and expression-PCA were not evaluated
on GTEx here. Consequently, the defensible conclusion is objective-specific:
ordinary expression geometry is strongest for exact paired-sample identity,
while compact whitened Bridge remains supported for biological search by the
independent frozen GTEx evidence. No GTEx computation, cross-species retrieval,
training, or representation tuning was performed.

## Retrospective GTEx representation controls

This extension asks whether the completed Bridge tissue-retrieval result
exceeds frozen conventional expression/PCA controls. It reuses, unchanged, the
6,064 eligible GTEx queries, 13 harmonized ARCHS4 tissues, conservative tissue
mapping, 333 GTEx-origin exclusions, and 510,376-reference corpus. The existing
raw Bridge, ABTT D=10, and Bridge-whitened Top-100 rankings and metrics remain
the source of truth. No method, dimensionality, mapping, threshold, or
transformation was selected after observing these results.

The GTEx input is the existing 6,314 × 15,165 float32 natural `log1p(TPM)`
matrix used to generate the frozen Bridge embeddings. Its sample order exactly
matches the GTEx manifest and its gene order exactly matches the canonical
Bridge vocabulary. The ARCHS4 PCA-512 basis and whitening parameters are the
reference-only transforms from the recount3 control; no GTEx sample contributes
to fitting.

### Identity versus biological retrieval

| Representation | Metric | Dim | recount3 exact R@10 | GTEx→ARCHS4 macro P@10 | GTEx→GTEx macro P@10 |
|---|---|---:|---:|---:|---:|
| Expression | Canberra | 15,165 | **98.04%** | 85.80% | 95.32% |
| Expression | Pearson | 15,165 | 96.09% | 90.11% | **96.12%** |
| PCA-512 | Canberra | 512 | 92.70% | 91.59% | 95.29% |
| PCA-white | Cosine | 512 | 92.17% | **92.08%** | 95.42% |
| PCA-white | Canberra | 512 | 92.70% | 91.59% | 95.29% |
| Bridge | Cosine | 512 | 77.22% | 86.73% | 90.61% |
| Bridge ABTT | Cosine | 502 | 85.41% | 90.60% | 93.35% |
| Bridge-white | Cosine | 512 | 94.48% | 91.21% | 94.86% |

Full expression remains best for exact cross-pipeline identity. It does not,
however, dominate the independent ARCHS4 tissue task: expression Canberra
falls to 85.80% tissue-macro P@10 and Pearson reaches 90.11%. The best result is
PCA-whitened cosine at 92.08%, 0.87 point above Bridge whitening. PCA Canberra
also exceeds Bridge whitening by 0.38 point. PCA and PCA-whitened Canberra have
identical rankings because positive coordinate-wise whitening scales cancel
exactly in the Canberra numerator/denominator ratio.

The fully labeled different-donor GTEx control is even clearer. Expression
Pearson leads at 96.12% macro P@10, followed by PCA-whitened cosine (95.42%),
expression Canberra (95.32%), PCA Canberra (95.29%), and Bridge whitening
(94.86%). All frozen conventional controls outperform raw Bridge and ABTT.

### GTEx→ARCHS4 micro and tissue-macro results

| Method | Micro annotated P@10 | Macro annotated P@10 | Macro Hit@10 | Macro Unknown@10 | Macro enrichment |
|---|---:|---:|---:|---:|---:|
| Expression Canberra | 92.76% | 85.80% | 24.64% | 72.99% | 130.2× |
| Expression Pearson | 93.34% | 90.11% | 23.12% | 74.82% | 127.8× |
| PCA-512 Canberra | **95.55%** | 91.59% | **25.35%** | 73.12% | 151.9× |
| PCA-white cosine | 94.93% | **92.08%** | 23.98% | 74.40% | **152.8×** |
| PCA-white Canberra | **95.55%** | 91.59% | **25.35%** | 73.12% | 151.9× |
| Raw Bridge cosine | 95.15% | 86.73% | 23.13% | 75.51% | 136.6× |
| Bridge ABTT D=10 | 95.39% | 90.60% | 22.49% | 76.08% | 141.1× |
| Bridge-white cosine | 94.97% | 91.21% | 24.77% | 73.61% | 147.4× |

Unknown-label rates remain 73–76%, so annotated precision and absolute hit
rate answer different questions. Raw Bridge has the highest sample-level
absolute Hit@10 (26.38%); expression Canberra is nearly tied (26.29%). The
imbalance-resistant macro result favors compact PCA rather than Bridge.

Across K, ABTT leads macro P@1 (93.37%), while PCA-white cosine leads P@10
(92.08%) and PCA Canberra leads P@100 (89.89%). There is no representation that
dominates every neighborhood depth.

### Per-tissue behavior and confusions

Brain and liver are perfect for every method; testis is essentially perfect.
PCA-white cosine leads or ties heart, lung, ovary, pancreas, stomach, and
testis. Expression Pearson is strongest for adipose (99.82%) and ties at 100%
for colon and pancreas. Raw Bridge is strongest for breast (73.80%), skeletal
muscle (99.88%), and spleen (73.33%). Representation sensitivity is largest for
stomach, breast, and adipose. Expression Pearson retrieves no confidently
labeled stomach neighbors at Top-10, so its stomach annotated precision is
undefined rather than zero; its absolute-hit/unknown metrics retain that fact.

The most frequent labeled expression errors are breast→adipose and
spleen→skeletal muscle. PCA reduces breast→adipose errors but increases the
reverse adipose→breast confusion. Raw Bridge instead has strong bidirectional
adipose↔breast confusion plus pancreas↔stomach; whitening removes much of the
pancreas/stomach error but introduces the same spleen→skeletal-muscle pattern
seen in expression/PCA. These are descriptive patterns, not post hoc tissue
redefinitions.

### Neighborhood overlap

Similar tissue accuracy does not imply identical retrieved samples. Relative
to Bridge-whitened Top-10 neighbors:

| Conventional method | Mean shared neighbors | Mean Jaccard | Zero overlap |
|---|---:|---:|---:|
| Expression Canberra | 4.01 / 10 | 0.281 | 7.34% |
| Expression Pearson | 4.59 / 10 | 0.332 | 4.98% |
| PCA-512 Canberra | 4.37 / 10 | 0.309 | 4.50% |

Overlap is lower when Bridge whitening's Top-1 is confidently wrong: mean
shared counts fall to 2.28, 3.20, and 3.26 respectively. Thus the methods often
reach the same tissue-level conclusion through materially different samples,
especially in failure cases.

### Compression and runtime

Expression uses 15,165 dimensions and 60,660 bytes/sample in float32, versus
512 dimensions and 2,048 bytes/sample for float32 PCA, and 1,024 bytes/sample
for stored float16 Bridge. This is 29.62-fold dimensional compression and up to
59.24-fold storage compression. For 510,376 references, the corresponding
vector stores are approximately 30.96 GB, 1.05 GB, and 0.52 GB.

On the available GPU, GTEx→ARCHS4 retrieval took 659 seconds for expression
Canberra, 42 seconds for expression Pearson, 50 seconds for PCA Canberra, 39
seconds for PCA-white cosine, and 46 seconds for PCA-white Canberra. Canberra
used a fused CUDA implementation whose real-data maximum absolute difference
from the established eager calculation was 0.000488 over 15,165 genes.

### Biological-value conclusion

The result does **not** support a general Bridge advantage over conventional
expression/PCA for tissue retrieval. Full expression is superior for exact
identity, expression Pearson is strongest in the fully labeled different-donor
control, and matched-dimensional PCA slightly exceeds Bridge whitening on the
primary GTEx→ARCHS4 tissue-macro endpoint. Therefore much of the observed broad
tissue organization is explainable by compact linear transcriptomic structure.

Bridge still provides practical and representational value: whitening raises
Bridge macro P@10 by 4.48 points over raw Bridge, Bridge is highly compressed,
and it retains distinct neighborhoods that can be useful for other transfer
objectives. But this experiment does not justify claiming biological retrieval
superiority over PCA. The strongest defensible interpretation is that Bridge
whitening exposes useful biology in a compact learned representation, while
PCA supplies an equally strong or slightly stronger linear baseline for this
human tissue task.

No cross-species retrieval, perturbation analysis, training, fine-tuning, or
post-GTEx method selection was performed.

## Contextual metadata audit of frozen GTEx retrievals

This exploratory extension classifies the deduplicated union of the existing
Top-100 ARCHS4 hits for expression Pearson, expression Canberra, PCA-512
Canberra, PCA-white cosine, raw Bridge cosine, and Bridge-white cosine. The
24,938 unique GSMs were each classified once; embeddings and retrieval were
not rerun. Prespecified regular expressions require explicit evidence in
`source_name_ch1`, sample title, or `characteristics_ch1`. Multiple state or
material labels are treated as conflicts and remain unknown. No corpus-wide
cached GSE title/summary/design resource was found, so series text was neither
fetched nor inferred.

### Annotation coverage before retrieval evaluation

| Axis | Label | GSMs | Fraction |
|---|---|---:|---:|
| State | tumor/cancer | 3,694 | 14.81% |
| State | normal/healthy | 1,983 | 7.95% |
| State | control | 2,265 | 9.08% |
| State | treated | 38 | 0.15% |
| State | unknown | 16,958 | **68.00%** |
| Material | primary tissue | 4,062 | 16.29% |
| Material | cell line | 374 | 1.50% |
| Material | unknown | 20,502 | **82.21%** |

State coverage is 32.00%; 1,268 GSMs (5.08%) contain explicit conflicting
state evidence and are included in unknown. Material coverage is 17.79%; only
nine samples conflict. These rates are reported before method comparison and
were not increased after inspecting performance.

GTEx `source_tissue` explicitly supports a primary-tissue query context, so
primary-tissue agreement is the defensible matched endpoint. The local GTEx
manifest does not establish healthy state sample by sample. Consequently,
normal/healthy among retrieved hits is reported as a descriptive **state
proxy**, not authoritative state agreement.

### Tissue-conditioned Top-10 context

The table conditions first on the existing conservative tissue match. Precision
uses only confidently annotated hits; unknown is always retained separately.

| Method | Primary-tissue precision | Material unknown | Normal/healthy proxy | State unknown |
|---|---:|---:|---:|---:|
| Expression Pearson | 79.94% | 95.30% | 60.34% | 74.67% |
| Expression Canberra | 76.65% | 93.74% | 74.57% | 82.12% |
| PCA-512 Canberra | 65.90% | 98.33% | 67.30% | 78.40% |
| PCA-white cosine | 70.85% | 98.51% | 61.58% | 76.20% |
| Raw Bridge cosine | **87.26%** | 92.79% | 67.94% | 83.96% |
| Bridge-white cosine | 75.10% | 95.03% | **74.41%** | 79.44% |

The material endpoint is very sparse after tissue conditioning: only 223 to
1,154 of roughly 14,000–16,000 tissue-matched Top-10 positions per method have
explicit material evidence. Absolute confirmed primary-tissue rates are just
1.06–6.29%. The state proxy is somewhat better covered but still 74.7–84.0%
unknown. Thus raw Bridge has the highest primary-tissue precision among known
hits and Bridge-white is nearly tied with expression Canberra on the
normal/healthy proxy, but neither observation supports a disease-state claim.

Across all Top-10 positions without tissue conditioning, expression Canberra
has 93.55% primary-tissue precision among known material labels, expression
Pearson 91.27%, PCA variants about 92.6%, raw Bridge 83.71%, and Bridge-white
87.99%; material unknownness remains 87.2–91.2%. Normal/healthy proxy precision
ranges from 41.78% (raw Bridge) to 52.29% (expression Canberra), with roughly
67.5–71.8% state unknownness.

The complete K=1/5/10/100 table, every deduplicated annotation with matched
rule/evidence, and deterministic representative Top-5 examples are saved under
`results/contextual_metadata/`. These submitter-text labels are a
high-precision exploratory layer, not disease ground truth. Missing keywords
cannot be interpreted as absence of a state, and the high unknown fractions
preclude using this audit to select a universal retrieval representation.
