# Unified Bridge ARCHS4 metadata database and Laya validation

## Outcome

The database is structurally suitable as a Bridge RAG metadata sidecar: every embedding row has exactly one GSM/global-index record; sample–study membership is many-to-many; raw source tables are preserved; canonical selections are materialized without deleting assertions; conflicts and model runs are first-class.

The 100-sample Laya run is validation-only. Its assertions are stored with `status=validation_only` and do not change production canonical metadata. Laya scores are raw model scores, not calibrated probabilities.

Official GEO brief metadata was retrieved for 59,791/59,804 relevant GSEs (99.978%); 13 bounded endpoint failures remain explicit in the failure table.

## Database validation

| Metric | Count |
|---|---:|
| samples | 940,455 |
| human | 510,709 |
| mouse | 429,746 |
| gses | 59,804 |
| multi_gse_samples | 210,140 |
| archs4_metadata | 940,455 |
| studies | 59,791 |
| geo_sample_extras | 754,944 |
| geometa_gsms | 438,119 |
| gemma_gsms | 177,455 |
| foreign_key_violations | 0 |

## Canonical coverage before Laya

| field                |   available |   total |    pct |
|:---------------------|------------:|--------:|-------:|
| organ                |      404266 |  940455 | 42.986 |
| disease              |      297603 |  940455 | 31.645 |
| gender               |      175090 |  940455 | 18.618 |
| age_group            |       78125 |  940455 |  8.307 |
| experimental_setting |      438119 |  940455 | 46.586 |
| perturbation         |      438119 |  940455 | 46.586 |
| cell_type            |       72985 |  940455 |  7.761 |
| strain               |       63227 |  940455 |  6.723 |
| genotype             |       98795 |  940455 | 10.505 |
| treatment            |       96172 |  940455 | 10.226 |
| dose                 |        5570 |  940455 |  0.592 |
| duration_development |       93309 |  940455 |  9.922 |

Because the pilot is held-out validation, canonical coverage after Laya is intentionally unchanged. The database contains 800 `validation_only` model assertions but no pilot prediction is promoted into `canonical_metadata`.

## GEO completeness

| field                |   available |   total_studies_retrieved |     pct |
|:---------------------|------------:|--------------------------:|--------:|
| title                |       59791 |                     59791 | 100.000 |
| summary              |       59791 |                     59791 | 100.000 |
| overall_design       |       59791 |                     59791 | 100.000 |
| pubmed_ids           |       48967 |                     59791 |  81.897 |
| study_type           |       59791 |                     59791 | 100.000 |
| submission_date      |       59791 |                     59791 | 100.000 |
| last_update_date     |       59791 |                     59791 | 100.000 |
| organisms            |       59791 |                     59791 | 100.000 |
| taxids               |       59791 |                     59791 | 100.000 |
| platforms            |       59791 |                     59791 | 100.000 |
| bioproject_relations |       57381 |                     59791 |  95.969 |
| sra_relations        |       33254 |                     59791 |  55.617 |
| other_relations      |       20251 |                     59791 |  33.870 |
| status               |       59791 |                     59791 | 100.000 |
| source_url           |       59791 |                     59791 | 100.000 |
| retrieved_at         |       59791 |                     59791 | 100.000 |
| source_sha256        |       59791 |                     59791 | 100.000 |

| field              |   available |   total_bridge_samples |   pct_of_bridge |
|:-------------------|------------:|-----------------------:|----------------:|
| treatment_protocol |      468214 |                 940455 |          49.786 |
| growth_protocol    |      464072 |                 940455 |          49.345 |
| description        |      518865 |                 940455 |          55.172 |

## Canonical source distribution

| Measure | Count |
|---|---:|
| organ_geometa | 383,647 |
| organ_gemma | 20,619 |
| gender_geometa | 142,401 |
| gender_gemma | 32,689 |
| cell_type_gemma | 72,985 |
| strain_gemma | 63,227 |
| genotype_gemma | 98,795 |

## Laya validation: ARCHS4 alone versus ARCHS4 + GEO

| field                |     A |     B |   delta_B_minus_A |
|:---------------------|------:|------:|------------------:|
| age_group            | 0.050 | 0.090 |             0.040 |
| experimental_setting | 0.100 | 0.250 |             0.150 |
| gender               | 0.760 | 0.790 |             0.030 |
| organ                | 0.420 | 0.390 |            -0.030 |

Predictions: 800 (100 samples × 4 fields × 2 conditions).
The held-out cohort is human-only because the downloaded GEOMeta release contains no mouse labels to use as hidden ground truth. It spans 100 unique GSEs, 45 Organ labels, and balanced simple/moderate/complex ARCHS4 metadata. Mouse inference was therefore not scored or presented as validation.
Disease was not forced into the typed-choice validation because the observed GEOMeta release contains 369 non-missing disease labels; that candidate set requires a hierarchical or retrieval-shortlisted design.

GEO context produced a mixed result: +15 percentage points for Experimental Setting, +4 for Age Group, +3 for Gender, and -3 for Organ. Gender is the only field with acceptable overall pilot accuracy (79% with GEO), but that is still below a safe automatic-fill standard. Organ and Gender scores show useful triage signal at the high end (Condition B Organ score 0.90–1.00: 23/25 correct; Gender score 0.70–1.00: 38/38 correct). Age Group and Experimental Setting remain unreliable across score bins. These small-bin results support review prioritization, not production thresholds.

## Score-bin results

| condition   | field                | score_bin   |   n |   accuracy |
|:------------|:---------------------|:------------|----:|-----------:|
| A           | organ                | all         | 100 |      0.420 |
| A           | organ                | 0.00-0.50   |  48 |      0.125 |
| A           | organ                | 0.50-0.60   |   7 |      0.143 |
| A           | organ                | 0.60-0.70   |   5 |      0.400 |
| A           | organ                | 0.70-0.80   |   2 |      0.000 |
| A           | organ                | 0.80-0.90   |   6 |      0.833 |
| A           | organ                | 0.90-1.00   |  32 |      0.875 |
| A           | gender               | all         | 100 |      0.760 |
| A           | gender               | 0.00-0.50   |  13 |      0.077 |
| A           | gender               | 0.50-0.60   |   6 |      0.500 |
| A           | gender               | 0.60-0.70   |  12 |      0.333 |
| A           | gender               | 0.70-0.80   |  16 |      0.938 |
| A           | gender               | 0.80-0.90   |  25 |      1.000 |
| A           | gender               | 0.90-1.00   |  28 |      1.000 |
| A           | age_group            | all         | 100 |      0.050 |
| A           | age_group            | 0.00-0.50   |  33 |      0.030 |
| A           | age_group            | 0.50-0.60   |  18 |      0.000 |
| A           | age_group            | 0.60-0.70   |  27 |      0.074 |
| A           | age_group            | 0.70-0.80   |  15 |      0.067 |
| A           | age_group            | 0.80-0.90   |   7 |      0.143 |
| A           | age_group            | 0.90-1.00   |   0 |    nan     |
| A           | experimental_setting | all         | 100 |      0.100 |
| A           | experimental_setting | 0.00-0.50   |  93 |      0.108 |
| A           | experimental_setting | 0.50-0.60   |   7 |      0.000 |
| A           | experimental_setting | 0.60-0.70   |   0 |    nan     |
| A           | experimental_setting | 0.70-0.80   |   0 |    nan     |
| A           | experimental_setting | 0.80-0.90   |   0 |    nan     |
| A           | experimental_setting | 0.90-1.00   |   0 |    nan     |
| B           | organ                | all         | 100 |      0.390 |
| B           | organ                | 0.00-0.50   |  60 |      0.150 |
| B           | organ                | 0.50-0.60   |   6 |      0.500 |
| B           | organ                | 0.60-0.70   |   3 |      0.667 |
| B           | organ                | 0.70-0.80   |   3 |      0.333 |
| B           | organ                | 0.80-0.90   |   3 |      0.333 |
| B           | organ                | 0.90-1.00   |  25 |      0.920 |
| B           | gender               | all         | 100 |      0.790 |
| B           | gender               | 0.00-0.50   |  19 |      0.421 |
| B           | gender               | 0.50-0.60   |  21 |      0.667 |
| B           | gender               | 0.60-0.70   |  22 |      0.864 |
| B           | gender               | 0.70-0.80   |  11 |      1.000 |
| B           | gender               | 0.80-0.90   |  15 |      1.000 |
| B           | gender               | 0.90-1.00   |  12 |      1.000 |
| B           | age_group            | all         | 100 |      0.090 |
| B           | age_group            | 0.00-0.50   |  76 |      0.066 |
| B           | age_group            | 0.50-0.60   |  11 |      0.091 |
| B           | age_group            | 0.60-0.70   |   6 |      0.167 |
| B           | age_group            | 0.70-0.80   |   4 |      0.250 |
| B           | age_group            | 0.80-0.90   |   3 |      0.333 |
| B           | age_group            | 0.90-1.00   |   0 |    nan     |
| B           | experimental_setting | all         | 100 |      0.250 |
| B           | experimental_setting | 0.00-0.50   |  29 |      0.172 |
| B           | experimental_setting | 0.50-0.60   |  52 |      0.308 |
| B           | experimental_setting | 0.60-0.70   |  19 |      0.211 |
| B           | experimental_setting | 0.70-0.80   |   0 |    nan     |
| B           | experimental_setting | 0.80-0.90   |   0 |    nan     |
| B           | experimental_setting | 0.90-1.00   |   0 |    nan     |

No production threshold is selected. Sparse bins and the checkpoint temperature warning require independent calibration/manual review before score-based acceptance.

## Remaining issues before scale

- Review unresolved GEOMeta/Gemma conflicts and mapping specificity.
- Validate the deterministic Gemma-to-GEOMeta exact mappings and the 32 exact-label UBERON mappings.
- Decide a hierarchical strategy for Disease rather than presenting 369 flat choices.
- Manually score representative prediction errors and unknowns, especially by metadata complexity and GEO-context gain.
- Do not annotate the remaining corpus until field-specific accuracy and score triage are acceptable.

## Files

The SQLite database, DDL, schema documentation, vocabulary/mapping exports, GEO manifest/failures, validation cohort, per-prediction output, score summary, and example SQL queries are all stored beside this report under the benchmark and `results/` directories.
