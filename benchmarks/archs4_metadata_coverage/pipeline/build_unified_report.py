#!/usr/bin/env python3
"""Render the final database and Laya validation report from saved results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/"results"
DATA_ROOT=ROOT.parents[1]/"data"; DB=DATA_ROOT/"archs4"/"bridge_metadata.sqlite"


def q(con,sql): return con.execute(sql).fetchone()[0]


def main():
    con=sqlite3.connect(DB)
    validation=json.loads((RESULTS/"database_validation.json").read_text())
    coverage=pd.read_csv(RESULTS/"final_metadata_coverage_summary.csv")
    scores=pd.read_csv(RESULTS/"laya_score_accuracy_summary.csv")
    overall=scores[scores.score_bin.eq("all")].pivot(index="field",columns="condition",values="accuracy")
    overall["delta_B_minus_A"]=overall.get("B")-overall.get("A")
    predictions=pd.read_csv(RESULTS/"laya_validation_predictions.csv")
    geo_study=pd.read_csv(RESULTS/"geo_study_metadata_completeness.csv")
    geo_extras=pd.read_csv(RESULTS/"geo_sample_extras_completeness.csv")
    source_counts=pd.read_sql_query("""SELECT
      SUM(organ_source='GEOMeta') AS organ_geometa, SUM(organ_source='Gemma') AS organ_gemma,
      SUM(gender_source='GEOMeta') AS gender_geometa, SUM(gender_source='Gemma') AS gender_gemma,
      SUM(cell_type_source='Gemma') AS cell_type_gemma, SUM(strain_source='Gemma') AS strain_gemma,
      SUM(genotype_source='Gemma') AS genotype_gemma FROM canonical_metadata""",con).iloc[0]
    lines=["# Unified Bridge ARCHS4 metadata database and Laya validation", "",
      "## Outcome", "",
      "The database is structurally suitable as a Bridge RAG metadata sidecar: every embedding row has exactly one GSM/global-index record; sample–study membership is many-to-many; raw source tables are preserved; canonical selections are materialized without deleting assertions; conflicts and model runs are first-class.","",
      "The 100-sample Laya run is validation-only. Its assertions are stored with `status=validation_only` and do not change production canonical metadata. Laya scores are raw model scores, not calibrated probabilities.","",
      f"Official GEO brief metadata was retrieved for {validation['studies']:,}/{validation['gses']:,} relevant GSEs ({100*validation['studies']/validation['gses']:.3f}%); 13 bounded endpoint failures remain explicit in the failure table.","",
      "## Database validation", "",
      "| Metric | Count |","|---|---:|"]
    for key,value in validation.items(): lines.append(f"| {key} | {value:,} |" if isinstance(value,int) else f"| {key} | {value} |")
    lines += ["","## Canonical coverage before Laya","",coverage.to_markdown(index=False,floatfmt=".3f"),"",
      "Because the pilot is held-out validation, canonical coverage after Laya is intentionally unchanged. The database contains 800 `validation_only` model assertions but no pilot prediction is promoted into `canonical_metadata`.","",
      "## GEO completeness","",geo_study.to_markdown(index=False,floatfmt=".3f"),"",geo_extras.to_markdown(index=False,floatfmt=".3f"),"",
      "## Canonical source distribution","","| Measure | Count |","|---|---:|"]
    for key,value in source_counts.items(): lines.append(f"| {key} | {int(value or 0):,} |")
    lines += ["","## Laya validation: ARCHS4 alone versus ARCHS4 + GEO","",
      overall.reset_index().to_markdown(index=False,floatfmt=".3f"),"",
      f"Predictions: {len(predictions):,} ({predictions.gsm.nunique()} samples × {predictions.field.nunique()} fields × {predictions.condition.nunique()} conditions).",
      "The held-out cohort is human-only because the downloaded GEOMeta release contains no mouse labels to use as hidden ground truth. It spans 100 unique GSEs, 45 Organ labels, and balanced simple/moderate/complex ARCHS4 metadata. Mouse inference was therefore not scored or presented as validation.",
      "Disease was not forced into the typed-choice validation because the observed GEOMeta release contains 369 non-missing disease labels; that candidate set requires a hierarchical or retrieval-shortlisted design.","",
      "GEO context produced a mixed result: +15 percentage points for Experimental Setting, +4 for Age Group, +3 for Gender, and -3 for Organ. Gender is the only field with acceptable overall pilot accuracy (79% with GEO), but that is still below a safe automatic-fill standard. Organ and Gender scores show useful triage signal at the high end (Condition B Organ score 0.90–1.00: 23/25 correct; Gender score 0.70–1.00: 38/38 correct). Age Group and Experimental Setting remain unreliable across score bins. These small-bin results support review prioritization, not production thresholds.","",
      "## Score-bin results","",scores.to_markdown(index=False,floatfmt=".3f"),"",
      "No production threshold is selected. Sparse bins and the checkpoint temperature warning require independent calibration/manual review before score-based acceptance.","",
      "## Remaining issues before scale","",
      "- Review unresolved GEOMeta/Gemma conflicts and mapping specificity.",
      "- Validate the deterministic Gemma-to-GEOMeta exact mappings and the 32 exact-label UBERON mappings.",
      "- Decide a hierarchical strategy for Disease rather than presenting 369 flat choices.",
      "- Manually score representative prediction errors and unknowns, especially by metadata complexity and GEO-context gain.",
      "- Do not annotate the remaining corpus until field-specific accuracy and score triage are acceptable.","",
      "## Files","",
      "The SQLite database, DDL, schema documentation, vocabulary/mapping exports, GEO manifest/failures, validation cohort, per-prediction output, score summary, and example SQL queries are all stored beside this report under the benchmark and `results/` directories."]
    (ROOT/"UNIFIED_METADATA_REPORT.md").write_text("\n".join(lines)+"\n")
    con.close()


if __name__=="__main__": main()
