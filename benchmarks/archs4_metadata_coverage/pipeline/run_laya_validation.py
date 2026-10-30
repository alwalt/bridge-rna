#!/usr/bin/env python3
"""Run the controlled 100-sample, two-condition Laya validation on the unified DB."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import laya
import torch
import transformers
from laya import Router


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA_ROOT = ROOT.parents[1] / "data"
DB = DATA_ROOT / "archs4" / "bridge_metadata.sqlite"
SEED = "bridge-laya-validation-v1"
RUN_ID = "laya-validation-v1"
FIELDS = {
    "organ": "Which canonical organ best describes this biological sample?",
    "gender": "Which canonical gender applies to the biological source?",
    "age_group": "Which canonical age group applies to the biological source?",
    "experimental_setting": "Was this sample generated in vivo, ex vivo, or in vitro?",
}


def hash_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def select_cohort(con: sqlite3.Connection) -> list[dict[str, str]]:
    rows = [dict(zip([x[0] for x in cur.description], row)) for cur in [con.execute("""
      SELECT g.gsm, g.gse, g.organ, g.gender, g.age_group,
             g.exp_setting AS experimental_setting,
             a.title, a.source_name_ch1, a.characteristics_ch1,
             a.extract_protocol_ch1, a.library_strategy, a.organism_ch1,
             x.treatment_protocol, x.growth_protocol, x.description,
             s.title AS gse_title, s.summary AS gse_summary,
             s.overall_design AS gse_overall_design
      FROM geometa g JOIN archs4_metadata a USING(gsm)
      LEFT JOIN geo_sample_extras x USING(gsm)
      LEFT JOIN studies s ON s.gse=g.gse
      WHERE g.organ IS NOT NULL AND g.gender IS NOT NULL AND g.age_group IS NOT NULL
        AND g.exp_setting IS NOT NULL
      GROUP BY g.gsm
    """)] for row in cur.fetchall()]
    for row in rows:
        n = len(row.get("characteristics_ch1") or "")
        row["complexity"] = "simple" if n <= 80 else "moderate" if n <= 220 else "complex"
        row["selection_hash"] = hash_key(row["gsm"])
        row["geo_context"] = "rich" if sum(bool(row.get(x)) for x in
            ["treatment_protocol","growth_protocol","description","gse_summary","gse_overall_design"]) >= 4 else "limited"
    chosen, used_gse = [], set()
    targets = {"simple":34, "moderate":33, "complex":33}
    for complexity, count in targets.items():
        subset = sorted((x for x in rows if x["complexity"] == complexity),
                        key=lambda x: (Counter(y["organ"] for y in chosen)[x["organ"]], x["selection_hash"]))
        per_organ = Counter()
        # First pass caps dominant organs, second pass fills any remainder.
        for cap in [2, 4, 999]:
            for row in subset:
                if row in chosen or row["gse"] in used_gse or per_organ[row["organ"]] >= cap:
                    continue
                chosen.append(row); used_gse.add(row["gse"]); per_organ[row["organ"]] += 1
                if sum(x["complexity"] == complexity for x in chosen) == count: break
            if sum(x["complexity"] == complexity for x in chosen) == count: break
    assert len(chosen) == 100 and len({x["gse"] for x in chosen}) == 100
    return sorted(chosen, key=lambda x: x["gsm"])


def evidence(row: dict[str, str], condition: str) -> str:
    def clip(field: str, limit: int) -> str:
        value = row.get(field) or ""
        return value if len(value) <= limit else value[:limit].rstrip() + " […]"
    lines = ["ARCHS4 SAMPLE CONTEXT", f"Species: {row.get('organism_ch1') or ''}",
             f"Title: {clip('title',240)}", f"Source: {clip('source_name_ch1',160)}",
             f"Characteristics: {clip('characteristics_ch1',600)}",
             f"Extraction/library: {clip('extract_protocol_ch1',300)}"]
    if condition == "B":
        lines += ["GEO SAMPLE EXTRAS", f"Treatment protocol: {clip('treatment_protocol',250)}",
                  f"Growth protocol: {clip('growth_protocol',250)}",
                  f"Description: {clip('description',250)}", "GEO STUDY CONTEXT",
                  f"Study title: {clip('gse_title',240)}",
                  f"Study summary: {clip('gse_summary',700)}",
                  f"Overall design: {clip('gse_overall_design',500)}"]
    return "\n".join(lines)


def main() -> None:
    assert torch.cuda.is_available(), "GPU validation requires CUDA"
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    cohort = select_cohort(con)
    vocab = {}
    for field in FIELDS:
        vocab[field] = [x[0] for x in con.execute(
            "SELECT canonical_label FROM canonical_terms WHERE field=? AND active=1 ORDER BY canonical_label", (field,))]
        assert vocab[field]
    with (RESULTS / "laya_validation_cohort_100.csv").open("w", newline="") as stream:
        fields = ["gsm","gse","complexity","geo_context","organ","gender","age_group","experimental_setting","selection_hash"]
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in cohort)

    router = Router(device="cuda")
    predictions = []
    started = time.monotonic()
    for number, row in enumerate(cohort, 1):
        for condition in ["A", "B"]:
            questions = {}
            for field, instruction in FIELDS.items():
                criteria = {label: f"The canonical {field} category {label}." for label in vocab[field]}
                criteria["Unknown"] = "The evidence is insufficient or no listed category applies."
                questions[field] = {"type":"choice", "instructions":instruction, "criteria":criteria}
            context = evidence(row, condition)
            result = router.predict(context, questions)
            for field in FIELDS:
                answer = result["answers"][field]
                selected = answer["choice"]
                truth = row[field]
                predictions.append({"annotation_run_id":RUN_ID,"gsm":row["gsm"],"gse":row["gse"],
                    "field":field,"condition":condition,"selected_label":selected,"true_label":truth,
                    "correct":int(selected.casefold()==truth.casefold()),
                    "model_score":answer.get("answer_confidence"),"raw_confidence":answer.get("confidence"),
                    "evidence_text":context,"model_route":result.get("routing",{}).get("model")})
        if number % 10 == 0:
            elapsed=time.monotonic()-started
            print(f"Laya {number}/100 elapsed={elapsed:.1f}s rate={number/elapsed:.2f} samples/s",flush=True)

    out = RESULTS / "laya_validation_predictions.csv"
    with out.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    bins=[(0.0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0000001)]
    summaries=[]
    for condition in ["A","B"]:
        for field in FIELDS:
            selected=[x for x in predictions if x["condition"]==condition and x["field"]==field]
            summaries.append({"condition":condition,"field":field,"score_bin":"all","n":len(selected),
                              "accuracy":sum(x["correct"] for x in selected)/len(selected)})
            for low,high in bins:
                items=[x for x in selected if x["model_score"] is not None and low<=x["model_score"]<high]
                summaries.append({"condition":condition,"field":field,"score_bin":f"{low:.2f}-{min(high,1):.2f}",
                                  "n":len(items),"accuracy":sum(x["correct"] for x in items)/len(items) if items else None})
    with (RESULTS/"laya_score_accuracy_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    run={"annotation_run_id":RUN_ID,"tool":"Laya","model_name":"convaiinnovations/laya",
         "model_version":getattr(laya,"__version__","unknown"),"model_commit":"4066d5d5fbf08b66c6757ddeedbd797bd7655bc0",
         "question_schema_version":"laya-geometa-choice-v1","canonical_vocabulary_version":"geometa-hf-d0cd1e1",
         "runtime_environment":json.dumps({"python":platform.python_version(),"torch":torch.__version__,
             "transformers":transformers.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}),
         "code_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
         "run_date":datetime.now(timezone.utc).isoformat(),
         "notes":"100 human samples; GEOMeta truth hidden from evidence. Scores are not calibrated probabilities."}
    (RESULTS/"laya_validation_run.json").write_text(json.dumps(run,indent=2))
    con.execute("""INSERT OR REPLACE INTO annotation_runs
      (annotation_run_id,tool,model_name,model_version,model_commit,question_schema_version,
       canonical_vocabulary_version,runtime_environment,code_commit,run_date,notes)
       VALUES (?,?,?,?,?,?,?,?,?,?,?)""", tuple(run[x] for x in ["annotation_run_id","tool","model_name",
       "model_version","model_commit","question_schema_version","canonical_vocabulary_version",
       "runtime_environment","code_commit","run_date","notes"]))
    term_lookup = {(r[0],r[1]):r[2] for r in con.execute(
        "SELECT field,canonical_label,canonical_id FROM canonical_terms")}
    assertions=[]
    for item in predictions:
        selected=item["selected_label"]
        cid=term_lookup.get((item["field"],selected))
        assertions.append((item["gsm"],item["field"],selected,selected if cid else None,cid,"Laya",
            "model_generated","convaiinnovations/laya",run["model_version"],run["model_commit"],
            item["model_score"],item["evidence_text"],f"validation_condition_{item['condition']}",
            None,None,RUN_ID,"bridge-metadata-v1",run["run_date"],"validation_only"))
    con.executemany("""INSERT INTO metadata_assertions
      (gsm,field,raw_value,canonical_value,canonical_id,source,annotation_method,model_name,
       model_version,model_commit,model_score,evidence_text,evidence_source,ontology_id,
       ontology_name,annotation_run_id,schema_version,assertion_timestamp,status)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",assertions)
    con.commit(); con.close()
    print(f"Saved {len(predictions)} predictions",flush=True)


if __name__ == "__main__":
    main()
