#!/usr/bin/env python3
"""Compare shortlisted and flat Laya Organ classification on ambiguous held-out cases."""

from __future__ import annotations

import csv
import json
import platform
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import laya
import pandas as pd
import torch
import transformers
from laya import Router


ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
OUT=RESULTS/"organ_recovery"
DATA_ROOT=ROOT.parents[1]/"data"
DB=DATA_ROOT/"archs4"/"bridge_metadata.sqlite"
RUN_ID="laya-organ-shortlist-v1"


def get_context(con:sqlite3.Connection,gsm:str)->str:
    row=con.execute("""
      SELECT a.organism_ch1,a.title,a.source_name_ch1,a.characteristics_ch1,a.extract_protocol_ch1,
             x.description,x.treatment_protocol,x.growth_protocol,
             group_concat(DISTINCT st.title),group_concat(DISTINCT st.summary),group_concat(DISTINCT st.overall_design)
      FROM archs4_metadata a LEFT JOIN geo_sample_extras x USING(gsm)
      LEFT JOIN sample_studies ss USING(gsm) LEFT JOIN studies st USING(gse)
      WHERE a.gsm=? GROUP BY a.gsm
    """,(gsm,)).fetchone()
    fields=["Species","GSM title","GSM source","GSM characteristics","Extraction protocol",
            "GEO GSM description","Treatment protocol","Growth protocol","GSE title","GSE summary","GSE overall design"]
    limits=[80,240,200,800,300,400,300,300,300,900,700]
    return "\n".join(f"{name}: {(value or '')[:limit]}" for name,value,limit in zip(fields,row,limits))


def bins(df:pd.DataFrame)->pd.DataFrame:
    cuts=[(0,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.000001)]
    rows=[]
    for mode in sorted(df["mode"].unique()):
        g=df[df["mode"]==mode]
        rows.append({"mode":mode,"score_bin":"all","n":len(g),"accuracy":g.correct.mean(),
                     "unknown_rate":g.selected_label.eq("Unknown").mean()})
        for low,high in cuts:
            x=g[(g.model_score>=low)&(g.model_score<high)]
            rows.append({"mode":mode,"score_bin":f"{low:.1f}-{min(high,1):.1f}","n":len(x),
                         "accuracy":x.correct.mean() if len(x) else None,
                         "unknown_rate":x.selected_label.eq("Unknown").mean() if len(x) else None})
    return pd.DataFrame(rows)


def main()->None:
    assert torch.cuda.is_available(),"CUDA is required for this bounded validation"
    val=pd.read_parquet(OUT/"deterministic_validation_predictions.parquet")
    cohort=val[(val.condition=="D")&(val.mapping_status=="ambiguous")].copy()
    cohort["candidate_labels"]=cohort.candidate_organs.map(json.loads)
    cohort["candidate_count"]=cohort.candidate_labels.map(len)
    cohort["true_in_shortlist"]=cohort.apply(lambda x:x.true_organ in x.candidate_labels,axis=1)
    cohort["candidate_provenance"]="strongest-tier direct anatomical aliases; field-aware deterministic mapper condition D"
    cohort[["gsm","species","true_organ","candidate_organs","candidate_count","true_in_shortlist",
            "evidence_source","evidence_field","evidence_text","candidate_provenance"]].to_csv(
                OUT/"laya_shortlist_validation_cohort.csv",index=False)
    labels=sorted(x[0] for x in sqlite3.connect(f"file:{DB}?mode=ro",uri=True).execute(
        "SELECT canonical_label FROM canonical_terms WHERE field='organ' AND active=1"))
    con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
    router=Router(device="cuda")
    predictions=[]; started=time.monotonic()
    for number,row in enumerate(cohort.itertuples(index=False),1):
        context=get_context(con,row.gsm)
        for mode,candidates in [("shortlist",row.candidate_labels),("flat_54",labels)]:
            criteria={x:f"The canonical GEOMeta Organ category {x}." for x in candidates}
            criteria["Unknown"]="Evidence is insufficient, contradictory, or none of the listed categories applies."
            question={"organ":{"type":"choice","instructions":
                "Choose the best canonical Organ for this sample. Prefer Unknown when the evidence does not support one choice.",
                "criteria":criteria}}
            answer=router.predict(context,question)["answers"]["organ"]
            selected=answer["choice"]
            predictions.append({"gsm":row.gsm,"species":row.species,"true_organ":row.true_organ,
                "mode":mode,"candidate_labels":json.dumps(candidates),"candidate_count":len(candidates),
                "true_in_shortlist":row.true_organ in candidates,"selected_label":selected,
                "correct":selected.casefold()==row.true_organ.casefold(),"model_score":answer.get("answer_confidence"),
                "raw_confidence":answer.get("confidence"),"evidence":context,"annotation_run_id":RUN_ID,
                "laya_version":getattr(laya,"__version__","unknown"),
                "model_commit":"4066d5d5fbf08b66c6757ddeedbd797bd7655bc0","validation_status":"validation_only"})
        if number%20==0:
            elapsed=time.monotonic()-started
            print(f"Laya ambiguous {number}/{len(cohort)} elapsed={elapsed:.1f}s rate={number/elapsed:.2f} samples/s",flush=True)
    pred=pd.DataFrame(predictions)
    pred.to_csv(OUT/"laya_shortlist_predictions.csv",index=False)
    candidate=pd.DataFrame([{"cohort_n":len(cohort),"candidate_recall":cohort.true_in_shortlist.mean(),
        "candidate_failure_n":int((~cohort.true_in_shortlist).sum()),"mean_candidate_count":cohort.candidate_count.mean(),
        "median_candidate_count":cohort.candidate_count.median(),"min_candidate_count":cohort.candidate_count.min(),
        "max_candidate_count":cohort.candidate_count.max()}])
    candidate.to_csv(OUT/"candidate_recall_analysis.csv",index=False)
    by_count=pred.groupby(["mode","candidate_count"],as_index=False).agg(n=("gsm","size"),accuracy=("correct","mean"),
        unknown_rate=("selected_label",lambda x:(x=="Unknown").mean()))
    by_count.to_csv(OUT/"laya_accuracy_by_candidate_count.csv",index=False)
    bins(pred).to_csv(OUT/"laya_score_bin_analysis.csv",index=False)
    by_organ=pred.groupby(["mode","true_organ"],as_index=False).agg(n=("gsm","size"),accuracy=("correct","mean"),
        unknown_rate=("selected_label",lambda x:(x=="Unknown").mean()))
    by_organ.to_csv(OUT/"laya_accuracy_by_organ.csv",index=False)

    all_d=val[val.condition=="D"]; resolved=all_d.mapping_status.eq("resolved")
    short=pred[pred["mode"]=="shortlist"].set_index("gsm")
    flat=pred[pred["mode"]=="flat_54"]
    laya_correct=sum(short.correct); laya_nonunknown=sum(~short.selected_label.eq("Unknown"))
    strategy=[
      {"strategy":"deterministic_only","evaluation_scope":"5000 held-out GEOMeta samples",
       "n":len(all_d),"coverage":resolved.mean(),"precision_among_labeled":all_d.loc[resolved,"correct"].mean(),
       "incorrect_label_rate":((resolved)&(~all_d.correct)).mean(),"unknown_or_ambiguous_rate":1-resolved.mean(),
       "samples_requiring_laya":0,"approx_runtime_seconds":0},
      {"strategy":"deterministic_plus_shortlist_laya","evaluation_scope":"5000 held-out; Laya only 140 ambiguous cases",
       "n":len(all_d),"coverage":(resolved.sum()+laya_nonunknown)/len(all_d),
       "precision_among_labeled":(all_d.loc[resolved,"correct"].sum()+laya_correct)/(resolved.sum()+laya_nonunknown),
       "incorrect_label_rate":(all_d.loc[resolved,"correct"].eq(False).sum()+laya_nonunknown-laya_correct)/len(all_d),
       "unknown_or_ambiguous_rate":1-(resolved.sum()+laya_nonunknown)/len(all_d),
       "samples_requiring_laya":len(short),"approx_runtime_seconds":time.monotonic()-started},
      {"strategy":"flat_54_laya_on_same_ambiguous_cases","evaluation_scope":"140 deterministic-ambiguous held-out cases only",
       "n":len(flat),"coverage":1-flat.selected_label.eq("Unknown").mean(),
       "precision_among_labeled":flat.loc[~flat.selected_label.eq("Unknown"),"correct"].mean(),
       "incorrect_label_rate":((~flat.selected_label.eq("Unknown"))&(~flat.correct)).mean(),
       "unknown_or_ambiguous_rate":flat.selected_label.eq("Unknown").mean(),
       "samples_requiring_laya":len(flat),"approx_runtime_seconds":time.monotonic()-started},
      {"strategy":"prior_flat_54_baseline","evaluation_scope":"previous independent 100-sample condition-B pilot",
       "n":100,"coverage":None,"precision_among_labeled":0.39,"incorrect_label_rate":None,
       "unknown_or_ambiguous_rate":None,"samples_requiring_laya":100,"approx_runtime_seconds":None}]
    pd.DataFrame(strategy).to_csv(OUT/"strategy_comparison.csv",index=False)
    run={"annotation_run_id":RUN_ID,"created_at":datetime.now(timezone.utc).isoformat(),"n":len(cohort),
        "device":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,
        "transformers":transformers.__version__,"laya":getattr(laya,"__version__","unknown"),
        "model_commit":"4066d5d5fbf08b66c6757ddeedbd797bd7655bc0",
        "git_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
        "elapsed_seconds":time.monotonic()-started,"score_semantics":"model_score; not calibrated confidence",
        "database_open_mode":"read_only","notes":"No prediction was promoted to canonical metadata."}
    (OUT/"laya_shortlist_run.json").write_text(json.dumps(run,indent=2))
    con.close()


if __name__=="__main__": main()
