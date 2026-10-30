#!/usr/bin/env python3
"""Audit conservative recovery of the GEOMeta Organ vocabulary.

The unified SQLite database is opened read-only.  Proposed labels are written only
to result files; this script never changes canonical_metadata or metadata_assertions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
RESULTS = ROOT / "results" / "organ_recovery"
DATA_ROOT = REPO_ROOT / "data"
DB = DATA_ROOT / "archs4" / "bridge_metadata.sqlite"
SEED = "bridge-organ-recovery-v1"
VALIDATION_N = 5000

# GEOMeta categories that have a clean or useful corresponding UBERON class.
# Head and Neck and Nasal remain intentionally unmapped: each is a heterogeneous
# GEOMeta category rather than one unambiguous anatomical structure.
UBERON = {
    "Adipose tissue":"UBERON:0001013", "Adrenal gland":"UBERON:0002369",
    "Blood":"UBERON:0000178", "Bone marrow":"UBERON:0002371",
    "Brain":"UBERON:0000955", "Brain: Amygdala":"UBERON:0001876",
    "Brain: Basal ganglia":"UBERON:0002420", "Brain: Cerebellum":"UBERON:0002037",
    "Brain: Cerebral cortex":"UBERON:0000956", "Brain: Choroid plexus":"UBERON:0001886",
    "Brain: Hippocampal formation":"UBERON:0002421", "Brain: Hypothalamus":"UBERON:0001898",
    "Brain: Medulla oblongata":"UBERON:0001896", "Brain: Midbrain":"UBERON:0001891",
    "Brain: Pons":"UBERON:0000988", "Brain: Spinal cord":"UBERON:0002240",
    "Brain: Thalamus":"UBERON:0001897", "Brain: White matter":"UBERON:0003544",
    "Breast":"UBERON:0000310", "Cervix":"UBERON:0000002",
    "Endometrium":"UBERON:0001295", "Epididymis":"UBERON:0001301",
    "Esophagus":"UBERON:0001043", "Eye":"UBERON:0000970",
    "Fallopian tube":"UBERON:0003889", "Gallbladder":"UBERON:0002110",
    "Heart":"UBERON:0000948", "Intestine":"UBERON:0000160",
    "Kidney":"UBERON:0002113", "Liver":"UBERON:0002107", "Lung":"UBERON:0002048",
    "Lymphoid tissue":"UBERON:0001744", "Nasopharynx":"UBERON:0001728",
    "Oropharynx":"UBERON:0001729", "Ovary":"UBERON:0000992",
    "Pancreas":"UBERON:0001264", "Pituitary gland":"UBERON:0000007",
    "Placenta":"UBERON:0001987", "Prostate":"UBERON:0002367",
    "Retina":"UBERON:0000966", "Salivary gland":"UBERON:0001044",
    "Skeletal muscle":"UBERON:0001134", "Skin":"UBERON:0002097",
    "Smooth muscle":"UBERON:0001135", "Stomach":"UBERON:0000945",
    "Synovium":"UBERON:0002018", "Testis":"UBERON:0000473",
    "Thyroid gland":"UBERON:0002046", "Tongue":"UBERON:0001723",
    "Umbilical vein":"UBERON:0002066", "Urinary bladder":"UBERON:0001255",
    "Vagina":"UBERON:0000996",
}

MANUAL = {
    "Adipose tissue":["adipose", "adipose tissue", "fat tissue"],
    "Adrenal gland":["adrenal", "adrenal gland"],
    "Blood":["whole blood", "peripheral blood", "blood"],
    "Bone marrow":["bone marrow"],
    "Brain":["whole brain", "brain"],
    "Brain: Amygdala":["amygdala", "amygdaloid"],
    "Brain: Basal ganglia":["basal ganglia", "basal ganglion", "striatum", "putamen", "caudate nucleus", "globus pallidus", "nucleus accumbens"],
    "Brain: Cerebellum":["cerebellum", "cerebellar"],
    "Brain: Cerebral cortex":["cerebral cortex", "cerebral cortical", "neocortex", "prefrontal cortex", "frontal cortex", "motor cortex", "somatosensory cortex"],
    "Brain: Choroid plexus":["choroid plexus"],
    "Brain: Hippocampal formation":["hippocampus", "hippocampal", "dentate gyrus", "ammon's horn"],
    "Brain: Hypothalamus":["hypothalamus", "hypothalamic"],
    "Brain: Medulla oblongata":["medulla oblongata"],
    "Brain: Midbrain":["midbrain", "mesencephalon", "ventral tegmental area", "substantia nigra"],
    "Brain: Pons":["pons", "pontine"],
    "Brain: Spinal cord":["spinal cord"],
    "Brain: Thalamus":["thalamus", "thalamic"],
    "Brain: White matter":["white matter"],
    "Breast":["breast", "mammary gland", "mammary tissue"],
    "Cervix":["uterine cervix", "cervical tissue", "cervix"],
    "Endometrium":["endometrium", "endometrial"],
    "Epididymis":["epididymis", "epididymal"],
    "Esophagus":["esophagus", "oesophagus", "esophageal", "oesophageal"],
    "Eye":["whole eye", "eyeball", "ocular tissue"],
    "Fallopian tube":["fallopian tube", "uterine tube", "oviduct"],
    "Gallbladder":["gallbladder", "gall bladder"],
    "Head and Neck":["head and neck", "head-and-neck", "head neck"],
    "Heart":["heart", "cardiac tissue", "myocardium", "myocardial"],
    "Intestine":["intestine", "intestinal", "colon", "colonic", "ileum", "ileal", "jejunum", "duodenum", "rectum", "colorectal"],
    "Kidney":["kidney", "renal tissue", "renal cortex", "renal medulla"],
    "Liver":["liver", "hepatic tissue"],
    "Lung":["lung", "pulmonary tissue"],
    "Lymphoid tissue":["lymphoid tissue", "lymph node", "tonsil", "spleen", "thymus"],
    "Nasal":["nasal tissue", "nasal mucosa", "nasal epithelium", "nasal cavity"],
    "Nasopharynx":["nasopharynx", "nasopharyngeal"],
    "Oropharynx":["oropharynx", "oropharyngeal"],
    "Ovary":["ovary", "ovarian tissue"],
    "Pancreas":["pancreas", "pancreatic tissue", "pancreatic islet", "islets of langerhans"],
    "Pituitary gland":["pituitary", "pituitary gland", "hypophysis"],
    "Placenta":["placenta", "placental tissue", "chorionic villus", "chorionic villi"],
    "Prostate":["prostate", "prostate gland", "prostatic tissue"],
    "Retina":["retina", "retinal tissue"],
    "Salivary gland":["salivary gland", "parotid gland", "submandibular gland", "sublingual gland"],
    "Skeletal muscle":["skeletal muscle", "striated muscle", "quadriceps", "gastrocnemius", "soleus muscle"],
    "Skin":["skin", "cutaneous tissue", "dermis", "epidermis"],
    "Smooth muscle":["smooth muscle"],
    "Stomach":["stomach", "gastric tissue", "gastric mucosa"],
    "Synovium":["synovium", "synovial membrane", "synovial tissue"],
    "Testis":["testis", "testes", "testicular tissue"],
    "Thyroid gland":["thyroid", "thyroid gland"],
    "Tongue":["tongue", "lingual tissue"],
    "Umbilical vein":["umbilical vein"],
    "Urinary bladder":["urinary bladder", "bladder tissue", "urothelium"],
    "Vagina":["vagina", "vaginal tissue"],
}

# Specific categories suppress their broad parent when both are found in one field.
PARENTS = {x:"Brain" for x in MANUAL if x.startswith("Brain:")}
PARENTS.update({"Retina":"Eye", "Nasopharynx":"Nasal", "Oropharynx":"Head and Neck"})

ARCH_FIELDS = [("source_name_ch1",1), ("characteristics_ch1",2), ("title",3), ("extract_protocol_ch1",4)]
GEO_SAMPLE_FIELDS = [("geo_description",2), ("treatment_protocol",5), ("growth_protocol",5)]
STUDY_FIELDS = [("gse_title",6), ("gse_summary",7), ("gse_overall_design",7)]
GEMMA_CATEGORIES = {"tissue", "organism part", "organ", "source organ", "brain region", "tissue type", "tissue origin", "tissue source", "source tissue"}


def hash_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def ols_term(obo_id: str) -> dict:
    url = "https://www.ebi.ac.uk/ols4/api/ontologies/uberon/terms?" + urllib.parse.urlencode({"obo_id":obo_id})
    with urllib.request.urlopen(url, timeout=30) as response:
        terms = json.load(response).get("_embedded", {}).get("terms", [])
    if not terms:
        raise RuntimeError(f"OLS4 returned no UBERON term for {obo_id}")
    return terms[0]


def build_vocabulary(con: sqlite3.Connection) -> tuple[list[dict], list[dict], dict]:
    canonical = {r[0]:r[1] for r in con.execute(
        "SELECT canonical_label,canonical_id FROM canonical_terms WHERE field='organ' AND active=1")}
    observed = dict(con.execute("SELECT organ,count(distinct gsm) FROM geometa WHERE organ IS NOT NULL GROUP BY organ"))
    assert set(canonical) == set(MANUAL) and len(canonical) == 54
    mappings, synonyms, alias_labels = [], [], defaultdict(set)
    for label in sorted(canonical):
        oid = UBERON.get(label)
        if oid:
            term = ols_term(oid)
            preferred = term.get("label")
            stripped=label.casefold().replace("brain: ", "")
            if label.casefold()==preferred.casefold(): relation="exact_label"
            elif stripped==preferred.casefold() or stripped.rstrip('s')==preferred.casefold().rstrip('s'):
                relation="equivalent_grouped_label"
            else: relation="equivalent_ontology_synonym"
            rationale = ("GEOMeta label matches the UBERON preferred label." if relation == "exact_label" else
                         "GEOMeta category is an equivalent common/synonym label; wording or Brain grouping differs.")
            status = "reliable"
            ols_synonyms = term.get("synonyms") or []
        else:
            preferred = None; relation = "no_clean_equivalent"; status = "unmapped"
            rationale = "Heterogeneous or underspecified GEOMeta category; no single UBERON class is asserted."
            ols_synonyms = []
        mappings.append({"canonical_id":canonical[label], "canonical_label":label,
            "observed_gsms":observed.get(label,0), "uberon_id":oid, "uberon_preferred_label":preferred,
            "mapping_status":status, "mapping_relation":relation, "is_exact":relation=="exact_label",
            "broader_or_narrower":"neither_equivalent" if oid else "not_applicable_unmapped", "mapping_rationale":rationale,
            "ontology_source":"EMBL-EBI OLS4 UBERON API"})
        seen = set()
        candidates = [(label, "geometa_label", True)]
        if preferred: candidates.append((preferred, "uberon_preferred_label", True))
        candidates += [(x, "uberon_synonym", False) for x in ols_synonyms]
        candidates += [(x, "manual_safe_alias", True) for x in MANUAL[label]]
        for alias, source, enabled in candidates:
            alias = re.sub(r"\s+", " ", alias.strip())
            key = alias.casefold()
            if not alias or (key,source) in seen: continue
            seen.add((key,source))
            # OLS synonym inventory is preserved, but only reviewed labels/manual aliases drive matching.
            synonyms.append({"canonical_id":canonical[label], "canonical_label":label,
                "alias":alias, "alias_normalized":key, "alias_source":source,
                "mapping_type":"direct_anatomical", "enabled_for_mapping":enabled,
                "uberon_id":oid, "review_status":"reviewed" if enabled else "inventory_only"})
            if enabled:
                alias_labels[key].add(label)
    alternatives="|".join(re.escape(x) for x in sorted(alias_labels,key=lambda x:(-len(x),x)))
    matcher={"regex":re.compile(r"(?<![A-Za-z0-9])(?:"+alternatives+r")(?![A-Za-z0-9])",re.I),
             "alias_labels":alias_labels}
    return mappings, synonyms, matcher


def matched_labels(text: str | None, matcher: dict) -> dict[str,str]:
    if not text: return {}
    found = {}
    for match in matcher["regex"].finditer(text):
        for label in matcher["alias_labels"][match.group(0).casefold()]:
            found.setdefault(label,match.group(0))
    for child, parent in PARENTS.items():
        if child in found and parent in found: found.pop(parent, None)
    return found


def map_record(row: dict, patterns: dict, condition: str) -> dict:
    fields = list(ARCH_FIELDS)
    if condition in {"B","C","D"}: fields += GEO_SAMPLE_FIELDS
    if condition in {"C","D"}: fields += STUDY_FIELDS
    evidence = []
    if condition == "D":
        for item in row.get("gemma_items", []):
            hits = matched_labels(item[1], patterns)
            for label, phrase in hits.items():
                evidence.append((0,label,phrase,"Gemma",item[0],item[1]))
    for field, priority in fields:
        value = row.get(field)
        for label, phrase in matched_labels(value, patterns).items():
            evidence.append((priority,label,phrase,"ARCHS4" if field in dict(ARCH_FIELDS) else "GEO",field,value))
    if not evidence:
        return {"mapping_status":"unknown", "proposed_organ":None, "candidate_organs":[],
                "evidence_source":None,"evidence_field":None,"evidence_text":None,"mapping_method":None}
    best = min(x[0] for x in evidence)
    strongest = [x for x in evidence if x[0] == best]
    candidates = sorted({x[1] for x in strongest})
    selected = candidates[0] if len(candidates)==1 else None
    supports = [x for x in strongest if x[1] == selected] if selected else strongest
    return {"mapping_status":"resolved" if selected else "ambiguous", "proposed_organ":selected,
        "candidate_organs":candidates, "evidence_source":"|".join(sorted({x[3] for x in supports})),
        "evidence_field":"|".join(sorted({x[4] for x in supports})),
        "evidence_text":" || ".join(dict.fromkeys((x[5] or "")[:1000] for x in supports)),
        "matched_phrase":"|".join(sorted({x[2] for x in supports})),
        "mapping_method":"boundary_safe_direct_synonym"}


BASE_QUERY = """
SELECT s.gsm,s.species,a.title,a.source_name_ch1,a.characteristics_ch1,a.extract_protocol_ch1,
       x.description AS geo_description,x.treatment_protocol,x.growth_protocol,
       st.title AS gse_title,st.summary AS gse_summary,st.overall_design AS gse_overall_design,
       cm.organ AS existing_organ,cm.organ_source
FROM samples s JOIN archs4_metadata a USING(gsm) JOIN canonical_metadata cm USING(gsm)
LEFT JOIN geo_sample_extras x USING(gsm)
LEFT JOIN sample_studies ss ON ss.gsm=s.gsm
LEFT JOIN studies st ON st.gse=ss.gse
"""


def get_gemma(con: sqlite3.Connection, gsms: list[str]) -> dict[str,list[tuple[str,str]]]:
    out = defaultdict(list)
    for start in range(0,len(gsms),800):
        batch=gsms[start:start+800]
        q="SELECT gsm,category,raw_value FROM gemma_annotations WHERE gsm IN (%s)" % ",".join("?"*len(batch))
        for gsm,cat,val in con.execute(q,batch):
            if (cat or "").casefold() in GEMMA_CATEGORIES and val: out[gsm].append((cat,val))
    return out


def validation_cohort(con: sqlite3.Connection) -> list[dict]:
    rows=[]
    cur=con.execute(BASE_QUERY.replace("FROM samples", ",gm.organ AS true_organ FROM samples") +
                    " JOIN geometa gm ON gm.gsm=s.gsm WHERE gm.organ IS NOT NULL GROUP BY s.gsm")
    names=[x[0] for x in cur.description]
    for values in cur:
        row=dict(zip(names,values))
        row["selection_hash"]=hash_key(row["gsm"]); rows.append(row)
    by=defaultdict(list)
    for row in rows: by[row["true_organ"]].append(row)
    chosen=[]
    # At least 20/category where available, then deterministic fill proportional to abundance.
    for label in sorted(by): chosen += sorted(by[label],key=lambda x:x["selection_hash"])[:20]
    used={x["gsm"] for x in chosen}
    chosen += sorted((x for x in rows if x["gsm"] not in used),key=lambda x:x["selection_hash"])[:VALIDATION_N-len(chosen)]
    assert len(chosen)==VALIDATION_N
    gemma=get_gemma(con,[x["gsm"] for x in chosen])
    for row in chosen: row["gemma_items"]=gemma.get(row["gsm"],[])
    return chosen


def summarize_validation(df: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    overall=[]
    for condition,group in df.groupby("condition"):
        resolved=group.mapping_status.eq("resolved")
        correct=group.correct.eq(True)
        overall.append({"condition":condition,"n":len(group),"resolved_n":int(resolved.sum()),
            "coverage":resolved.mean(),"accuracy_among_mapped":correct[resolved].mean() if resolved.any() else None,
            "incorrect_mapping_rate":((resolved)&(~correct)).mean(),
            "ambiguity_rate":group.mapping_status.eq("ambiguous").mean(),
            "unknown_rate":group.mapping_status.eq("unknown").mean()})
    per=[]
    for (condition,label),group in df.groupby(["condition","true_organ"]):
        resolved=group.mapping_status.eq("resolved"); predicted=group.proposed_organ.eq(label)
        tp=(resolved&predicted).sum(); all_pred=((df.condition==condition)&df.mapping_status.eq("resolved")&df.proposed_organ.eq(label)).sum()
        per.append({"condition":condition,"canonical_organ":label,"ground_truth_n":len(group),
            "resolved_n":int(resolved.sum()),"recall":tp/len(group),"predicted_n":int(all_pred),
            "precision":tp/all_pred if all_pred else None})
    return pd.DataFrame(overall),pd.DataFrame(per)


def main() -> None:
    RESULTS.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True); con.row_factory=sqlite3.Row
    mappings,synonyms,patterns=build_vocabulary(con)
    pd.DataFrame(mappings).to_csv(RESULTS/"geometa_organ_uberon_mapping.csv",index=False)
    pd.DataFrame(synonyms).to_csv(RESULTS/"organ_synonym_dictionary.csv",index=False)
    pd.DataFrame(mappings)[["canonical_id","canonical_label","observed_gsms"]].to_csv(RESULTS/"geometa_organ_vocabulary.csv",index=False)

    cohort=validation_cohort(con)
    cohort_export=[{k:v for k,v in r.items() if k not in {"gemma_items","gse_summary","gse_overall_design","characteristics_ch1"}} for r in cohort]
    pd.DataFrame(cohort_export).to_csv(RESULTS/"deterministic_validation_cohort_5000.csv",index=False)
    predictions=[]
    for row in cohort:
        for condition in "ABCD":
            result=map_record(row,patterns,condition)
            predictions.append({"gsm":row["gsm"],"species":row["species"],"true_organ":row["true_organ"],
                "condition":condition,**result,"correct":result["proposed_organ"]==row["true_organ"] if result["proposed_organ"] else False})
    val=pd.DataFrame(predictions)
    val["candidate_organs"]=val.candidate_organs.map(json.dumps)
    val.to_parquet(RESULTS/"deterministic_validation_predictions.parquet",index=False)
    overall,per=summarize_validation(val)
    overall.to_csv(RESULTS/"deterministic_validation_summary.csv",index=False)
    per.to_csv(RESULTS/"deterministic_validation_by_organ.csv",index=False)

    canonical_ids={r["canonical_label"]:r["canonical_id"] for r in mappings}
    uberon_ids={r["canonical_label"]:r["uberon_id"] for r in mappings}
    schema=pa.schema([(x,pa.string()) for x in ["gsm","species","proposed_canonical_organ","canonical_id",
        "mapping_status","candidate_organs","mapping_method","evidence_text","evidence_field","evidence_source",
        "matched_phrase","uberon_id"]])
    writer=pq.ParquetWriter(RESULTS/"missing_organ_recovery_predictions.parquet",schema,compression="zstd")
    status=Counter(); by_species=Counter(); by_label=Counter(); by_source=Counter(); buffer=[]
    cur=con.execute(BASE_QUERY+" WHERE cm.organ IS NULL GROUP BY s.gsm")
    names=[x[0] for x in cur.description]
    processed=0; started=time.monotonic()
    while True:
        raw=cur.fetchmany(5000)
        if not raw: break
        rows=[dict(zip(names,x)) for x in raw]
        gemma=get_gemma(con,[x["gsm"] for x in rows])
        for row in rows:
            row["gemma_items"]=gemma.get(row["gsm"],[])
            result=map_record(row,patterns,"D")
            label=result["proposed_organ"]
            item={"gsm":row["gsm"],"species":row["species"],"proposed_canonical_organ":label,
                "canonical_id":canonical_ids.get(label),"mapping_status":result["mapping_status"],
                "candidate_organs":json.dumps(result["candidate_organs"]),"mapping_method":result.get("mapping_method"),
                "evidence_text":result.get("evidence_text"),"evidence_field":result.get("evidence_field"),
                "evidence_source":result.get("evidence_source"),"matched_phrase":result.get("matched_phrase"),
                "uberon_id":uberon_ids.get(label)}
            buffer.append(item); status[result["mapping_status"]]+=1
            by_species[(row["species"],result["mapping_status"])]+=1
            if label: by_label[label]+=1
            if result.get("evidence_source"): by_source[result["evidence_source"]]+=1
        writer.write_table(pa.Table.from_pylist(buffer,schema=schema)); buffer=[]
        processed += len(rows)
        if processed%50000==0:
            elapsed=time.monotonic()-started
            print(f"missing-organ {processed}/536189 elapsed={elapsed:.1f}s rate={processed/elapsed:.1f}/s",flush=True)
    writer.close()
    summary=[]
    for (species_name,state),n in sorted(by_species.items()): summary.append({"dimension":"species_status","group":species_name,"value":state,"n":n})
    for state,n in sorted(status.items()): summary.append({"dimension":"overall_status","group":"all","value":state,"n":n})
    for label,n in sorted(by_label.items()): summary.append({"dimension":"proposed_organ","group":"all","value":label,"n":n})
    for source,n in sorted(by_source.items()): summary.append({"dimension":"evidence_source","group":"all","value":source,"n":n})
    pd.DataFrame(summary).to_csv(RESULTS/"missing_organ_recovery_summary.csv",index=False)

    audit={"run_id":"organ-recovery-v1","created_at":datetime.now(timezone.utc).isoformat(),
        "database":DB.relative_to(REPO_ROOT).as_posix(),"database_open_mode":"read_only","seed":SEED,"validation_n":VALIDATION_N,
        "vocabulary_n":54,"uberon_reliable_n":sum(bool(x["uberon_id"]) for x in mappings),
        "ontology_api":"https://www.ebi.ac.uk/ols4/api/ontologies/uberon/terms",
        "rules":{"precedence":"Gemma anatomy > ARCHS4 source > characteristics/GEO description > title > protocols > study context",
            "boundaries":"negative alphanumeric lookarounds; no raw substring matching",
            "ambiguity":"multiple labels at strongest evidence tier returns ambiguous",
            "specificity":"listed child label suppresses its broad parent only within the same field",
            "study_context":"used only if no stronger sample-level or Gemma evidence exists"},
        "missing_status_counts":dict(status),"limitations":[
            "Direct anatomical aliases only; cell-type-to-organ inference is intentionally excluded.",
            "Unknown means no recognized 54-category alias, not proof that the source contains no free-text anatomical clue.",
            "GEOMeta ground truth is human-only, so held-out precision is not independently established for mouse."]}
    (RESULTS/"organ_recovery_provenance.json").write_text(json.dumps(audit,indent=2))
    con.close()


if __name__ == "__main__":
    main()
