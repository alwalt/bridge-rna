#!/usr/bin/env python3
"""Build the normalized Bridge ARCHS4 metadata SQLite database from audited sources."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
RESULTS = ROOT / "results"
DATA_ROOT = REPO_ROOT / "data"
DB = DATA_ROOT / "archs4" / "bridge_metadata.sqlite"
SCHEMA = Path(__file__).with_name("schema.sql")
LOCATIONS = REPO_ROOT / "embeddings/archs4/sample_locations.parquet"
MANIFEST = REPO_ROOT / "data/manifests/archs4_sample_metadata_v2.5.parquet"
H5 = {
    "human": REPO_ROOT / "data/archs4/human_gene_v2.5.h5",
    "mouse": REPO_ROOT / "data/archs4/mouse_gene_v2.5.h5",
}
COMBINED = RESULTS / "combined_coverage.parquet"
GEO_CACHE = DATA_ROOT / "geo" / "geo_family_soft"
GEOMETA = DATA_ROOT / "geometa"
GEMMA = DATA_ROOT / "gemma" / "samples"
SCHEMA_VERSION = "bridge-metadata-v1"
VOCAB_VERSION = "geometa-hf-d0cd1e1"

H5_FIELDS = [
    "geo_accession", "series_id", "organism_ch1", "taxid_ch1", "platform_id",
    "title", "source_name_ch1", "characteristics_ch1", "molecule_ch1",
    "library_strategy", "library_source", "library_selection", "instrument_model",
    "channel_count", "type", "sample", "extract_protocol_ch1", "data_processing",
    "relation", "singlecellprobability", "submission_date", "last_update_date",
    "status", "contact_name", "contact_institute", "contact_address", "contact_city",
    "contact_zip", "contact_country",
]
GEOMETA_MAP = {
    "Organ": "organ", "Disease": "disease", "Gender": "gender",
    "Age_Group": "age_group", "Exp_Setting": "experimental_setting",
    "GSE_Pert": "perturbation",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value).strip()
    return value or None


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    return con


def setup_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing database: {path}")
    con = connect(path)
    con.executescript(SCHEMA.read_text())
    return con


def import_samples(con: sqlite3.Connection) -> set[str]:
    locations = pd.read_parquet(LOCATIONS).sort_values("global_index")
    metadata = pd.read_parquet(MANIFEST, columns=["gsm", "series_id", "organism_ch1"])
    merged = locations.merge(metadata, left_on="geo_accession", right_on="gsm", validate="one_to_one")
    assert len(merged) == 940455
    assert merged.global_index.tolist() == list(range(len(merged)))
    species = merged.organism_ch1.map({"Homo sapiens": "human", "Mus musculus": "mouse"})
    assert species.notna().all()
    con.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?)", zip(
        merged.gsm, merged.global_index.astype(int), species, merged.species_id.astype(int),
        merged.shard_idx.astype(int), merged.shard_file, merged.row_in_shard.astype(int)))
    mappings = []
    for gsm, raw in zip(merged.gsm, merged.series_id):
        for gse in sorted(set(re.findall(r"GSE\d+", str(raw).upper()))):
            mappings.append((gsm, gse))
    con.executemany("INSERT OR IGNORE INTO sample_studies VALUES (?,?)", mappings)
    con.commit()
    print(f"samples={len(merged):,} sample_studies={len(mappings):,}", flush=True)
    return set(merged.gsm)


def import_archs4(con: sqlite3.Connection, required: set[str], chunk_size: int = 5000) -> None:
    placeholders = ",".join("?" for _ in range(len(H5_FIELDS) + 3))
    sql = f"INSERT INTO archs4_metadata VALUES ({placeholders})"
    total = 0
    for species, path in H5.items():
        started = time.monotonic()
        with h5py.File(path, "r") as handle:
            group = handle["meta/samples"]
            accessions = [text(x) for x in group["geo_accession"][:]]
            indices = [i for i, gsm in enumerate(accessions) if gsm in required]
            for offset in range(0, len(indices), chunk_size):
                selected = indices[offset:offset + chunk_size]
                columns = {}
                for field in H5_FIELDS:
                    values = group[field][selected]
                    columns[field] = [float(x) if field == "singlecellprobability" else text(x) for x in values]
                rows = []
                for row_idx, source_idx in enumerate(selected):
                    gsm = accessions[source_idx]
                    rows.append((gsm,) + tuple(columns[field][row_idx] for field in H5_FIELDS) + (path.name, "2.5"))
                con.executemany(sql, rows)
                con.commit()
                total += len(rows)
                if total % 50000 < chunk_size:
                    elapsed = time.monotonic() - started
                    print(f"ARCHS4 {species}: {offset + len(selected):,}/{len(indices):,} total={total:,} elapsed={elapsed:.1f}s", flush=True)
    assert total == len(required), (total, len(required))


def parse_soft(path: Path):
    entity = None
    accession = None
    fields: dict[str, list[str]] = {}
    with gzip.open(path, "rt", errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if line.startswith("^") and " = " in line:
                if entity:
                    yield entity, accession, fields
                label, accession = line[1:].split(" = ", 1)
                entity, fields = label.upper(), {}
            elif entity and line.startswith("!") and " = " in line:
                label, value = line[1:].split(" = ", 1)
                prefix = f"{entity.capitalize()}_"
                if label.startswith(prefix):
                    label = label[len(prefix):]
                key = re.sub(r"_\d+$", "", label).lower().replace("zip/postal_code", "zip")
                fields.setdefault(key, []).append(value.strip())
        if entity:
            yield entity, accession, fields


def joined(fields: dict[str, list[str]], key: str) -> str | None:
    values = [x.strip() for x in fields.get(key, []) if x.strip()]
    return " | ".join(dict.fromkeys(values)) or None


def relation_parts(values: list[str]) -> tuple[str | None, str | None, str | None]:
    bio, sra, other = [], [], []
    for value in values:
        lower = value.lower()
        (bio if "bioproject" in lower else sra if "sra" in lower else other).append(value)
    combine = lambda x: " | ".join(dict.fromkeys(x)) or None
    return combine(bio), combine(sra), combine(other)


def import_geo(con: sqlite3.Connection, required: set[str]) -> None:
    manifest_path = RESULTS / "geo_download_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    if not manifest.empty:
        cols = ["gse", "status", "http_status", "bytes", "sha256", "url", "error", "retrieved_at"]
        con.executemany("INSERT OR REPLACE INTO geo_downloads VALUES (?,?,?,?,?,?,?,?)",
                        manifest.reindex(columns=cols).where(pd.notna(manifest), None).itertuples(index=False, name=None))
    studies, extras = [], []
    files = sorted(GEO_CACHE.glob("GSE*_metadata.soft.txt.gz"), key=lambda p: int(p.name.split("_", 1)[0][3:]))
    retrieved = now()
    for number, path in enumerate(files, 1):
        gse = path.name.split("_", 1)[0]
        bucket = re.sub(r"\d{1,3}$", "nnn", gse)
        with gzip.open(path, "rb") as source:
            digest = hashlib.sha256(source.read()).hexdigest()
        for entity, accession, fields in parse_soft(path):
            if entity == "SERIES" and accession == gse:
                bio, sra, other = relation_parts(fields.get("relation", []))
                studies.append((gse, joined(fields,"title"), joined(fields,"summary"),
                    joined(fields,"overall_design"), joined(fields,"pubmed_id"), joined(fields,"type"),
                    joined(fields,"submission_date"), joined(fields,"last_update_date"),
                    joined(fields,"sample_organism") or joined(fields,"platform_organism"),
                    joined(fields,"sample_taxid") or joined(fields,"platform_taxid"),
                    joined(fields,"platform_id"), bio, sra, other, joined(fields,"status"),
                    f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=self&view=brief&form=text",
                    retrieved, digest))
            elif entity == "SAMPLE" and accession in required:
                treatment, growth, description = (joined(fields, x) for x in
                    ["treatment_protocol_ch1", "growth_protocol_ch1", "description"])
                if treatment or growth or description:
                    source_locator = path.relative_to(REPO_ROOT).as_posix()
                    extras.append((accession, treatment, growth, description,
                        joined(fields,"last_update_date"), gse, source_locator, retrieved))
        if len(studies) >= 500:
            con.executemany("INSERT OR REPLACE INTO studies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", studies)
            con.executemany("INSERT OR REPLACE INTO geo_sample_extras VALUES (?,?,?,?,?,?,?,?)", extras)
            con.commit(); studies.clear(); extras.clear()
        if number % 5000 == 0:
            print(f"GEO parsed {number:,}/{len(files):,}", flush=True)
    con.executemany("INSERT OR REPLACE INTO studies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", studies)
    con.executemany("INSERT OR REPLACE INTO geo_sample_extras VALUES (?,?,?,?,?,?,?,?)", extras)
    con.commit()
    print(f"GEO files parsed={len(files):,}", flush=True)


def canonical_id(field: str, label: str) -> str:
    slug = hashlib.sha1(f"{field}\0{label}".encode()).hexdigest()[:16]
    return f"geometa:{field}:{slug}"


def geometa_text(value: object) -> str | None:
    value = text(value)
    return None if value is None or value.casefold() in {"na", "n/a", "nan", "null"} else value


def import_geometa(con: sqlite3.Connection, required: set[str]) -> dict[str, dict[str, str]]:
    vocabulary: dict[str, dict[str, str]] = {x: {} for x in GEOMETA_MAP.values()}
    source_frames = []
    for path in sorted(GEOMETA.glob("*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        source_frames.append((path, frame))
        for source_col, field in GEOMETA_MAP.items():
            for raw_value in frame[source_col].astype(str).str.strip().unique():
                value = geometa_text(raw_value)
                if value:
                    vocabulary[field][value.casefold()] = value
    terms = []
    mappings = []
    for field, values in vocabulary.items():
        for label in sorted(values.values(), key=str.casefold):
            cid = canonical_id(field, label)
            terms.append((cid,field,label,"GEOMeta",None,None,None,VOCAB_VERSION,1))
            mappings.append(("GEOMeta",field,label,cid,"identity","valid",None,"Observed release value"))
    con.executemany("INSERT INTO canonical_terms VALUES (?,?,?,?,?,?,?,?,?)", terms)
    con.executemany("""INSERT INTO canonical_mappings
        (source,field,raw_value,canonical_id,mapping_method,mapping_status,ontology_id,notes)
        VALUES (?,?,?,?,?,?,?,?)""", mappings)
    con.commit()
    uberon_path = RESULTS / "uberon_mappings.csv"
    if uberon_path.exists():
        uberon = pd.read_csv(uberon_path).where(lambda x: pd.notna(x), None)
        for row in uberon.itertuples(index=False):
            if row.mapping_status == "reliable":
                con.execute("""UPDATE canonical_terms SET ontology_id=?, ontology_label=?, ontology_name=?
                    WHERE field='organ' AND canonical_label=? AND source_vocabulary='GEOMeta'""",
                    (row.ontology_id,row.ontology_label,row.ontology_name,row.canonical_label))
                con.execute("""UPDATE canonical_mappings SET ontology_id=?, notes=?
                    WHERE source='GEOMeta' AND field='organ' AND raw_value=?""",
                    (row.ontology_id,row.mapping_method,row.canonical_label))
        con.commit()
    row_id = 0
    assertion_rows = []
    for path, frame in source_frames:
        rows = []
        for record in frame.to_dict("records"):
            gsm = text(record.get("GSM_ID")); gse = text(record.get("GSE_ID"))
            if gsm not in required:
                continue
            row_id += 1
            rows.append((row_id, gsm, gse, geometa_text(record.get("year")), geometa_text(record.get("contact_country")),
                geometa_text(record.get("instrument_model")), geometa_text(record.get("RNA_Library")),
                geometa_text(record.get("Exp_Setting")), geometa_text(record.get("GSE_Pert")), geometa_text(record.get("Disease")),
                geometa_text(record.get("Organ")), geometa_text(record.get("Gender")), geometa_text(record.get("Age_Group")),
                path.name, "d0cd1e117143b32a392bbe3130e1c4a2a3c8e740"))
            for source_col, field in GEOMETA_MAP.items():
                value = geometa_text(record.get(source_col))
                if not value: continue
                cid = canonical_id(field, value)
                assertion_rows.append((gsm, field, value, value, cid, "GEOMeta", "curated_resource",
                    None,None,None,None,None,str(path.name),None,None,None,SCHEMA_VERSION,None,"active"))
        con.executemany("INSERT INTO geometa VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.executemany("""INSERT INTO metadata_assertions
            (gsm,field,raw_value,canonical_value,canonical_id,source,annotation_method,
             model_name,model_version,model_commit,model_score,evidence_text,evidence_source,
             ontology_id,ontology_name,annotation_run_id,schema_version,assertion_timestamp,status)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", assertion_rows)
        assertion_rows.clear(); con.commit()
        print(f"GEOMeta imported {path.name}: {len(frame):,}", flush=True)
    return vocabulary


def mapped_field(category: str) -> str | None:
    c = category.casefold().strip()
    if any(x in c for x in ["tissue", "organism part", "brain region", "source organ"]): return "organ"
    if c in {"disease", "disease state", "diagnosis", "disease model", "disease staging"}: return "disease"
    if c in {"sex", "gender", "biological sex"}: return "gender"
    if "cell type" in c or c == "cell subtype": return "cell_type"
    if "strain" in c: return "strain"
    if "genotype" in c: return "genotype"
    if any(x in c for x in ["treatment", "drug", "exposure"]): return "treatment"
    if any(x in c for x in ["dose", "concentration"]): return "dose"
    if any(x in c for x in ["timepoint", "time point", "duration", "developmental stage", "age"]): return "duration_development"
    if c in {"condition", "group", "phenotype", "growth condition", "study design"}: return "experimental_setting"
    return None


def import_gemma(con: sqlite3.Connection, required: set[str], vocabulary: dict[str, dict[str, str]]) -> None:
    insert = """INSERT INTO gemma_annotations
      (gsm,gse,annotation_scope,category,category_uri,raw_value,value_uri,factor_id,
       factor_type,factor_value_ontology_id,gemma_sample_id,source_file)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
    assertion_insert = """INSERT INTO metadata_assertions
      (gsm,field,raw_value,canonical_value,canonical_id,source,annotation_method,
       model_name,model_version,model_commit,model_score,evidence_text,evidence_source,
       ontology_id,ontology_name,annotation_run_id,schema_version,assertion_timestamp,status)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    files = sorted(GEMMA.glob("*.json"))
    annotation_rows, assertion_rows, valid_mappings = [], [], set()
    gender_synonyms = {"female sex":"female", "male sex":"male", "woman":"female", "man":"male"}
    def record_mapping(field: str | None, value: str, ontology_id: str | None) -> None:
        if not field or field not in vocabulary: return
        key = gender_synonyms.get(value.casefold(), value.casefold()) if field == "gender" else value.casefold()
        label = vocabulary[field].get(key)
        if label:
            valid_mappings.add(("Gemma",field,value,canonical_id(field,label),"exact_normalized",
                                "valid",ontology_id,"Unique exact match to observed GEOMeta vocabulary"))
    for number, path in enumerate(files, 1):
        try: payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError): continue
        for sample in payload.get("data", payload if isinstance(payload, list) else []) or []:
            gsm = text((sample.get("accession") or {}).get("accession") or sample.get("shortName"))
            if not gsm or not gsm.startswith("GSM") or gsm not in required: continue
            description = text((sample.get("sample") or {}).get("description")) or ""
            match = re.search(r"(GSE\d+)", description)
            gse = match.group(1) if match else path.stem
            sample_obj = sample.get("sample") or sample
            sid = text(sample_obj.get("id") or sample.get("id"))
            for item in sample_obj.get("characteristics") or sample.get("characteristics") or []:
                category, value = text(item.get("category")), text(item.get("value"))
                if not value: continue
                row = (gsm,gse,"sample_characteristic",category,text(item.get("categoryUri")),value,
                       text(item.get("valueUri")),None,None,None,sid,path.name)
                annotation_rows.append(row)
                field = mapped_field(category or "")
                record_mapping(field, value, text(item.get("valueUri")))
                if field:
                    assertion_rows.append((gsm,field,value,None,None,"Gemma","curated_resource",None,None,None,None,
                        None,path.name,text(item.get("valueUri")),None,None,SCHEMA_VERSION,None,"active"))
            factors = sample_obj.get("factorValueObjects") or sample_obj.get("fvbasicVOs") or []
            for factor in factors:
                category_obj = factor.get("experimentalFactorCategory") or {}
                category = text(category_obj.get("category") or category_obj.get("value"))
                chars = factor.get("characteristics") or []
                values = chars or [{"value": factor.get("value") or factor.get("summary")}]
                for item in values:
                    value = text(item.get("value")); item_category = text(item.get("category")) or category
                    if not value: continue
                    value_uri = text(item.get("valueUri"))
                    annotation_rows.append((gsm,gse,"experimental_factor",item_category,
                        text(item.get("categoryUri") or category_obj.get("categoryUri") or category_obj.get("valueUri")),
                        value,value_uri,text(factor.get("experimentalFactorId")),text(factor.get("experimentalFactorType")),
                        text(factor.get("ontologyId")),sid,path.name))
                    field = mapped_field(item_category or "")
                    record_mapping(field, value, value_uri)
                    if field:
                        assertion_rows.append((gsm,field,value,None,None,"Gemma","curated_resource",None,None,None,None,
                            None,path.name,value_uri,None,None,SCHEMA_VERSION,None,"active"))
        if len(annotation_rows) >= 50000 or number == len(files):
            con.executemany(insert, annotation_rows); con.executemany(assertion_insert, assertion_rows)
            con.commit(); annotation_rows.clear(); assertion_rows.clear()
        if number % 1000 == 0:
            print(f"Gemma parsed {number:,}/{len(files):,}", flush=True)
    con.executemany("""INSERT OR IGNORE INTO canonical_mappings
      (source,field,raw_value,canonical_id,mapping_method,mapping_status,ontology_id,notes)
      VALUES (?,?,?,?,?,?,?,?)""", sorted(valid_mappings, key=lambda row: tuple("" if x is None else str(x) for x in row)))
    con.execute("""UPDATE metadata_assertions
      SET canonical_id=(SELECT m.canonical_id FROM canonical_mappings m
                        WHERE m.source='Gemma' AND m.field=metadata_assertions.field
                          AND m.raw_value=metadata_assertions.raw_value),
          canonical_value=(SELECT t.canonical_label FROM canonical_mappings m
                           JOIN canonical_terms t USING(canonical_id)
                           WHERE m.source='Gemma' AND m.field=metadata_assertions.field
                             AND m.raw_value=metadata_assertions.raw_value)
      WHERE source='Gemma' AND EXISTS (
        SELECT 1 FROM canonical_mappings m WHERE m.source='Gemma'
          AND m.field=metadata_assertions.field AND m.raw_value=metadata_assertions.raw_value)""")
    con.commit()


def clean_choice(value: object, vocabulary: dict[str, str], synonyms: dict[str, str] | None = None) -> str | None:
    value = text(value)
    if not value: return None
    synonyms = synonyms or {}
    candidates = [x.strip() for x in value.split(" | ") if x.strip()]
    mapped = {vocabulary.get(synonyms.get(x.casefold(), x.casefold())) for x in candidates}
    mapped.discard(None)
    return next(iter(mapped)) if len(mapped) == 1 else None


def build_canonical(con: sqlite3.Connection, vocabulary: dict[str, dict[str, str]]) -> None:
    frame = pd.read_parquet(COMBINED)
    rows = []
    for r in frame.itertuples(index=False):
        values, sources = {}, {}
        source_pairs = {
            "organ": (r.geometa_tissue, r.gemma_tissue), "disease": (r.geometa_disease, r.gemma_disease),
            "gender": (r.geometa_sex, r.gemma_sex), "age_group": (r.geometa_age_development, r.gemma_age_development),
            "experimental_setting": (r.geometa_experimental_factor_group, None),
            "perturbation": (r.geometa_treatment_perturbation, None),
        }
        for field, (geo_value, gemma_value) in source_pairs.items():
            if text(geo_value): values[field], sources[field] = text(geo_value), "GEOMeta"
            elif text(gemma_value):
                synonyms = {"female sex":"female", "male sex":"male", "woman":"female", "man":"male"} if field == "gender" else None
                mapped = clean_choice(gemma_value, vocabulary.get(field, {}), synonyms)
                values[field], sources[field] = mapped, "Gemma" if mapped else None
            else: values[field], sources[field] = None, None
        for field, value in {
            "cell_type":r.gemma_cell_type, "strain":r.gemma_strain, "genotype":r.gemma_genotype,
            "treatment":r.gemma_treatment_perturbation, "dose":r.gemma_dose,
            "duration_development":text(r.gemma_duration) or text(r.gemma_age_development),
        }.items():
            values[field], sources[field] = text(value), "Gemma" if text(value) else None
        order = ["organ","disease","gender","age_group","experimental_setting","perturbation",
                 "cell_type","strain","genotype","treatment","dose","duration_development"]
        rows.append((r.gsm, *(values[x] for x in order), *(sources[x] for x in order), 0))
    con.executemany("INSERT INTO canonical_metadata VALUES (" + ",".join("?" for _ in range(26)) + ")", rows)
    # Conservatively load only explicitly marked obvious conflicts from the prior audit.
    conflicts_path = RESULTS / "geometa_gemma_conflicts.parquet"
    if conflicts_path.exists():
        conflicts = pd.read_parquet(conflicts_path)
        conflicts = conflicts[conflicts.relation == "obvious_conflict"]
        out = []
        for r in conflicts.itertuples(index=False):
            out.append((r.gsm,r.field,"GEOMeta",r.geometa_value,"Gemma",r.gemma_value,"unresolved",r.relation))
        con.executemany("""INSERT INTO metadata_conflicts
          (gsm,field,source_a,value_a,source_b,value_b,status,notes) VALUES (?,?,?,?,?,?,?,?)""", out)
        con.execute("UPDATE canonical_metadata SET has_conflict=1 WHERE gsm IN (SELECT gsm FROM metadata_conflicts)")
    con.commit()


def write_validation(con: sqlite3.Connection) -> None:
    checks = {}
    for name, sql in {
        "samples":"SELECT COUNT(*) FROM samples", "human":"SELECT COUNT(*) FROM samples WHERE species='human'",
        "mouse":"SELECT COUNT(*) FROM samples WHERE species='mouse'", "gses":"SELECT COUNT(DISTINCT gse) FROM sample_studies",
        "multi_gse_samples":"SELECT COUNT(*) FROM (SELECT gsm FROM sample_studies GROUP BY gsm HAVING COUNT(*)>1)",
        "archs4_metadata":"SELECT COUNT(*) FROM archs4_metadata", "studies":"SELECT COUNT(*) FROM studies",
        "geo_sample_extras":"SELECT COUNT(*) FROM geo_sample_extras", "geometa_gsms":"SELECT COUNT(DISTINCT gsm) FROM geometa",
        "gemma_gsms":"SELECT COUNT(DISTINCT gsm) FROM gemma_annotations",
    }.items(): checks[name] = con.execute(sql).fetchone()[0]
    assert checks["samples"] == checks["archs4_metadata"] == 940455
    assert con.execute("SELECT COUNT(*) FROM samples s JOIN archs4_metadata a USING(gsm) WHERE s.gsm != a.geo_accession").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM samples WHERE global_index < 0 OR global_index >= 940455").fetchone()[0] == 0
    checks["foreign_key_violations"] = len(con.execute("PRAGMA foreign_key_check").fetchall())
    (RESULTS / "database_validation.json").write_text(json.dumps(checks, indent=2))
    coverage = []
    for field in ["organ","disease","gender","age_group","experimental_setting","perturbation",
                  "cell_type","strain","genotype","treatment","dose","duration_development"]:
        count = con.execute(f"SELECT COUNT(*) FROM canonical_metadata WHERE {field} IS NOT NULL AND {field} != ''").fetchone()[0]
        coverage.append({"field":field,"available":count,"total":checks["samples"],"pct":100*count/checks["samples"]})
    pd.DataFrame(coverage).to_csv(RESULTS / "final_metadata_coverage_summary.csv", index=False)
    pd.read_sql_query("SELECT * FROM canonical_terms ORDER BY field,canonical_label", con).to_csv(
        RESULTS / "canonical_terms.csv", index=False)
    pd.read_sql_query("SELECT * FROM canonical_mappings WHERE source='GEOMeta' ORDER BY field,raw_value", con).to_csv(
        RESULTS / "geometa_canonical_mappings.csv", index=False)
    pd.read_sql_query("SELECT * FROM canonical_mappings WHERE source='Gemma' ORDER BY field,raw_value", con).to_csv(
        RESULTS / "gemma_canonical_mappings.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DB)
    parser.add_argument("--local-only", action="store_true",
                        help="Build reusable local ARCHS4/GEOMeta/Gemma staging DB, then stop before GEO.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a --local-only staging DB after GEO downloads finish.")
    args = parser.parse_args()
    if args.local_only and args.resume:
        raise SystemExit("Choose only one of --local-only or --resume")
    output = args.output.resolve(); temporary = output.with_suffix(".sqlite.tmp")
    if args.resume:
        if not temporary.exists(): raise FileNotFoundError(f"No staged database: {temporary}")
        con = connect(temporary)
    else:
        if temporary.exists():
            raise FileExistsError(f"Refusing to overwrite staged database: {temporary}")
        con = setup_database(temporary)
    try:
        if args.resume:
            required = {x[0] for x in con.execute("SELECT gsm FROM samples")}
            if len(required) != 940455: raise RuntimeError("Staged samples table is incomplete")
            vocabulary: dict[str, dict[str, str]] = {x: {} for x in GEOMETA_MAP.values()}
            for field, label in con.execute("SELECT field,canonical_label FROM canonical_terms WHERE source_vocabulary='GEOMeta'"):
                vocabulary[field][label.casefold()] = label
        else:
            required = import_samples(con)
            import_archs4(con, required)
            vocabulary = import_geometa(con, required)
            import_gemma(con, required, vocabulary)
            if args.local_only:
                con.execute("PRAGMA optimize"); con.commit(); con.close()
                print(f"Local staging complete: {temporary} ({temporary.stat().st_size:,} bytes)", flush=True)
                return
        import_geo(con, required)
        if con.execute("SELECT COUNT(*) FROM canonical_metadata").fetchone()[0] == 0:
            build_canonical(con, vocabulary)
        write_validation(con)
        con.execute("PRAGMA optimize"); con.commit(); con.close()
        if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
        temporary.replace(output)
        print(f"Database complete: {output} ({output.stat().st_size:,} bytes)", flush=True)
    except Exception:
        con.close()
        raise


if __name__ == "__main__":
    main()
