#!/usr/bin/env python3
"""Audit ARCHS4 v2.5 metadata against selectively retrieved GEO SOFT metadata."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data"
WORK = DATA_ROOT / "geo" / "geo_archs4_audit_100"
RESULTS = ROOT / "results"
MANIFEST = REPO_ROOT / "data/manifests/archs4_sample_metadata_v2.5.parquet"
H5_FILES = {
    "human": REPO_ROOT / "data/archs4/human_gene_v2.5.h5",
    "mouse": REPO_ROOT / "data/archs4/mouse_gene_v2.5.h5",
}
COMBINED = ROOT / "results/combined_coverage.parquet"
SEED = "bridge-archs4-geo-audit-v1"

ARCHS4_SAMPLE_CATEGORIES = {
    "geo_accession": "sample_identifier",
    "series_id": "sample_identifier",
    "organism_ch1": "sample_identifier",
    "taxid_ch1": "sample_identifier",
    "platform_id": "sample_identifier",
    "title": "biological_metadata",
    "source_name_ch1": "biological_metadata",
    "characteristics_ch1": "biological_metadata",
    "molecule_ch1": "technical_metadata",
    "library_strategy": "technical_metadata",
    "library_source": "technical_metadata",
    "library_selection": "technical_metadata",
    "instrument_model": "technical_metadata",
    "channel_count": "technical_metadata",
    "type": "technical_metadata",
    "sample": "technical_metadata",
    "extract_protocol_ch1": "technical_metadata",
    "data_processing": "technical_metadata",
    "relation": "technical_metadata",
    "singlecellprobability": "technical_metadata",
    "submission_date": "technical_metadata",
    "last_update_date": "technical_metadata",
    "status": "technical_metadata",
    "contact_name": "submitter_metadata",
    "contact_institute": "submitter_metadata",
    "contact_address": "submitter_metadata",
    "contact_city": "submitter_metadata",
    "contact_zip": "submitter_metadata",
    "contact_country": "submitter_metadata",
}

DERIVED_FIELDS = {
    "geo_accession", "series_id", "organism_ch1", "title", "source_name_ch1",
    "characteristics_ch1", "platform_id", "library_strategy", "library_source",
    "instrument_model", "molecule_ch1", "submission_date", "singlecellprobability",
}

GEO_FIELD_DESCRIPTIONS = {
    "geo_accession": "Stable GEO accession.",
    "title": "Submitter-provided sample or study title.",
    "summary": "Study goals and context.",
    "overall_design": "Experimental design, groups, controls, and replication.",
    "pubmed_id": "Associated PubMed identifier.",
    "type": "GEO record or study assay type.",
    "status": "Public/release status.",
    "submission_date": "GEO submission date.",
    "last_update_date": "Last GEO record update date.",
    "source_name_ch1": "Submitter-provided biological source name.",
    "organism_ch1": "Sample organism.",
    "taxid_ch1": "NCBI taxonomy identifier.",
    "characteristics_ch1": "Repeatable free-text biological characteristics.",
    "treatment_protocol_ch1": "Sample treatment or perturbation protocol.",
    "growth_protocol_ch1": "Growth, culture, husbandry, or collection protocol.",
    "extract_protocol_ch1": "Extraction and library-preparation protocol.",
    "description": "Additional sample description.",
    "data_processing": "Data processing and alignment description.",
    "platform_id": "GEO platform accession.",
    "platform_organism": "Organism represented by study platform.",
    "sample_organism": "Organisms represented by study samples.",
    "sample_id": "GSM accessions belonging to a study.",
    "series_id": "GSE accessions containing a sample.",
    "relation": "Relations to SRA, BioSample, BioProject, or other records.",
    "library_strategy": "Sequencing library strategy.",
    "library_source": "Sequencing library source.",
    "library_selection": "Sequencing library selection method.",
    "instrument_model": "Sequencing instrument.",
    "molecule_ch1": "Extracted molecule type.",
}


def clean(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def normalized(value: object) -> str:
    text = clean(value).lower()
    text = re.sub(r"\s*[|,;]\s*", " | ", text)
    return re.sub(r"\s+", " ", text).strip(" |")


def hash_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def first_gse(value: object) -> str:
    matches = re.findall(r"GSE\d+", clean(value).upper())
    return matches[0] if matches else ""


def select_cohort() -> pd.DataFrame:
    frame = pd.read_parquet(COMBINED)
    frame["coverage_group"] = "neither"
    frame.loc[frame.geometa_matched & ~frame.gemma_matched, "coverage_group"] = "geometa_only"
    frame.loc[~frame.geometa_matched & frame.gemma_matched, "coverage_group"] = "gemma_only"
    frame.loc[frame.geometa_matched & frame.gemma_matched, "coverage_group"] = "both"
    frame["gse"] = frame.series_id.map(first_gse)
    frame["complexity"] = pd.cut(
        frame.characteristics_ch1.fillna("").str.len(),
        bins=[-1, 80, 220, float("inf")], labels=["simple", "moderate", "complex"]
    ).astype(str)
    frame["selection_hash"] = frame.gsm.map(hash_key)
    targets = {
        ("human", "both"): 13, ("human", "geometa_only"): 13,
        ("human", "gemma_only"): 12, ("human", "neither"): 12,
        ("mouse", "gemma_only"): 25, ("mouse", "neither"): 25,
    }
    picked = []
    used_gse: set[str] = set()
    for (species, group), count in targets.items():
        candidates = frame[(frame.species == species) & (frame.coverage_group == group)].copy()
        candidates["tissue_key"] = candidates.union_tissue.fillna("").str.lower().str.split(" | ").str[0]
        chosen = []
        complexity_targets = {
            level: count // 3 + (offset < count % 3)
            for offset, level in enumerate(["simple", "moderate", "complex"])
        }
        for level, level_count in complexity_targets.items():
            subset = candidates[candidates.complexity == level].copy()
            # Interleave tissue groups deterministically before falling back to hash order.
            subset = subset.sort_values(["tissue_key", "selection_hash"])
            subset["within_tissue_rank"] = subset.groupby("tissue_key").cumcount()
            subset = subset.sort_values(["within_tissue_rank", "selection_hash"])
            level_chosen = 0
            for _, row in subset.iterrows():
                if not row.gse or row.gse in used_gse:
                    continue
                chosen.append(row)
                used_gse.add(row.gse)
                level_chosen += 1
                if level_chosen == level_count:
                    break
            if level_chosen != level_count:
                raise RuntimeError(f"Could not select {level_count} {level} samples for {(species, group)}")
        if len(chosen) != count:
            raise RuntimeError(f"Could not select {count} unique-GSE samples for {(species, group)}")
        picked.extend(chosen)
    result = pd.DataFrame(picked).sort_values(["species", "coverage_group", "gsm"])
    assert len(result) == 100 and result.gsm.nunique() == 100 and result.gse.nunique() == 100
    return result.reset_index(drop=True)


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def extract_h5_rows(cohort: pd.DataFrame) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for species, path in H5_FILES.items():
        wanted = set(cohort.loc[cohort.species == species, "gsm"])
        with h5py.File(path, "r") as handle:
            group = handle["meta/samples"]
            accessions = [decode(x) for x in group["geo_accession"][:]]
            indices = sorted(i for i, gsm in enumerate(accessions) if gsm in wanted)
            if len(indices) != len(wanted):
                raise RuntimeError(f"Missing {len(wanted) - len(indices)} {species} cohort GSMs in H5")
            for i in indices:
                output[accessions[i]] = {}
            for field in group.keys():
                values = group[field][indices]
                for i, value in zip(indices, values):
                    output[accessions[i]][field] = decode(value)
        print(f"Read {len(wanted)} {species} metadata rows from {path.name}", flush=True)
    return output


def fetch_soft(accession: str) -> str:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{accession}.soft.txt"
    if path.exists():
        return path.read_text(errors="replace")
    response = requests.get(
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
        params={"acc": accession, "targ": "self", "view": "brief", "form": "text"},
        headers={"User-Agent": "BridgeRNA metadata audit (research use)"}, timeout=60,
    )
    response.raise_for_status()
    if not response.text.startswith("^"):
        raise RuntimeError(f"Unexpected GEO response for {accession}: {response.text[:120]!r}")
    path.write_text(response.text)
    time.sleep(0.34)
    return response.text


def parse_soft(text: str) -> tuple[str, dict[str, list[str]]]:
    entity = ""
    fields: dict[str, list[str]] = defaultdict(list)
    for line in text.splitlines():
        if line.startswith("^") and " = " in line:
            entity = line[1:].split(" = ", 1)[0].lower()
        elif line.startswith("!") and " = " in line:
            label, value = line[1:].split(" = ", 1)
            prefix = f"{entity.capitalize()}_"
            if label.startswith(prefix):
                label = label[len(prefix):]
            label = re.sub(r"_\d+$", "", label).lower().replace("zip/postal_code", "zip")
            fields[label].append(value.strip())
    return entity, dict(fields)


def comparison(arch: str, geo: str) -> str:
    if not arch and not geo:
        return "missing_both"
    if geo and not arch:
        return "only_GEO"
    if arch and not geo:
        return "only_ARCHS4"
    if arch == geo:
        return "already_present_identical"
    if normalized(arch) == normalized(geo):
        return "already_present_equivalent"
    if normalized(arch) in normalized(geo) or normalized(geo) in normalized(arch):
        return "GEO_adds_information" if len(geo) > len(arch) else "ARCHS4_adds_information"
    return "GEO_adds_information"


def evidence_flags(arch: dict[str, str], sample: dict[str, list[str]], study: dict[str, list[str]]) -> dict[str, bool]:
    arch_text = " ".join(clean(arch.get(x)) for x in ["title", "source_name_ch1", "characteristics_ch1"])
    added_parts = []
    for field in ["treatment_protocol_ch1", "growth_protocol_ch1", "description"]:
        value = clean(sample.get(field, []))
        if value and normalized(value) not in normalized(arch_text):
            added_parts.append(value)
    study_summary = clean(study.get("summary", []))
    design = clean(study.get("overall_design", []))
    added = " ".join(added_parts + [study_summary, design]).lower()
    patterns = {
        "genotype_context": r"genotyp|knockout|knock[- ]?down|wild[- ]?type|transgenic|mutant|cre\b|\bko\b",
        "disease_context": r"disease|cancer|tumou?r|carcinoma|infection|diabet|healthy|control|case\b|syndrome",
        "experimental_role": r"control|reference|treated|untreated|vehicle|baseline|case\b|wild[- ]?type|mock",
        "tissue_context": r"tissue|organ|brain|liver|blood|lung|heart|kidney|muscle|cell type",
        "treatment_context": r"treat|dose|expos|stimulat|inject|administ|vehicle|drug|compound|incubat",
    }
    return {
        "gse_summary": bool(study_summary),
        "overall_design": bool(design),
        "pubmed_id": bool(clean(study.get("pubmed_id", []))),
        "additional_gsm_characteristics": comparison(clean(arch.get("characteristics_ch1")), clean(sample.get("characteristics_ch1", []))) == "GEO_adds_information",
        "treatment_protocol": bool(clean(sample.get("treatment_protocol_ch1", []))),
        "growth_protocol": bool(clean(sample.get("growth_protocol_ch1", []))),
        "sample_description": bool(clean(sample.get("description", []))),
        **{key: bool(re.search(pattern, added)) for key, pattern in patterns.items()},
        "geo_adds_any_laya_evidence": bool(added_parts or study_summary or design),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cohort = select_cohort()
    cohort[["gsm", "gse", "species", "coverage_group", "complexity", "selection_hash"]].to_csv(
        RESULTS / "archs4_vs_geo_cohort_100.csv", index=False
    )
    arch_rows = extract_h5_rows(cohort)
    geo_samples: dict[str, dict[str, list[str]]] = {}
    geo_studies: dict[str, dict[str, list[str]]] = {}
    for number, row in enumerate(cohort.itertuples(), 1):
        _, geo_samples[row.gsm] = parse_soft(fetch_soft(row.gsm))
        _, geo_studies[row.gse] = parse_soft(fetch_soft(row.gse))
        if number % 10 == 0:
            print(f"Fetched/cached {number}/100 GSM+GSE pairs", flush=True)

    audit = []
    gain_rows = []
    sample_fields = sorted(set(ARCHS4_SAMPLE_CATEGORIES) | set().union(*(x.keys() for x in geo_samples.values())))
    study_fields = sorted(set().union(*(x.keys() for x in geo_studies.values())))
    cohort_lookup = cohort.set_index("gsm")
    for gsm, row in cohort_lookup.iterrows():
        arch = arch_rows[gsm]
        gsm_geo = geo_samples[gsm]
        gse_geo = geo_studies[row.gse]
        base = {"gsm": gsm, "gse": row.gse, "species": row.species,
                "coverage_group": row.coverage_group, "complexity": row.complexity}
        for field in sample_fields:
            arch_value = clean(arch.get(field))
            geo_value = clean(gsm_geo.get(field, []))
            audit.append({**base, "entity": "GSM", "field": field,
                          "archs4_value": arch_value, "geo_value": geo_value,
                          "classification": comparison(arch_value, geo_value)})
        for field in study_fields:
            arch_value = clean(arch.get("series_id")) if field == "geo_accession" else ""
            geo_value = clean(gse_geo.get(field, []))
            audit.append({**base, "entity": "GSE", "field": field,
                          "archs4_value": arch_value, "geo_value": geo_value,
                          "classification": comparison(arch_value, geo_value)})
        gain_rows.append({**base, **evidence_flags(arch, gsm_geo, gse_geo)})

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_parquet(RESULTS / "archs4_vs_geo_sample_audit.parquet", index=False)
    summary = (audit_frame.groupby(["entity", "field", "classification"], dropna=False)
               .size().rename("samples").reset_index())
    denominators = audit_frame.groupby(["entity", "field"]).size().rename("total").reset_index()
    summary = summary.merge(denominators, on=["entity", "field"])
    summary["pct"] = 100 * summary.samples / summary.total
    summary.to_csv(RESULTS / "archs4_vs_geo_field_summary.csv", index=False)

    gains = pd.DataFrame(gain_rows)
    gain_summary = []
    id_cols = {"gsm", "gse", "species", "coverage_group", "complexity"}
    for field in [x for x in gains.columns if x not in id_cols]:
        for species in ["all", "human", "mouse"]:
            subset = gains if species == "all" else gains[gains.species == species]
            count = int(subset[field].sum())
            gain_summary.append({"species": species, "geo_addition": field,
                                 "samples": count, "total": len(subset),
                                 "pct_samples_where_useful": 100 * count / len(subset)})
    pd.DataFrame(gain_summary).to_csv(RESULTS / "geo_laya_information_gain.csv", index=False)
    gains.to_csv(RESULTS / "geo_laya_information_gain_by_sample.csv", index=False)

    schema_rows = []
    h5_counts = {field: 0 for field in ARCHS4_SAMPLE_CATEGORIES}
    for gsm in cohort.gsm:
        for field in h5_counts:
            h5_counts[field] += bool(clean(arch_rows[gsm].get(field)))
    for field, category in ARCHS4_SAMPLE_CATEGORIES.items():
        schema_rows.append({"source": "ARCHS4_v2.5_H5", "entity": "sample", "field": field,
                            "category": category, "present_in_h5": True,
                            "retained_in_derived_parquet": field in DERIVED_FIELDS,
                            "nonempty_in_audit_100": h5_counts[field],
                            "description": ""})
    for field in ["metadata_archs4_file", "archs4_version"]:
        schema_rows.append({"source": "local_derived", "entity": "sample", "field": field,
                            "category": "provenance", "present_in_h5": False,
                            "retained_in_derived_parquet": True, "nonempty_in_audit_100": 100,
                            "description": "Local extraction provenance."})
    for field in ["title", "summary", "overall_design", "pubmed_id", "type", "submission_date",
                  "last_update_date", "platform_id", "platform_organism", "sample_organism",
                  "relation", "status"]:
        schema_rows.append({"source": "not_present", "entity": "study", "field": field,
                            "category": "study_level_metadata", "present_in_h5": False,
                            "retained_in_derived_parquet": False, "nonempty_in_audit_100": 0,
                            "description": "ARCHS4 v2.5 H5 contains no GSE entity table."})
    pd.DataFrame(schema_rows).to_csv(RESULTS / "archs4_metadata_schema.csv", index=False)

    geo_schema = []
    for entity, records in [("GSM", geo_samples), ("GSE", geo_studies)]:
        counts = Counter()
        repeats = Counter()
        for record in records.values():
            for field, values in record.items():
                if any(clean(x) for x in values): counts[field] += 1
                if len(values) > 1: repeats[field] += 1
        for field in sorted(counts):
            geo_schema.append({"source": "NCBI_GEO_SOFT_brief", "entity": entity,
                               "field": field, "nonempty_records_in_audit_100": counts[field],
                               "repeatable_records_in_audit_100": repeats[field],
                               "description": GEO_FIELD_DESCRIPTIONS.get(field, "Observed GEO SOFT attribute.")})
    pd.DataFrame(geo_schema).to_csv(RESULTS / "geo_metadata_schema.csv", index=False)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "cohort_size": 100, "human": 50, "mouse": 50, "unique_gse": 100,
        "archs4_h5": {k: v.relative_to(REPO_ROOT).as_posix() for k, v in H5_FILES.items()},
        "derived_manifest": MANIFEST.relative_to(REPO_ROOT).as_posix(), "geo_endpoint": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
        "geo_parameters": {"targ": "self", "view": "brief", "form": "text"},
        "metadata_only": True, "expression_downloaded": False,
        "limitations": ["Information-gain context categories use conservative keyword evidence checks, not Laya.",
                        "The 100-record cohort is deterministic and stratified, not a prevalence-weighted random sample."],
    }
    (RESULTS / "archs4_vs_geo_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Saved {len(audit_frame):,} field comparisons and {len(gains):,} sample gain rows", flush=True)


if __name__ == "__main__":
    main()
