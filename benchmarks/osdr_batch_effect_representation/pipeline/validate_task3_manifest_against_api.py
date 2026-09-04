#!/usr/bin/env python3
"""Audit the Task 3 sample/contrast manifests against NASA OSDR BIOdata API metadata.

This is deliberately read-only with respect to the Task 3 manifests. Discrepancies
are reported, classified, and assigned an impact; they are never auto-corrected.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "task3_manifest_api_validation"
OUT.mkdir(parents=True, exist_ok=True)
STUDIES = [47, 48, 137, 168, 173, 242, 245]
API = "https://visualization.osdr.nasa.gov/biodata/api/v2"


def clean(value: object) -> str:
    if pd.isna(value) or str(value).strip().lower() in {"", "nan", "{not available}", "{not applicable}"}:
        return "not_reported"
    return re.sub(r"\s+", " ", str(value).strip())


def norm(value: object) -> str:
    value = clean(value).lower().replace("female", "f").replace("male", "m")
    return re.sub(r"[^a-z0-9]+", "", value)


def fetch_api() -> tuple[pd.DataFrame, dict]:
    frames, datasets = [], {}
    params = [
        ("study.characteristics", ""), ("study.factor value", ""),
        ("study.parameter value", ""), ("study.comment", ""),
        ("assay", ""), ("investigation.study assays", ""), ("format", "csv"),
    ]
    for study in STUDIES:
        accession = f"OSD-{study}"
        response = requests.get(
            f"{API}/query/metadata/", params=[("id.accession", accession), *params], timeout=120
        )
        response.raise_for_status()
        cache = OUT / f"api_{accession}_sample_metadata.csv"
        cache.write_bytes(response.content)
        frame = pd.read_csv(cache, dtype=str, low_memory=False)
        # Wildcard queries may return non-RNA assays. Keep the RNA-seq assay used
        # by Task 3 and collapse exact duplicate sample rows only.
        tech = frame.get("investigation.study assays.study assay technology type", pd.Series("", index=frame.index))
        assay = frame.get("id.assay name", pd.Series("", index=frame.index))
        frame = frame[tech.str.contains("RNA Sequencing", case=False, na=False) |
                      assay.str.contains("rna-sequencing", case=False, na=False)].copy()
        frame = frame.drop_duplicates("id.sample name", keep="last")
        frames.append(frame)

        ds = requests.get(f"{API}/dataset/{accession}/", timeout=120)
        ds.raise_for_status()
        payload = ds.json()[accession]
        datasets[accession] = payload
        (OUT / f"api_{accession}_dataset.json").write_text(json.dumps(payload, indent=2))
    api = pd.concat(frames, ignore_index=True, sort=False)
    api.to_csv(OUT / "api_sample_metadata_combined.csv", index=False)
    return api, datasets


def api_mission(osd: str, sample: str, datasets: dict) -> str:
    project = clean(datasets[osd]["metadata"].get("project identifier"))
    if osd == "OSD-168":
        return "RR3" if "_RR3_" in sample else "RR1"
    if osd == "OSD-173":
        return clean(datasets[osd]["metadata"].get("mission", {}).get("name"))
    return project.replace("RR-", "RR")


def local_mission_base(value: str) -> str:
    if value.startswith("RR1_"): return "RR1"
    if value.startswith("RR6_"): return "RR6"
    return value.replace("_", "-")


def stratum(sample: str, osd: str) -> str:
    if osd == "OSD-245": return "ISS-T" if "ISS-T" in sample else "LAR"
    if osd == "OSD-48":
        return "carcass" if re.search(r"_(?:FLT|GC)_C_", sample) else "immediate"
    return "not_applicable"


def api_duration(row: pd.Series) -> str:
    for col in ["study.factor value.duration", "study.parameter value.duration",
                "study.parameter value.exposure duration"]:
        value = clean(row.get(col))
        if value != "not_reported": return value
    return "not_reported"


def api_library(row: pd.Series) -> str:
    value = clean(row.get("assay.parameter value.library selection"))
    if "ribo" in value.lower(): return "ribodepleted"
    if "polya" in norm(value): return "polyA"
    return value


def api_seq(row: pd.Series) -> str:
    layout = clean(row.get("assay.parameter value.library layout"))
    length = clean(row.get("assay.parameter value.read length"))
    instrument = clean(row.get("assay.parameter value.sequencing instrument"))
    return "; ".join(x for x in [layout, length, instrument] if x != "not_reported") or "not_reported"


def api_facility(osd: str, datasets: dict) -> str:
    descriptions = datasets[osd]["metadata"].get("study protocol description", [])
    if not isinstance(descriptions, list): descriptions = [descriptions]
    text = " ".join(map(str, descriptions)).lower()
    if "uc davis genome center" in text: return "UC Davis"
    if "genelab sample processing lab" in text: return "GeneLab SPL"
    return "not_reported"


def comparison_status(field: str, local: str, api: str) -> tuple[str, str, str]:
    if api == "not_reported":
        return "not_comparable", "missing/ambiguous API metadata", "annotation_only"
    if local == "not_reported":
        return "mismatch", "our parsing/mapping logic", "annotation_only"
    if field == "mission": equal = local_mission_base(local) == api
    elif field == "sequencing_configuration":
        # Compare layout/read length. Facility is separately flagged when the
        # API does not expose it as a structured sample-level field.
        expected_layout = "PAIRED" if local.startswith("PE") else "SINGLE"
        bp = re.search(r"(\d+)bp", local)
        equal = expected_layout in api and (not bp or bp.group(1) in api)
    else: equal = norm(local) == norm(api)
    if equal: return "match", "none", "none"
    impact = "sample_membership_or_contrast" if field in {"OSD", "condition", "within_study_stratum"} else "annotation_only"
    return "mismatch", "our parsing/mapping logic", impact


def main() -> None:
    api, datasets = fetch_api()
    local = pd.read_csv(RESULTS / "sample_manifest.csv", dtype=str).fillna("not_reported")
    members = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv", dtype=str)
    contrasts = pd.read_csv(RESULTS / "task3b_contrast_summary.csv", dtype=str)
    local = local.merge(members[["sample_id", "contrast_id"]], on="sample_id", validate="one_to_one")
    api = api.rename(columns={"id.accession": "api_OSD", "id.sample name": "sample_id"})
    merged = local.merge(api, on="sample_id", how="left", validate="one_to_one", indicator=True)

    rows = []
    for _, r in merged.iterrows():
        osd, sample = r["OSD"], r["sample_id"]
        api_osd = clean(r.get("api_OSD"))
        fields = {
            "OSD": (osd, api_osd),
            "mission": (clean(r["mission"]), api_mission(osd, sample, datasets) if api_osd != "not_reported" else "not_reported"),
            "within_study_stratum": (stratum(sample, osd), stratum(sample, api_osd) if api_osd != "not_reported" else "not_reported"),
            "condition": (clean(r["condition"]), {"Space Flight": "FLT", "Ground Control": "GC"}.get(clean(r.get("study.factor value.spaceflight")), clean(r.get("study.factor value.spaceflight")))),
            "strain": (clean(r["strain"]), clean(r.get("study.characteristics.strain"))),
            "sex": (clean(r["sex"]), clean(r.get("study.characteristics.sex"))),
            "flight_duration": (clean(r.get("flight_duration", contrasts.set_index("contrast_id").get("flight_duration", pd.Series()).get(r["contrast_id"], "not_reported"))), api_duration(r)),
            "library_preparation": (clean(r["library_preparation"]), api_library(r)),
            "sequencing_configuration": (clean(r["sequencing_parameters"]), api_seq(r)),
            "sequencing_facility": (clean(r["sequencing_facility"]), api_facility(osd, datasets)),
            "sample_identifier": (sample, clean(r.get("assay.sample name", r.get("sample_id")))),
            "animal_identifier": ("not_reported", clean(r.get("study.comment.alsda subject id", r.get("study.comment.rfid")))),
        }
        for field, (lv, av) in fields.items():
            status, cause, impact = comparison_status(field, lv, av)
            if field == "sequencing_configuration" and status == "match":
                # Layout/read length match. The local label also contains a
                # nominal requested depth that is not consistently structured
                # in the API, so record this as a partial match.
                status, cause, impact = "partial_match", "missing/ambiguous API metadata", "annotation_only"
            rows.append({"contrast_id": r["contrast_id"], "sample_id": sample, "field": field,
                         "local_value": lv, "api_value": av, "status": status,
                         "discrepancy_source": cause, "impact": impact})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "sample_field_validation_long.csv", index=False)

    # One row per contrast and field, retaining all distinct local/API values.
    summary = (audit.groupby(["contrast_id", "field"], sort=False)
               .agg(local_value=("local_value", lambda s: " | ".join(sorted(set(map(str, s))))),
                    api_value=("api_value", lambda s: " | ".join(sorted(set(map(str, s))))),
                    match_status=("status", lambda s: "match" if set(s) == {"match"} else
                                  ("partial_match" if set(s) <= {"match", "partial_match"} else
                                   ("not_comparable" if set(s) == {"not_comparable"} else "mismatch"))),
                    discrepancy_source=("discrepancy_source", lambda s: " | ".join(sorted(set(s)-{"none"})) or "none"),
                    impact=("impact", lambda s: " | ".join(sorted(set(s)-{"none"})) or "none"),
                    samples=("sample_id", "nunique"))
               .reset_index())
    summary.to_csv(OUT / "contrast_field_validation_long.csv", index=False)

    mismatch = summary[summary.match_status.ne("match")].copy()
    mismatch.to_csv(OUT / "discrepancies.csv", index=False)
    counts = (summary.groupby(["field", "match_status"], dropna=False).size()
              .rename("contrast_fields").reset_index())
    counts.to_csv(OUT / "validation_summary.csv", index=False)

    trace = pd.DataFrame([
        {"stage": "API dataset metadata", "evidence": "OSD-245 project identifier and project title identify RR-6; SpaceX-13 mission.", "finding": "RR-6"},
        {"stage": "API sample metadata", "evidence": "ISS-T rows have ~60 day duration, carcass dissection, and FLT euthanasia on ISS; LAR rows have ~30 day duration and upon-euthanasia dissection on Earth.", "finding": "ISS-T and LAR are within-RR-6 strata"},
        {"stage": "Original local parser", "evidence": "mission(245, sample) returned RR3 when 'ISS-T' occurred, else RR6.", "finding": "local parsing/mapping bug: ISS-T was incorrectly interpreted as a mission"},
        {"stage": "Impact", "evidence": "The same 20 ISS-T and 19 LAR FLT/GC samples remained paired within separate strata; only the mission label was wrong.", "finding": "annotation and mission-boundary summaries affected; sample membership, FLT/GC, and contrast construction unaffected"},
        {"stage": "Corrected parser", "evidence": "OSD-245 now maps to RR6_ISS_T or RR6_LAR.", "finding": "consistent with API; no discrepancy silently altered by this audit"},
    ])
    trace.to_csv(OUT / "osd245_rr3_root_cause_trace.csv", index=False)

    report = ["# Task 3 manifest validation against OSDR API", "",
              f"- Local samples audited: {local.sample_id.nunique()}",
              f"- Local contrasts audited: {local.contrast_id.nunique()}",
              f"- Samples found by exact API sample name: {(merged['_merge'] == 'both').sum()}/{len(merged)}", "",
              "The audit is non-mutating: discrepancies below were reported but not applied to the Task 3 manifest.", "",
              "## OSD-245 root cause", "",
              "The API identifies OSD-245 as RR-6. `ISS-T` means ISS-terminal and `LAR` means live-animal return; both are strata within RR-6. The original local `mission()` parser incorrectly treated the substring `ISS-T` as evidence for RR-3. This was a local mapping-logic error, not a source-metadata error. It changed annotation and mission-boundary summaries, but not sample membership, FLT/GC assignment, or the two OSD-245 contrast memberships.", "",
              "## Important comparability limitation", "",
              "Sequencing facility is recovered from the API dataset protocol descriptions. Layout/read length/instrument are structured per sample; nominal requested depth is not consistently structured, so configuration rows are marked partial matches when layout and read length agree."]
    (OUT / "validation_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print("\nValidation counts:\n", counts.to_string(index=False))
    print("\nDiscrepancies/non-comparable fields:\n", mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
