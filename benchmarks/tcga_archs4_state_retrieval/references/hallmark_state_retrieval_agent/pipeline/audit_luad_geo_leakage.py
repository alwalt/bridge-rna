#!/usr/bin/env python3
"""Conservative GEO provenance/leakage audit for frozen LUAD centroid top-100 hits."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

from common import CONFIG, RESULTS, WORK, save_json

OUT = RESULTS / "luad_focused_audit"
CACHE = WORK / "geo_series_quick"
URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=self&form=text&view=quick"
TCGA_BARCODE = re.compile(r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9-]+)?\b", re.I)
STRONG_REUSE = re.compile(
    r"(?:downloaded|obtained|retrieved|derived|selected|imported|acquired).{0,80}(?:TCGA|The Cancer Genome Atlas)"
    r"|(?:TCGA|The Cancer Genome Atlas).{0,80}(?:downloaded|obtained|retrieved|derived|selected|imported|acquired)"
    r"|(?:TCGA cohort|TCGA samples?|TCGA RNA-?seq)", re.I | re.S)
INDEPENDENT = re.compile(
    r"(?:patients?|subjects?|participants?|donors?).{0,120}(?:recruited|enrolled|consented|underwent|collected|obtained|resected)"
    r"|(?:collected|obtained|resected).{0,120}(?:patients?|subjects?|participants?|donors?)"
    r"|institutional review board|informed consent"
    r"|(?:cell lines?|organoids?).{0,100}(?:treated|transfected|cultured|generated|sequenced)"
    r"|(?:independent|novel|our|RNA-?seq).{0,40}cohort"
    r"|tumou?r specimens? (?:were )?acquired"
    r"|fresh frozen tumou?r tissue from \d+ patients?"
    r"|(?:selected|collected) \d+ .{0,60}(?:tissues?|samples?)"
    r"|(?:tissue blocks?|specimens?).{0,60}resected from"
    r"|\d+ .{0,40}(?:tissues?|specimens?).{0,80}(?:underwent|performed) RNA", re.I | re.S)


def excerpt(text: str, match: re.Match | None, width: int = 220) -> str:
    if match is None:
        return ""
    left = max(0, match.start() - width // 2)
    right = min(len(text), match.end() + width // 2)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def fetch(gse: str) -> tuple[str, str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{gse}.txt"
    if not path.exists():
        response = requests.get(URL.format(gse=gse), timeout=45,
                                headers={"User-Agent": "BridgeRNA-Hallmark-benchmark-provenance-audit/1.0"})
        response.raise_for_status()
        path.write_text(response.text)
    return path.read_text(errors="replace"), str(path)


def classify(text: str) -> dict:
    barcode = TCGA_BARCODE.search(text)
    reuse = STRONG_REUSE.search(text)
    mention = re.search(r"\bTCGA\b|The Cancer Genome Atlas", text, re.I)
    independent = INDEPENDENT.search(text)
    if barcode or reuse:
        status = "TCGA-derived/reused"
        evidence = excerpt(text, barcode or reuse)
    elif independent:
        status = "independent GEO cohort"
        evidence = excerpt(text, independent)
    else:
        status = "ambiguous/insufficient metadata"
        evidence = excerpt(text, mention or independent)
    return {"series_status": status, "tcga_barcode_found": bool(barcode),
            "tcga_reuse_language_found": bool(reuse), "tcga_mention_found": bool(mention),
            "affirmative_independence_evidence_found": bool(independent), "series_evidence_excerpt": evidence}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(RESULTS / "tcga33_retrieval/centroid_top100_hits.csv")
    hits = hits[hits.cancer_type.eq("LUAD")].copy()
    assert len(hits) == 200 and hits.groupby("method").size().eq(100).all()
    hits["gse_accessions"] = hits.gse.fillna("").map(lambda x: re.findall(r"GSE\d+", x))
    series = sorted({g for values in hits.gse_accessions for g in values})
    series_rows = []
    for n, gse in enumerate(series, 1):
        text, cache_path = fetch(gse)
        row = {"gse": gse, "geo_url": URL.format(gse=gse), "cached_quick_text": cache_path,
               "quick_text_characters": len(text), **classify(text)}
        series_rows.append(row)
        print(f"[{n}/{len(series)}] {gse}: {row['series_status']}", flush=True)
    s = pd.DataFrame(series_rows)
    s.to_csv(OUT / "luad_geo_series_provenance_audit.csv", index=False)

    tcga = pd.read_parquet(CONFIG["tcga_manifest"])
    query_ids = set(tcga.sample_id.astype(str)) | set(tcga.patient_id.astype(str))
    out = []
    for hit in hits.itertuples(index=False):
        local_text = " ".join(str(getattr(hit, x, "")) for x in ("gsm", "title", "source_name", "characteristics"))
        duplicate_ids = sorted(q for q in query_ids if q and q in local_text)
        rows = s[s.gse.isin(hit.gse_accessions)]
        statuses = set(rows.series_status)
        if duplicate_ids or "TCGA-derived/reused" in statuses or TCGA_BARCODE.search(local_text) or STRONG_REUSE.search(local_text):
            status = "TCGA-derived/reused"
        elif "independent GEO cohort" in statuses:
            status = "independent GEO cohort"
        else:
            status = "ambiguous/insufficient metadata"
        out.append({**hit._asdict(), "audit_status": status,
                    "exact_tcga_query_identifier_duplicates": ";".join(duplicate_ids),
                    "sample_metadata_tcga_barcode": bool(TCGA_BARCODE.search(local_text)),
                    "sample_metadata_tcga_reuse_language": bool(STRONG_REUSE.search(local_text)),
                    "series_audit_statuses": ";".join(sorted(statuses)),
                    "series_geo_urls": ";".join(rows.geo_url),
                    "series_evidence_excerpts": " || ".join(rows.series_evidence_excerpt.fillna(""))})
    audit = pd.DataFrame(out).drop(columns=["gse_accessions"])
    audit.to_csv(OUT / "luad_centroid_top100_leakage_audit.csv", index=False)
    audit.to_parquet(OUT / "luad_centroid_top100_leakage_audit.parquet", index=False)

    filtered = (audit[~audit.audit_status.eq("TCGA-derived/reused")]
                .sort_values(["method", "rank"]).groupby("method", as_index=False).head(5))
    filtered.to_csv(OUT / "luad_centroid_top5_excluding_tcga_reuse.csv", index=False)
    summary = {
        "audited_hit_rows": len(audit), "unique_gsm": int(audit.gsm.nunique()),
        "unique_gse_accessions": len(series), "hit_status_counts": audit.audit_status.value_counts().to_dict(),
        "series_status_counts": s.series_status.value_counts().to_dict(),
        "exact_tcga_identifier_duplicate_hits": int(audit.exact_tcga_query_identifier_duplicates.ne("").sum()),
        "strong_tcga_reuse_hits": int(audit.audit_status.eq("TCGA-derived/reused").sum()),
        "interpretation_recomputed_excluding_flagged": bool(audit.audit_status.eq("TCGA-derived/reused").any()),
        "classification_policy": "TCGA barcode or explicit reuse/import language => derived; affirmative original-cohort recruitment/collection or experimental-generation language => independent even if TCGA is mentioned only as an external comparison; otherwise ambiguous.",
        "important_caveat": "Absence of TCGA language is not treated as proof of independence.",
    }
    save_json(OUT / "luad_leakage_audit_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
