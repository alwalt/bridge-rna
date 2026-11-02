#!/usr/bin/env python3
"""Join Bridge ARCHS4 IDs to GEOMeta and Gemma and audit usable fields."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
RESULTS = ROOT / "results"
DATA_ROOT = REPO_ROOT / "data"
FIELDS = [
    "tissue", "disease", "disease_status", "cell_type", "sex",
    "age_development", "strain", "genotype", "treatment_perturbation",
    "experimental_factor_group", "dose", "duration",
]


def present(x: pd.Series) -> pd.Series:
    return x.notna() & x.astype(str).str.strip().ne("") & ~x.astype(str).str.lower().isin({"nan", "na", "n/a", "none", "unknown", "not reported"})


def unique_join(values) -> str | None:
    vals = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        value = str(value).strip()
        if value and value.lower() not in {"nan", "none", "unknown", "not reported"} and value not in vals:
            vals.append(value)
    return " | ".join(vals) if vals else None


def present_value(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() not in {"", "nan", "na", "n/a", "none", "unknown", "not reported"}


def load_bridge() -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_parquet(REPO_ROOT / "data/manifests/archs4_sample_metadata_v2.5.parquet")
    loc = pd.read_parquet(REPO_ROOT / "embeddings/archs4/sample_locations.parquet")
    assert len(meta) == len(loc) == 940455
    loc = loc.rename(columns={"geo_accession": "gsm"})
    assert loc.global_index.tolist() == list(range(len(loc)))
    check = meta[["gsm"]].merge(loc[["gsm", "global_index", "species_id"]], on="gsm", validate="one_to_one")
    assert len(check) == len(meta)
    bridge = loc.merge(meta, on="gsm", validate="one_to_one")
    bridge["species"] = bridge.organism_ch1.map({"Homo sapiens": "human", "Mus musculus": "mouse"})
    bridge_gse = bridge[["gsm", "species", "series_id"]].copy()
    bridge_gse["gse"] = bridge_gse.series_id.fillna("").astype(str).str.split(",")
    bridge_gse = bridge_gse.explode("gse")
    bridge_gse["gse"] = bridge_gse.gse.str.strip()
    bridge_gse = bridge_gse[bridge_gse.gse.str.match(r"^GSE\d+$", na=False)].drop_duplicates()
    return bridge, bridge_gse


def load_geometa() -> pd.DataFrame:
    frames = []
    for p in sorted((DATA_ROOT / "geometa").glob("GEO_HUMAN_500K_*.csv")):
        x = pd.read_csv(p, low_memory=False)
        x["geometa_split"] = p.stem.removeprefix("GEO_HUMAN_500K_")
        frames.append(x)
    g = pd.concat(frames, ignore_index=True).drop_duplicates("GSM_ID", keep="first")
    g = g.rename(columns={
        "GSM_ID": "gsm", "GSE_ID": "geometa_gse", "Organ": "geometa_tissue",
        "Disease": "geometa_disease", "Gender": "geometa_sex",
        "Age_Group": "geometa_age_development", "Exp_Setting": "geometa_experimental_factor_group",
        "GSE_Pert": "geometa_treatment_perturbation",
    })
    # GEOMeta's public CSV represents healthy/control samples with Disease=Normal.
    g["geometa_disease_status"] = np.where(
        ~present(g["geometa_disease"]), None,
        np.where(g.geometa_disease.astype(str).str.lower().eq("normal"), "control/normal", "disease"),
    )
    for f in FIELDS:
        col = f"geometa_{f}"
        if col not in g:
            g[col] = None
    keep = ["gsm", "geometa_gse", "geometa_split"] + [f"geometa_{f}" for f in FIELDS]
    return g[keep]


def classify_category(category: str) -> set[str]:
    c = re.sub(r"[_-]+", " ", (category or "").strip().lower())
    out = set()
    if any(k in c for k in ["tissue", "organism part", "organ system", "brain region"]): out.add("tissue")
    if c == "region": out.add("tissue")
    if "cell type" in c: out.add("cell_type")
    if any(k in c for k in ["disease", "diagnosis"]): out.update(["disease", "disease_status"])
    if c in {"sex", "gender", "biological sex"}: out.add("sex")
    if "age" in c or "developmental stage" in c: out.add("age_development")
    if "strain" in c: out.add("strain")
    if any(k in c for k in ["genotype", "genetic variation", "variation"]): out.add("genotype")
    if any(k in c for k in ["treatment", "perturbation", "drug", "agent", "exposure", "challenge", "transfected"]): out.add("treatment_perturbation")
    if any(k in c for k in ["dose", "dosage", "concentration"]): out.add("dose")
    if any(k in c for k in ["duration", "timepoint", "time point", "time in"]): out.add("duration")
    if any(k in c for k in ["group", "condition", "phenotype", "study design"]): out.add("experimental_factor_group")
    return out


def load_gemma() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalogue = pd.read_parquet(DATA_ROOT / "gemma" / "datasets.parquet")
    rows = []
    category_counts = defaultdict(int)
    for path in sorted((DATA_ROOT / "gemma" / "samples").glob("GSE*.json")):
        try:
            data = json.loads(path.read_text()).get("data", [])
        except Exception:
            continue
        gse = path.stem
        for x in data:
            acc = (x.get("accession") or {}).get("accession") or x.get("shortName")
            if not acc or not re.fullmatch(r"GSM\d+", str(acc)):
                continue
            values = defaultdict(list)
            sample = x.get("sample") or {}
            for ch in sample.get("characteristics") or []:
                category = ch.get("category") or ""
                category_counts[("sample_characteristic", category)] += 1
                for f in classify_category(category): values[f].append(ch.get("value"))
            factors = []
            for fv in sample.get("factorValueObjects") or sample.get("fvbasicVOs") or []:
                cat = (fv.get("experimentalFactorCategory") or {}).get("category") or ""
                category_counts[("experimental_factor", cat)] += 1
                value = fv.get("summary") or fv.get("value")
                factors.append(f"{cat}: {value}")
                for f in classify_category(cat): values[f].append(value)
            if factors:
                values["experimental_factor_group"].extend(factors)
            # Reference-role statements are useful disease/control status.
            disease_text = " ".join(str(v) for v in values.get("disease", []))
            if disease_text:
                remainder = re.sub(r"reference (subject|substance) role|control|normal|healthy|no disease|no rejection|wild.?type", "", disease_text, flags=re.I)
                status = "disease/case" if re.search(r"[a-z]{3,}", remainder, re.I) else "control/reference"
                values["disease_status"].append(status)
            row = {"gsm": str(acc), "gemma_gse": gse, "gemma_sample_name": x.get("name")}
            for f in FIELDS:
                row[f"gemma_{f}"] = unique_join(values.get(f, []))
            rows.append(row)
    samples = pd.DataFrame(rows)
    if len(samples):
        samples = samples.groupby("gsm", as_index=False).agg({c: unique_join for c in samples.columns if c != "gsm"})
    cats = pd.DataFrame([
        {"source": k[0], "category": k[1], "occurrences": v, "mapped_fields": " | ".join(sorted(classify_category(k[1])))}
        for k, v in category_counts.items()
    ]).sort_values(["source", "occurrences"], ascending=[True, False])
    return samples, catalogue, cats


def norm(value) -> str | None:
    if value is None or pd.isna(value): return None
    synonyms = {"f": "female", "m": "male", "female sex": "female", "male sex": "male", "female biological sex": "female", "male biological sex": "male", "woman": "female", "women": "female", "man": "male", "men": "male"}
    tokens = []
    for raw in str(value).split("|"):
        x = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
        x = synonyms.get(x, x)
        if x and x not in tokens:
            tokens.append(x)
    return " | ".join(sorted(tokens)) or None


def audit_unmatched(unmatched: pd.DataFrame) -> pd.DataFrame:
    sample = unmatched.sample(n=min(25, len(unmatched)), random_state=20260925).copy()
    out = []
    anatomy = re.compile(r"tissue\s*:|cell type\s*:|blood|brain|liver|lung|kidney|heart|muscle|skin|breast|colon|intestin|ovary|testis|bone|marrow|cell line|fibroblast|neuron|lymphocyte|macrophage|adipose|spleen|oocyte|epiderm|jejun|embry", re.I)
    disease = re.compile(r"cancer|tumou?r|carcinoma|leukemia|lymphoma|disease|infect|diabet|obes|control|healthy|normal|wild.?type", re.I)
    treatment = re.compile(r"treat|dose|expos|stimulat|transfect|knock.?out|knockdown|siRNA|drug|vehicle|control|hours?|days?|diet|infect|inject|agent\s*:|saline|baseline", re.I)
    role = re.compile(r"control|vehicle|untreated|treated|case|patient|healthy|normal|wild.?type|mutant|baseline", re.I)
    for _, r in sample.iterrows():
        gsm = r.gsm
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?targ=self&acc={gsm}&form=text&view=full"
        try:
            text = urlopen(url, timeout=45).read().decode("utf-8", "replace")
        except Exception as e:
            text = f"FETCH_ERROR: {e!r}"
        fields = defaultdict(list)
        for line in text.splitlines():
            m = re.match(r"!Sample_([^=]+)\s*=\s*(.*)", line)
            if m: fields[m.group(1).strip()].append(m.group(2).strip())
        biological = " | ".join(sum((v for k, v in fields.items() if any(z in k for z in ["title", "source_name", "characteristics", "description", "treatment_protocol", "growth_protocol"])), []))
        out.append({
            "gsm": gsm, "species": r.species, "gse": r.series_id,
            "geo_title": unique_join(fields.get("title", [])),
            "geo_source": unique_join(fields.get("source_name_ch1", [])),
            "geo_characteristics": unique_join(fields.get("characteristics_ch1", [])),
            "geo_treatment_protocol": unique_join(fields.get("treatment_protocol_ch1", [])),
            "geo_description": unique_join(fields.get("description", [])),
            "tissue_recoverable": bool(anatomy.search(biological)),
            "disease_state_recoverable": bool(disease.search(biological)),
            "treatment_recoverable": bool(treatment.search(biological)),
            "experimental_role_recoverable": bool(role.search(biological)),
            "geo_fetch_ok": not text.startswith("FETCH_ERROR"), "geo_url": url,
        })
    return pd.DataFrame(out)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bridge, bridge_gse = load_bridge()
    geometa = load_geometa()
    gemma, catalogue, cats = load_gemma()
    cats.to_csv(RESULTS / "gemma_observed_schema.csv", index=False)

    base = bridge[["global_index", "gsm", "species", "series_id", "title", "source_name_ch1", "characteristics_ch1"]]
    combined = base.merge(geometa, on="gsm", how="left", validate="one_to_one")
    combined = combined.merge(gemma, on="gsm", how="left", validate="one_to_one")
    combined["geometa_matched"] = present(combined["geometa_gse"])
    combined["gemma_matched"] = present(combined["gemma_gse"])
    combined["union_matched"] = combined.geometa_matched | combined.gemma_matched
    combined["intersection_matched"] = combined.geometa_matched & combined.gemma_matched
    for f in FIELDS:
        combined[f"union_{f}"] = combined[f"geometa_{f}"].where(present(combined[f"geometa_{f}"]), combined[f"gemma_{f}"])

    combined.to_parquet(RESULTS / "combined_coverage.parquet", index=False)
    combined[combined.geometa_matched].to_parquet(RESULTS / "bridge_geometa_matches.parquet", index=False)
    combined[combined.gemma_matched].to_parquet(RESULTS / "bridge_gemma_matches.parquet", index=False)
    unmatched = combined[~combined.union_matched].copy()
    unmatched.to_parquet(RESULTS / "unmatched_gsms.parquet", index=False)

    coverage_rows = []
    for species, x in combined.groupby("species"):
        for source in ["geometa", "gemma", "union", "intersection"]:
            n = int(x[f"{source}_matched"].sum())
            coverage_rows.append({"level": "GSM", "species": species, "source": source, "total": len(x), "matched": n, "unmatched": len(x)-n, "coverage_pct": 100*n/len(x)})
        bg = bridge_gse[bridge_gse.species.eq(species)]
        total_gses = set(bg.gse)
        # A species-level GSE match requires at least one matched Bridge GSM of
        # that species; this avoids crediting mixed-species series by accession.
        sx = combined[combined.species.eq(species)]
        gm_gsms = set(sx.loc[sx.geometa_matched, "gsm"])
        ge_gsms = set(sx.loc[sx.gemma_matched, "gsm"])
        gm_gses = set(bg.loc[bg.gsm.isin(gm_gsms), "gse"])
        observed_gemma_gses = set(bg.loc[bg.gsm.isin(ge_gsms), "gse"])
        for source, vals in [("geometa",gm_gses),("gemma",observed_gemma_gses),("union",gm_gses|observed_gemma_gses),("intersection",gm_gses&observed_gemma_gses)]:
            coverage_rows.append({"level":"GSE","species":species,"source":source,"total":len(total_gses),"matched":len(vals),"unmatched":len(total_gses)-len(vals),"coverage_pct":100*len(vals)/len(total_gses)})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(RESULTS / "coverage_summary.csv", index=False)

    complete = []
    for species, x in combined.groupby("species"):
        for source in ["geometa", "gemma", "union"]:
            matched = x[x[f"{source}_matched"]]
            for f in FIELDS:
                n = int(present(matched[f"{source}_{f}"]).sum()) if len(matched) else 0
                complete.append({"species":species,"source":source,"field":f,"matched_gsms":len(matched),"available":n,"availability_pct_of_matched":100*n/len(matched) if len(matched) else 0,"availability_pct_of_bridge":100*n/len(x)})
    pd.DataFrame(complete).to_csv(RESULTS / "field_level_completeness.csv", index=False)

    conflicts = []
    both = combined[combined.intersection_matched]
    for f in ["tissue", "disease", "sex", "cell_type", "treatment_perturbation"]:
        a, b = both[f"geometa_{f}"], both[f"gemma_{f}"]
        for idx in both.index:
            av, bv = a.at[idx], b.at[idx]
            ap, bp = present_value(av), present_value(bv)
            if ap and bp:
                if str(av).strip() == str(bv).strip():
                    relation = "exact"
                elif norm(av) == norm(bv):
                    relation = "normalized"
                elif f == "sex":
                    relation = "obvious_conflict"
                elif f == "treatment_perturbation" and norm(av) in {"yes", "no"}:
                    relation = "not_comparable_granularity"
                else:
                    relation = "different_vocabulary_or_candidate_conflict"
            elif ap: relation = "missing_gemma"
            elif bp: relation = "missing_geometa"
            else: relation = "missing_both"
            conflicts.append({"gsm":both.at[idx,"gsm"],"species":both.at[idx,"species"],"field":f,"geometa_value":av,"gemma_value":bv,"relation":relation})
    conflict_df = pd.DataFrame(conflicts)
    conflict_df.to_parquet(RESULTS / "geometa_gemma_conflicts.parquet", index=False)
    conflict_df.groupby(["field","relation"]).size().rename("n").reset_index().to_csv(RESULTS / "agreement_summary.csv", index=False)

    examples = combined[combined.union_matched].copy()
    examples["useful_fields"] = sum(present(examples[f"union_{f}"]).astype(int) for f in FIELDS)
    examples.sort_values(["useful_fields","gsm"], ascending=[False,True]).head(20).to_csv(RESULTS / "retrieval_ready_examples.csv", index=False)
    audit_unmatched(unmatched).to_csv(RESULTS / "unmatched_geo_audit_25.csv", index=False)

    provenance = {
        "created_date": "2026-09-25", "random_seed": 20260925,
        "bridge_embedding_manifest": "embeddings/archs4/embedding_manifest.json",
        "bridge_locations": "embeddings/archs4/sample_locations.parquet",
        "bridge_metadata": "data/manifests/archs4_sample_metadata_v2.5.parquet",
        "geometa_release": "https://huggingface.co/datasets/binchenlab/GEOMeta",
        "geometa_revision": "d0cd1e117143b32a392bbe3130e1c4a2a3c8e740",
        "gemma_api": API if False else "https://gemma.msl.ubc.ca/rest/v2",
        "gemma_catalogue_unique_accessions": len(catalogue),
        "gemma_overlapping_gse_sample_endpoint_404s": int(len(pd.read_csv(DATA_ROOT / "gemma" / "fetch_failures.csv"))),
        "limitations": ["GEOMeta analysis reflects the three metadata CSVs publicly downloadable at the pinned revision, not the larger sample count stated in the preprint.", "Gemma field mapping is based on observed category labels; free-text BioSource alone is not counted as a structured tissue annotation.", "Agreement marked different_vocabulary_or_conflict requires review and is not automatically a biological contradiction."],
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(coverage.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
