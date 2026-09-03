#!/usr/bin/env python3
"""Audit the expanded ENCODE pool without preprocessing or rerunning Task 1A.

Exposure is delegated to scripts/data_audit/check_exposure.py by importing its
manifest loader and exact GSE-matching helper. Exact GSM and study-level GSE
matches remain separate classifications.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_audit.check_exposure import DEFAULT_MANIFEST, load_manifest, rows_for_gse

HERE = Path(__file__).resolve().parents[1]
SEARCH_CACHE = HERE / "work" / "expanded_encode_search.json"
EXPERIMENT_CACHE = HERE / "work" / "encode_experiment_metadata.json"
FILE_METADATA = ROOT / "data" / "encode" / "metadata_expanded.tsv"

# Only direct or explicitly prespecified anatomical matches are candidates.
# Generic kidney, bowel, adipose, brain, and cortical-plate labels are excluded
# because their GTEx counterparts are anatomically more specific.
DIRECT_GTEX_MATCHES = {
    "adrenal gland": "Adrenal Gland",
    "gastrocnemius": "Muscle - Skeletal",
    "skeletal muscle tissue": "Muscle - Skeletal",
    "heart": "Heart - Atrial Appendage; Heart - Left Ventricle",
    "left cerebral cortex": "Brain - Cortex",
    "layer of hippocampus": "Brain - Hippocampus",
    "liver": "Liver",
    "lung": "Lung",
    "ovary": "Ovary",
    "sigmoid colon": "Colon - Sigmoid",
    "stomach": "Stomach",
    "mammary gland": "Breast - Mammary Tissue",
    "testis": "Testis",
    "pancreas": "Pancreas",
    "frontal cortex": "Brain - Frontal Cortex (BA9)",
    "subcutaneous adipose tissue": "Adipose - Subcutaneous",
    "cerebellum": "Brain - Cerebellum",
    "spleen": "Spleen",
}
DISEASE_RE = re.compile(
    r"5xFAD|Alzheimer|disease model|transgenic|knockout|knock-out|mutant|\bTg\b",
    re.IGNORECASE,
)
GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)
GSE_RE = re.compile(r"GSE\d+", re.IGNORECASE)


def biosamples(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for replicate in experiment.get("replicates", []):
        biosample = replicate.get("library", {}).get("biosample", {})
        if isinstance(biosample, dict) and biosample.get("accession"):
            found[biosample["accession"]] = biosample
    return list(found.values())


def joined(values) -> str | None:
    values = sorted({str(x).strip() for x in values if x is not None and str(x).strip()})
    return "; ".join(values) if values else None


def health_status(experiment: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    samples = biosamples(experiment)
    strains, text, modifications, treatments = [], [], [], []
    for sample in samples:
        donor = sample.get("donor", {}) if isinstance(sample.get("donor", {}), dict) else {}
        strains.extend([donor.get("strain_name"), donor.get("strain_background")])
        text.extend([
            sample.get("description"), sample.get("summary"), donor.get("description"),
            donor.get("strain_name"), donor.get("strain_background"),
        ])
        modifications.extend(sample.get("genetic_modifications", []) or [])
        modifications.extend(sample.get("model_organism_donor_modifications", []) or [])
        modifications.extend(donor.get("genetic_modifications", []) or [])
        treatments.extend(sample.get("treatments", []) or [])
    text.extend([experiment.get("description"), experiment.get("biosample_summary")])
    reasons = []
    if DISEASE_RE.search("; ".join(str(x) for x in text if x)):
        reasons.append("disease/transgenic model text")
    if modifications:
        reasons.append("genetic modification")
    if treatments:
        reasons.append("treatment")
    if experiment.get("perturbed") is True:
        reasons.append("ENCODE perturbed=true")
    return not reasons, joined(reasons), joined(strains)


def exposure_status(
    gsms: list[str], gses: list[str], manifest: pd.DataFrame,
) -> tuple[str, str | None, str | None]:
    """Classify exact-sample exposure first, then study exposure.

    Presence means presence in the training split. Validation/unused rows do not
    count as pretraining exposure. Available identifiers with no training match
    establish ``fully_unseen``; only absent identifiers remain unresolved.
    """
    gsm_splits: dict[str, list[str]] = {}
    for gsm in gsms:
        rows = manifest.loc[manifest["gsm"].eq(gsm)]
        gsm_splits[gsm] = sorted(set(rows["split"].dropna().astype(str)))
    exact_seen = sorted(gsm for gsm, splits in gsm_splits.items() if "train" in splits)
    if exact_seen:
        return "exact_sample_seen", joined(exact_seen), None

    gse_train_counts: dict[str, int] = {}
    for gse in gses:
        rows = rows_for_gse(gse, manifest)
        gse_train_counts[gse] = int(rows["split"].eq("train").sum())
    study_seen = sorted(gse for gse, count in gse_train_counts.items() if count > 0)
    if study_seen:
        detail = "; ".join(f"{gse}:{gse_train_counts[gse]} train samples" for gse in study_seen)
        return "same_study_seen", joined(study_seen), detail

    if gsms or gses:
        checked = joined([*(f"GSM:{x}" for x in gsms), *(f"GSE:{x}" for x in gses)])
        return "fully_unseen", checked, "no exact GSM or GSE training match"
    return "unresolved", None, "ENCODE provides neither a GSM nor GSE"


def main() -> None:
    search = json.loads(SEARCH_CACHE.read_text()).get("@graph", [])
    details = json.loads(EXPERIMENT_CACHE.read_text())
    file_meta = pd.read_csv(FILE_METADATA, sep="\t", low_memory=False)
    quant = file_meta[file_meta["Output type"].eq("gene quantifications")].copy()
    quant_by_experiment = quant.groupby("Experiment accession")["File accession"].agg(lambda x: ";".join(sorted(set(x))))
    exposure_manifest = load_manifest(DEFAULT_MANIFEST)

    rows = []
    for item in search:
        accession = item["accession"]
        experiment = details[accession]
        samples = biosamples(experiment)
        healthy, exclusion_reason, strain = health_status(experiment)
        encode_tissue = experiment.get("biosample_ontology", {}).get("term_name")
        dbxrefs = experiment.get("dbxrefs", []) or []
        gsms = sorted(set(GSM_RE.findall(";".join(dbxrefs))))
        gses = sorted(set(GSE_RE.findall(";".join(dbxrefs))))
        # ENCODE experiment-level quantification is the analysis sample. A unique
        # experiment-level GSM can therefore be checked exactly; absent/ambiguous
        # GSM metadata cannot be replaced with a study-level GSE check.
        gsms = [x.upper() for x in gsms]
        gses = [x.upper() for x in gses]
        overlap, matched_identifier, overlap_detail = exposure_status(gsms, gses, exposure_manifest)
        rows.append({
            "experiment_accession": accession,
            "encode_tissue": encode_tissue,
            "gtex_tissue": DIRECT_GTEX_MATCHES.get(encode_tissue),
            "assay_type": experiment.get("assay_title"),
            "healthy_non_transgenic": healthy,
            "health_exclusion_reason": exclusion_reason,
            "strain": strain,
            "biological_samples": len(samples),
            "biosample_accessions": joined(s.get("accession") for s in samples),
            "gene_quantification_available": accession in quant_by_experiment.index,
            "gene_quantification_file": quant_by_experiment.get(accession),
            "gsm": joined(gsms),
            "gse": joined(gses),
            "exposure_class": overlap,
            "matched_pretraining_identifier": matched_identifier,
            "exposure_detail": overlap_detail,
            "direct_gtex_match": encode_tissue in DIRECT_GTEX_MATCHES,
        })

    audit = pd.DataFrame(rows).sort_values(["direct_gtex_match", "encode_tissue", "assay_type", "experiment_accession"], ascending=[False, True, True, True])
    eligible = audit[audit["healthy_non_transgenic"] & audit["direct_gtex_match"]].copy()
    summary = (
        eligible.groupby(["encode_tissue", "gtex_tissue", "assay_type"], dropna=False)
        .agg(
            eligible_experiments=("experiment_accession", "nunique"),
            eligible_mouse_samples=("experiment_accession", "size"),
            underlying_biosamples=("biological_samples", "sum"),
            expression_ready_samples=("gene_quantification_available", lambda x: int(x.sum())),
            missing_gene_quantification=("gene_quantification_available", lambda x: int((~x).sum())),
            exact_sample_seen=("exposure_class", lambda x: int((x == "exact_sample_seen").sum())),
            same_study_seen=("exposure_class", lambda x: int((x == "same_study_seen").sum())),
            fully_unseen=("exposure_class", lambda x: int((x == "fully_unseen").sum())),
            unresolved=("exposure_class", lambda x: int((x == "unresolved").sum())),
        )
        .reset_index()
        .sort_values(["fully_unseen", "eligible_experiments", "encode_tissue"], ascending=[False, False, True])
    )
    tissue_summary = (
        eligible.groupby(["encode_tissue", "gtex_tissue"], dropna=False)
        .agg(
            eligible_experiments=("experiment_accession", "nunique"),
            expression_ready_experiments=("gene_quantification_available", lambda x: int(x.sum())),
            exact_sample_seen=("exposure_class", lambda x: int((x == "exact_sample_seen").sum())),
            same_study_seen=("exposure_class", lambda x: int((x == "same_study_seen").sum())),
            fully_unseen=("exposure_class", lambda x: int((x == "fully_unseen").sum())),
            unresolved=("exposure_class", lambda x: int((x == "unresolved").sum())),
        )
        .reset_index()
        .sort_values(["fully_unseen", "expression_ready_experiments", "encode_tissue"], ascending=[False, False, True])
    )

    results = HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(results / "expanded_encode_cohort_audit.parquet", index=False)
    audit.to_csv(results / "expanded_encode_cohort_audit.csv", index=False)
    eligible.to_parquet(results / "expanded_encode_direct_match_candidates.parquet", index=False)
    eligible.to_csv(results / "expanded_encode_direct_match_candidates.csv", index=False)
    summary.to_csv(results / "expanded_encode_candidate_tissue_summary.csv", index=False)
    tissue_summary.to_csv(results / "expanded_encode_tissue_exposure_summary.csv", index=False)
    ready = eligible[eligible["gene_quantification_available"]].copy()
    fully_unseen = ready[ready["exposure_class"].eq("fully_unseen")].copy()
    unseen_per_tissue = fully_unseen.groupby("encode_tissue")["experiment_accession"].transform("size")
    primary = fully_unseen[unseen_per_tissue >= 2].copy()
    singleton_reserve = fully_unseen[unseen_per_tissue < 2].copy()
    secondary = ready[ready["exposure_class"].eq("same_study_seen")].copy()
    fully_unseen.to_parquet(results / "proposed_all_fully_unseen_profiles.parquet", index=False)
    primary.to_parquet(results / "proposed_primary_fully_unseen_replicated.parquet", index=False)
    singleton_reserve.to_parquet(results / "proposed_fully_unseen_singleton_reserve.parquet", index=False)
    secondary.to_parquet(results / "proposed_secondary_same_study_seen.parquet", index=False)
    print(f"files.txt/search pool: {len(search)} experiments")
    print(f"Healthy/non-transgenic: {audit.healthy_non_transgenic.sum()}")
    print(f"Healthy with a direct GTEx match: {len(eligible)}")
    print(f"Expression-ready direct matches: {eligible.gene_quantification_available.sum()}")
    print("\nCandidate tissue/assay summary\n")
    print(summary.to_string(index=False))
    print("\nCorrected tissue-level exposure summary\n")
    print(tissue_summary.to_string(index=False))
    print(f"\nPrimary proposal: {len(primary)} fully unseen, expression-ready profiles "
          f"across {primary.encode_tissue.nunique()} tissues with >=2 profiles/tissue")
    print(f"Singleton reserve: {len(singleton_reserve)} profiles")
    print(f"Same-study secondary stratum: {len(secondary)} profiles")


if __name__ == "__main__":
    main()
