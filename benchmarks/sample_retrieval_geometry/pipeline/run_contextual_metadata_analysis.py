#!/usr/bin/env python3
"""High-precision metadata context audit of cached GTEx -> ARCHS4 Top-100 hits.

Rules are declared below and applied once to the union of retrieved GSMs.  This
script never reads expression or embeddings and never changes the frozen ranks.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_gtex_biological_retrieval import ARCHS4_TISSUES, GTEX_TO_ARCHS4, norm_text

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/contextual_metadata"
WORK = HERE / "work"
BASE = Path("/home/walt/bridge-rna/benchmarks/mouse_encode/results/task1a_balanced_geometry")
GTEX_MANIFEST = BASE / "gtex_human_sample_manifest.parquet"
METHODS = [
    "Expression Pearson", "Expression Canberra", "PCA-512 Canberra",
    "PCA-white cosine", "Raw Bridge cosine", "Bridge-white cosine",
]
KS = [1, 5, 10, 100]

# Frozen before any method-level summaries are calculated. Each regex requires
# explicit sample text; accession/name-only heuristics are intentionally absent.
STATE_RULES = {
    "tumor_cancer": {
        "malignant_terms": r"\b(?:tumou?r|cancer|carcinoma|adenocarcinoma|sarcoma|leukemia|leukaemia|lymphoma|melanoma|glioma|neoplasm|malignan(?:t|cy))\b",
    },
    "normal_healthy": {
        "normal_terms": r"\b(?:normal|healthy|non[- ]?diseased|non[- ]?malignant|benign)\b",
    },
    "treated": {
        "treatment_terms": r"\b(?:treated with|drug[- ]treated|treatment\s*:|exposed to|stimulated with|irradiated|transfected with|knockdown|siRNA[- ]treated|CRISPR[- ]treated)\b",
    },
    "control": {
        "control_terms": r"\b(?:control|ctrl|vehicle|untreated|mock|baseline|placebo|wild[- ]?type|scrambled|non[- ]?targeting)\b",
    },
}
MATERIAL_RULES = {
    "primary_tissue": {
        "primary_tissue_terms": r"\b(?:primary tissue|primary human (?:tissue|cells?)|biopsy|surgical (?:specimen|tissue)|tissue specimen|post[- ]?mortem|whole blood|peripheral blood)\b",
    },
    "cell_line": {
        "cell_line_terms": r"\b(?:cell line|cell-line|immortali[sz]ed|transformed cell|cultured cells?)\b|(?:^|[,;])\s*cell line\s*:",
    },
}
FIELDS = ["source_name_ch1", "title", "characteristics_ch1"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_for(row: pd.Series, rules: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for label, label_rules in rules.items():
        hits = []
        for rule_name, pattern in label_rules.items():
            rx = re.compile(pattern, re.I)
            for field in FIELDS:
                text = str(row.get(field, "") or "")
                match = rx.search(text)
                if match:
                    lo, hi = max(0, match.start() - 45), min(len(text), match.end() + 45)
                    hits.append(f"{rule_name}|{field}|{text[lo:hi]}")
        if hits:
            found[label] = hits
    return found


def classify(row: pd.Series, rules: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    found = evidence_for(row, rules)
    labels = sorted(found)
    if len(labels) == 1:
        label = labels[0]
        return label, "high", " || ".join(found[label])
    if len(labels) > 1:
        return "unknown", "conflict", " || ".join(f"{k}: {'; '.join(v)}" for k, v in found.items())
    return "unknown", "none", ""


def load_frozen_rankings() -> dict[str, np.ndarray]:
    conventional = {
        "Expression Canberra": 0,
        "Expression Pearson": 1,
        "PCA-512 Canberra": 2,
        "PCA-white cosine": 3,
    }
    tops = {}
    for method, idx in conventional.items():
        z = np.load(WORK / f"gtex_representation_controls_top100_{idx}.npz")
        assert tuple(z["indices"].shape) == (6064, 100)
        tops[method] = z["indices"]
    bridge = np.load(WORK / "gtex_archs4_top100.npz")
    tops["Raw Bridge cosine"] = bridge["indices_0"]
    tops["Bridge-white cosine"] = bridge["indices_2"]
    for method, indices in tops.items():
        assert np.all(np.apply_along_axis(lambda x: len(np.unique(x)) == 100, 1, indices)), f"duplicate Top-100 reference: {method}"
    return {m: tops[m] for m in METHODS}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "config.json").read_text())
    inp = {k: Path(v) for k, v in cfg["inputs"].items()}
    emb = inp["embedding_dir"]
    loc = pd.read_parquet(emb / "sample_locations.parquet")
    sman = pd.read_parquet(inp["sample_manifest"], columns=["gsm", "species"])
    human = loc.merge(sman, left_on="geo_accession", right_on="gsm", validate="one_to_one")
    human = human[norm_text(human.species).eq("human")].reset_index(drop=True)
    meta = pd.read_parquet(inp["sample_metadata"]).drop_duplicates("gsm").set_index("gsm").reindex(human.geo_accession).reset_index()
    meta = meta.rename(columns={meta.columns[0]: "gsm"})
    source = norm_text(meta.source_name_ch1).fillna("")
    tissue_all = np.full(len(meta), "", object)
    for label, pattern in ARCHS4_TISSUES.items():
        mask = source.str.match(pattern, na=False).to_numpy()
        assert not np.any((tissue_all != "") & mask)
        tissue_all[mask] = label
    leak_text = meta.title.fillna("").astype(str) + " " + meta.characteristics_ch1.fillna("").astype(str)
    leak = leak_text.str.contains(r"GTEX[- _]|Genotype-Tissue Expression|\bGTEx\b", case=False, regex=True, na=False).to_numpy()
    ref_indices = np.flatnonzero(~leak)
    assert len(ref_indices) == 510376

    gtex = pd.read_parquet(GTEX_MANIFEST).reset_index(drop=True)
    gtex["archs4_tissue"] = gtex.tissue.map(GTEX_TO_ARCHS4)
    frozen = pd.read_csv(HERE / "results/gtex_biological_retrieval/tissue_harmonization.csv")
    eligible = set(frozen.loc[frozen.eligible, "archs4_tissue"])
    qmeta = gtex[gtex.archs4_tissue.isin(eligible)].reset_index(drop=True)
    assert len(qmeta) == 6064 and qmeta.source_tissue.notna().all()

    tops = load_frozen_rankings()
    # Indices in caches address the GTEx-excluded reference array.
    unique_local = np.unique(np.concatenate([a.ravel() for a in tops.values()]))
    unique_global = ref_indices[unique_local]
    ann = meta.iloc[unique_global].copy().reset_index(drop=True)
    ann["archs4_global_index"] = unique_global
    ann["archs4_local_index"] = unique_local
    ann["tissue"] = tissue_all[unique_global]
    state = ann.apply(lambda r: classify(r, STATE_RULES), axis=1)
    material = ann.apply(lambda r: classify(r, MATERIAL_RULES), axis=1)
    ann[["state_label", "state_confidence", "state_evidence"]] = pd.DataFrame(state.tolist(), index=ann.index)
    ann[["material_label", "material_confidence", "material_evidence"]] = pd.DataFrame(material.tolist(), index=ann.index)
    ann.to_parquet(OUT / "deduplicated_hit_annotations.parquet", index=False)
    lookup = ann.set_index("archs4_local_index")

    coverage = []
    for axis, col in [("state", "state_label"), ("material", "material_label")]:
        counts = ann[col].value_counts(dropna=False)
        for label, count in counts.items():
            coverage.append({"axis": axis, "label": label, "samples": int(count), "fraction": count / len(ann)})
    pd.DataFrame(coverage).to_csv(OUT / "annotation_coverage.csv", index=False)
    confidence_qc = []
    for axis in ["state", "material"]:
        x = ann[f"{axis}_confidence"].value_counts(dropna=False)
        for status, count in x.items():
            confidence_qc.append({"axis": axis, "status": status, "samples": int(count), "fraction": count / len(ann)})
    pd.DataFrame(confidence_qc).to_csv(OUT / "annotation_confidence_qc.csv", index=False)

    rows = []
    hit_rows = []
    for method, inds in tops.items():
        for k in KS:
            flat = inds[:, :k].ravel()
            hit = lookup.loc[flat].reset_index()
            hit["query_index"] = np.repeat(np.arange(len(qmeta)), k)
            hit["rank"] = np.tile(np.arange(1, k + 1), len(qmeta))
            hit["query_tissue"] = np.repeat(qmeta.archs4_tissue.to_numpy(), k)
            tissue_match = hit.tissue.eq(hit.query_tissue)
            for conditioned, mask in [(False, np.ones(len(hit), bool)), (True, tissue_match.to_numpy())]:
                subset = hit.loc[mask]
                for axis, col, target in [("material", "material_label", "primary_tissue"), ("state_proxy", "state_label", "normal_healthy")]:
                    known = subset[col].ne("unknown")
                    denom = int(known.sum())
                    same = int(subset.loc[known, col].eq(target).sum())
                    rows.append({
                        "method": method, "k": k, "tissue_conditioned": conditioned,
                        "axis": axis, "query_target": target, "retrieved_hits": len(subset),
                        "confident_hits": denom, "agreement_precision": same / denom if denom else np.nan,
                        "absolute_agreement_rate": same / len(subset) if len(subset) else np.nan,
                        "unknown_fraction": 1 - denom / len(subset) if len(subset) else np.nan,
                    })
            if k == 100:
                hit["method"] = method
                hit_rows.append(hit[["method", "query_index", "rank", "gsm", "series_id", "tissue", "query_tissue", "state_label", "material_label"]])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "contextual_agreement.csv", index=False)
    # The repeated per-hit table is a regenerable cache, not a compact result.
    pd.concat(hit_rows, ignore_index=True).to_parquet(WORK / "contextual_metadata_top100_labels.parquet", index=False)

    compact = metrics[(metrics.k.isin([1, 10, 100])) & metrics.tissue_conditioned].copy()
    compact.to_csv(OUT / "compact_comparison.csv", index=False)

    # Deterministic examples: for every method take its first query where Top-1
    # is tissue-matched and confidently annotated, then show ranks 1..5.
    examples = []
    for method, inds in tops.items():
        chosen = None
        for qi, local in enumerate(inds[:, 0]):
            r = lookup.loc[local]
            if r.tissue == qmeta.loc[qi, "archs4_tissue"] and (r.state_label != "unknown" or r.material_label != "unknown"):
                chosen = qi
                break
        if chosen is None:
            continue
        for rank, local in enumerate(inds[chosen, :5], 1):
            r = lookup.loc[local]
            examples.append({
                "selection_rule": "lowest query index with tissue-matched, confidently context-annotated Top-1",
                "method": method, "query_index": chosen, "gtex_sample": qmeta.loc[chosen, "sample_id"],
                "query_tissue": qmeta.loc[chosen, "archs4_tissue"], "rank": rank,
                "gsm": r.gsm, "gse": r.series_id, "retrieved_tissue": r.tissue,
                "state": r.state_label, "state_confidence": r.state_confidence, "state_evidence": r.state_evidence,
                "material": r.material_label, "material_confidence": r.material_confidence, "material_evidence": r.material_evidence,
                "source_name_ch1": r.source_name_ch1, "title": r.title, "characteristics_ch1": r.characteristics_ch1,
            })
    pd.DataFrame(examples).to_csv(OUT / "representative_examples.csv", index=False)

    rules = {"state_rules": STATE_RULES, "material_rules": MATERIAL_RULES,
             "conflict_policy": "More than one label on an axis is unknown/conflict; no match is unknown/none.",
             "query_context": "GTEx source_tissue explicitly supports primary_tissue. GTEx query normal/healthy state is not asserted sample-wise; normal_healthy is reported only as a proxy rate, not authoritative agreement."}
    (OUT / "annotation_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    provenance = {
        "analysis": "contextual metadata audit of frozen GTEx to ARCHS4 Top-100 rankings",
        "queries": len(qmeta), "unique_retrieved_gsms": len(ann), "methods": METHODS, "ks": KS,
        "sample_metadata": str(inp["sample_metadata"]), "sample_metadata_sha256": sha256(inp["sample_metadata"]),
        "series_metadata": "No cached corpus-wide GSE title/summary/design asset was found; none was fetched or inferred.",
        "retrieval_rerun": False, "embedding_rerun": False,
        "limitations": [
            "Submitter-provided GEO text is heterogeneous; labels are exploratory and high precision rather than authoritative.",
            "Absence of a keyword is not evidence of absence of a context.",
            "State labels are mutually exclusive; explicit cross-state conflicts remain unknown.",
            "GTEx query state is not labeled healthy at sample level, so normal_healthy is a descriptive proxy rather than a validated agreement endpoint.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"unique_retrieved_gsms": len(ann), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
