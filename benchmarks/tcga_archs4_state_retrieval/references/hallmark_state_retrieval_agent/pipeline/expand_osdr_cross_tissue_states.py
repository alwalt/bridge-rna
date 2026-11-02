#!/usr/bin/env python3
"""Resolve broad cross-tissue OSDR retrieval states from saved metadata only."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "osdr_atlas"
BASE = SOURCE / "muscle_liver_audit"
OUT = BASE / "expanded_cross_tissue"
FIG = OUT / "figures"
METHODS = ["Bridge-512 cosine", "Hallmark-50 cosine", "Hallmark signed top10 overlap"]

spec = importlib.util.spec_from_file_location("focused_audit", ROOT / "pipeline" / "audit_osdr_muscle_liver.py")
focused = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(focused)


def text(row: pd.Series) -> str:
    return " | ".join(str(row.get(c, "")) for c in ["title", "source_name", "characteristics"]).lower()


def hit(s: str, pattern: str) -> bool:
    return bool(re.search(pattern, s, flags=re.I))


def expanded_category(row: pd.Series) -> tuple[str, str]:
    """Return category and conservative metadata rationale."""
    base = row["base_category"]
    if base != "other cross-tissue state":
        harmonized = {"ambiguous/unresolved": "unresolved",
                      "drug/toxicant exposure": "drug/toxicant response"}.get(base, base)
        return harmonized, "retained from focused audit"
    s = text(row)
    adipose = hit(s, r"adipos|adipocyte|preadipocyte|white fat|brown fat|\bwat\b|\bbat\b")
    inflammatory = hit(s, r"inflamm|\blps\b|lipopolysacchar|cytokine|interferon|\btnf\b|immune stimul|infection|infected")
    if adipose and inflammatory:
        return "adipose inflammatory state", "adipose plus explicit inflammatory/stimulation metadata"
    if adipose:
        return "adipose metabolic state", "explicit adipose/adipocyte metadata"
    if hit(s, r"skeletal muscle|vastus|soleus|quadriceps|gastrocnemius|tibialis|digitorum|exercise|training|trained"):
        return "skeletal-muscle/exercise state", "explicit muscle or exercise metadata"
    if hit(s, r"fibroblast|stromal|stroma|mesenchymal|synovial|\bmsc(?:[_ -]|$)"):
        return "fibroblast/stromal state", "explicit fibroblast/stromal/mesenchymal metadata"
    if hit(s, r"infect|viral|virus|bacter|\blps\b|lipopolysacchar|immune stimul"):
        return "infection/immune stimulation", "explicit infection or immune-stimulation metadata"
    if hit(s, r"bone marrow|plasma cell|bmpc|lymphocyte|monocyte|macrophage|neutrophil|allograft|immune|inflamm|myeloma|mgus"):
        return "immune/inflammatory state", "explicit immune, marrow, inflammatory, or allograft metadata"
    if hit(s, r"\bdrug\b|drug exposure|toxic|toxin|inhibitor|agonist|antagonist|chemotherapy|doxorubicin|cisplatin|acetaminophen|\bapap\b"):
        return "drug/toxicant response", "explicit drug, treatment, or toxicant metadata"
    if hit(s, r"oxidative|reactive oxygen|\bros\b|hypoxi|stress response|heat shock|unfolded protein"):
        return "oxidative/stress-response state", "explicit oxidative, hypoxic, or stress metadata"
    if hit(s, r"mitochond|metabolic|glycoly|respiration|oxidative phosphorylation"):
        return "mitochondrial/metabolic state", "explicit mitochondrial or metabolic metadata"
    if hit(s, r"prolifer|cell.?cycle|mitotic|tumou?r|cancer|carcinoma|melanoma|adenoma|sarcoma|leukemia|hnscc|neoplasm"):
        return "proliferative/tumor state", "explicit proliferation or tumor metadata"
    if hit(s, r"epithelial|breast|pancrea|kidney|renal|pituitary|adrenal|brain|lung|colon|prostate"):
        return "other identifiable tissue state", "explicit non-target tissue metadata without a supported functional label"
    return "unresolved", "metadata insufficient for a more specific category"


def dominant_summary(series: pd.Series) -> str:
    names: list[str] = []
    for value in series.dropna().astype(str):
        for token in value.split(";"):
            name = token.strip().split("=")[0].strip()
            if name:
                names.append(name)
    if not names:
        return ""
    return "; ".join(pd.Series(names).value_counts().index[:7])


def make_heatmap(values: pd.DataFrame, strict: pd.DataFrame, groups: pd.DataFrame, domain: str) -> None:
    domain_groups = groups.loc[groups.domain == domain, "group_id"].tolist()
    order = (values[values.domain == domain].groupby("expanded_category").standardized_effect
             .apply(lambda x: x.abs().max()).sort_values(ascending=False).index.tolist())
    labels = dict(zip(groups.group_id, groups.study + "\n" + groups.group_id.str.replace("OSDRG", "G", regex=False)))
    fig, axes = plt.subplots(3, 1, figsize=(max(12, .68 * len(domain_groups)), max(12, .43 * len(order) * 3)), sharex=True)
    for ax, method in zip(axes, METHODS):
        sub = values[(values.domain == domain) & (values.method == method)]
        mat = sub.pivot(index="expanded_category", columns="group_id", values="standardized_effect").reindex(
            index=order, columns=domain_groups).fillna(0)
        sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5, linewidths=.2,
                    cbar_kws={"label": "standardized FLT−GC enrichment"})
        sig = strict[(strict.domain == domain) & (strict.method == method)]
        marked = set(zip(sig.expanded_category, sig.group_id))
        for yi, state in enumerate(order):
            for xi, gid in enumerate(domain_groups):
                if (state, gid) in marked:
                    ax.text(xi + .5, yi + .5, "●", ha="center", va="center", fontsize=7, color="black")
        ax.set_title(method + "  (● existing strict replicate-consistent enrichment)", loc="left", fontsize=10)
        ax.set_ylabel("")
    axes[-1].set_xticklabels([labels[g] for g in domain_groups], rotation=55, ha="right", fontsize=8)
    axes[-1].set_xlabel("Matched OSDR experimental group")
    fig.suptitle(f"{domain.title()}: expanded ARCHS4 biological-state retrieval shifts", y=1.003, fontsize=14)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{domain}_expanded_state_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(BASE / "eligible_muscle_liver_groups.csv")
    diff = pd.read_parquet(SOURCE / "flt_vs_gc_differential_retrieval.parquet").merge(
        groups[["group_id", "study", "tissue", "domain"]], on="group_id", how="inner")
    diff["base_category"] = diff.apply(focused.category, axis=1)
    assigned = diff.apply(expanded_category, axis=1, result_type="expand")
    diff[["expanded_category", "category_rationale"]] = assigned

    broad = diff[diff.base_category == "other cross-tissue state"].copy()
    audit_cols = ["group_id", "study", "tissue", "domain", "method", "reference_row", "gsm", "gse",
                  "title", "source_name", "characteristics", "rrf_delta_flt_minus_gc", "permutation_p", "bh_q",
                  "flt_replicate_prevalence", "gc_replicate_prevalence", "expanded_category", "category_rationale"]
    broad[audit_cols].to_csv(OUT / "all_cross_tissue_hit_metadata_audit.csv", index=False)
    (broad.sort_values("rrf_delta_flt_minus_gc", ascending=False)
     .drop_duplicates(["gsm", "gse"])[["gsm", "gse", "title", "source_name", "characteristics",
                                          "expanded_category", "category_rationale"]]
     .to_csv(OUT / "unique_cross_tissue_gsm_annotations.csv", index=False))

    strict_mask = ((diff.bh_q <= .10) & (diff.rrf_delta_flt_minus_gc > 0) &
                   (diff.flt_replicate_prevalence >= .75) & (diff.gc_replicate_prevalence <= .25))
    strict = diff[strict_mask].copy()
    saved_strict = pd.read_csv(BASE / "strict_flt_enriched_hits.csv")
    contributions = saved_strict[["group", "method", "ARCHS4 GSM", "dominant shared Hallmarks",
                                  "top5pct_flt_hub_gse", "loho_tests", "loho_survival_fraction"]].rename(
        columns={"group": "group_id", "ARCHS4 GSM": "gsm"})
    strict = strict.merge(contributions, on=["group_id", "method", "gsm"], how="left")
    strict["sensitivity_status"] = np.where(
        strict.method.str.contains("Hallmark") & strict.loho_tests.isna(),
        "not tested in saved dominant-Hallmark-removal analysis",
        np.where(strict.loho_tests.notna(), "tested", "not applicable to Bridge"))

    values = (diff.groupby(["domain", "method", "group_id", "expanded_category"], as_index=False)
              .rrf_delta_flt_minus_gc.sum())
    values["standardized_effect"] = values.groupby("method").rrf_delta_flt_minus_gc.transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else 1))
    values.to_csv(OUT / "expanded_category_group_standardized_effects.csv", index=False)

    strict_cross = strict[strict.base_category == "other cross-tissue state"].copy()
    state_domains = strict_cross.groupby("expanded_category").domain.nunique()
    state_studies = strict_cross.groupby(["domain", "expanded_category"]).study.nunique()
    rec = (strict_cross.groupby("expanded_category", as_index=False)
           .agg(n_hits=("gsm", "size"), n_gsm=("gsm", "nunique"), n_gse=("gse", "nunique"),
                n_groups=("group_id", "nunique"), n_studies=("study", "nunique"),
                tissues=("domain", lambda x: "; ".join(sorted(set(x)))),
                methods=("method", lambda x: "; ".join(sorted(set(x)))),
                representative_gsm=("gsm", lambda x: "; ".join(pd.unique(x)[:5])),
                representative_gse=("gse", lambda x: "; ".join(pd.unique(x)[:5])),
                dominant_hallmarks=("dominant shared Hallmarks", dominant_summary),
                hub_driven_hits=("top5pct_flt_hub_gse", lambda x: int(pd.Series(x).fillna(False).sum())),
                max_effect=("rrf_delta_flt_minus_gc", "max"), min_q=("bh_q", "min")))
    def recurrence_class(row: pd.Series) -> str:
        state = row.expanded_category
        if state_domains.get(state, 0) > 1:
            return "recurring across different tissues"
        d = row.tissues.split("; ")[0]
        if state_studies.get((d, state), 0) > 1:
            return "recurring across independent studies of the same tissue"
        return "study-specific state"
    rec["recurrence_class"] = rec.apply(recurrence_class, axis=1)
    rec["primarily_hub_driven"] = rec.hub_driven_hits > (rec.n_hits / 2)
    rec.to_csv(OUT / "strict_cross_tissue_state_recurrence.csv", index=False)

    representatives = strict_cross.sort_values(["expanded_category", "rrf_delta_flt_minus_gc"], ascending=[True, False])
    representatives = representatives.groupby("expanded_category", group_keys=False).head(5)
    representatives[["expanded_category", "study", "group_id", "domain", "method", "gsm", "gse", "title",
                     "source_name", "characteristics", "rrf_delta_flt_minus_gc", "bh_q",
                     "flt_replicate_prevalence", "gc_replicate_prevalence", "dominant shared Hallmarks",
                     "top5pct_flt_hub_gse", "sensitivity_status", "loho_survival_fraction"]].to_csv(
        OUT / "recurrent_state_representative_hits.csv", index=False)

    hallmark = strict_cross[strict_cross.method.str.contains("Hallmark")].copy()
    hallmark[["expanded_category", "study", "group_id", "domain", "method", "gsm", "gse",
              "dominant shared Hallmarks", "sensitivity_status", "loho_survival_fraction"]].to_csv(
        OUT / "hallmark_cross_tissue_contributions_and_sensitivity.csv", index=False)

    sns.set_theme(style="white", font_scale=.85)
    for d in ["muscle", "liver"]:
        make_heatmap(values, strict, groups, d)

    provenance = {
        "inference_rerun": False,
        "statistical_tests_changed": False,
        "source_differential_rows": len(diff),
        "broad_cross_tissue_rows_audited": len(broad),
        "unique_cross_tissue_gsm": int(broad.gsm.nunique()),
        "strict_cross_tissue_hits": len(strict_cross),
        "classification_basis": "saved title, source_name, and characteristics text only",
        "sensitivity_note": "Strict Hallmark cross-tissue hits did not intersect the saved centroid-top5 LOHO table; marked not tested.",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
