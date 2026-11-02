#!/usr/bin/env python3
"""Focused biological audit of saved OSDR muscle/liver retrieval outputs.

This script performs no model inference. It consumes the frozen OSDR atlas tables,
applies the previously declared strict differential-retrieval criteria, assigns
conservative metadata-supported biological categories, and creates audit figures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "osdr_atlas"
OUT = SOURCE / "muscle_liver_audit"
FIG = OUT / "figures"
METHODS = ["Bridge-512 cosine", "Hallmark-50 cosine", "Hallmark signed top10 overlap"]


def domain(tissue: str) -> str | None:
    text = str(tissue).lower()
    if "liver" in text:
        return "liver"
    if re.search(r"muscle|quadriceps|gastrocnemius|soleus|tibialis|digitorum", text):
        return "muscle"
    return None


def has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


def category(row: pd.Series) -> str:
    text = " | ".join(str(row.get(c, "")) for c in ["title", "source_name", "characteristics"])
    d = row["domain"]
    if d == "muscle":
        if has(text, r"exercise|training|trained|post.?exercise|endurance|resistance exercise|running|cycling"):
            return "exercise/training"
        if has(text, r"atrophy|disuse|immobili|unload|denerv|bed.?rest|cachexia"):
            return "atrophy/disuse"
        if has(text, r"injur|trauma|cardiotoxin|ischemi|regenerat|muscle damage"):
            return "muscle injury"
        if has(text, r"myositis|muscular dystrophy|duchenne|inflamm.{0,20}muscle"):
            return "inflammatory muscle state"
        if has(text, r"obes|diabet|insulin resist|metabolic syndrome"):
            return "metabolic disease"
        if has(text, r"\baging\b|\baged\b|elderly|old donor|age[ _-]?(6[5-9]|[7-9][0-9])"):
            return "aging"
        if has(text, r"skeletal muscle|vastus|soleus|quadriceps|gastrocnemius|tibialis|digitorum|biceps brachii|abductor hallucis|\bmuscle\b"):
            return "normal/unspecified skeletal muscle"
        if has(text, r"fibroblast|adipose|tumou?r|carcinoma|adenoma|bone marrow|synovial|kidney|renal"):
            return "other cross-tissue state"
        return "ambiguous/unresolved"
    if has(text, r"fatty liver|steatos|nafld|mafld|nash|obes|diabet|insulin resist|metabolic syndrome"):
        return "fatty/metabolic liver disease"
    if has(text, r"fibros|cirrho"):
        return "fibrosis"
    if has(text, r"hepatocellular|cholangiocarc|liver cancer|hepatic cancer|liver tumou?r|hepatoma"):
        return "cancer"
    if has(text, r"hepatitis|inflamm.{0,20}liver|liver.{0,20}inflamm"):
        return "inflammatory liver state"
    if has(text, r"acetaminophen|\bapap\b|toxic|toxin|drug|ethanol|alcohol|ccl4|carbon tetrachloride"):
        return "drug/toxicant exposure"
    if has(text, r"liver injury|hepatic injury|liver failure|hepatic failure|hepatectomy|resection|ischemi"):
        return "liver injury"
    if has(text, r"\bliver\b|hepatocyte|hepatic"):
        return "normal/unspecified liver"
    if has(text, r"adipose|breast|melanoma|tumou?r|carcinoma"):
        return "other cross-tissue state"
    return "ambiguous/unresolved"


def dominant_hallmarks(hits: pd.DataFrame) -> pd.DataFrame:
    h = hits[(hits.query_type == "individual") & (hits.condition == "FLT")].copy()
    h = h[h.method.str.contains("Hallmark", na=False)]
    h = h[h.dominant_shared_hallmarks.notna() & (h.dominant_shared_hallmarks != "")]
    if h.empty:
        return pd.DataFrame(columns=["group_id", "method", "gsm", "dominant_shared_hallmarks"])
    def mode_join(x: pd.Series) -> str:
        tokens = []
        for value in x.astype(str):
            tokens.extend(t.strip() for t in value.split(";") if t.strip())
        if not tokens:
            return ""
        counts = pd.Series(tokens).value_counts()
        return "; ".join(counts.index[:5])
    return (h.groupby(["group_id", "method", "gsm"], as_index=False)
             .dominant_shared_hallmarks.agg(mode_join))


def plot_heatmap(values: pd.DataFrame, strict: pd.DataFrame, groups: pd.DataFrame, d: str) -> None:
    cats = [c for c in values.loc[values.domain == d, "category"].unique() if c != "ambiguous/unresolved"]
    gids = groups.loc[groups.domain == d, "group_id"].tolist()
    labels = dict(zip(groups.group_id, groups.study + "\n" + groups.group_id.str.replace("OSDRG", "G", regex=False)))
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(max(12, .65 * len(gids)), 3.2 * len(METHODS)), sharex=True)
    for ax, method in zip(axes, METHODS):
        sub = values[(values.domain == d) & (values.method == method)]
        mat = sub.pivot(index="category", columns="group_id", values="standardized_effect").reindex(index=cats, columns=gids).fillna(0)
        sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5, ax=ax,
                    cbar_kws={"label": "standardized FLT−GC enrichment"}, linewidths=.25)
        sig = strict[(strict.domain == d) & (strict.method == method)]
        sig_cells = set(zip(sig.category, sig.group_id))
        for yi, cat in enumerate(mat.index):
            for xi, gid in enumerate(mat.columns):
                if (cat, gid) in sig_cells:
                    ax.text(xi + .5, yi + .5, "●", ha="center", va="center", fontsize=7, color="black")
        ax.set_title(method + "  (● strict replicate-consistent FLT enrichment)", loc="left", fontsize=10)
        ax.set_ylabel("")
    axes[-1].set_xticklabels([labels[g] for g in gids], rotation=55, ha="right", fontsize=8)
    axes[-1].set_xlabel("Matched OSDR experimental group")
    fig.suptitle(f"{d.title()}: ARCHS4 biological-state shift in flight", y=1.005, fontsize=14)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{d}_category_enrichment_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_strongest(strict: pd.DataFrame, groups: pd.DataFrame, d: str) -> list[str]:
    sub = strict[strict.domain == d].copy()
    strength = sub.groupby("group_id").rrf_delta_flt_minus_gc.sum().sort_values(ascending=False)
    selected = strength.head(5).index.tolist()
    if not selected:
        return []
    plot = sub[sub.group_id.isin(selected)].sort_values("rrf_delta_flt_minus_gc", ascending=False)
    plot["state"] = plot["category"] + " — " + plot["gse"].fillna("no GSE") + " / " + plot["gsm"]
    plot = plot.groupby(["group_id", "method"], group_keys=False).head(6)
    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 3.2 * len(selected)), squeeze=False)
    study = groups.set_index("group_id").study.to_dict()
    palette = dict(zip(METHODS, ["#355C7D", "#C06C84", "#6C5B7B"]))
    for ax, gid in zip(axes[:, 0], selected):
        p = plot[plot.group_id == gid].sort_values("rrf_delta_flt_minus_gc")
        ax.barh(np.arange(len(p)), p.rrf_delta_flt_minus_gc, color=p.method.map(palette))
        ax.set_yticks(np.arange(len(p)), [s if len(s) < 88 else s[:85] + "…" for s in p.state], fontsize=7)
        ax.set_title(f"{study[gid]} / {gid}", loc="left", fontsize=10)
        ax.set_xlabel("FLT−GC reciprocal-rank enrichment")
        for y, q in enumerate(p.bh_q):
            ax.text(p.rrf_delta_flt_minus_gc.iloc[y], y, f"  q={q:.3g}", va="center", fontsize=7)
        sns.despine(ax=ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="upper right", frameon=False)
    fig.suptitle(f"Strongest replicate-consistent {d} flight retrieval shifts", y=1.002, fontsize=14)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{d}_strongest_group_retrieval_shifts.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(SOURCE / "eligible_group_manifest.csv")
    groups["domain"] = groups.tissue.map(domain)
    groups = groups[groups.domain.notna()].copy()
    groups.to_csv(OUT / "eligible_muscle_liver_groups.csv", index=False)

    diff = pd.read_parquet(SOURCE / "flt_vs_gc_differential_retrieval.parquet")
    diff = diff.merge(groups[["group_id", "study", "tissue", "domain"]], on="group_id", how="inner")
    diff["category"] = diff.apply(category, axis=1)
    strict_mask = ((diff.bh_q <= .10) & (diff.rrf_delta_flt_minus_gc > 0) &
                   (diff.flt_replicate_prevalence >= .75) & (diff.gc_replicate_prevalence <= .25))
    strict = diff[strict_mask].copy()

    hits = pd.read_parquet(SOURCE / "individual_and_centroid_top100.parquet")
    hm = dominant_hallmarks(hits)
    strict = strict.merge(hm, on=["group_id", "method", "gsm"], how="left")
    strict["dominant_shared_hallmarks"] = strict.dominant_shared_hallmarks.fillna("")
    strict["same_tissue"] = strict.apply(
        lambda r: bool(has(" | ".join(map(str, [r.title, r.source_name, r.characteristics])),
                           r"liver|hepat") if r.domain == "liver" else
                       has(" | ".join(map(str, [r.title, r.source_name, r.characteristics])),
                           r"muscle|vastus|soleus|quadriceps|gastrocnemius|tibialis|digitorum|biceps|abductor")), axis=1)
    strict["title_source"] = strict.title.fillna("") + " | " + strict.source_name.fillna("")

    hubs = pd.read_csv(SOURCE / "archs4_hub_frequency.csv")
    hub_cut = hubs.groupby(["method", "condition"]).occurrences.transform(lambda x: x.quantile(.95))
    hub_gse = set(hubs.loc[(hubs.condition == "FLT") & (hubs.occurrences >= hub_cut), ["method", "gse"]].itertuples(index=False, name=None))
    strict["top5pct_flt_hub_gse"] = [(m, g) in hub_gse for m, g in zip(strict.method, strict.gse)]

    loho = pd.read_csv(SOURCE / "hallmark_leave_one_dominant_out.csv")
    loho = (loho[loho.condition == "FLT"].groupby(["group_id", "gsm"], as_index=False)
            .agg(loho_tests=("survives_top100", "size"), loho_survival_fraction=("survives_top100", "mean")))
    strict = strict.merge(loho, on=["group_id", "gsm"], how="left")

    requested = strict.rename(columns={
        "study": "OSDR study", "group_id": "group", "gsm": "ARCHS4 GSM", "gse": "GSE",
        "title_source": "title/source", "flt_replicate_prevalence": "FLT prevalence",
        "gc_replicate_prevalence": "GC prevalence", "rrf_delta_flt_minus_gc": "FLT-GC retrieval effect",
        "bh_q": "q-value", "same_tissue": "same tissue?", "dominant_shared_hallmarks": "dominant shared Hallmarks",
    })
    columns = ["OSDR study", "group", "tissue", "method", "ARCHS4 GSM", "GSE", "title/source",
               "FLT prevalence", "GC prevalence", "FLT-GC retrieval effect", "q-value", "same tissue?",
               "dominant shared Hallmarks", "category", "top5pct_flt_hub_gse", "loho_tests", "loho_survival_fraction"]
    requested[columns].sort_values(["tissue", "OSDR study", "method", "q-value"]).to_csv(
        OUT / "strict_flt_enriched_hits.csv", index=False)

    # Preserve every selected group, including groups with zero strict hits.
    counts = strict.groupby(["group_id", "method"]).size().unstack(fill_value=0).reindex(columns=METHODS, fill_value=0)
    coverage = groups.merge(counts, left_on="group_id", right_index=True, how="left").fillna({m: 0 for m in METHODS})
    coverage["any_strict_hit"] = coverage[METHODS].sum(axis=1) > 0
    coverage.to_csv(OUT / "group_strict_hit_coverage.csv", index=False)

    recurrence = (strict.groupby(["domain", "category"], as_index=False)
                  .agg(n_hits=("gsm", "size"), n_gse=("gse", "nunique"), n_groups=("group_id", "nunique"),
                       n_studies=("study", "nunique"), methods=("method", lambda x: "; ".join(sorted(set(x)))),
                       max_effect=("rrf_delta_flt_minus_gc", "max"), min_q=("bh_q", "min")))
    recurrence.to_csv(OUT / "category_recurrence.csv", index=False)
    gse_recurrence = (strict.groupby(["domain", "category", "gse"], as_index=False)
                      .agg(n_hits=("gsm", "size"), n_groups=("group_id", "nunique"),
                           n_studies=("study", "nunique"),
                           groups=("group_id", lambda x: "; ".join(sorted(set(x)))),
                           methods=("method", lambda x: "; ".join(sorted(set(x)))),
                           max_effect=("rrf_delta_flt_minus_gc", "max"), min_q=("bh_q", "min"),
                           top5pct_flt_hub_gse=("top5pct_flt_hub_gse", "max")))
    gse_recurrence.to_csv(OUT / "gse_recurrence_and_hub_audit.csv", index=False)

    # Category enrichment from all saved candidates, standardized within each method.
    values = (diff[diff.category != "ambiguous/unresolved"]
              .groupby(["domain", "method", "group_id", "category"], as_index=False)
              .rrf_delta_flt_minus_gc.sum())
    values["standardized_effect"] = values.groupby("method").rrf_delta_flt_minus_gc.transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else 1))
    values.to_csv(OUT / "category_group_standardized_effects.csv", index=False)

    sns.set_theme(style="white", font_scale=.9)
    for d in ["muscle", "liver"]:
        plot_heatmap(values, strict, groups, d)
    strongest = {d: plot_strongest(strict, groups, d) for d in ["muscle", "liver"]}

    summary = {
        "inputs": {"source": str(SOURCE), "inference_rerun": False},
        "strict_definition": {"bh_q_max": .10, "effect": ">0", "flt_prevalence_min": .75, "gc_prevalence_max": .25},
        "eligible_groups": groups.domain.value_counts().to_dict(),
        "groups_with_strict_hits": strict.groupby("domain").group_id.nunique().to_dict(),
        "strict_hits": strict.groupby("domain").size().to_dict(),
        "strongest_groups": strongest,
        "limitations": [
            "Metadata categories are conservative text-supported annotations, not disease labels.",
            "No strict hit is interpreted as disease equivalence or causal mechanism.",
            "LOHO survival is available only where the saved centroid top-five sensitivity table covers the GSM.",
            "Category heatmaps aggregate candidate-level reciprocal-rank effects and are exploratory.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
