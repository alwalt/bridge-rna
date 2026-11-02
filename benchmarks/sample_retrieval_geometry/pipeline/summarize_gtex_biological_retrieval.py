#!/usr/bin/env python3
"""Build compact presentation tables from completed GTEx retrieval outputs."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "gtex_biological_retrieval"
METHODS = ["Raw cosine", "ABTT D=10", "Whitening", "NICDM", "Local Scaling"]

pt = pd.read_csv(OUT / "per_tissue_metrics.csv")
harm = pd.read_csv(OUT / "tissue_harmonization.csv")
qc = pd.read_csv(OUT / "archs4_tissue_annotation_qc.csv")
summary = pd.read_csv(OUT / "archs4_annotation_summary.csv")

p10 = pt[pt.k.eq(10)].copy()
precision = p10.pivot(index="tissue", columns="method", values="annotated_precision")
unknown = p10.pivot(index="tissue", columns="method", values="unknown_rate")
queries = p10.groupby("tissue").queries.first()
support = qc.set_index("tissue")
table = pd.DataFrame({"tissue": sorted(precision.index)}).set_index("tissue")
table["gtex_queries"] = queries
table["archs4_references"] = support.references
table["archs4_gses"] = support.gses
for method in METHODS:
    key = method.lower().replace(" ", "_").replace("=", "")
    table[f"{key}_p_at_10"] = precision[method]
    table[f"{key}_unknown_at_10"] = unknown[method]
table.reset_index().to_csv(OUT / "per_tissue_p10_comparison.csv", index=False)

raw = precision["Raw cosine"]
diagnostics = []
for method in ["ABTT D=10", "Whitening", "NICDM", "Local Scaling"]:
    delta = (precision[method] - raw).sort_values()
    for category, series in [("largest_degradation", delta.head(3)), ("largest_improvement", delta.tail(3).sort_values(ascending=False))]:
        for tissue, value in series.items():
            diagnostics.append({"method": method, "criterion": category, "tissue": tissue,
                                "raw_p_at_10": raw[tissue], "method_p_at_10": precision.loc[tissue, method],
                                "delta_p_at_10": value, "queries": queries[tissue],
                                "method_unknown_at_10": unknown.loc[tissue, method]})
pd.DataFrame(diagnostics).to_csv(OUT / "objective_tissue_rank_changes.csv", index=False)

represented = set(harm.archs4_tissue.dropna())
represented_labeled = int(qc.loc[qc.tissue.isin(represented), "references"].sum())
eligible = set(harm.loc[harm.eligible, "archs4_tissue"])
eligible_labeled = int(qc.loc[qc.tissue.isin(eligible), "references"].sum())
summary["gtex_represented_labeled_references"] = represented_labeled
summary["gtex_represented_coverage_of_full_corpus"] = represented_labeled / summary.references_after_exclusion
summary["eligible_gtex_tissue_labeled_references"] = eligible_labeled
summary["eligible_gtex_tissue_coverage_of_full_corpus"] = eligible_labeled / summary.references_after_exclusion
summary.to_csv(OUT / "archs4_annotation_summary.csv", index=False)
print("Wrote GTEx presentation tables")
