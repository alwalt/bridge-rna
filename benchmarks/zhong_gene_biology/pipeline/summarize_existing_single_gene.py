#!/usr/bin/env python3
"""Summarize completed single-gene results without rerunning any model or probe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "zhong_gene_biology"
RESULTS = BENCH / "results"

EXPECTED_TERMS = {"GO": 56, "OMIM": 103}
REPRESENTATIONS = {
    "E0": "E0_static",
    "E1": "E1_null_L12",
    "E2": "E2_GTEx200_L12",
}

ZHONG_REFERENCES = [
    ("GenePT-Model3", "literature", 0.926, 0.877),
    ("BioConceptVec-FastText", "literature", 0.909, 0.871),
    ("Mashup", "PPI", 0.891, 0.782),
    ("ProtT5", "amino acid", 0.845, 0.691),
    ("ESMB1", "amino acid", 0.841, 0.685),
    ("scGPT-human", "single-cell expression", 0.787, 0.744),
]


def main() -> None:
    rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for task, expected_n in EXPECTED_TERMS.items():
        reference_terms: set[str] | None = None
        for short, full_name in REPRESENTATIONS.items():
            path = RESULTS / f"{short}_{task}.csv"
            df = pd.read_csv(path)
            required = {"term", "AUROC", "AUPRC", "n_test_pos", "n_test_neg"}
            if not required.issubset(df.columns):
                raise AssertionError(f"{path} missing columns: {sorted(required - set(df.columns))}")
            if len(df) != expected_n or df["term"].nunique() != expected_n:
                raise AssertionError(
                    f"{task}/{short}: expected {expected_n} unique terms, got "
                    f"{len(df)} rows and {df['term'].nunique()} unique terms"
                )
            if df["term"].duplicated().any() or df[["AUROC", "AUPRC"]].isna().any().any():
                raise AssertionError(f"{task}/{short}: duplicate terms or missing final metrics")
            terms = set(df["term"].astype(str))
            if reference_terms is None:
                reference_terms = terms
            elif terms != reference_terms:
                raise AssertionError(f"{task}: representation term sets differ")

            rows.append(
                {
                    "Task": task,
                    "Representation": short,
                    "Representation_full": full_name,
                    "N_terms": int(len(df)),
                    "Mean_AUROC": float(df["AUROC"].mean()),
                    "Median_AUROC": float(df["AUROC"].median()),
                    "SD_AUROC": float(df["AUROC"].std(ddof=1)),
                    "Mean_AUPRC": float(df["AUPRC"].mean()),
                    "Median_AUPRC": float(df["AUPRC"].median()),
                    "SD_AUPRC": float(df["AUPRC"].std(ddof=1)),
                }
            )
            inputs.append(
                {
                    "path": str(path.resolve()),
                    "task": task,
                    "representation": short,
                    "rows": int(len(df)),
                    "unique_terms": int(df["term"].nunique()),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "single_gene_zhong_matched_summary.csv", index=False)

    comparison_rows = [
        {"Model": model, "Input_type": input_type, "GO_mean_AUROC": go, "OMIM_mean_AUROC": omim}
        for model, input_type, go, omim in ZHONG_REFERENCES
    ]
    for short in REPRESENTATIONS:
        go = summary[(summary["Task"] == "GO") & (summary["Representation"] == short)].iloc[0]
        omim = summary[(summary["Task"] == "OMIM") & (summary["Representation"] == short)].iloc[0]
        comparison_rows.append(
            {
                "Model": f"Bridge {short}",
                "Input_type": "bulk-expression FM",
                "GO_mean_AUROC": float(go["Mean_AUROC"]),
                "OMIM_mean_AUROC": float(omim["Mean_AUROC"]),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(RESULTS / "zhong_literature_comparison.csv", index=False)

    provenance = {
        "status": "complete",
        "operation": "read-only aggregation of existing final per-term results",
        "aggregation_unit": "one unweighted final held-out test metric per biological term",
        "sd_definition": "sample standard deviation across terms (ddof=1)",
        "not_averaged": [
            "CV-fold training scores",
            "individual predictions",
            "terms weighted by gene count",
            "predictions pooled across terms or tasks",
        ],
        "expected_terms": EXPECTED_TERMS,
        "inputs": inputs,
        "literature_reference_note": "Zhong values were supplied explicitly for comparison and were not recomputed.",
    }
    (RESULTS / "single_gene_zhong_matched_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
