#!/usr/bin/env python3
"""Validate headline final-report claims against frozen machine-readable outputs."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "benchmarks/drug_discovery/FINAL_SCIENTIFIC_REPORT.md"


def close(value: float, expected: float, tolerance: float = 5e-7) -> None:
    assert abs(float(value) - expected) <= tolerance, (value, expected)


def main() -> None:
    report = REPORT.read_text()

    disease = pd.read_csv(
        ROOT / "benchmarks/deweerd_replication/results/evaluation/disease_gene_recovery.csv"
    )
    primary = disease[
        (disease["module_size"] == 500)
        & (disease["reference_definition"] == "all_v7_associations")
    ]
    expected_recovery = {
        ("GSE138614", "Bridge"): 72,
        ("GSE138614", "DE"): 87,
        ("GSE101794", "Bridge"): 70,
        ("GSE101794", "DE"): 79,
        ("GSE72509", "Bridge"): 106,
        ("GSE72509", "DE"): 72,
    }
    observed = {
        (row.accession, row.method): int(row.known_genes_recovered)
        for row in primary.itertuples()
    }
    assert observed == expected_recovery

    overall = pd.read_csv(
        ROOT
        / "benchmarks/drug_discovery/results/expanded_chembl_sensitivity/partition_overall_summary.csv"
    ).set_index("partition")
    close(overall.loc["Bridge-only", "recurrence_fraction"], 0.435307)
    close(overall.loc["DE-only", "recurrence_fraction"], 0.117259)
    close(overall.loc["Bridge-only", "osdr_jaccard"], 0.158969)
    close(overall.loc["DE-only", "osdr_jaccard"], 0.016042)
    close(overall.loc["Bridge-only", "targetable_fraction"], 0.163337)
    close(overall.loc["DE-only", "targetable_fraction"], 0.146111)

    drug = pd.read_csv(
        ROOT
        / "benchmarks/drug_discovery/results/expanded_chembl_sensitivity/expanded_vs_restrictive.csv"
    ).set_index("condition")
    close(drug.loc["Muscle atrophy", "bridge_mean_pairwise_top25_jaccard_expanded"], 0.164080)
    close(drug.loc["Muscle atrophy", "de_mean_pairwise_top25_jaccard_expanded"], 0.006803)
    close(drug.loc["Muscle atrophy", "bridge_mean_pairwise_top25_jaccard_restrictive"], 0.714286)

    radiation = pd.read_csv(
        ROOT
        / "benchmarks/drug_discovery/results/radiation_context_sensitivity/bridge_vs_de_stability.csv"
    ).set_index("context_distance")
    all_pairs = radiation.loc["all"]
    assert int(all_pairs["pairs"]) == 21
    assert int(all_pairs["bridge_higher_pairs"]) == 18
    close(all_pairs["mean_bridge_minus_de"], 0.105187)
    close(all_pairs["wilcoxon_pvalue"], 0.000018)

    lincs = json.loads(
        (ROOT / "benchmarks/direction_aware_reversal/results/validation_summary.json").read_text()
    )
    gse = json.loads(
        (ROOT / "benchmarks/gse264130_reversal/results/validation_summary.json").read_text()
    )
    assert lincs["valid"] and not lincs["frozen_input_hash_mismatches"]
    assert gse["valid"] and not gse["benchmark6_input_hash_mismatches"]
    assert not gse["frozen_lincs_hash_mismatches"]
    assert gse["drugs"] == 372 and gse["cells"] == 2
    assert gse["bh_significant_expression"] == 0
    assert gse["bh_significant_latent"] == 0
    assert gse["strict_cross_study_candidates"] == 0

    required_text = [
        "43.5%", "11.7%", "0.159", "0.016", "0.164", "0.0068",
        "18/21", "+0.105", "237 Bridge genes", "15 genes", "minimum q=0.9973",
    ]
    missing = [claim for claim in required_text if claim not in report]
    assert not missing, f"Missing report claims: {missing}"

    print("Final report claims validated against frozen outputs.")


if __name__ == "__main__":
    main()
