#!/usr/bin/env python3
"""Validate expanded-ChEMBL Evaluation 2 outputs without touching frozen analyses."""

from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results/evaluation2_expanded"

qc = json.loads((OUT / "drug_target_universe_qc.json").read_text())
edges = pd.read_csv(OUT / "chembl37_broad_drug_target_edges.csv.gz")
enrichment = pd.read_parquet(OUT / "drug_enrichment_gt10.parquet")
audit = pd.read_csv(OUT / "published_drug_recovery.csv")

assert not edges.duplicated(["drug_id", "target_gene"]).any()
assert edges.drug_id.nunique() == qc["broad_universe"]["drugs_with_edges"]
assert edges.target_gene.nunique() == qc["broad_universe"]["targets"]
assert len(edges) == qc["broad_universe"]["edges"]
assert enrichment.groupby(["accession", "method"]).size().eq(qc["broad_universe"]["drugs_gt10_targets"]).all()
assert set(enrichment.method) == {"Bridge", "DE"}
assert len(audit) == 30 and audit.reference_drug.nunique() == 15
assert (enrichment.fdr + 1e-15 >= enrichment.pvalue).all()
assert qc["selection_frozen_before_reference_drug_audit"] is True

summary = {"valid": True, "edges": len(edges), "targets": edges.target_gene.nunique(),
           "drugs": edges.drug_id.nunique(), "eligible_gt10": qc["broad_universe"]["drugs_gt10_targets"],
           "enrichment_rows": len(enrichment), "reference_audit_rows": len(audit),
           "evaluation1_rerun": False, "bridge_rerun": False, "de_rerun": False,
           "phase2_rerun": False}
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
