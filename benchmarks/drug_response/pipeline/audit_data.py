#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, RESULTS, SHARED, checksum, write_json


def main() -> None:
    expression_path = Path(CONFIG["source"]["expression"])
    response_path = Path(CONFIG["source"]["response"])
    response = pd.read_csv(response_path, low_memory=False)
    expression = pd.read_csv(expression_path, low_memory=False)
    meta = ["depmap_id", "cell_line_display_name", "lineage_1", "lineage_2",
            "lineage_3", "lineage_6", "lineage_4"]
    genes = [column for column in expression if column not in meta]
    values = expression[genes].to_numpy(dtype=np.float32)
    canonical = pd.read_csv(SHARED / "data/ensembl/canonical_genes.csv")
    bulkformer = pd.read_csv(SHARED / "model/BulkFormer/data/bulkformer_gene_info.csv")
    screen_counts = response.groupby(["Dataset Version", "Drug ID"]).size()
    audit = {
        "expression_rows": len(expression), "expression_genes": len(genes),
        "expression_duplicate_cell_ids": int(expression.depmap_id.duplicated().sum()),
        "expression_missing_values": int(np.isnan(values).sum()),
        "expression_negative_values": int((values < 0).sum()),
        "expression_min": float(values.min()), "expression_max": float(values.max()),
        "response_rows": len(response), "cell_lines": int(response.ModelID.nunique()),
        "gdsc_drug_ids": int(response["Drug ID"].nunique()),
        "pubchem_cids": int(response.cid.nunique()),
        "dataset_version_drug_screens": int(screen_counts.size),
        "screen_observations_min": int(screen_counts.min()),
        "screen_observations_median": float(screen_counts.median()),
        "screen_observations_max": int(screen_counts.max()),
        "exact_duplicate_rows": int(response.duplicated().sum()),
        "duplicate_cell_drug_id_rows": int(response.duplicated(["ModelID", "Drug ID"], False).sum()),
        "duplicate_cell_version_drug_rows": int(response.duplicated(
            ["ModelID", "Dataset Version", "Drug ID"], False).sum()),
        "bridge_symbol_overlap": len(set(genes) & set(canonical.gene_symbol.astype(str))),
        "bulkformer_symbol_overlap": len(set(genes) & set(bulkformer.gene_symbol.astype(str))),
        "all_response_cells_in_expression": bool(set(response.ModelID) <= set(expression.depmap_id)),
        "dataset_versions": response["Dataset Version"].value_counts().to_dict(),
        "ic50_summary": response.IC50.describe().to_dict(),
        "auc_summary": response.AUC.describe().to_dict(),
        "z_score_summary": response["Z score"].describe().to_dict(),
    }
    write_json(RESULTS / "dataset_audit.json", audit)
    manifest = {
        "zenodo_record": CONFIG["source"]["zenodo_record"],
        "assets": [
            {"path": str(expression_path), "bytes": expression_path.stat().st_size,
             "sha256": checksum(expression_path), "zenodo_md5": "0cdf07dfd599142efbfb625a661d4a92"},
            {"path": str(response_path), "bytes": response_path.stat().st_size,
             "sha256": checksum(response_path), "zenodo_md5": "7b81c3a153c2a19b508e175c1bbdcc57"},
        ],
    }
    write_json(RESULTS / "asset_manifest.json", manifest)
    response.assign(screen_id=response["Dataset Version"].astype(str) + ":" +
                    response["Drug ID"].astype(str)).to_parquet(
                        RESULTS / "observation_manifest.parquet", index=False)
    expression[meta].to_csv(RESULTS / "cell_line_manifest.csv", index=False)
    pd.DataFrame({"gene_symbol": genes,
                  "in_bridge": [g in set(canonical.gene_symbol.astype(str)) for g in genes],
                  "in_bulkformer": [g in set(bulkformer.gene_symbol.astype(str)) for g in genes]}).to_csv(
                      RESULTS / "gene_overlap.csv", index=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
