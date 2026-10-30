#!/usr/bin/env python3
"""Refresh compact public caches from the validated research benchmark outputs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks" / "zhong_gene_biology" / "results"
DEST = Path(__file__).resolve().parent / "cached"

FILES = {
    "bridge_E0_GO.csv": SOURCE / "E0_GO.csv",
    "bridge_E1_GO.csv": SOURCE / "E1_GO.csv",
    "bridge_E2_GO.csv": SOURCE / "E2_GO.csv",
    "bridge_E0_OMIM.csv": SOURCE / "E0_OMIM.csv",
    "bridge_E1_OMIM.csv": SOURCE / "E1_OMIM.csv",
    "bridge_E2_OMIM.csv": SOURCE / "E2_OMIM.csv",
    "bulkformer_50M_GO.csv": SOURCE / "bulkformer_static" / "BulkFormer-50M_GO.csv",
    "bulkformer_147M_GO.csv": SOURCE / "bulkformer_static" / "BulkFormer-147M_GO.csv",
    "bulkformer_50M_OMIM.csv": SOURCE / "bulkformer_static" / "BulkFormer-50M_OMIM.csv",
    "bulkformer_147M_OMIM.csv": SOURCE / "bulkformer_static" / "BulkFormer-147M_OMIM.csv",
    "single_gene_expected_summary.csv": SOURCE / "single_gene_zhong_matched_summary.csv",
    "bulkformer_expected_summary.csv": SOURCE / "bulkformer_static" / "summary.csv",
    "zhong_literature_reference.csv": SOURCE / "zhong_literature_comparison.csv",
    "pair_outer_fold_results.csv": SOURCE / "pair_outer_fold_results_all.csv",
    "pair_expected_summary.csv": SOURCE / "pair_representation_summary.csv",
    "pair_expected_best.csv": SOURCE / "final_pair_best_conditions.csv",
    "pair_leakage_qc.csv": SOURCE / "final_pair_leakage_qc.csv",
    "bridge_provenance.json": SOURCE / "final_benchmark_provenance.json",
    "bulkformer_provenance.json": SOURCE / "bulkformer_static" / "provenance.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for public_name, source in FILES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        target = DEST / public_name
        shutil.copyfile(source, target)
        records.append(
            {
                "file": public_name,
                "source": str(source.relative_to(ROOT)),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    manifest = {
        "mode": "cached validated artifacts; no model inference or classifier fitting",
        "source_benchmark": "benchmarks/zhong_gene_biology",
        "files": records,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(records)} cached artifacts and {DEST / 'manifest.json'}")


if __name__ == "__main__":
    main()
