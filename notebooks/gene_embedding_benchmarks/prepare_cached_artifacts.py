#!/usr/bin/env python3
"""Refresh compact public caches from the validated research benchmark outputs."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks" / "zhong_gene_biology" / "results"
DEST = Path(__file__).resolve().parent / "cached"
ZHONG_DEST = DEST / "zhong"
ZHONG_REPOSITORY = "https://github.com/ylaboratory/gene-embedding-benchmarks"
ZHONG_COMMIT = "d1320026a2a4ee033d49517f91e2d1c2ccc8df1e"
ZHONG_RAW = f"https://raw.githubusercontent.com/ylaboratory/gene-embedding-benchmarks/{ZHONG_COMMIT}"

ZHONG_FILES = [
    "data/embed_meta.csv",
    "results/tsvs/single_gene_go_AUROC.tsv",
    "results/tsvs/single_gene_omim_AUROC.tsv",
]
for task in ("ng", "sl", "tf"):
    for operation in ("concat", "product", "sum"):
        ZHONG_FILES.extend(
            [
                f"results/tsvs/paired_gene_{task}_{operation}_outerloop_AUC.tsv",
                f"results/tsvs/paired_gene_{task}_{operation}_outerloop_avgs.tsv",
            ]
        )

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
    ZHONG_DEST.mkdir(parents=True, exist_ok=True)
    zhong_records = []
    for repository_path in ZHONG_FILES:
        filename = Path(repository_path).name
        target = ZHONG_DEST / filename
        url = f"{ZHONG_RAW}/{repository_path}"
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
        record = {
            "file": filename,
            "repository_path": repository_path,
            "source_url": url,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
        zhong_records.append(record)
        records.append(
            {
                "file": f"zhong/{filename}",
                "source": url,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    zhong_provenance = {
        "repository": ZHONG_REPOSITORY,
        "commit": ZHONG_COMMIT,
        "commit_url": f"{ZHONG_REPOSITORY}/commit/{ZHONG_COMMIT}",
        "download_date": date.today().isoformat(),
        "selection": (
            "Published common-intersection GO/OMIM AUROC tables and paired-gene "
            "outer-fold AUROC/summary tables used by the Zhong paper figures; no embeddings."
        ),
        "files": zhong_records,
    }
    provenance_path = ZHONG_DEST / "provenance.json"
    provenance_path.write_text(json.dumps(zhong_provenance, indent=2) + "\n")
    records.append(
        {
            "file": "zhong/provenance.json",
            "source": "generated from pinned download metadata",
            "bytes": provenance_path.stat().st_size,
            "sha256": sha256(provenance_path),
        }
    )
    manifest = {
        "mode": "cached validated artifacts; no model inference or classifier fitting",
        "source_benchmark": "benchmarks/zhong_gene_biology",
        "files": records,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {len(records)} cached artifacts "
        f"({len(zhong_records)} pinned Zhong downloads) and {DEST / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
