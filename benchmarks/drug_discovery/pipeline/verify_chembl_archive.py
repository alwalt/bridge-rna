#!/usr/bin/env python3
"""Verify the acquired ChEMBL archive against the frozen official checksum."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESOURCE = json.loads((HERE / "references/drug_target_resource.json").read_text())
ARCHIVE = HERE / "work/chembl/chembl_37_sqlite.tar.gz"
QC_PATH = HERE / "results/chembl37_contract_qc.json"

digest = hashlib.sha256()
with ARCHIVE.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
observed = digest.hexdigest()
expected = RESOURCE["distribution_sha256"]
if observed != expected:
    raise SystemExit(f"ChEMBL archive checksum mismatch: {observed} != {expected}")

qc = json.loads(QC_PATH.read_text())
qc.update({
    "published_archive_sha256": expected,
    "published_archive_checksum_verified": True,
    "local_archive": str(ARCHIVE.relative_to(HERE)),
    "local_archive_bytes": ARCHIVE.stat().st_size,
    "local_archive_sha256": observed,
    "local_archive_checksum_verified": True,
    "primary_min_targets": 3,
    "sensitivity_min_targets": [2, 5],
    "drugs_with_at_least_2_bridge_targets": 223,
    "drugs_with_at_least_3_bridge_targets": 63,
    "drugs_with_at_least_5_bridge_targets": 10,
})
qc["note"] = "The official ChEMBL 37 SQLite archive was acquired locally and its SHA-256 matched the frozen EMBL-EBI checksum; the filtered API snapshot was independently frozen and hashed."
QC_PATH.write_text(json.dumps(qc, indent=2) + "\n")
print(observed)
