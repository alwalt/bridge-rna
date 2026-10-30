#!/usr/bin/env python3
"""Download relevant NCBI GEO metadata-only SOFT records with resume bookkeeping."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data"
MANIFEST = REPO_ROOT / "data/manifests/archs4_sample_metadata_v2.5.parquet"
CACHE = DATA_ROOT / "geo" / "geo_family_soft"
RESULTS = ROOT / "results"


def gses() -> list[str]:
    frame = pd.read_parquet(MANIFEST, columns=["series_id"])
    values: set[str] = set()
    for raw in frame.series_id:
        values.update(re.findall(r"GSE\d+", str(raw).upper()))
    return sorted(values, key=lambda x: int(x[3:]))


ENDPOINT = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"


def url_for(gse: str, target: str) -> str:
    return f"{ENDPOINT}?acc={gse}&targ={target}&view=brief&form=text"


def valid(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", errors="replace") as stream:
            prefix = stream.read(4096)
        return f"^SERIES = {path.name.split('_', 1)[0]}" in prefix
    except (OSError, EOFError):
        return False


def download(gse: str, timeout: int = 30) -> dict[str, object]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{gse}_metadata.soft.txt.gz"
    if path.exists() and valid(path):
        return {"gse": gse, "status": "cached", "http_status": 200,
                "bytes": path.stat().st_size, "url": url_for(gse, "self") + " ; " + url_for(gse, "gsm"), "error": ""}
    temporary = path.with_suffix(".gz.part")
    try:
        parts = []
        for target in ["self", "gsm"]:
            response = None
            last_error = None
            success = False
            for attempt in range(3):
                try:
                    response = requests.get(ENDPOINT,
                        params={"acc":gse,"targ":target,"view":"brief","form":"text"},
                        timeout=timeout, headers={"User-Agent":"BridgeRNA metadata research audit"})
                    response.raise_for_status()
                    if not response.content.startswith(b"^"):
                        raise ValueError(f"unexpected {target} response")
                    success = True
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 2: time.sleep(1.0 + attempt)
            if not success or response is None:
                raise last_error or RuntimeError(f"failed {target} response")
            parts.append(response.content)
        payload = b"\n".join(parts)
        digest = hashlib.sha256(payload).hexdigest()
        with gzip.open(temporary, "wb", compresslevel=6) as output:
            output.write(payload)
        if not valid(temporary):
            raise ValueError("download is not valid metadata-only GEO SOFT")
        temporary.replace(path)
        return {"gse": gse, "status": "downloaded", "http_status": 200,
                "bytes": path.stat().st_size, "sha256": digest,
                "url": url_for(gse, "self") + " ; " + url_for(gse, "gsm"), "error": ""}
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"gse": gse, "status": "failed", "http_status": status,
                "bytes": 0, "url": url_for(gse, "self") + " ; " + url_for(gse, "gsm"), "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    targets = gses()
    if args.limit is not None:
        targets = targets[:args.limit]
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    print(f"GEO family metadata: {len(targets):,} targets, {args.workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, gse): gse for gse in targets}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            rows.append(future.result())
            if completed % 500 == 0 or completed == len(targets):
                elapsed = time.monotonic() - started
                rate = completed / elapsed
                eta = (len(targets) - completed) / rate if rate else 0
                failures = sum(x["status"] == "failed" for x in rows)
                print(f"{completed:,}/{len(targets):,} elapsed={elapsed:.1f}s rate={rate:.1f}/s "
                      f"eta={eta/60:.1f}m failures={failures}", flush=True)
    frame = pd.DataFrame(rows).sort_values("gse")
    frame["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    frame.to_csv(RESULTS / "geo_download_manifest.csv", index=False)
    frame[frame.status == "failed"].to_csv(RESULTS / "geo_download_failures.csv", index=False)
    (RESULTS / "geo_download_provenance.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "targets": len(targets),
        "workers": args.workers, "cache": CACHE.relative_to(REPO_ROOT).as_posix(),
        "format": "GEO accession viewer brief SOFT (GSE self + linked GSM metadata), locally gzip-compressed",
        "expression_or_supplementary_downloads": False,
    }, indent=2))
    if (frame.status == "failed").any():
        raise SystemExit(f"Completed with {(frame.status == 'failed').sum()} failures; rerun to retry")


if __name__ == "__main__":
    main()
