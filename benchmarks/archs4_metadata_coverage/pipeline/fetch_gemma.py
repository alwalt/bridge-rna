#!/usr/bin/env python3
"""Fetch Gemma's public dataset catalogue and Bridge-overlapping sample metadata.

The cache is resumable and deliberately limits collection to GSEs represented in
the frozen Bridge ARCHS4 manifest. No expression matrices are requested.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data"
OUT = DATA_ROOT / "gemma"
API = "https://gemma.msl.ubc.ca/rest/v2"
WORKERS = 32


def get_json(url: str, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        try:
            r = requests.get(url, timeout=120, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_catalogue() -> pd.DataFrame:
    path = OUT / "datasets.parquet"
    if path.exists():
        return pd.read_parquet(path)
    count = int(get_json(f"{API}/datasets/count")["data"])
    offsets = list(range(0, count, 100))
    pages: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {
            pool.submit(get_json, f"{API}/datasets?limit=100&offset={off}"): off
            for off in offsets
        }
        for done, fut in enumerate(as_completed(futs), 1):
            off = futs[fut]
            pages[off] = fut.result()["data"]
            if done % 25 == 0 or done == len(futs):
                print(f"catalogue pages {done}/{len(futs)}", flush=True)
    rows = []
    for off in offsets:
        for x in pages[off]:
            rows.append(
                {
                    "gemma_id": x.get("id"),
                    "gse": x.get("accession") or x.get("shortName"),
                    "name": x.get("name"),
                    "description": x.get("description"),
                    "species": (x.get("taxon") or {}).get("scientificName"),
                    "ncbi_taxon_id": (x.get("taxon") or {}).get("ncbiId"),
                    "bioassay_count": x.get("bioAssayCount") or x.get("numberOfBioAssays"),
                    "characteristics_json": json.dumps(x.get("characteristics") or []),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates("gse")
    df.to_parquet(path, index=False)
    return df


def fetch_one(gse: str) -> tuple[str, dict | None, str | None]:
    dest = OUT / "samples" / f"{gse}.json"
    if dest.exists():
        return gse, None, None
    try:
        payload = get_json(f"{API}/datasets/{gse}/samples")
        dest.write_text(json.dumps(payload, separators=(",", ":")))
        return gse, payload, None
    except Exception as e:
        return gse, None, repr(e)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples").mkdir(exist_ok=True)
    catalogue = fetch_catalogue()
    bridge = pd.read_parquet(REPO_ROOT / "data/manifests/archs4_sample_metadata_v2.5.parquet")
    bridge_gses = set()
    for value in bridge["series_id"].dropna().astype(str):
        bridge_gses.update(v.strip() for v in value.split(",") if v.strip())
    targets = sorted(set(catalogue.gse.dropna()) & bridge_gses)
    pending = [x for x in targets if not (OUT / "samples" / f"{x}.json").exists()]
    print(
        f"Gemma datasets={len(catalogue):,}; Bridge GSEs={len(bridge_gses):,}; "
        f"overlap={len(targets):,}; pending sample records={len(pending):,}",
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_one, gse): gse for gse in pending}
        started = time.monotonic()
        for done, fut in enumerate(as_completed(futs), 1):
            gse, _, error = fut.result()
            if error:
                failures.append({"gse": gse, "error": error})
            if done % 100 == 0 or done == len(futs):
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed else 0
                eta = (len(futs) - done) / rate if rate else 0
                print(
                    f"sample metadata {done}/{len(futs)} elapsed={elapsed:.0f}s "
                    f"rate={rate:.1f}/s eta={eta:.0f}s failures={len(failures)}",
                    flush=True,
                )
    pd.DataFrame(failures, columns=["gse", "error"]).to_csv(
        OUT / "fetch_failures.csv", index=False
    )


if __name__ == "__main__":
    main()
