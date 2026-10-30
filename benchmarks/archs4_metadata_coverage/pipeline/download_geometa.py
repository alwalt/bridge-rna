#!/usr/bin/env python3
"""Download only the metadata CSVs from the official GEOMeta release."""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parents[1] / "data"
OUT = DATA_ROOT / "geometa"
BASE = "https://huggingface.co/datasets/binchenlab/GEOMeta/resolve/main/data"
FILES = [
    "GEO_HUMAN_500K_training.csv",
    "GEO_HUMAN_500K_test.csv",
    "GEO_HUMAN_500K_test_2024.csv",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"exists {dest} ({dest.stat().st_size:,} bytes)", flush=True)
            continue
        print(f"downloading {name}", flush=True)
        urlretrieve(f"{BASE}/{name}?download=true", dest)
        print(f"saved {dest} ({dest.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
