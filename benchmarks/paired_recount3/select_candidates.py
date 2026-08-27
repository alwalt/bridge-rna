#!/usr/bin/env python3
"""Select reproducible ARCHS4 candidates or finalize matched recount3 pairs.

Candidate selection happens before recount3 metadata matching. Final selection
happens only among successfully matched GSMs and is balanced across GSEs.
Neither mode reads expression data or changes the master sample manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/sample_manifest.parquet"
DEFAULT_H5 = REPO_ROOT / "data/archs4/human_gene_v2.5.h5"
DEFAULT_OUTPUT = HERE / "outputs/candidate_samples.parquet"
COHORT_RULES = {
    "unseen_sample_seen_study": lambda d: (
        d["split"].eq("unseen")
        & d["study_exposure"].eq("seen_study")
        & d["mapping_status"].eq("mapped_single")
    ),
    "strict_unseen_single_gse": lambda d: (
        d["split"].eq("unseen")
        & d["study_exposure"].eq("unseen_study")
        & d["mapping_status"].eq("mapped_single")
    ),
}
ACCESSION_RE = re.compile(r"\b(?:SRR|ERR|DRR|SRX|ERX|DRX|SRS|ERS|DRS)\d+\b", re.I)


def stable_score(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def balanced_select(frame: pd.DataFrame, n: int, cap: int, seed: int) -> pd.DataFrame:
    """Deterministically sample while limiting representation from each GSE."""
    ranked = frame.assign(
        _score=frame["gsm"].astype(str).map(lambda x: stable_score(seed, x))
    ).sort_values("_score")
    if n >= len(ranked):
        return ranked.drop(columns="_score")
    selected = ranked.groupby("gse", sort=False, group_keys=False).head(cap)
    return selected.sort_values("_score").head(n).drop(columns="_score")


def add_archs4_relations(frame: pd.DataFrame, h5_path: Path) -> pd.DataFrame:
    wanted = set(frame["gsm"])
    rows: list[dict[str, str]] = []
    with h5py.File(h5_path, "r") as handle:
        gsm_values = handle["meta/samples/geo_accession"][:]
        normalized = [value.decode() if isinstance(value, bytes) else str(value) for value in gsm_values]
        positions = np.asarray([i for i, gsm in enumerate(normalized) if gsm in wanted])
        relation_values = handle["meta/samples/relation"][positions]
        for index, value in zip(positions, relation_values):
            gsm = normalized[index]
            raw = value.decode() if isinstance(value, bytes) else str(value)
            accessions = sorted(set(match.upper() for match in ACCESSION_RE.findall(raw)))
            rows.append({
                "gsm": gsm,
                "archs4_relation": raw,
                "archs4_sra_accessions": ",".join(accessions),
            })
    relation = pd.DataFrame(rows)
    return frame.merge(relation, on="gsm", how="left", validate="one_to_one")


def build_candidates(config: dict, manifest_path: Path, h5_path: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    species = config["species"]
    frames = []
    for cohort in config["cohorts"]:
        if cohort not in COHORT_RULES:
            raise ValueError(f"Unsupported cohort in config: {cohort}")
        eligible = manifest.loc[
            COHORT_RULES[cohort](manifest) & manifest["species"].eq(species)
        ].copy()
        eligible["cohort"] = cohort
        eligible["gse"] = eligible["gse_candidates_str"].astype(str)
        requested = int(config["candidate_samples_per_cohort"])
        chosen = balanced_select(
            eligible, requested, int(config["max_candidates_per_gse"]),
            int(config["random_seed"]),
        )
        print(f"{cohort}: eligible={len(eligible):,}, candidates={len(chosen):,}")
        frames.append(chosen)
    candidates = pd.concat(frames, ignore_index=True)
    if candidates["gsm"].duplicated().any():
        raise ValueError("A GSM was selected into more than one benchmark cohort")
    keep = ["sample_id", "gsm", "gse", "cohort", "split", "study_exposure",
            "mapping_status", "species", "global_index"]
    keep = [column for column in keep if column in candidates]
    return add_archs4_relations(candidates[keep], h5_path)


def finalize_pairs(config: dict, matches_path: Path) -> pd.DataFrame:
    matches = pd.read_parquet(matches_path)
    required = {"gsm", "gse", "cohort", "match_status"}
    missing = required - set(matches)
    if missing:
        raise ValueError(f"Match table lacks columns: {sorted(missing)}")
    matched = matches.loc[matches["match_status"].eq("matched")].copy()
    frames = []
    for cohort in config["cohorts"]:
        available = matched.loc[matched["cohort"].eq(cohort)].copy()
        chosen = balanced_select(
            available, int(config["target_pairs_per_cohort"]),
            int(config["max_final_pairs_per_gse"]), int(config["random_seed"]),
        )
        print(f"{cohort}: matched={available['gsm'].nunique():,}, final={len(chosen):,}")
        frames.append(chosen)
    result = pd.concat(frames, ignore_index=True)
    if result["gsm"].duplicated().any():
        raise ValueError("Final paired GSMs are not unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["candidates", "finalize"], default="candidates")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archs4-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--matches", type=Path, help="Required for --mode finalize")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "candidates":
        result = build_candidates(config, args.manifest, args.archs4_h5)
    else:
        if args.matches is None:
            raise SystemExit("--matches is required with --mode finalize")
        result = finalize_pairs(config, args.matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    result.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Saved {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
