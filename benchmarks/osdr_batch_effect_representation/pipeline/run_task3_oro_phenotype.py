#!/usr/bin/env python3
"""Relate fixed Task 3 BridgeRNA response modes to animal-level liver ORO."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
OUT = RESULTS / "task3_oro_phenotype"
WORK = HERE / "work" / "task3_oro"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
API = "https://visualization.osdr.nasa.gov/biodata/api/v2"
STUDIES = [47, 48, 137]
COLORS = {"GC": "#4C78A8", "FLT": "#E45756"}
MODE_COLORS = {"Mode 1": "#3B82B8", "Mode 2": "#D26A4B"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def obtain_oro_files() -> dict[str, dict[str, str]]:
    provenance = {}
    for study in STUDIES:
        response = requests.get(f"{API}/dataset/OSD-{study}/files/", timeout=120)
        response.raise_for_status()
        files = response.json()[f"OSD-{study}"]["files"]
        candidates = [(name, item["URL"]) for name, item in files.items()
                      if name.endswith("Histology_TRANSFORMED.csv")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one transformed histology file for OSD-{study}: {candidates}")
        name, url = candidates[0]
        path = WORK / name
        if not path.exists():
            payload = requests.get(url, timeout=120)
            payload.raise_for_status()
            path.write_bytes(payload.content)
        provenance[f"OSD-{study}"] = {"file": str(path.relative_to(HERE)), "url": url, "sha256": sha256(path)}
    return provenance


def load_oro(provenance: dict[str, dict[str, str]]) -> pd.DataFrame:
    frames = []
    for osd, item in provenance.items():
        frame = pd.read_csv(HERE / item["file"])
        frame.columns = frame.columns.str.strip()
        frame["sample_id"] = frame["Sample name"].astype(str).str.strip()
        frame["OSD"] = osd
        frame["oro_positivity_percent"] = pd.to_numeric(frame["ORO Positivity (%)"], errors="coerce")
        frame["oro_condition"] = frame["Spaceflight"].map({"Flight": "FLT", "Ground": "GC"})
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out.sample_id.duplicated().any():
        raise RuntimeError("Duplicate ORO sample identifiers")
    return out


def hedges_g(flt: np.ndarray, gc: np.ndarray) -> float:
    n1, n0 = len(flt), len(gc)
    if n1 < 2 or n0 < 2: return np.nan
    pooled = np.sqrt(((n1 - 1) * flt.var(ddof=1) + (n0 - 1) * gc.var(ddof=1)) / (n1 + n0 - 2))
    if not np.isfinite(pooled) or pooled == 0: return np.nan
    d = (flt.mean() - gc.mean()) / pooled
    correction = 1 - 3 / (4 * (n1 + n0) - 9)
    return float(correction * d)


def exact_permutation_difference(values: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Enumerate all 4/3 mode allocations; descriptive because contrasts cluster by mission."""
    from itertools import combinations
    values = values.dropna(subset=[metric]).reset_index(drop=True)
    observed = (values.loc[values["mode"].eq("Mode 1"), metric].mean() -
                values.loc[values["mode"].eq("Mode 2"), metric].mean())
    x = values[metric].to_numpy(float)
    null = []
    for idx in combinations(range(len(x)), int(values["mode"].eq("Mode 1").sum())):
        mask = np.zeros(len(x), dtype=bool); mask[list(idx)] = True
        null.append(x[mask].mean() - x[~mask].mean())
    null = np.asarray(null)
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return float(observed), p


def build_tables(oro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    membership = pd.read_csv(RESULTS / "task3b_contrast_sample_membership.csv")
    contrast = pd.read_csv(RESULTS / "task3b_contrast_summary.csv")
    modes = pd.read_csv(RESULTS / "task3c_cluster_assignments.csv")[["contrast_id", "geometry_cluster"]]
    eligible = membership[membership.OSD.isin([f"OSD-{x}" for x in STUDIES])].copy()
    matched = eligible.merge(oro[["OSD", "sample_id", "oro_condition", "oro_positivity_percent",
                                  "Space mission", "Dissection"]],
                             on=["OSD", "sample_id"], how="left", validate="one_to_one", indicator=True)
    matched = matched.merge(modes, on="contrast_id", validate="many_to_one")
    matched["mode"] = matched.geometry_cluster.map({1: "Mode 1", 2: "Mode 2"})
    matched["condition_match"] = matched.condition.eq(matched.oro_condition)
    matched["matching_confidence"] = np.where(
        matched._merge.eq("both") & matched.condition_match & matched.oro_positivity_percent.notna(),
        "exact sample/animal identifier; condition concordant", "unresolved")
    if not matched.matching_confidence.str.startswith("exact").all():
        bad = matched[~matched.matching_confidence.str.startswith("exact")]
        raise RuntimeError(f"Unresolved ORO matches:\n{bad.to_string(index=False)}")
    matched.drop(columns=["_merge"], inplace=True)

    rows = []
    meta = contrast.set_index("contrast_id")
    for cid, frame in matched.groupby("contrast_id", sort=False):
        flt = frame.loc[frame.condition.eq("FLT"), "oro_positivity_percent"].to_numpy(float)
        gc = frame.loc[frame.condition.eq("GC"), "oro_positivity_percent"].to_numpy(float)
        row = meta.loc[cid]
        delta = float(flt.mean() - gc.mean())
        rows.append({
            "OSD": row.OSD, "mission": row.mission, "contrast_id": cid,
            "BridgeRNA_mode": frame["mode"].iloc[0], "n_FLT": len(flt), "n_GC": len(gc),
            "mean_ORO_FLT": float(flt.mean()), "mean_ORO_GC": float(gc.mean()), "delta_ORO": delta,
            "percent_change_vs_GC": float(100 * delta / gc.mean()),
            "log_response_ratio": float(np.log(flt.mean() / gc.mean())),
            "hedges_g": hedges_g(flt, gc),
            "matching_confidence": "all animals exact by OSDR sample name; FLT/GC concordant",
            "dissection": " | ".join(sorted(frame.Dissection.dropna().unique())),
        })
    effects = pd.DataFrame(rows).sort_values(["BridgeRNA_mode", "OSD", "contrast_id"])

    descriptive = []
    for mode, frame in effects.groupby("BridgeRNA_mode"):
        for metric in ["delta_ORO", "percent_change_vs_GC", "hedges_g"]:
            descriptive.append({"mode": mode, "metric": metric, "contrasts": len(frame),
                                "mean": frame[metric].mean(), "median": frame[metric].median(),
                                "min": frame[metric].min(), "max": frame[metric].max(),
                                "positive": int(frame[metric].gt(0).sum()), "negative": int(frame[metric].lt(0).sum())})
    for metric in ["delta_ORO", "percent_change_vs_GC", "hedges_g"]:
        diff, p = exact_permutation_difference(effects.rename(columns={"BridgeRNA_mode": "mode"}), metric)
        descriptive.append({"mode": "Mode 1 minus Mode 2", "metric": metric, "contrasts": len(effects),
                            "mean": diff, "median": np.nan, "min": np.nan, "max": np.nan,
                            "positive": np.nan, "negative": np.nan, "exact_label_permutation_p": p})
    return matched, effects, pd.DataFrame(descriptive)


def plot_individual(matched: pd.DataFrame, effects: pd.DataFrame) -> None:
    order = effects.sort_values(["BridgeRNA_mode", "OSD", "contrast_id"]).contrast_id.tolist()
    labels = [x.split("__")[0] + "\n" + x.split("__")[1].replace("OSD-", "OSD-") +
              "\n" + effects.set_index("contrast_id").loc[x, "BridgeRNA_mode"] for x in order]
    rng = np.random.default_rng(1200132)
    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    for i, cid in enumerate(order):
        z = matched[matched.contrast_id.eq(cid)]
        for j, condition in enumerate(["GC", "FLT"]):
            vals = z.loc[z.condition.eq(condition), "oro_positivity_percent"].to_numpy()
            xpos = i + (j - .5) * .28
            jitter = rng.uniform(-.045, .045, len(vals))
            ax.scatter(xpos + jitter, vals, s=38, color=COLORS[condition], alpha=.82,
                       edgecolor="white", linewidth=.4, zorder=3)
            ax.hlines(vals.mean(), xpos-.10, xpos+.10, color="black", lw=2.1, zorder=4)
        ax.plot([i-.14, i+.14], [z.loc[z.condition.eq("GC"), "oro_positivity_percent"].mean(),
                                 z.loc[z.condition.eq("FLT"), "oro_positivity_percent"].mean()],
                color="#777777", lw=1, zorder=2)
    ax.set(xticks=np.arange(len(order)), xticklabels=labels, ylabel="Oil Red O positivity (%)",
           title="Animal-level hepatic lipid staining in exact Task 3 FLT/GC contrasts")
    ax.grid(axis="y", alpha=.2)
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=c, label=k) for k,c in COLORS.items()],
              frameon=False, ncol=2)
    fig.savefig(OUT / "oro_individual_flt_vs_gc.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "oro_individual_flt_vs_gc.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_effects(effects: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for ax, metric, ylabel in zip(axes, ["delta_ORO", "hedges_g"],
                                  ["ΔORO (FLT − GC; percentage points)", "Hedges g (FLT − GC)"]):
        for mode, xpos in [("Mode 1", 0), ("Mode 2", 1)]:
            z = effects[effects.BridgeRNA_mode.eq(mode)]
            jitter = np.linspace(-.10, .10, len(z)) if len(z)>1 else np.array([0.])
            ax.scatter(xpos+jitter, z[metric], s=72, color=MODE_COLORS[mode], edgecolor="white", zorder=3)
            for x, (_, row) in zip(xpos+jitter, z.iterrows()):
                ax.annotate(f"{row.OSD}\n{row.contrast_id.split('__')[0]}", (x,row[metric]),
                            xytext=(0,6), textcoords="offset points", ha="center", fontsize=7)
            ax.hlines(z[metric].mean(), xpos-.19, xpos+.19, color="black", lw=2)
        ax.axhline(0, color="#444444", ls="--", lw=1)
        ax.set(xticks=[0,1], xticklabels=["Mode 1","Mode 2"], ylabel=ylabel)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Oil Red O response grouped by fixed BridgeRNA response mode")
    fig.savefig(OUT / "oro_effect_by_bridgerna_mode.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "oro_effect_by_bridgerna_mode.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    provenance = obtain_oro_files()
    oro = load_oro(provenance)
    matched, effects, descriptive = build_tables(oro)
    matched.to_csv(OUT / "oro_animal_matches.csv", index=False)
    effects.to_csv(OUT / "oro_contrast_effects.csv", index=False)
    descriptive.to_csv(OUT / "oro_mode_descriptive_summary.csv", index=False)
    within_mission = (effects.groupby(["OSD", "mission", "BridgeRNA_mode"], as_index=False)
                      .agg(contrasts=("contrast_id", "nunique"),
                           mean_delta_ORO=("delta_ORO", "mean"),
                           mean_percent_change=("percent_change_vs_GC", "mean"),
                           mean_hedges_g=("hedges_g", "mean"),
                           contrast_ids=("contrast_id", lambda s: " | ".join(s))))
    within_mission.to_csv(OUT / "oro_within_mission_summary.csv", index=False)
    pd.DataFrame([
        {"OSD": "OSD-168", "status": "excluded_from_ORO_biological_analysis",
         "reason": "RR-1/RR-3 RNA aliquots reused for ERCC/resequencing; not an independent animal cohort",
         "independent_contrasts": 0}
    ]).to_csv(OUT / "oro_excluded_resequencing_material.csv", index=False)
    plot_individual(matched, effects)
    plot_effects(effects)
    provenance.update({"analysis": {
        "effect_direction": "FLT minus GC", "standardized_effect": "Hedges g using pooled within-contrast SD",
        "relative_effects": ["percent change from GC mean", "natural-log response ratio"],
        "mode_assignments": "unchanged task3c_cluster_assignments.csv",
        "OSD-168": "excluded as non-independent ERCC/resequencing material",
        "inference_or_training": "none",
    }})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Matched {len(matched)}/{len(matched)} Task 3 animals across {len(effects)} biological contrasts")
    print(effects.to_string(index=False))
    print("\nMode summary:\n", descriptive.to_string(index=False))


if __name__ == "__main__":
    main()
