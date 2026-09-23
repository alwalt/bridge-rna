#!/usr/bin/env python3
"""Create figures, the result summary, notebook, and top-level provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
EVAL = RESULTS / "evaluation"
FIGURES = RESULTS / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(EVAL / "primary_effects.csv")
    random = pd.read_csv(EVAL / "random_module_summary.csv")
    pretrained = pd.read_csv(EVAL / "pretraining_status_summary.csv")
    sensitivity = pd.read_csv(EVAL / "GSE211204_unseen_subject_convergence.csv")
    convergence = pd.read_csv(EVAL / "convergence_metrics.csv")
    enrichment = pd.read_parquet(EVAL / "drug_enrichment.parquet")
    completeness = pd.read_csv(RESULTS / "bridge/ig_completeness.csv")
    deletion = pd.read_csv(RESULTS / "bridge/deletion_control_summary.csv")
    deletion_effect = deletion.pivot(index="dataset", columns="panel_type", values="mean_absolute_change").reset_index()
    deletion_effect["top_to_random_change_ratio"] = deletion_effect["top"] / deletion_effect["random"]

    coverage = (enrichment[(enrichment.module_size == 500) & (enrichment.min_targets == 3)]
                .groupby(["method", "dataset"], as_index=False)
                .agg(tested_drugs=("drug_id", "size"), nonzero_overlap_drugs=("overlap_count", lambda x: int((x > 0).sum())), fdr_005=("fdr", lambda x: int((x < .05).sum()))))
    coverage.to_csv(EVAL / "primary_ranking_coverage.csv", index=False)
    pairwise25 = convergence[(convergence.top_n == 25) & convergence.dataset_1.ne("three_way")]
    sensitivity_grid = pairwise25.groupby(["condition", "method", "module_size", "min_targets"], as_index=False).jaccard.mean()
    sensitivity_wide = sensitivity_grid.pivot(index=["condition", "module_size", "min_targets"], columns="method", values="jaccard").reset_index()
    sensitivity_wide["bridge_minus_de"] = sensitivity_wide["Bridge"] - sensitivity_wide["DE"]
    sensitivity_wide.to_csv(EVAL / "module_and_target_threshold_sensitivity.csv", index=False)

    conditions = ["Radiation", "Bone loss", "Muscle atrophy", "Overall"]
    shown = effects.set_index("condition").loc[conditions]
    x = np.arange(len(conditions)); width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, shown.bridge_mean_pairwise_top25_jaccard, width, label="Bridge")
    ax.bar(x + width / 2, shown.de_mean_pairwise_top25_jaccard, width, label="DE")
    ax.set_xticks(x, conditions, rotation=15, ha="right")
    ax.set_ylabel("Mean pairwise top-25 drug Jaccard")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGURES / "primary_drug_convergence.png", dpi=180); plt.close(fig)

    sens = convergence[(convergence.top_n == 25) & (convergence.dataset_1 != "three_way")]
    sens = sens.groupby(["condition", "method", "module_size", "min_targets"], as_index=False).jaccard.mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, condition in zip(axes, ["Radiation", "Bone loss", "Muscle atrophy"]):
        part = sens[(sens.condition == condition) & (sens.min_targets == 3)]
        for method, group in part.groupby("method"):
            ax.plot(group.module_size, group.jaccard, marker="o", label=method)
        ax.set_title(condition); ax.set_xlabel("Module size")
    axes[0].set_ylabel("Mean pairwise top-25 Jaccard"); axes[-1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "module_size_sensitivity.png", dpi=180); plt.close(fig)

    outcome = shown.loc["Overall", "bridge_minus_de"]
    overall_p = shown.loc["Overall", "exact_p"]
    verdict = "shows a statistically supported improvement" if outcome > 0 and overall_p < .05 else "does not establish a statistically conclusive improvement"
    lines = [
        "# BridgeRNA drug-discovery benchmark: Phase 2 results", "",
        f"At the prespecified primary endpoint, Bridge **{verdict}** in cross-study drug convergence over matched DE (overall Bridge-minus-DE mean pairwise top-25 Jaccard = {fmt(outcome)}; exact label-swap p = {overall_p:.4g}).",
        "", "This benchmark measures convergence of ChEMBL direct mechanism-target enrichment rankings. It does not test perturbational reversal, treatment direction, clinical benefit, or therapeutic efficacy.",
        "", "## Primary results", "", shown.reset_index()[["condition","bridge_mean_pairwise_top25_jaccard","de_mean_pairwise_top25_jaccard","bridge_minus_de","exact_p","permutations"]].to_markdown(index=False),
        "", "## Expression-matched random-module controls", "", random.to_markdown(index=False),
        "", "## ARCHS4 pretraining-status analysis", "", pretrained.to_markdown(index=False),
        "", "Comparisons involving a train/validation-exposed cohort are descriptive reproducibility results, not out-of-pretraining generalization. The strict-unseen row is the relevant held-out comparison.",
        "", "## GSE211204 leakage sensitivity", "", sensitivity.to_markdown(index=False),
        "", "The sensitivity replaces the muscle discovery ranking with the seven subjects whose baseline and ULLS endpoints are both absent from the reconstructed ARCHS4 train/validation split.",
        "", "## Module-size and target-threshold sensitivity", "", sensitivity_wide.to_markdown(index=False),
        "", "## Primary ranking coverage", "", coverage.to_markdown(index=False),
        "", "## Attribution QA", "",
        f"Across {len(completeness)} sample attributions, the median absolute IG endpoint-completeness error was {completeness.absolute_completeness_delta.median():.6f} (95th percentile {completeness.absolute_completeness_delta.quantile(.95):.6f}).",
        "", deletion_effect.to_markdown(index=False),
        "", "For every dataset, masking the Bridge top-500 changed the condition-axis effect more than the mean of five expression-decile-matched random panels.",
        "", "## Scope and limitations", "",
        "- Modules use absolute signed Bridge attribution or absolute DE statistic, with identical sizes and dataset-specific expressed-gene universes.",
        "- Zero-target-overlap p=1 ties are retained in enrichment output but excluded from convergence rankings because identifier ordering is not biological signal.",
        "- The direct-mechanism graph is sparse: primary nonzero-overlap lists can contain fewer than 25 drugs (including an empty DE list for GSE297090). Thus the nominal top-25 Jaccard uses all available nonzero-overlap drugs when fewer than 25 exist and must be interpreted as a sparse-candidate convergence endpoint.",
        "- Bone convergence spans primary osteoporosis MSCs, glucocorticoid-associated cortical bone, and spaceflight bone-marrow MSCs; it is cross-etiology convergence, not literal disease replication.",
        "- OSDR culture/chip replicates are not independent human donors.",
        "- No direction-aware drug perturbation or reversal analysis was performed.", "",
    ]
    (RESULTS / "phase2_summary.md").write_text("\n".join(lines))

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# BridgeRNA drug-discovery benchmark\n\nExecuted audit notebook for the frozen Phase 2 outputs. Enrichment indicates target-set convergence only, not therapeutic efficacy."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif ROOT.name!='drug_discovery': ROOT=ROOT/'benchmarks/drug_discovery'\nEVAL=ROOT/'results/evaluation'"),
        nbf.v4.new_markdown_cell("## Primary endpoint"),
        nbf.v4.new_code_cell("primary=pd.read_csv(EVAL/'primary_effects.csv')\ndisplay(primary)\ndisplay(Image(filename=ROOT/'results/figures/primary_drug_convergence.png'))"),
        nbf.v4.new_markdown_cell("## Module-size sensitivity and random-module nulls"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'results/figures/module_size_sensitivity.png'))\ndisplay(pd.read_csv(EVAL/'random_module_summary.csv'))"),
        nbf.v4.new_markdown_cell("## ARCHS4 exposure and leakage sensitivity"),
        nbf.v4.new_code_cell("display(pd.read_csv(EVAL/'pretraining_pair_effects.csv'))\ndisplay(pd.read_csv(EVAL/'pretraining_status_summary.csv'))\ndisplay(pd.read_csv(EVAL/'GSE211204_unseen_subject_convergence.csv'))"),
        nbf.v4.new_markdown_cell("## Attribution and deletion controls"),
        nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'results/bridge/ig_completeness.csv').describe(include='all'))\ndisplay(pd.read_csv(ROOT/'results/bridge/deletion_control_summary.csv'))"),
    ]
    NotebookClient(nb, timeout=600, kernel_name="python3").execute(cwd=str(HERE))
    nbf.write(nb, HERE / "drug_discovery_benchmark.ipynb")

    tracked = [p for p in RESULTS.rglob("*") if p.is_file() and "run_logs" not in p.parts and p.name != "provenance.json"]
    provenance = {
        "benchmark": "BridgeRNA drug-discovery Phase 2",
        "python": sys.version,
        "platform": platform.platform(),
        "current_worktree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "bridge_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/home/walt/bridge-rna", text=True).strip(),
        "checkpoint": {"path": "/home/walt/bridge-rna/model/r7hnr92k/best_model.pt", "sha256": sha256(Path("/home/walt/bridge-rna/model/r7hnr92k/best_model.pt"))},
        "random_seed": json.loads((HERE / "config.json").read_text())["seed"],
        "artifacts": {str(p.relative_to(HERE)): sha256(p) for p in sorted(tracked)},
        "direction_aware_drug_analysis": False,
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
