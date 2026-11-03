#!/usr/bin/env python3
"""Build and execute the cached-evidence cross-species MTLE notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd


EVIDENCE = {
    "primary_direction_metrics.csv": "results/cross_species_direction_metrics.csv",
    "primary_manifest.csv": "results/final_primary_manifest.csv",
    "mouse_hallmark_recovery_values.csv": "results/hallmark_head/mouse_hallmark_recovery_values.csv",
    "mouse_hallmark_recovery_metrics.csv": "results/hallmark_head/mouse_hallmark_recovery_metrics.csv",
    "phase_resolved_hallmarks.csv": "results/hallmark_head/epilepsy_energy_three_phase_hallmark_states.csv",
    "strict_hallmark_correspondence.csv": "results/hallmark_head/strict_hallmark_correspondence.csv",
    "kim_validation_counts.csv": "results/kim_external_validation/kim_validation_counts.csv",
    "kim_independent_replication.csv": "results/kim_external_validation/independent_bridge_replication.csv",
    "network_method_summary.csv": "results/network_final/network_method_summary.csv",
    "network_functional_validation.csv": "results/network_final/edge_functional_validation.csv",
    "network_rewiring_metrics.csv": "results/network_final/rewiring_conservation_metrics.csv",
}

FIGURES = {
    "mouse_hallmark_recovery_summary.png": "figures/hallmark_head/mouse_hallmark_recovery_summary.png",
    "mouse_hallmark_recovery_summary.pdf": "figures/hallmark_head/mouse_hallmark_recovery_summary.pdf",
    "epilepsy_energy_three_phase_hallmark_states.png": "figures/hallmark_head/epilepsy_energy_three_phase_hallmark_states.png",
    "epilepsy_energy_three_phase_hallmark_states.pdf": "figures/hallmark_head/epilepsy_energy_three_phase_hallmark_states.pdf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
    }
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        """# Cross-species MTLE: pathway transfer and functional organization

This study tests whether frozen BridgeRNA representations identify corresponding transcriptional programs between human mesial temporal lobe epilepsy (MTLE) and phase-specific mouse kainate epilepsy. The strict evaluation uses studies absent from Bridge pretraining, valid natural `log1p(TPM)` inputs, no fitted cross-species alignment, and matched raw-expression and PCA baselines.

The report distinguishes three questions: cross-species sample-direction alignment, zero-shot recovery of mouse pathway state by a human-trained Hallmark head, and functional organization of contextual gene networks. All displayed results are recomputed from compact frozen tables; no model inference is performed."""))
    cells.append(nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

PACKAGE = Path.cwd()
if not (PACKAGE / 'cached_evidence').exists():
    candidate = PACKAGE / 'notebooks' / 'cross_species_mtle'
    if candidate.exists():
        PACKAGE = candidate
CACHE = PACKAGE / 'cached_evidence'
FIGURES = PACKAGE / 'figures'
assert CACHE.exists() and FIGURES.exists()
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
print(f'Cached evidence: {CACHE}')"""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Data and preprocessing

- Human GSE256068/PRJNA1077986: 58 MTLE cases and 10 controls, whole hippocampal tissue, predominantly MTLE with hippocampal sclerosis.
- Mouse GSE319769/PRJNA1425071: intra-amygdala kainate and matched vehicle/control, with 6 animals per condition at Day 3 (acute), Day 7 (intermediate), and Day 14 (chronic). Phases use independent animals.
- All 104 samples are study-unseen to Bridge pretraining.
- Human inputs use deposited NCBI TPM and cover 15,141/15,165 canonical ortholog genes; mouse TPM was calculated from HTSeq counts with Ensembl-112 exon-union lengths and covers all 15,165 genes.
- Bridge receives natural `log1p(TPM)` in canonical vocabulary order. No cross-species alignment or domain adaptation is fitted."""))
    cells.append(nbf.v4.new_code_cell(
        """manifest = pd.read_csv(CACHE / 'primary_manifest.csv')
assert len(manifest) == 104
assert manifest['gsm'].nunique() == 104
cohort = manifest.groupby(['species', 'phenotype', 'phase'], dropna=False).size().rename('N').reset_index()
display(cohort)
print('Unique biological samples:', len(manifest))"""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Frozen sample representations and cross-species disease directions

For human samples, $\\Delta_H=\\mathrm{mean}(z_{MTLE})-\\mathrm{mean}(z_{control})$. For each mouse phase, $\\Delta_{M,k}=\\mathrm{mean}(z_{kainate,k})-\\mathrm{mean}(z_{vehicle,k})$. The primary statistic is cosine similarity. Uncertainty uses 5,000 biological-sample bootstraps; inference uses 10,000 within-study/phase phenotype-label permutations."""))
    cells.append(nbf.v4.new_code_cell(
        """directions = pd.read_csv(CACHE / 'primary_direction_metrics.csv')
order = ['Acute', 'Intermediate', 'Chronic']
bridge = directions.query("method == 'Bridge'").set_index('mouse_phase').loc[order].reset_index()
expected = np.array([0.532620, 0.275335, 0.235620])
assert np.allclose(bridge['observed_cosine'], expected, atol=5e-7)
display(bridge[['mouse_phase','observed_cosine','bootstrap_ci_low','bootstrap_ci_high',
                'permutation_p_one_sided','standardized_vs_null']].round(4))

fig, ax = plt.subplots(figsize=(7.2, 4.3))
methods = ['Bridge', 'Raw log1p(TPM)', 'PCA-50']
colors = {'Bridge':'#ca4b3f','Raw log1p(TPM)':'#777777','PCA-50':'#4c78a8'}
x = np.arange(3); width = 0.24
for j, method in enumerate(methods):
    q = directions.query('method == @method').set_index('mouse_phase').loc[order]
    y = q['observed_cosine'].to_numpy()
    lo = y - q['bootstrap_ci_low'].to_numpy(); hi = q['bootstrap_ci_high'].to_numpy() - y
    ax.errorbar(x + (j-1)*width, y, yerr=np.vstack([lo,hi]), fmt='o', ms=7,
                color=colors[method], capsize=3, label=method)
ax.axhline(0, color='#444', lw=.8)
ax.set(xticks=x, xticklabels=order, ylabel='Human–mouse disease-direction cosine',
       title='Strict study-unseen cross-species geometry')
ax.legend(frameon=False); ax.grid(axis='y', color='#e8e8e8')
fig.tight_layout()
fig.savefig(FIGURES / 'primary_direction_comparison.png', dpi=300)
fig.savefig(FIGURES / 'primary_direction_comparison.pdf')
plt.show()"""))
    cells.append(nbf.v4.new_markdown_cell(
        """Bridge produced the largest acute point estimate (0.5326 versus approximately 0.0975 for raw expression and PCA), but no phase exceeded its own phenotype-label null. The positive bootstrap interval quantifies sampling stability conditional on the observed labels and does not supersede the permutation test."""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Zero-shot Hallmark recovery in mouse epilepsy

A frozen 512→50 Ridge head trained on human GTEx samples predicts ssGSEA-like Hallmark activity from Bridge sample embeddings. Applying it unchanged to mouse samples tests zero-shot functional-state transfer. Disease effects are kainate minus phase-matched control. Because the head was trained against expression-derived pathway targets, agreement with conventional GSEA is transfer/recovery evidence rather than an independent assay."""))
    cells.append(nbf.v4.new_code_cell(
        """recovery = pd.read_csv(CACHE / 'mouse_hallmark_recovery_values.csv')
metrics = pd.read_csv(CACHE / 'mouse_hallmark_recovery_metrics.csv')
assert len(recovery) == 150
assert recovery.groupby('phase')['hallmark'].nunique().eq(50).all()
mouse = metrics.query("comparison == 'Mouse GSEA ↔ Bridge'").set_index('phase').loc[order]
display(mouse[['pearson','spearman','same_sign_fraction']].round(4))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True, sharey=True)
limx = 1.08 * recovery['mouse_gsea_nes'].abs().max()
limy = 1.08 * recovery['mouse_head_delta'].abs().max()
for ax, phase in zip(axes, order):
    q = recovery.query('phase == @phase')
    same = np.sign(q.mouse_gsea_nes) == np.sign(q.mouse_head_delta)
    ax.scatter(q.loc[same,'mouse_gsea_nes'], q.loc[same,'mouse_head_delta'], s=30,
               color='#27866a', edgecolor='white', lw=.4, label='Same sign')
    ax.scatter(q.loc[~same,'mouse_gsea_nes'], q.loc[~same,'mouse_head_delta'], s=30,
               facecolor='none', edgecolor='#c64f45', lw=1, label='Opposite sign')
    ax.axhline(0,color='#555',lw=.7); ax.axvline(0,color='#555',lw=.7)
    m = mouse.loc[phase]
    ax.text(.04,.96,f"ρ={m.spearman:.2f}\\nsame sign={100*m.same_sign_fraction:.0f}%",
            transform=ax.transAxes,va='top')
    ax.set(title=phase, xlim=(-limx,limx), ylim=(-limy,limy), xlabel='Mouse GSEA NES')
    ax.grid(color='#eeeeee',lw=.5)
axes[0].set_ylabel('Mouse Bridge Hallmark Δ')
axes[-1].legend(frameon=False, loc='lower right')
fig.suptitle('Zero-shot recovery of the mouse Hallmark landscape', fontweight='bold')
fig.tight_layout()
fig.savefig(FIGURES / 'mouse_hallmark_recovery_scatter.png', dpi=300)
fig.savefig(FIGURES / 'mouse_hallmark_recovery_scatter.pdf')
plt.show()"""))
    cells.append(nbf.v4.new_code_cell(
        """comparisons = ['Mouse GSEA ↔ Bridge', 'Human ↔ mouse GSEA', 'Human ↔ mouse Bridge']
colors = ['#27866a', '#7f858e', '#ca5145']
x = np.arange(3); width = 0.23
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for i, (comparison, color) in enumerate(zip(comparisons, colors)):
    q = metrics.query('comparison == @comparison').set_index('phase').loc[order]
    ax.bar(x + (i-1)*width, q['spearman'], width, color=color, label=comparison)
ax.axhline(0, color='#333333', lw=.9)
ax.set(xticks=x, xticklabels=order, ylim=(-.75,.82),
       ylabel='Spearman correlation across 50 Hallmarks',
       title='Recovery of mouse biology does not imply human–mouse conservation')
ax.grid(axis='y', color='#e8e8e8', lw=.6)
ax.legend(frameon=False, loc='lower center')
fig.tight_layout()
fig.savefig(FIGURES / 'hallmark_recovery_vs_conservation.png', dpi=300)
fig.savefig(FIGURES / 'hallmark_recovery_vs_conservation.pdf')
plt.show()"""))
    cells.append(nbf.v4.new_markdown_cell(
        """Across all 50 Hallmarks, mouse GSEA versus mouse Bridge-head Spearman correlation was 0.6596 acute, 0.3466 intermediate, and 0.5673 chronic; sign agreement was 78%, 92%, and 86%. This demonstrates functional transfer to mouse without mouse retraining. The comparison panel makes the boundary explicit: both conventional GSEA and the Bridge head show negative human–mouse rank correlation in this strict cohort. Mouse functional recovery therefore does not establish global human–mouse conservation."""))
    cells.append(nbf.v4.new_code_cell(
        """strict_head = pd.read_csv(CACHE / 'strict_hallmark_correspondence.csv')
head_cos = strict_head.query("statistic == 'cosine'").set_index('phase').loc[order]
display(head_cos[['observed','bootstrap_ci_low','bootstrap_ci_high',
                  'permutation_p_one_sided','family_q']].round(4))"""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Phase-resolved injury, energy, and signaling programs

The focused pathway panel uses 22 prespecified epilepsy-relevant Hallmarks. Filled Bridge markers indicate biological-sample bootstrap intervals excluding zero; filled GSEA markers indicate BH q<0.05. These support criteria are not calibrated equivalents. Within each biological block, pathways are ordered by their observed Bridge phase swing, so the figure is descriptive rather than a confirmatory pathway ranking."""))
    cells.append(nbf.v4.new_code_cell(
        """phase_table = pd.read_csv(CACHE / 'phase_resolved_hallmarks.csv')
assert len(phase_table) == 22
display(phase_table[['biological_group','hallmark','bridge_acute','bridge_intermediate',
                     'bridge_chronic','gsea_acute','gsea_intermediate','gsea_chronic']].round(3))
display(Image(filename=str(FIGURES / 'epilepsy_energy_three_phase_hallmark_states.png'), width=1100))"""))
    cells.append(nbf.v4.new_markdown_cell(
        """The head captured an acute injury/stress state involving unfolded-protein response, p53, interferons, apoptosis, TNF/NF-kB, IL-6/JAK/STAT3, and hypoxia, with several responses weakening by chronic disease. It also resolved dynamic energy/redox programs—acute decreases followed by intermediate/chronic increases in peroxisome, fatty-acid metabolism, and oxidative phosphorylation—and growth/injury programs including intermediate-peaking mTORC1 and attenuating TGF-beta. These phases represent different animals and are not longitudinal within-animal trajectories."""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Contextual organization and literature concordance

The final pre-mapped comparison with Kim et al. supported neuroinflammation through both contextual embeddings and acute IG, with TNF signaling receiving contextual support. The exact gliogenesis terms did not pass the original correction procedures. Bridge contextual networks were assessed against density-matched Pearson and Spearman coexpression; annotation co-membership measures functional organization, not physical interaction."""))
    cells.append(nbf.v4.new_code_cell(
        """kim = pd.read_csv(CACHE / 'kim_validation_counts.csv')
network = pd.read_csv(CACHE / 'network_method_summary.csv')
functional = pd.read_csv(CACHE / 'network_functional_validation.csv')
display(kim)
display(network[['method','rewiring_cosine_min','rewiring_cosine_max',
                 'top1000_overlap_edges_total','top_rewiring_functional_tests_q05']].round(4))
bridge_top = functional[functional.edge_set.str.startswith('Bridge contextual L12') &
                        functional.edge_set.str.contains('top_rewiring_union')]
display(bridge_top[['edge_set','library','fold_enrichment','permutation_p','q']].round(4))"""))
    cells.append(nbf.v4.new_markdown_cell(
        """Bridge-specific top rewiring edges were enriched approximately 3.8–4.2-fold for GO, 4.6–5.0-fold for KEGG, and 14.1–14.6-fold for Hallmark co-membership after removing matched coexpression top edges. Signed human–mouse rewiring cosines nevertheless remained near zero, placing the positive result at the level of functional organization rather than conserved disease rewiring."""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Limitations

- Human disease is confounded with tissue procurement: cases are surgical samples and controls are older post-mortem samples with higher mitochondrial contribution.
- The primary design contains one study per species, predominantly MTLE with hippocampal sclerosis, and six animals per mouse condition.
- Tissue acquisition and cellular composition differ between species; no fitted alignment was used to remove those differences.
- Hallmark sets overlap, and the human-trained head targets expression-derived pathway scores; Hallmark agreement is not independent biological validation.
- Contextual gene inference used 250 permutations, limiting gene-level null resolution. Network annotation enrichment is not PPI validation.
- IG measures model reliance rather than causality. Genes and pathways are not independent biological replicates."""))
    cells.append(nbf.v4.new_markdown_cell(
        """## Conclusion

Frozen BridgeRNA representations produced suggestive acute-weighted cross-species geometry but did not pass the prespecified phenotype-label null. The clearest transferable result was higher-order functional organization: a human GTEx-trained Hallmark head recovered much of the mouse epilepsy pathway landscape zero-shot, identified coherent phase-resolved injury, metabolic, and growth-signaling states, and contextual networks recovered functional relationships beyond matched coexpression. Neuroinflammation was the strongest cross-species literature-concordant program. These findings support program-level transfer and organization while stopping short of conserved individual genes, conserved disease-network rewiring, or global human–mouse disease-state alignment."""))
    nb["cells"] = cells
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    research = args.research_root.resolve()
    repo = research.parents[1]
    out = repo / "notebooks" / "cross_species_mtle"
    cache = out / "cached_evidence"
    figures = out / "figures"
    cache.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    copied = []
    for name, rel in EVIDENCE.items():
        src = research / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst = cache / name
        shutil.copy2(src, dst)
        copied.append((src, dst))
    for name, rel in FIGURES.items():
        src = research / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst = figures / name
        shutil.copy2(src, dst)
        copied.append((src, dst))

    # Bounded verification of all headline values used by the report.
    direction = pd.read_csv(cache / "primary_direction_metrics.csv")
    bridge = direction.query("method == 'Bridge'").set_index("mouse_phase")
    assert np.allclose(bridge.loc[["Acute", "Intermediate", "Chronic"], "observed_cosine"],
                       [0.5326203516, 0.2753352041, 0.2356203798], atol=1e-9)
    manifest = pd.read_csv(cache / "primary_manifest.csv")
    assert len(manifest) == 104 and manifest["gsm"].nunique() == 104
    recovery = pd.read_csv(cache / "mouse_hallmark_recovery_values.csv")
    assert len(recovery) == 150 and recovery.groupby("phase")["hallmark"].nunique().eq(50).all()
    focused = pd.read_csv(cache / "phase_resolved_hallmarks.csv")
    assert len(focused) == 22

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_revision": git_value(repo, "rev-parse", "HEAD"),
        "source_branch": git_value(repo, "branch", "--show-current"),
        "research_benchmark": "benchmarks/cross_species_mtle",
        "approval_scope": "Supplementary translational stress test approved by explicit benchmark-notebook invocation.",
        "cohort": {"human_control": 10, "human_mtle": 58,
                   "mouse_per_phase_control": 6, "mouse_per_phase_kainate": 6,
                   "mouse_phases": ["Acute", "Intermediate", "Chronic"]},
        "preprocessing": "natural log1p(TPM), canonical 15,165-gene order; no fitted cross-species alignment",
        "checkpoint_sha256_recorded": "f3491bea...5541",
        "hallmark_probe_sha256_recorded": "f3f6c2dd...8931",
        "seeds_and_resampling": {"primary_bootstrap": 5000, "primary_label_permutations": 10000},
        "verification": ["104 unique GSMs", "9 primary method-phase rows",
                         "150 mouse Hallmark phase rows; 50 per phase", "22 focused Hallmarks",
                         "headline Bridge cosines matched cached source within 1e-9"],
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__},
        "artifacts": [{"package_path": str(dst.relative_to(out)),
                       "source_path": str(src.relative_to(repo)), "sha256": sha256(dst),
                       "bytes": dst.stat().st_size} for src, dst in copied],
        "limitations": ["One strict study per species", "Human procurement confounded with phenotype",
                        "Six animals per mouse condition", "Hallmark head targets expression-derived scores",
                        "Primary phenotype-label endpoint nonsignificant"],
    }
    (cache / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (cache / "requirements.txt").write_text(
        "numpy==2.2.6\npandas==2.2.3\nmatplotlib==3.10.7\n"
        "nbformat==5.10.4\nnbconvert==7.16.6\nipykernel\n"
    )

    nb_path = out / "cross_species_mtle.ipynb"
    nbf.write(notebook(), nb_path)
    if args.execute:
        subprocess.run([
            "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300", nb_path.name,
        ], cwd=out, check=True)
    print(f"Wrote {nb_path}")


if __name__ == "__main__":
    main()
