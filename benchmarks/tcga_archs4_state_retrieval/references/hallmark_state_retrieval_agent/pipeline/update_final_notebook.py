#!/usr/bin/env python3
"""Append the benchmark-closure section and execute the report notebook."""
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

root = Path(__file__).resolve().parents[1]
path = root / "hallmark_state_retrieval.ipynb"
nb = nbformat.read(path, as_version=4)
tag = "## Final benchmark closure"
nb.cells = [c for c in nb.cells if tag not in c.get("source", "") and "final_leakage_audit" not in c.get("source", "")]
nb.cells.extend([
    nbformat.v4.new_markdown_cell(tag + "\n\nBenchmark #9 is complete. The final validation uses frozen outputs only and distinguishes global Bridge-512 retrieval, Hallmark-50 functional-state retrieval, and species-conditioned search."),
    nbformat.v4.new_code_cell(
        "final_leakage_audit = pd.read_csv(ROOT/'results/luad_focused_audit/luad_centroid_top100_leakage_audit.csv')\n"
        "display(final_leakage_audit.groupby(['method','audit_status']).size().rename('hits').reset_index())\n"
        "display(pd.read_csv(ROOT/'results/luad_focused_audit/luad_rank_overlap.csv'))\n"
        "display(pd.read_csv(ROOT/'results/luad_focused_audit/luad_selection_similarity_distributions.csv'))"
    ),
    nbformat.v4.new_code_cell(
        "display(Image(filename=str(ROOT/'results/tcga_pan_cancer_positive_control/figures/cancer_similarity_matrices_final.png')))\n"
        "display(Image(filename=str(ROOT/'results/luad_focused_audit/luad_centroid_bridge_vs_hallmark_scatter.png')))\n"
        "display(Image(filename=str(ROOT/'results/luad_focused_audit/luad_hallmark_terms_bridge512_vs_hallmark50.png')))"
    ),
    nbformat.v4.new_markdown_cell(
        "**Final conclusion.** Bridge-512 is the primary global transcriptomic search; Hallmark-50 is a complementary functional-state view. TCGA validates broad functional geometry, while individual retrieval and OSDR preserve important negative results. Combined human+mouse search is species-dominated, so within-species and explicit forced cross-species searches must remain separate. Similarity does not establish mechanism, disease equivalence, or causality."
    ),
])
ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(root)}})
nbformat.write(nb, path)
print(f"executed {path}")
