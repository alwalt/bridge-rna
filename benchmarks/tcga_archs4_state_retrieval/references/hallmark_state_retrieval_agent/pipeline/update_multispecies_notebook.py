#!/usr/bin/env python3
"""Append the frozen multispecies results section to the benchmark notebook."""
from pathlib import Path
import nbformat

root=Path(__file__).resolve().parents[1];path=root/'hallmark_state_retrieval.ipynb';nb=nbformat.read(path,as_version=4)
tag='## Human + mouse ARCHS4 correction'
nb.cells=[c for c in nb.cells if tag not in c.get('source','') and 'multispecies_summary_table' not in c.get('source','')]
nb.cells.extend([
 nbformat.v4.new_markdown_cell(tag+"\n\nThe compatible atlas contains 510,709 human and 429,746 mouse samples. Earlier OSDR results are retained as a forced mouse→human cross-species benchmark. Mouse Hallmark scores are zero-shot predictions from the human GTEx-trained head."),
 nbformat.v4.new_code_cell("multispecies_summary_table = pd.read_csv(ROOT/'results/osdr_multispecies/representative_topk_comparison.csv')\ndisplay(multispecies_summary_table.query(\"reference_set == 'combined' and k == 10\"))\nprint(json.loads((ROOT/'results/archs4_multispecies/manifold/provenance.json').read_text()))"),
 nbformat.v4.new_code_cell("display(Image(filename=str(ROOT/'results/osdr_multispecies/figures/liver_largest_study_per_sample_heatmap.png')))\ndisplay(Image(filename=str(ROOT/'results/osdr_multispecies/figures/muscle_largest_study_per_sample_heatmap.png')))\ndisplay(Image(filename=str(ROOT/'results/tcga_pan_cancer_positive_control/figures/cancer_similarity_matrices.png')))"),
 nbformat.v4.new_markdown_cell("**Conclusion.** Combined-atlas OSDR retrieval is overwhelmingly mouse-first. Hallmark-50 improves within-species tissue/state coherence and reproduces TCGA functional cancer geometry, but the global ARCHS4 manifold shows stronger species separation rather than broad human–mouse mixing. See `MULTISPECIES_FINAL_REPORT.md` for the full audit and limitations.")])
nbformat.write(nb,path)
