# Asset and provenance manifest

Large assets remain in `/home/walt/bridge-rna` and are not duplicated in the
benchmark directory. Exact paths, byte sizes, SHA-256 hashes, and Zenodo MD5
checksums are saved in `results/asset_manifest.json`.

| Asset | Provenance | Use |
|---|---|---|
| `drug_response_ic50.csv` | Zenodo 15744294 | response observations |
| `drug_response_expr_data.csv` | Zenodo 15744294 | baseline cell-line expression |
| `model/r7hnr92k/best_model.pt` | shared Bridge repository | frozen Bridge-45.6M |
| `model/BulkFormer/BulkFormer_50M.pt` | official release, shared repository | frozen BulkFormer-50M |
| `model/BulkFormer/model/BulkFormer_147M.pt` | official release, shared repository | frozen BulkFormer-147M |
| `data/ensembl/canonical_genes.csv` | shared Bridge repository | ordered 15,165-gene vocabulary |
| `model/BulkFormer/data/bulkformer_gene_info.csv` | official release | native BulkFormer vocabulary |

Generated embeddings and model-aligned matrices are regenerable caches under
`work/` and are git-ignored.
