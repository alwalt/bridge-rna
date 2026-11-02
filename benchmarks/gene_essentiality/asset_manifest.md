# Asset manifest

Large source/model assets are reused in place outside this worktree. No source
dataset or checkpoint was copied or modified.

| Asset | External path | Use |
|---|---|---|
| Released expression pickle | `/home/walt/bridge-rna/benchmarks/gene_property_prediction/data/source/gene_essentiality_expr_data.pkl` | Authoritative expression input |
| Released dependency pickle | `/home/walt/bridge-rna/benchmarks/gene_property_prediction/data/source/gene_essentiality_score.pkl` | Authoritative target input |
| Bridge checkpoint | `/home/walt/bridge-rna/model/r7hnr92k/best_model.pt` | Frozen encoder |
| Bridge configuration | `/home/walt/bridge-rna/model/r7hnr92k/config.json` | Architecture/input contract |
| Bridge vocabulary | `data/ensembl/canonical_genes.csv` | Fixed 15,165-token order |
| HGNC mapping snapshot | `/home/walt/bridge-rna/data/annotations/hgnc/hgnc_complete_set_2026-08-27.tsv` | Ensembl-to-Bridge mapping |
| BulkFormer-147M checkpoint | `/home/walt/bridge-rna/model/BulkFormer/model/BulkFormer_147M.pt` | Frozen encoder |
| BulkFormer vocabulary | `/home/walt/bridge-rna/model/BulkFormer/data/bulkformer_gene_info.csv` | Fixed 20,010-token order |
| BulkFormer graph and weights | `/home/walt/bridge-rna/model/BulkFormer/data/G_tcga.pt`, `/home/walt/bridge-rna/model/BulkFormer/data/G_tcga_weight.pt` | Published model graph |
| Derived shared work | `/home/walt/bridge-rna/benchmarks/gene_essentiality/work` | Regenerable matrices, float16 contextual-token caches, fold models, and out-of-fold predictions |

Checkpoint and source checksums, cache shapes, layers, transformations, and
completion state are recorded in `results/dataset_audit.json` and each shared
cache's `provenance.json`.
