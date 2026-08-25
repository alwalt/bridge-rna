## This used the preprocessed parquet files, it's not currently used in the pipeline
from pathlib import Path
import pandas as pd

protein_csv = Path("protein_coding_genes.csv")
ortholog_file = Path("orthologs_one2one.txt")

# Parquet paths (genes × samples, log1p(TPM))
val_human_parquet = Path("../archs4/train_orthologs/val/expression_val_human.parquet")
val_mouse_parquet = Path("../archs4/train_orthologs/val/expression_val_mouse.parquet")

# -------------------------------------------------
# 1. Load protein-coding gene symbols
# -------------------------------------------------
pc_genes = set()
with open(protein_csv) as f:
    next(f)  # header
    for line in f:
        symbol = line.split(",")[0].strip().upper()
        if symbol:
            pc_genes.add(symbol)

# -------------------------------------------------
# 2. Scan ortholog table
# -------------------------------------------------
all_human_genes = []
pc_matches = []
non_pc_matches = []

with open(ortholog_file) as f:
    next(f)  # header
    for line in f:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 5:
            continue

        human_gene = fields[4].strip().upper()
        if not human_gene:
            continue

        all_human_genes.append(human_gene)

        if human_gene in pc_genes:
            pc_matches.append(human_gene)
        else:
            non_pc_matches.append(human_gene)

# -------------------------------------------------
# 3. Summaries
# -------------------------------------------------
all_unique = set(all_human_genes)
pc_unique = set(pc_matches)
non_pc_unique = set(non_pc_matches)

print("\n===== PROTEIN-CODING SUMMARY =====")
print(f"Total ortholog rows: {len(all_human_genes):,}")
print(f"Unique human ortholog genes: {len(all_unique):,}")
print(f"Protein-coding ortholog genes (unique): {len(pc_unique):,}")
print(f"Non-protein-coding ortholog genes (unique): {len(non_pc_unique):,}")

if all_unique:
    print(f"Percent protein-coding: {100*len(pc_unique)/len(all_unique):.2f}%")

# -------------------------------------------------
# 4. Show quick examples of non-protein-coding genes
# -------------------------------------------------
print("\nExamples of non-protein-coding ortholog genes:")
for g in sorted(non_pc_unique)[:10]:
    print("  ", g)

# -------------------------------------------------
# 5. Zero-expression gene analysis from val parquets
# -------------------------------------------------
print("\n===== ZERO-EXPRESSION GENE ANALYSIS (val split) =====")

# Load expression data (genes × samples)
print("\nLoading human val expression...")
h_expr = pd.read_parquet(val_human_parquet)
print(f"  Shape: {h_expr.shape} (genes × samples)")

print("Loading mouse val expression...")
m_expr = pd.read_parquet(val_mouse_parquet)
print(f"  Shape: {m_expr.shape} (genes × samples)")

# Genes with zero expression across ALL samples
human_zero_genes = set(h_expr.index[h_expr.sum(axis=1) == 0])
mouse_zero_genes = set(m_expr.index[m_expr.sum(axis=1) == 0])
both_zero_genes = human_zero_genes & mouse_zero_genes

print(f"\nHuman: {len(human_zero_genes):,} / {h_expr.shape[0]:,} genes have zero expression in all {h_expr.shape[1]:,} samples")
print(f"Mouse: {len(mouse_zero_genes):,} / {m_expr.shape[0]:,} genes have zero expression in all {m_expr.shape[1]:,} samples")
print(f"Both:  {len(both_zero_genes):,} genes have zero expression in BOTH species")

# Show examples
if both_zero_genes:
    print(f"\nExamples of genes with zero expression in both species:")
    for g in sorted(both_zero_genes)[:20]:
        print(f"  {g}")

# Check overlap with non-protein-coding
# (gene names in parquet are original case, ortho table was uppercased)
both_zero_upper = {g.upper() for g in both_zero_genes}
zero_and_non_pc = both_zero_upper & non_pc_unique
zero_and_pc = both_zero_upper & pc_unique

print(f"\nOf the {len(both_zero_genes):,} zero-in-both genes:")
print(f"  Non-protein-coding: {len(zero_and_non_pc):,}")
print(f"  Protein-coding:     {len(zero_and_pc):,}")
print(f"  Not in ortho table: {len(both_zero_upper) - len(zero_and_non_pc) - len(zero_and_pc):,}")
