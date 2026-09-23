#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(limma)
  library(data.table)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", args[grep("^--file=", args)])
here <- normalizePath(file.path(dirname(script_arg), ".."))
prep <- file.path(here, "work", "prepared")
out <- file.path(here, "results", "differential_expression")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
manifest <- fread(file.path(here, "results", "manifests", "locked_samples.csv"))
genes <- fread("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv")$gene_symbol

for (accession in c("GSE138614", "GSE101794", "GSE72509")) {
  samples <- manifest[manifest$accession == accession, ]
  rows <- fread(file.path(prep, paste0(accession, "_matrix_rows.csv")))
  samples <- merge(samples, rows, by = "sample_id", sort = FALSE)
  samples <- samples[order(samples$matrix_row), ]
  expression_frame <- fread(file.path(prep, paste0(accession, "_log1p_tpm.csv.gz")))
  stopifnot(identical(expression_frame$sample_id, samples$sample_id))
  expression <- t(as.matrix(expression_frame[, ..genes]))
  colnames(expression) <- samples$sample_id
  rownames(expression) <- genes

  if (accession == "GSE138614") {
    # Donor is the independent unit. Average log-expression across tissue pieces
    # within each donor; cases and controls are different people.
    donor_ids <- unique(samples$donor)
    donor_expression <- sapply(donor_ids, function(d) rowMeans(expression[, samples$donor == d, drop = FALSE]))
    donor_role <- vapply(donor_ids, function(d) unique(samples$role[samples$donor == d]), character(1))
    expression <- donor_expression
    role <- factor(donor_role, levels = c("control", "case"))
  } else {
    role <- factor(samples$role, levels = c("control", "case"))
  }
  design <- model.matrix(~ role)
  fit <- eBayes(lmFit(expression, design), trend = TRUE, robust = TRUE)
  table <- topTable(fit, coef = "rolecase", number = Inf, sort.by = "none")
  result <- data.frame(accession = accession, gene = rownames(table),
                       logFC = table$logFC, t = table$t, pvalue = table$P.Value,
                       fdr = table$adj.P.Val, stringsAsFactors = FALSE)
  result$absolute_score <- abs(result$t)
  result <- result[order(-result$absolute_score, result$gene), ]
  result$rank <- seq_len(nrow(result))
  fwrite(result, file.path(out, paste0(accession, "_gene_ranking.csv.gz")))
}

provenance <- list(
  input = "natural log1p(TPM), canonical 15,165 genes; identical locked samples as Bridge",
  method = "limma empirical-Bayes linear model with trend=TRUE and robust=TRUE",
  design = "case versus control",
  ms_independence = "mean log-expression per donor before fitting; 7 case versus 5 control donors",
  ranking = "absolute moderated t statistic",
  covariates = "unadjusted to match de Weerd case-control contrasts; recorded cohort covariates are limitations"
)
jsonlite::write_json(provenance, file.path(out, "provenance.json"), auto_unbox = TRUE, pretty = TRUE)
