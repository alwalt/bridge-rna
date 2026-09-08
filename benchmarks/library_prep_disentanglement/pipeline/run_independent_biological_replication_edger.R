#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(edgeR))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3) stop("usage: script counts.csv.gz memberships.csv output.csv.gz")
counts <- as.matrix(read.csv(gzfile(args[1]), row.names=1, check.names=FALSE))
members <- read.csv(args[2], stringsAsFactors=FALSE)
results <- list()
for (cid in unique(members$contrast_id)) {
  meta <- members[members$contrast_id == cid, , drop=FALSE]
  x <- round(counts[, meta$sample_id, drop=FALSE])
  group <- relevel(factor(meta$condition), "GC")
  design <- model.matrix(~ group)
  y <- DGEList(counts=x, group=group)
  keep <- filterByExpr(y, design=design)
  y <- calcNormFactors(y[keep,], method="TMM")
  if (ncol(x) == 2) {
    # One FLT and one GC provide no residual df. Retain a descriptive edgeR
    # ranking using a conservative fixed BCV=0.4 (dispersion=0.16), and flag it.
    test <- exactTest(y, pair=c("GC", "FLT"), dispersion=0.16)
    tab <- topTags(test, n=Inf, sort.by="none")$table
    # exactTest's topTags table has no QL F statistic. A monotone
    # -log10(P) magnitude supplies a descriptive signed ranking only.
    tab$F <- (-log10(pmax(tab$PValue, .Machine$double.xmin)))^2
    inference <- "exactTest_fixed_BCV_0.4_no_replication"
  } else {
    y <- estimateDisp(y, design, robust=TRUE)
    fit <- glmQLFit(y, design, robust=TRUE)
    tab <- topTags(glmQLFTest(fit, coef="groupFLT"), n=Inf, sort.by="none")$table
    inference <- "robust_QLF"
  }
  tab$gene_symbol <- rownames(tab)
  tab$signed_statistic <- sign(tab$logFC) * sqrt(pmax(tab$F, 0))
  tab$contrast_id <- cid
  tab$inference_method <- inference
  tab$tested <- TRUE
  all <- data.frame(gene_symbol=rownames(counts), stringsAsFactors=FALSE)
  all <- merge(all, tab, by="gene_symbol", all.x=TRUE, sort=FALSE)
  all$tested[is.na(all$tested)] <- FALSE
  results[[cid]] <- all
  cat(sprintf("[edgeR] %s samples=%d tested=%d method=%s\n", cid, ncol(x), sum(keep), inference))
}
write.csv(do.call(rbind, results), gzfile(args[3]), row.names=FALSE)
