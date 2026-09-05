#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(edgeR))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4) stop("usage: script counts.csv metadata.csv analysis output.csv")
counts <- as.matrix(read.csv(args[1], row.names=1, check.names=FALSE))
meta <- read.csv(args[2], stringsAsFactors=TRUE)
counts <- counts[, meta$sample_id, drop=FALSE]
y <- DGEList(counts=round(counts))

if (args[3] == "tcell") {
  meta$donor <- factor(meta$donor)
  meta$library_prep <- relevel(factor(meta$library_prep), "polyA")
  design <- model.matrix(~ donor + library_prep, meta)
  coef_name <- "library_prepribo"
  sign_multiplier <- 1
} else if (args[3] == "rr1") {
  meta$animal <- factor(meta$animal)
  meta$measurement_B <- as.numeric(meta$measurement == "OSD168")
  meta$interaction <- meta$measurement_B * as.numeric(meta$flight_status == "FLT")
  design <- model.matrix(~ animal + measurement_B + interaction, meta)
  coef_name <- "interaction"
  # edgeR interaction is (OSD168 FLT-GC) - (OSD48 FLT-GC).
  # Requested reporting direction is OSD48 minus OSD168.
  sign_multiplier <- -1
} else stop("unknown analysis")

if (qr(design)$rank != ncol(design)) stop("design matrix is rank deficient")
keep <- filterByExpr(y, design=design)
y <- calcNormFactors(y[keep,,keep.lib.sizes=FALSE])
y <- estimateDisp(y, design, robust=TRUE)
fit <- glmQLFit(y, design, robust=TRUE)
test <- glmQLFTest(fit, coef=which(colnames(design)==coef_name))
tab <- topTags(test, n=Inf, sort.by="none")$table
tab$gene_symbol <- rownames(tab)
tab$logFC <- tab$logFC * sign_multiplier
# Guard against machine-precision negative QL F values (observed magnitude
# below 5e-9 for five RR1 genes); mathematically these are zero.
tab$signed_statistic <- sign(tab$logFC) * sqrt(pmax(tab$F, 0))
tab$instability_statistic <- sqrt(pmax(tab$F, 0))
tab$tested <- TRUE
all <- data.frame(gene_symbol=rownames(counts), stringsAsFactors=FALSE)
all <- merge(all, tab, by="gene_symbol", all.x=TRUE, sort=FALSE)
all$tested[is.na(all$tested)] <- FALSE
write.csv(all, args[4], row.names=FALSE)
cat(sprintf("[%s] samples=%d genes_input=%d genes_tested=%d coefficient=%s\n", args[3], ncol(counts), nrow(counts), sum(keep), coef_name))
