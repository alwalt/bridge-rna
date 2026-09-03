#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(edgeR))
args <- commandArgs(trailingOnly=TRUE)
counts_df <- read.delim(args[[1]], check.names=FALSE, stringsAsFactors=FALSE)
gene_id <- counts_df[[1]]
counts <- as.matrix(counts_df[,-1,drop=FALSE]); rownames(counts) <- gene_id
meta <- read.csv(args[[2]], stringsAsFactors=FALSE)
counts <- counts[,meta$GSM,drop=FALSE]
meta$condition <- factor(meta$role, levels=c("pre_control","post_exercise"))
paired <- all(!is.na(meta$subject_id)) && all(nchar(meta$subject_id)>0) &&
          all(table(meta$subject_id)==2) && length(unique(meta$subject_id)) < nrow(meta)
if (paired) {
  meta$subject_id <- factor(meta$subject_id)
  design <- model.matrix(~ subject_id + condition, data=meta)
} else design <- model.matrix(~ condition, data=meta)
y <- DGEList(counts=counts)
keep <- filterByExpr(y, design=design)
if (sum(keep) < 2) stop("Too few expressed genes after filterByExpr")
y <- calcNormFactors(y[keep,,keep.lib.sizes=FALSE])
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design)
coef_name <- grep("conditionpost_exercise", colnames(design), value=TRUE)
if (length(coef_name)!=1) stop("Condition coefficient not found")
tt <- topTags(glmQLFTest(fit, coef=which(colnames(design)==coef_name)), n=Inf, sort.by="none")$table
out <- data.frame(gene_id=rownames(tt), log2_fold_change=tt$logFC,
 log_cpm=tt$logCPM, quasi_likelihood_f=tt$F, p_value=tt$PValue,
 adjusted_p_value=tt$FDR, stringsAsFactors=FALSE)
write.csv(out, args[[3]], row.names=FALSE)
cat(sprintf("edgeR complete: samples=%d genes_available=%d genes_tested=%d paired=%s\n",
 ncol(counts), nrow(counts), nrow(out), paired))
