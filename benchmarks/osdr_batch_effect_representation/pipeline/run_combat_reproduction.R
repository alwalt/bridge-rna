#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: run_combat_reproduction.R <work> <results> <r-lib>")
work <- args[[1]]
results <- args[[2]]
.libPaths(c(args[[3]], .libPaths()))
suppressPackageStartupMessages(library(sva))

message("[ComBat] loading matrices")
raw <- as.matrix(read.csv(gzfile(file.path(work, "combat_raw_counts.csv.gz")), row.names = 1, check.names = FALSE))
logexpr <- as.matrix(read.csv(gzfile(file.path(work, "combat_log2_normalized.csv.gz")), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(results, "sample_manifest.csv"), check.names = FALSE)
stopifnot(identical(colnames(raw), meta$sample_id), identical(colnames(logexpr), meta$sample_id))

# Constant rows cannot inform PCA and can destabilize empirical-Bayes fitting.
keep_cont <- apply(logexpr, 1, var) > 0
keep_count <- rowSums(raw) > 0 & apply(raw, 1, var) > 0
logexpr <- logexpr[keep_cont, , drop = FALSE]
# RSEM estimated counts may be fractional; ComBat_seq requires integer counts.
# Rounding is explicit and recorded as a reproduction limitation.
raw_int <- round(pmax(raw[keep_count, , drop = FALSE], 0))

median_ratio <- function(counts) {
  eligible <- apply(counts > 0, 1, all)
  gm <- exp(rowMeans(log(counts[eligible, , drop = FALSE])))
  sf <- apply(sweep(counts[eligible, , drop = FALSE], 1, gm, "/"), 2, median)
  sweep(counts, 2, sf, "/")
}

pca_rows <- list()
var_rows <- list()
run_pca <- function(mat, method, batch_name) {
  fit <- prcomp(t(mat), center = TRUE, scale. = FALSE)
  variance <- fit$sdev^2 / sum(fit$sdev^2)
  coords <- data.frame(sample_id = rownames(fit$x), method = method,
                       batch_variable = batch_name, PC1 = fit$x[, 1], PC2 = fit$x[, 2],
                       stringsAsFactors = FALSE)
  vars <- data.frame(method = method, batch_variable = batch_name,
                     PC = seq_along(variance), variance_explained = variance,
                     cumulative_variance = cumsum(variance))
  list(coords = coords, vars = vars)
}

for (batch_name in c("library_preparation", "mission")) {
  batch <- factor(meta[[batch_name]])
  # Preserve the declared FLT/GC biological condition while estimating batch.
  mod <- model.matrix(~ condition, data = meta)
  message(sprintf("[ComBat] %s", batch_name))
  corrected <- ComBat(dat = logexpr, batch = batch, mod = mod,
                      par.prior = TRUE, prior.plots = FALSE)
  ans <- run_pca(corrected, "ComBat", batch_name)
  pca_rows[[length(pca_rows) + 1]] <- ans$coords
  var_rows[[length(var_rows) + 1]] <- ans$vars

  message(sprintf("[ComBat-seq] %s", batch_name))
  corrected_counts <- ComBat_seq(counts = raw_int, batch = batch,
                                 group = factor(meta$condition), full_mod = TRUE)
  corrected_norm <- median_ratio(corrected_counts)
  ans <- run_pca(log2(corrected_norm + 1), "ComBat-seq", batch_name)
  pca_rows[[length(pca_rows) + 1]] <- ans$coords
  var_rows[[length(var_rows) + 1]] <- ans$vars
}

write.csv(do.call(rbind, pca_rows), file.path(results, "combat_pca_coordinates.csv"), row.names = FALSE)
write.csv(do.call(rbind, var_rows), file.path(results, "combat_pca_variance.csv"), row.names = FALSE)
message("[complete] wrote ComBat reproduction coordinates and variance")
