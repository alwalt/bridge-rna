#!/usr/bin/env Rscript
# Download project-level recount3 gene assays, retain selected runs, transform
# coverage to read counts, aggregate runs to GSM, and save sparse Matrix Market.

suppressPackageStartupMessages({
  library(recount3)
  library(Matrix)
  library(SummarizedExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (!length(hit)) return(default)
  if (hit == length(args)) stop(flag, " requires a value")
  args[[hit + 1]]
}
input <- value_after("--pairs", "benchmarks/paired_recount3/outputs/final_pairs.csv")
output_dir <- value_after("--output-dir", "benchmarks/paired_recount3/outputs/recount3_counts")
annotation <- value_after("--annotation", "gencode_v29")
recount_url <- value_after(
  "--recount3-url", "https://recount-opendata.s3.amazonaws.com/recount3/release"
)
if (!file.exists(input)) stop("Missing final-pairs CSV: ", input)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
pairs <- read.csv(input, stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("gsm", "recount3_project", "recount3_project_home", "recount3_run_ids") %in% names(pairs))) {
  stop("Final-pairs CSV lacks recount3 identifiers")
}
if (any(grepl(",", pairs$recount3_project))) stop("Multi-project GSMs must be resolved before download")

blocks <- list()
sample_meta <- list()
gene_ids <- NULL
gene_metadata <- NULL
for (project in unique(pairs$recount3_project)) {
  selected <- pairs[pairs$recount3_project == project, ]
  project_info <- unique(selected[, c("recount3_project", "recount3_project_home")])
  info <- data.frame(
    project = project_info$recount3_project,
    project_home = project_info$recount3_project_home,
    organism = "human", project_type = "data_sources", file_source = "sra",
    stringsAsFactors = FALSE
  )
  message("Downloading project ", project, " for ", nrow(selected), " GSMs")
  rse <- create_rse(info, type = "gene", annotation = annotation,
                    recount3_url = recount_url)
  run_to_gsm <- do.call(rbind, lapply(seq_len(nrow(selected)), function(i) {
    runs <- strsplit(selected$recount3_run_ids[[i]], ",", fixed = TRUE)[[1]]
    data.frame(external_id = runs[nzchar(runs)], gsm = selected$gsm[[i]])
  }))
  ids <- as.character(colData(rse)$external_id)
  keep <- which(ids %in% run_to_gsm$external_id)
  if (!length(keep)) next
  counts <- transform_counts(rse)[, keep, drop = FALSE]
  colnames(counts) <- ids[keep]
  map <- run_to_gsm$gsm[match(colnames(counts), run_to_gsm$external_id)]
  # Sum all runs for the same GSM in count space.
  unique_gsms <- unique(map)
  design <- sparseMatrix(
    i = seq_along(map), j = match(map, unique_gsms), x = 1,
    dims = c(length(map), length(unique_gsms)),
    dimnames = list(colnames(counts), unique_gsms)
  )
  aggregated <- as(counts, "dgCMatrix") %*% design
  colnames(aggregated) <- unique_gsms
  current_genes <- rownames(rse)
  if (is.null(gene_ids)) {
    gene_ids <- current_genes
    gene_metadata <- as.data.frame(rowData(rse), stringsAsFactors = FALSE)
    gene_metadata$recount3_gene_id <- current_genes
  }
  if (!identical(gene_ids, current_genes)) stop("Gene annotation/order differs across projects")
  blocks[[length(blocks) + 1]] <- aggregated
  sample_meta[[length(sample_meta) + 1]] <- data.frame(
    gsm = colnames(aggregated), recount3_project = project,
    observed_run_count = as.integer(table(map)[colnames(aggregated)])
  )
}
if (!length(blocks)) stop("No selected recount3 runs were found")
matrix <- do.call(cbind, blocks)
if (anyDuplicated(colnames(matrix))) stop("A GSM was emitted by multiple projects")
writeMM(matrix, file.path(output_dir, "counts.mtx"))
writeLines(gene_ids, file.path(output_dir, "genes.txt"))
write.csv(gene_metadata, file.path(output_dir, "gene_metadata.csv"), row.names = FALSE)
writeLines(colnames(matrix), file.path(output_dir, "samples.txt"))
write.csv(do.call(rbind, sample_meta), file.path(output_dir, "sample_metadata.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
message("Saved ", nrow(matrix), " genes x ", ncol(matrix), " GSMs to ", output_dir)
