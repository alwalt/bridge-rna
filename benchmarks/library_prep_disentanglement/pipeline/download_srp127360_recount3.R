#!/usr/bin/env Rscript
# Download recount3 gene counts for the controlled SRP127360 test dataset.
suppressPackageStartupMessages({library(recount3); library(Matrix); library(SummarizedExperiment)})
args <- commandArgs(trailingOnly=TRUE)
out <- if (length(args)) args[[1]] else "benchmarks/library_prep_disentanglement/work/srp127360"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
info <- data.frame(project="SRP127360", project_home="data_sources/sra",
                   organism="human", project_type="data_sources", file_source="sra")
message("[download] recount3 SRP127360")
rse <- create_rse(info, type="gene", annotation="gencode_v29",
                  recount3_url="https://recount-opendata.s3.amazonaws.com/recount3/release")
runs <- c("SRR6410613","SRR6410614","SRR6410611","SRR6410612",
          "SRR6410617","SRR6410618","SRR6410615","SRR6410616",
          "SRR6410605","SRR6410606","SRR6410603","SRR6410604",
          "SRR6410609","SRR6410610","SRR6410607","SRR6410608")
ids <- as.character(colData(rse)$external_id); keep <- which(ids %in% runs)
if (length(keep) != 16) stop("Expected 16 runs; found ", length(keep))
counts <- as(transform_counts(rse)[, keep, drop=FALSE], "dgCMatrix")
colnames(counts) <- ids[keep]
writeMM(counts, file.path(out, "counts.mtx"))
writeLines(rownames(rse), file.path(out, "genes.txt"))
writeLines(colnames(counts), file.path(out, "samples.txt"))
gm <- as.data.frame(rowData(rse)); gm$recount3_gene_id <- rownames(rse)
write.csv(gm, file.path(out, "gene_metadata.csv"), row.names=FALSE)
write.csv(as.data.frame(colData(rse)[keep,]), file.path(out, "recount3_metadata.csv"), row.names=FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "session_info.txt"))
message("[complete] ", nrow(counts), " genes x ", ncol(counts), " runs")
