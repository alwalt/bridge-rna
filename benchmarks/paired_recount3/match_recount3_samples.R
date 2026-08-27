#!/usr/bin/env Rscript
# Match ARCHS4 GSM candidates to recount3 using recount3 metadata only.
# This script downloads metadata, never expression. It emits CSV because the
# official recount3 R environment does not require the optional Arrow package.

suppressPackageStartupMessages({
  library(recount3)
})

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0) return(default)
  if (hit == length(args)) stop(flag, " requires a value")
  args[[hit + 1]]
}

input <- value_after("--candidates", "benchmarks/paired_recount3/outputs/candidate_samples.csv")
output <- value_after("--output", "benchmarks/paired_recount3/outputs/recount3_matches.csv")
recount_url <- value_after(
  "--recount3-url", "https://recount-opendata.s3.amazonaws.com/recount3/release"
)
max_projects <- as.integer(value_after("--max-projects", "0"))
workers <- as.integer(value_after("--workers", "8"))
metadata_cache <- value_after(
  "--metadata-cache", "benchmarks/paired_recount3/outputs/recount3_metadata_cache"
)
if (!file.exists(input)) stop("Missing candidate CSV: ", input)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
dir.create(metadata_cache, recursive = TRUE, showWarnings = FALSE)

candidates <- read.csv(input, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("gsm", "gse", "cohort")
if (!all(required %in% names(candidates))) stop("Candidate CSV lacks required columns")
targets <- unique(toupper(candidates$gsm))

message("Loading recount3 human sample catalog...")
catalog <- available_samples(organism = "human", recount3_url = recount_url)
catalog <- catalog[catalog$project_type == "data_sources" & catalog$file_source == "sra", ]
projects <- unique(catalog[, c("project", "project_home", "organism")])
if (max_projects > 0) projects <- head(projects, max_projects)
message("Scanning metadata for ", nrow(projects), " SRA projects; no expression is downloaded.")

hits <- list()
failures <- list()
# Resolve any ARCHS4 relations that already provide run accessions directly.
if ("archs4_sra_accessions" %in% names(candidates)) {
  for (i in seq_len(nrow(candidates))) {
    accessions <- strsplit(candidates$archs4_sra_accessions[[i]], ",", fixed = TRUE)[[1]]
    runs <- accessions[grepl("^(SRR|ERR|DRR)[0-9]+$", accessions)]
    direct_rows <- match(runs, catalog$external_id, nomatch = 0L)
    direct <- catalog[direct_rows[direct_rows > 0L], , drop = FALSE]
    if (nrow(direct)) hits[[length(hits) + 1]] <- list(
      gsm = candidates$gsm[[i]],
      recount3_project = paste(sort(unique(direct$project)), collapse = ","),
      recount3_project_home = paste(sort(unique(direct$project_home)), collapse = ","),
      recount3_run_ids = paste(sort(unique(direct$external_id)), collapse = ","),
      run_count = length(unique(direct$external_id))
    )
  }
}
scan_project <- function(i) {
  info <- projects[i, , drop = FALSE]
  urls <- tryCatch(
    locate_url(project = info$project, project_home = info$project_home,
               type = "metadata", organism = "human", recount3_url = recount_url),
    error = function(e) NULL
  )
  if (is.null(urls)) return(list(hits = list(), failure = NULL))
  # Only the original SRA table is needed for GSM/SRX/SRR aliases. Avoid the
  # larger QC/prediction bundle during this metadata-only discovery pass.
  sra_url <- urls[grepl("/sra[.]sra[.]", urls)]
  if (!length(sra_url)) return(list(hits = list(), failure = NULL))
  meta <- tryCatch({
    local_file <- file.path(metadata_cache, basename(sra_url[[1]]))
    if (!file.exists(local_file)) {
      temporary <- paste0(local_file, ".", Sys.getpid(), ".tmp")
      download.file(sra_url[[1]], temporary, mode = "wb", quiet = TRUE)
      file.rename(temporary, local_file)
    }
    read.delim(local_file, stringsAsFactors = FALSE, check.names = FALSE)
  }, error = function(e) {
    NULL
  })
  if (is.null(meta) || nrow(meta) == 0) return(list(
    hits = list(), failure = data.frame(
      project = info$project, reason = "download_or_read_failed", stringsAsFactors = FALSE
    )
  ))
  character_columns <- names(meta)[vapply(meta, is.character, logical(1))]
  if (length(character_columns) == 0) return(list(hits = list(), failure = NULL))
  text <- apply(meta[, character_columns, drop = FALSE], 1, paste, collapse = " | ")
  project_hits <- list()
  found_gsms <- unique(unlist(regmatches(text, gregexpr("GSM[0-9]+", text))))
  for (gsm in found_gsms[found_gsms %in% targets]) {
    row_idx <- which(grepl(paste0("(^|[^0-9])", gsm, "([^0-9]|$)"), text))
    run_col <- intersect(c("external_id", "run", "sra.run", "run_accession"), names(meta))
    runs <- if (length(run_col)) unique(as.character(meta[row_idx, run_col[[1]]])) else character()
    runs <- runs[grepl("^(SRR|ERR|DRR)[0-9]+$", runs)]
    project_hits[[length(project_hits) + 1]] <- list(
      gsm = gsm, recount3_project = as.character(info$project),
      recount3_project_home = as.character(info$project_home),
      recount3_run_ids = paste(sort(runs), collapse = ","), run_count = length(runs)
    )
  }
  list(hits = project_hits, failure = NULL)
}
workers <- max(1L, workers)
message("Using ", workers, " metadata workers; cache: ", metadata_cache)
if (.Platform$OS.type == "unix" && workers > 1) {
  scanned <- parallel::mclapply(seq_len(nrow(projects)), scan_project, mc.cores = workers)
} else {
  scanned <- lapply(seq_len(nrow(projects)), scan_project)
}
for (item in scanned) {
  hits <- c(hits, item$hits)
  if (!is.null(item$failure)) failures[[length(failures) + 1]] <- item$failure
}
message("Finished project scan; matched GSMs=",
        length(unique(vapply(hits, `[[`, character(1), "gsm"))))

if (length(hits)) {
  hit_df <- unique(do.call(rbind, lapply(hits, as.data.frame, stringsAsFactors = FALSE)))
  # Keep all project/run candidates in one row and flag ambiguity rather than choosing silently.
  hit_df <- aggregate(
    cbind(recount3_project, recount3_project_home, recount3_run_ids) ~ gsm,
    hit_df, function(x) paste(sort(unique(x[nzchar(x)])), collapse = ",")
  )
  hit_df$run_count <- vapply(strsplit(hit_df$recount3_run_ids, ",", fixed = TRUE),
                             function(x) sum(nzchar(x)), integer(1))
  hit_df$project_count <- vapply(strsplit(hit_df$recount3_project, ",", fixed = TRUE),
                                 function(x) sum(nzchar(x)), integer(1))
} else {
  hit_df <- data.frame(gsm = character(), recount3_project = character(),
                       recount3_project_home = character(), recount3_run_ids = character(),
                       run_count = integer(), project_count = integer())
}

result <- merge(candidates, hit_df, by = "gsm", all.x = TRUE, sort = FALSE)
result$match_status <- ifelse(
  is.na(result$recount3_project), "not_found",
  ifelse(result$project_count > 1, "multiple_projects",
         ifelse(result$run_count == 0, "matched_no_run", "matched"))
)
write.csv(result, output, row.names = FALSE, na = "")
message("Saved: ", output)
print(table(result$cohort, result$match_status, useNA = "ifany"))
if (length(failures)) {
  failure_path <- sub("[.]csv$", "_metadata_failures.csv", output)
  write.csv(do.call(rbind, failures), failure_path, row.names = FALSE)
  message("Metadata read failures: ", failure_path)
}
