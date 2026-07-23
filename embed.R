#!/usr/bin/env Rscript
# Module to download datasets via Elia's found benchmark
#
suppressPackageStartupMessages({
  library(argparser)
})

# arg parsing
source("common/cli.R")
p <- arg_parser("download module")
p <- add_base_args(p)                      # --output_dir, --name
p <- add_stage_args(p, "INTG8")  # the stage I/O contract
# your own method params — argparser directly (its add_argument requires `help`):
p <- add_argument(p, "--method", type = "character", help = "number of PCs")
p <- add_argument(p, "--k_anchor", type = "integer", help = "number of PCs")
args <- parse_args(p)                      # argparser's own parser

# from properties input, get batch variable
props <- yaml::read_yaml(args$properties_info)
if (is.null(props$batch_var) || props$batch_var == "") {
  stop("batch_var is required in properties_info for selection_type 'seurat_vst_batch'")
}
args$batch_variable <- props$batch_var

# logging
cat(sprintf("Full command: %s\n", paste(commandArgs(trailingOnly = FALSE), collapse = " ")))
cat(sprintf("LOG: command line args\n----------------------------------\n"))
for (i in 1:length(args))
  cat(sprintf("  %s: %s\n", names(args)[i], args[[i]]))
cat(sprintf("----------------------------------\n"))


