#!/usr/bin/env Rscript
# Knit the checkpoint analysis reports for every model.
#
# For each model in MODELS this renders two per-model PDFs into reports/:
#   trajectory_analysis_<model>.pdf
#   <evals>_eval_performance_analysis_<model>.pdf
#
# Run from anywhere:
#
#     Rscript Data-Tools/CheckPoint/render_all.R
#
# Compare the strategies at a different evaluation budget (default 500) with
# --evals; the value is baked into the performance-report file name:
#
#     Rscript Data-Tools/CheckPoint/render_all.R --evals 250
#
# Optionally restrict to specific models (after any --evals flag):
#
#     Rscript Data-Tools/CheckPoint/render_all.R --evals 250 RF MLP

MODELS <- c("RF", "ET", "GB", "KNN", "KSVC", "MLP")

# Hard-evaluation budget the performance report compares strategies at.
EVALS <- 500L

# Directory holding this script (and the .Rmd reports + aggregated CSV).
args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_all[grep("^--file=", args_all)])
script_dir <- if (length(file_arg)) normalizePath(dirname(file_arg)) else getwd()

# Positional args: optional `--evals <N>` then an optional model list.
sel <- commandArgs(trailingOnly = TRUE)

ev_idx <- which(sel == "--evals")
if (length(ev_idx)) {
  if (ev_idx == length(sel))
    stop("--evals requires a value, e.g. --evals 250")
  EVALS <- suppressWarnings(as.integer(sel[ev_idx + 1]))
  if (is.na(EVALS))
    stop("--evals value must be an integer")
  sel <- sel[-c(ev_idx, ev_idx + 1)]
}

if (length(sel)) {
  unknown <- setdiff(sel, MODELS)
  if (length(unknown))
    stop("Unknown model(s): ", paste(unknown, collapse = ", "))
  MODELS <- sel
}

csv_path <- file.path(script_dir, "checkpoint_trajectories.csv")
if (!file.exists(csv_path))
  stop("Missing ", csv_path, " -- run aggregator.py first.")

out_dir <- file.path(script_dir, "reports")
dir.create(out_dir, showWarnings = FALSE)

render_report <- function(report, output_file, params) {
  rmarkdown::render(
    input = file.path(script_dir, report),
    params = params,
    output_file = output_file,
    output_dir = out_dir,
    quiet = TRUE,
    envir = new.env())
}

for (model in MODELS) {
  message(sprintf("Rendering trajectory_analysis for %s ...", model))
  render_report("trajectory_analysis.Rmd",
                sprintf("trajectory_analysis_%s.pdf", model),
                list(model = model))

  message(sprintf("Rendering eval_performance_analysis (%d evals) for %s ...",
                  EVALS, model))
  render_report("eval_performance_analysis.Rmd",
                sprintf("%d_eval_performance_analysis_%s.pdf", EVALS, model),
                list(model = model, evals = EVALS))
}

message("Done. Reports written to ", out_dir)
