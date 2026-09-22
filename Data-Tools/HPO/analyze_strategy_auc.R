#!/usr/bin/env Rscript

# This script produces statistics and plots comparing the performance of strategies by model.
# This includes the Friedman test, post-hoc pairwise comparisons, and win count across seeds. 
#
# Usage:
#   Rscript analyze_strategy_auc.R results.csv analysis_output_dir
#   Rscript analyze_strategy_auc.R "data/*.csv" analysis_output_dir
#
# Required packages:
#   install.packages(c("data.table", "ggplot2"))

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("Usage: Rscript analyze_strategy_auc.R <CSV file or glob> [output directory]")
}

input_pattern <- args[[1L]]
out_dir <- if (length(args) >= 2L) args[[2L]] else "auc_analysis"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "condition_plots"), recursive = TRUE, showWarnings = FALSE)

files <- Sys.glob(input_pattern)
if (!length(files) && file.exists(input_pattern)) files <- input_pattern
if (!length(files)) stop("No CSV files matched: ", input_pattern)

message("Reading ", length(files), " CSV file(s)...")
dt <- rbindlist(lapply(files, fread), use.names = TRUE, fill = TRUE,
                idcol = "source_file")

required <- c(
  "pop", "mut_rate", "variance", "strategy", "tpe_prob", "model",
  "task", "seed", "generation", "hard_evals", "test_auc"
)
missing_cols <- setdiff(required, names(dt))
if (length(missing_cols)) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Treat these columns as identifiers even when fread inferred numeric types.
id_cols <- c("pop", "mut_rate", "variance", "strategy", "tpe_prob",
             "model", "task", "seed")
dt[, (id_cols) := lapply(.SD, as.character), .SDcols = id_cols]
dt[, `:=`(
  generation = as.numeric(generation),
  hard_evals = as.numeric(hard_evals),
  test_auc = as.numeric(test_auc)
)]
dt <- dt[is.finite(generation) & is.finite(test_auc)]
if (!nrow(dt)) stop("No rows with finite generation and test_auc were found.")

# tpe_prob is tied to strategy, not an independent treatment.
strategy_map <- unique(dt[, .(strategy, tpe_prob)])
strategy_map[, n_tpe_values := uniqueN(tpe_prob), by = strategy]
if (strategy_map[, any(n_tpe_values > 1L)]) {
  warning("At least one strategy has multiple tpe_prob values; inspect strategy_tpe_mapping.csv")
}
fwrite(strategy_map, file.path(out_dir, "strategy_tpe_mapping.csv"))

condition_cols <- c("task", "model", "pop", "mut_rate", "variance")
run_cols <- c(condition_cols, "seed", "strategy")

# If duplicate rows exist at the same checkpoint, collapse them to one median value.
checkpoint <- dt[, .(
  test_auc = median(test_auc, na.rm = TRUE),
  hard_evals = max(hard_evals, na.rm = TRUE)
), by = c(run_cols, "generation")]

# The last available checkpoint is the row with the highest generation per run.
# max(hard_evals) breaks a tie if generation is duplicated.
setorder(checkpoint, task, model, pop, mut_rate, variance, seed, strategy,
         generation, hard_evals)
final <- checkpoint[, .SD[.N], by = run_cols]
fwrite(final, file.path(out_dir, "final_auc_by_run.csv"))

# Friedman requires a complete block: every included seed must contain every
# strategy present in that experimental condition.
condition_sizes <- final[, .(n_strategies_condition = uniqueN(strategy)),
                         by = condition_cols]
block_sizes <- final[, .(n_strategies_seed = uniqueN(strategy)),
                     by = c(condition_cols, "seed")]
block_sizes <- merge(block_sizes, condition_sizes, by = condition_cols)
block_sizes[, complete_block := n_strategies_seed == n_strategies_condition]
fwrite(block_sizes, file.path(out_dir, "block_completeness.csv"))

complete_keys <- block_sizes[complete_block == TRUE,
                            c(condition_cols, "seed"), with = FALSE]
final_complete <- merge(final, complete_keys,
                        by = c(condition_cols, "seed"), all = FALSE)

friedman_one <- function(x) {
  n_blocks <- uniqueN(x$seed)
  n_strategies <- uniqueN(x$strategy)
  expected_n <- n_blocks * n_strategies

  if (n_blocks < 2L || n_strategies < 2L || nrow(x) != expected_n) {
    return(data.table(
      n_blocks = n_blocks, n_strategies = n_strategies,
      statistic = NA_real_, df = NA_integer_, p_value = NA_real_,
      kendalls_w = NA_real_, status = "insufficient_or_unbalanced"
    ))
  }

  wide <- dcast(x, seed ~ strategy, value.var = "test_auc")
  score_matrix <- as.matrix(wide[, -"seed"])
  if (anyNA(score_matrix)) {
    return(data.table(
      n_blocks = n_blocks, n_strategies = n_strategies,
      statistic = NA_real_, df = NA_integer_, p_value = NA_real_,
      kendalls_w = NA_real_, status = "missing_values"
    ))
  }

  test <- friedman.test(score_matrix)
  q <- unname(test$statistic)
  data.table(
    n_blocks = n_blocks,
    n_strategies = n_strategies,
    statistic = q,
    df = unname(test$parameter),
    p_value = test$p.value,
    kendalls_w = q / (n_blocks * (n_strategies - 1))
  )
}

friedman_results <- final_complete[, friedman_one(.SD), by = condition_cols]
friedman_results[, p_holm_across_conditions := p.adjust(p_value, method = "holm")]
fwrite(friedman_results, file.path(out_dir, "friedman_results.csv"))

# Paired post-hoc tests are only interpreted when the omnibus Friedman test is
# significant. We still write all comparisons so the decision rule is explicit.
pairwise_one <- function(x) {
  strategies <- sort(unique(x$strategy))
  pairs <- combn(strategies, 2L, simplify = FALSE)
  ans <- rbindlist(lapply(pairs, function(pair) {
    a <- x[strategy == pair[[1L]], .(seed, auc_a = test_auc)]
    b <- x[strategy == pair[[2L]], .(seed, auc_b = test_auc)]
    z <- merge(a, b, by = "seed")
    delta <- z$auc_a - z$auc_b
    p <- if (nrow(z) >= 2L && any(delta != 0)) {
      suppressWarnings(wilcox.test(z$auc_a, z$auc_b, paired = TRUE,
                                   exact = FALSE)$p.value)
    } else NA_real_
    data.table(
      strategy_1 = pair[[1L]], strategy_2 = pair[[2L]], n_pairs = nrow(z),
      median_difference_1_minus_2 = median(delta, na.rm = TRUE),
      p_value = p
    )
  }))
  ans[, p_holm := p.adjust(p_value, method = "holm")]
  ans
}

posthoc <- final_complete[, pairwise_one(.SD), by = condition_cols]
fwrite(posthoc, file.path(out_dir, "pairwise_wilcoxon_holm.csv"))

# A winner is the strategy with the highest final test AUC in a complete seed
# block. Exact ties receive fractional wins summing to one within the block.
winners <- final_complete[, {
  best <- max(test_auc, na.rm = TRUE)
  tied <- strategy[test_auc == best]
  .(strategy = tied, fractional_win = 1 / length(tied), best_test_auc = best,
    n_tied = length(tied))
}, by = c(condition_cols, "seed")]

win_counts <- winners[, .(
  wins = sum(fractional_win),
  outright_wins = sum(n_tied == 1L),
  tied_wins = sum(n_tied > 1L)
), by = c(condition_cols, "strategy")]

# Retain strategies with zero wins so plots and tables do not omit them.
win_grid <- unique(final_complete[, c(condition_cols, "strategy"), with = FALSE])
win_counts <- merge(win_grid, win_counts,
                    by = c(condition_cols, "strategy"), all.x = TRUE)
win_counts[is.na(wins), `:=`(wins = 0, outright_wins = 0L, tied_wins = 0L)]
fwrite(winners, file.path(out_dir, "winners_by_seed.csv"))
fwrite(win_counts, file.path(out_dir, "win_counts.csv"))

# Median and interquartile trajectory across seeds at each checkpoint.
trajectory_summary <- checkpoint[, .(
  median_test_auc = median(test_auc, na.rm = TRUE),
  q25 = quantile(test_auc, 0.25, na.rm = TRUE),
  q75 = quantile(test_auc, 0.75, na.rm = TRUE),
  n_seeds = uniqueN(seed)
), by = c(condition_cols, "strategy", "generation")]
fwrite(trajectory_summary, file.path(out_dir, "trajectory_summary.csv"))

safe_name <- function(x) {
  x <- paste(x, collapse = "_")
  gsub("[^A-Za-z0-9._-]+", "-", x)
}

# Write a three-panel PNG for each task/model/hyperparameter condition.
condition_table <- unique(final_complete[, ..condition_cols])
for (i in seq_len(nrow(condition_table))) {
  key <- condition_table[i]
  selector <- rep(TRUE, nrow(final_complete))
  for (col in condition_cols) selector <- selector & final_complete[[col]] == key[[col]]
  f <- final_complete[selector]

  selector_ts <- rep(TRUE, nrow(trajectory_summary))
  for (col in condition_cols) selector_ts <- selector_ts & trajectory_summary[[col]] == key[[col]]
  ts <- trajectory_summary[selector_ts]

  selector_w <- rep(TRUE, nrow(win_counts))
  for (col in condition_cols) selector_w <- selector_w & win_counts[[col]] == key[[col]]
  wc <- win_counts[selector_w]

  title <- paste(
    paste0("task=", key$task), paste0("model=", key$model),
    paste0("pop=", key$pop), paste0("mutation rate=", key$mut_rate),
    paste0("variance=", key$variance), sep = " | "
  )

  p_traj <- ggplot(ts, aes(generation, median_test_auc, colour = strategy,
                           fill = strategy, group = strategy)) +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 0.8) +
    labs(title = title, subtitle = "Median test AUC; ribbon = seed IQR",
         x = "Generation", y = "Test AUC", colour = "Strategy", fill = "Strategy") +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom")

  p_final <- ggplot(f, aes(strategy, test_auc, colour = strategy)) +
    geom_boxplot(outlier.shape = NA, colour = "grey35") +
    geom_point(position = position_jitter(width = 0.10, height = 0), size = 2) +
    labs(title = title, subtitle = "Final checkpoint for each seed",
         x = "Strategy", y = "Final test AUC") +
    theme_bw(base_size = 12) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 35, hjust = 1))

  p_wins <- ggplot(wc, aes(strategy, wins, fill = strategy)) +
    geom_col(width = 0.75) +
    geom_text(aes(label = format(wins, trim = TRUE)), vjust = -0.25) +
    labs(title = title, subtitle = "Fractional wins split exact ties",
         x = "Strategy", y = "Wins across seeds") +
    theme_bw(base_size = 12) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 35, hjust = 1))

  stem <- safe_name(unlist(key, use.names = FALSE))
  ggsave(file.path(out_dir, "condition_plots", paste0(stem, "_trajectory.png")),
         p_traj, width = 10, height = 6, dpi = 300)
  ggsave(file.path(out_dir, "condition_plots", paste0(stem, "_final_auc.png")),
         p_final, width = 9, height = 6, dpi = 300)
  ggsave(file.path(out_dir, "condition_plots", paste0(stem, "_wins.png")),
         p_wins, width = 9, height = 6, dpi = 300)
}

message("Analysis complete. Results written to: ", normalizePath(out_dir))
message("Important: inspect block_completeness.csv before interpreting Friedman results.")
