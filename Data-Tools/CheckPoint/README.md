# CheckPoint analysis

Tools for analysing the per-generation **best-so-far trajectories** produced by
the HPO runs (the `checkpoints.csv` file written in every
`Results/.../Seed_<n>/` directory). See
[`Results/README.md`](../../Results/README.md) for the run layout and the
experimental factors (population size, mutation/variance, strategy, model, task,
seed).

## Files

| File                          | What it does                                                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `aggregator.py`               | Walks `Results/` and stacks every `checkpoints.csv` into one tidy long-format CSV.                              |
| `checkpoint_trajectories.csv` | Output of `aggregator.py`: one row per (run, generation). **Input to both reports.**                           |
| `trajectory_analysis.Rmd`     | Best-so-far Test AUC vs accumulated `hard_evals` — median line + Q1–Q3 ribbon, one line per strategy.          |
| `eval_performance_analysis.Rmd` | Test AUC distributions across strategies at a chosen evaluation budget, with five-number summaries and Friedman tests. |
| `render_all.R`                | Knits both reports to PDF for every model.                                                                      |
| `reports/`                    | Rendered PDFs: `trajectory_analysis_<MODEL>.pdf` and `<EVALS>_eval_performance_analysis_<MODEL>.pdf`.          |

## `checkpoint_trajectories.csv`

One row per (run, generation). Columns:

| Column       | Meaning                                                                 |
| ------------ | ----------------------------------------------------------------------- |
| `pop`        | population size (`25`, `50`)                                             |
| `mut_rate`   | per-hyperparameter mutation probability, as a decimal (`0.25`, `0.50`)  |
| `variance`   | Gaussian mutation variance, as a decimal (`0.25`, `0.50`)               |
| `strategy`   | `TPE25`/`TPE50`/`TPE75`/`TPE100`/`EA_Explore`/`EA_Exploit`              |
| `tpe_prob`   | probability of TPE-guided offspring (decimal; `0.0` for pure EA)        |
| `model`      | `RF` / `ET` / `GB` / `KNN` / `KSVC` / `MLP`                             |
| `task`       | `HOSP_READM_90` / `LOS_extended` / `discharge_Home`                     |
| `seed`       | replicate seed `0`–`20`                                                  |
| `generation` | generation index (`-1` = initial population)                            |
| `hard_evals` | accumulated hard evaluations at that generation                         |
| `val_auc`    | validation AUC of the best-so-far individual                            |
| `train_auc`  | its training AUC                                                        |
| `test_auc`   | its held-out test AUC                                                   |

## Reports

- **`trajectory_analysis.Rmd`** — for each (task, population size,
  mutation/variance) configuration, a single-panel figure of the best-so-far
  **Test AUC** over accumulated `hard_evals`. Each line is the median across the
  21 seeds; the shaded band is the inter-quartile range (Q1–Q3). Lines are
  coloured by search strategy. Population size sets the checkpoint resolution
  (Pop 25 → 20 points, Pop 50 → 10 points), so each population keeps its own
  `hard_evals` grid.
- **`eval_performance_analysis.Rmd`** — the **Test AUC** distribution across
  strategies at a chosen evaluation budget (`evals` parameter, default `500`), as
  raincloud plots faceted by population size and mutation/variance, plus
  five-number summary tables and Friedman tests across strategies (paired by
  seed) within each population/mutation-variance cell. `evals` must land on the
  checkpoint grid (Pop 25 → 25, 50, …, 500; Pop 50 → 50, 100, …, 500); a value a
  population never checkpoints at simply drops that population from the report.

Both reports take a `model` parameter (default `"RF"`) and cover all three tasks
in one document; `eval_performance_analysis.Rmd` additionally takes `evals`.

## Generating the reports

1. **Build the input CSV** (only needed when `Results/` changes):

   ```bash
   python Data-Tools/CheckPoint/aggregator.py
   ```

2. **Knit the reports.** `pandoc` is bundled inside RStudio rather than on the
   PATH, so point R at it first, then render both reports for all six models:

   ```bash
   export RSTUDIO_PANDOC="/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/$(uname -m | sed 's/arm64/aarch64/')"
   Rscript Data-Tools/CheckPoint/render_all.R
   ```

   Compare strategies at a different evaluation budget (default `500`) with
   `--evals`; the value is baked into the performance-report file name (e.g.
   `250_eval_performance_analysis_RF.pdf`):

   ```bash
   Rscript Data-Tools/CheckPoint/render_all.R --evals 250
   ```

   Restrict to specific models by passing them as arguments (after any
   `--evals` flag):

   ```bash
   Rscript Data-Tools/CheckPoint/render_all.R --evals 250 RF MLP
   ```

   PDFs are written to `Data-Tools/CheckPoint/reports/`.

Alternatively, open either `.Rmd` in **RStudio** and click **Knit** (RStudio
supplies pandoc automatically); use *Knit with Parameters* to pick the model
(and, for `eval_performance_analysis.Rmd`, the evaluation budget).
