# TPOT comparison

This folder runs TPOT over the same four-stage linear CASH space used by `Source/CASH`:

1. feature scaler (or passthrough),
2. feature engineering transformer (or passthrough),
3. feature selector (or passthrough), and
4. a mandatory predictor.

Every stage uses the current `Source/ML` registry and its project-defined hyperparameter
space. Classification chooses among RF, ET, KSVC, GB, KNN, and MLP; regression uses the
corresponding regressors with SVR in place of KSVC.

This matches the full CASH pipeline and evaluation space, while intentionally retaining
TPOT's own evolutionary search algorithm. The comparison therefore changes the optimizer,
not the operators, hyperparameter domains, preprocessing policy, or validation objective.

## Shared protocol

- Dataset: `Data/SO/combined.csv`
- Outcomes: `LOS_extended`, `discharge_Home`, `HOSP_READM_90`
- Seeds: 0 through 20
- Train/test split: 70/30, stratified for classification and shuffled for regression
- Validation: five stratified folds for classification or five shuffled folds for regression
- Objective: validation ROC-AUC for classification or R2 for regression; complexity is
  not an optimization objective
- Budget: 500 candidate evaluations per run
- Populations: 25, 50, and 100, using TPOT generation counts 20, 10, and 5

Unlike the repository's custom EA, TPOT includes its initial population in the
`generations` limit. These settings therefore request 25x20, 50x10, and 100x5 = 500
candidates, while producing the same 20, 10, and 5 checkpoint resolutions.

As in CASH, the base preprocessor only numericizes and orders columns. Scaling is an
evolved decision. When protected passthrough columns exist, the evolved scaler is applied
only to the designated continuous clinical columns. Every base preprocessor is fit inside
its CV training fold; the final one is fit only on the full training partition.

## Cluster use

Generate the population-specific job arrays and preview submission:

```bash
python Experiments/tpot_comparison/generate_sb_files.py
Experiments/tpot_comparison/submit_all.sh --dry-run
```

Submit all arrays, or one population:

```bash
Experiments/tpot_comparison/submit_all.sh
Experiments/tpot_comparison/submit_all.sh Pop25
```

`run_tpot_exps.sb` is a directly-submittable Pop25 convenience script. The generated
files under `Pop25`, `Pop50`, and `Pop100` are the canonical full sweep.

The scripts retain the dedicated `tpotenv` environment. TPOT's periodic state is stored
under each run directory so an interrupted array can resume. A run is only treated as
complete when `best_results.json` exists.

## Outputs

Each `Task_<outcome>/Seed_<seed>` directory contains:

- `configuration.json`
- `best_results.json` (completion marker)
- `best_pipeline.pkl`
- `archive.json`
- `checkpoints.csv`
- `results_eval_<N>.json`
- `tpot_state/` (TPOT resume data)
- `failed.json` when a run raises an exception

Summarize a results tree with:

```bash
python Experiments/tpot_comparison/analyze_results.py \
    --results-root /home/hernandezj45/Repos/TPEC/Results_TPOT
```
