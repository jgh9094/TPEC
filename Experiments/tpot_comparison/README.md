# TPOT comparison

This folder runs TPOT as a CASH baseline over the same six classifier families used by
the current TPEC experiments: RF, ET, KSVC, GB, KNN, and MLP. Each run searches across
all six families and their project-defined hyperparameter spaces.

## Shared protocol

- Dataset: `Data/SO/combined.csv`
- Outcomes: `LOS_extended`, `discharge_Home`, `HOSP_READM_90`
- Seeds: 0 through 20
- Train/test split: 70/30, stratified by outcome
- Validation: five stratified folds built from the training partition
- Objective: validation ROC-AUC; complexity is not an optimization objective
- Budget: 500 candidate evaluations per run
- Populations: 25, 50, and 100, using TPOT generation counts 20, 10, and 5

Unlike the repository's custom EA, TPOT includes its initial population in the
`generations` limit. These settings therefore request 25x20, 50x10, and 100x5 = 500
candidates, while producing the same 20, 10, and 5 checkpoint resolutions.

Continuous clinical columns are scaled independently inside each CV fold. The final
pipeline is fitted using a preprocessor trained only on the full training partition.

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
