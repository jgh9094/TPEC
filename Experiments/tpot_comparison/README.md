# TPOT comparison

This experiment compares two TPOT pipeline topologies while keeping the operator families,
hyperparameter spaces, data split, objectives, budget, outcomes, and seeds fixed:

- `Constrained`: the four-stage CASH-compatible linear pipeline
  `scaler -> transformer -> selector -> predictor` using the current `Source/ML` spaces.
- `Graph`: a flexible directed-acyclic graph whose root is a predictor and whose inner nodes
  may be predictors, scalers, transformers, or selectors from those same `Source/ML` spaces.

Both conditions use the same project-defined hyperparameter ranges and conversion logic,
including the same column-aware scaler wrappers. `PolynomialFeatures` is explicitly excluded
from both search spaces. The intended experimental difference is pipeline topology: fixed
four-stage linear pipelines versus variable DAGs with up to 15 nodes.

## Protocol

- Dataset: the coordinated NumPy arrays in `Data-Tools/spinal_surgery/`
- Outcomes: `LOS_extended`, `discharge_Home`, and `HOSP_READM_90`
- Seeds: 1 through 21
- Runs: 63 per search space, 126 total
- Train/test split: 70/30, stratified for classification
- Cross-validation: five stratified folds on the training partition
- Population size: 25
- Evolutionary generations after the initial population: 19
- TPOT generation argument: 20, because current TPOT counts the initial population as
  generation 0
- Requested evaluations: `25 * (19 + 1) = 500` per run
- Objectives: maximize validation AUROC and minimize TPOT's fitted-model
  `complexity_scorer`, returned together by one custom CV objective
- Final pipeline: highest validation AUROC, with lower complexity breaking exact AUROC ties
- Operator and hyperparameter source: the current `Source/ML` registries in both conditions

Data are loaded directly from `X.npy`, `y.npy`, `feature_names.npy`, and
`output_names.npy`; `combined.csv` is not used. `patient_info.npy` is deliberately excluded
so patient identifiers and surgery timestamps cannot become modeling features. The stored
`los_greater_than_7d` outcome is exposed under the experiment's established
`LOS_extended` name, and the three lab names containing spaces are normalized to the
underscore names expected by the preprocessing configuration. The 14 designated continuous
columns are placed first and passed through unscaled, and all remaining columns pass through
unchanged. Before optimization,
the fixed CASH preprocessor is fitted independently outside the candidate pipeline on every
outer CV training fold and used to transform only that fold's training and validation
partitions. A custom TPOT `other_objective_functions` callback
evaluates every candidate on these same cached fold matrices and returns both mean validation
performance and mean fitted complexity. Preprocessing is never inserted into the candidate ML
pipeline. After selection, the outer-training preprocessor and selected pipeline are fitted and
saved as separate artifacts. Scaling remains an evolved decision in both spaces.

## HPC use

The experiment is pinned to TPOT 1.1.0, the latest PyPI release when this configuration was
created. Install it in the existing `tpotenv` environment before submission rather than
installing packages independently inside each job:

```bash
conda activate tpotenv
python -m pip install -r Experiments/tpot_comparison/requirements.txt
python -c 'import tpot; print(tpot.__version__)'
```

Generate and preview the two arrays:

```bash
python Experiments/tpot_comparison/generate_sb_files.py
Experiments/tpot_comparison/submit_all.sh --dry-run
```

Submit both arrays or one search space:

```bash
Experiments/tpot_comparison/submit_all.sh
Experiments/tpot_comparison/submit_all.sh Constrained
Experiments/tpot_comparison/submit_all.sh Graph
```

The canonical batch files are `Constrained/tpot.sb` and `Graph/tpot.sb`. Each is a 63-task
array covering three outcomes and 21 seeds. A run is complete only when `best_results.json`
exists and its search-space revision matches the current protocol. The arrays request four
days of wall time, and each candidate pipeline evaluation has a five-minute limit by default.
On resubmission, current completed runs exit immediately; incomplete,
failed, or older-revision runs have their TPOT-owned artifacts removed and restart from
scratch. TPOT checkpoint resumption is intentionally disabled because TPOT
1.1.0 can repeat the full generation budget after loading a partial population. A
per-replicate file lock prevents two submissions from writing the same run concurrently.

## Outputs

Each `Pop25/<search-space>/Task_<outcome>/Seed_<seed>` directory contains:

- `configuration.json`
- `fit_timing.json`, measuring wall time spent inside `TPOTEstimator.fit()`
- `best_results.json`, written last as the completion marker
- `evaluation_budget.json`, confirming that exactly 500 distinct pipelines were evaluated
- `best_pipeline.json`, with the selected pipeline representation and deep parameters
- `best_pipeline.pkl`
- `preprocessor.pkl`, fitted on the outer training partition and intentionally separate from
  `best_pipeline.pkl`
- `archive.json`, including TPOT evaluation errors when exposed
- `pareto_front.json`
- `checkpoints.csv`
- `results_eval_<N>.json`
- `failed.json` if the run raises an exception

`best_results.json` records the complete fitted pipeline representation, deep parameters,
train/CV/test AUROC, AUPRC, balanced accuracy, complexity, TPOT version, fit time,
Pareto-front size, and counts of invalid or timed-out evaluations when `Eval Error` is
available. AUROC remains the optimization objective; AUPRC and balanced accuracy are
post-selection diagnostics for the final pipeline.

Summarize both result sets with:

```bash
python Experiments/tpot_comparison/analyze_results.py \
    --results-root /home/sainia3/TPEC/Results_TPOT_SpinalSurgery_CustomCVObjective
```

In addition to the printed aggregate summary and half-violin plot, the analyzer writes
`Experiments/tpot_comparison/tpot_run_summary.csv`. This file contains one row per completed
replicate with total evaluations, condition, task, seed, train/validation/test AUROC,
complexity, the evaluation and generation where the selected pipeline first appeared, and
TPOT fit runtime. It includes both the 100-evaluation (`Constrained_100`, `Graph_100`) and
500-evaluation (`Constrained`, `Graph`) result directories by default. The CSV condition names
are `tpot_c` for constrained pipelines and `tpot_g` for graph pipelines; the `evals` column
distinguishes the budgets. Use `--run-output-csv PATH` to choose a different destination and
`--run-search-spaces ...` to override the included directories. The initial population is
reported as generation `1`.

The custom-CV experiment uses a separate results root so completed results from earlier
TPOT preprocessing protocols cannot be reused or mixed into this comparison.
