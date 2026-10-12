#!/usr/bin/env python3
"""
Generate the SLURM batch (.sb) files for the Optuna HPO baseline sweep.

This is the Optuna counterpart to ``Experiments/HPO/TPEC/generate_sb_files.py``. It keeps every
setting that carries over from the TPEC condition identical (data, task/seed grid, evaluation
budgets, train proportion, initial random-search size, wall time) and swaps the EA-specific
search settings for the one dimension that distinguishes Optuna runs: the sampler.

The sweep varies three dimensions, each encoded by a directory level under
``Experiments/HPO/Optuna``:

    <budget>_Evals / <sampler> / <model>.sb

  * Evaluation budget -> 100_Evals / 200_Evals / 300_Evals   (passed as --n_trials)
  * Sampler           -> CMAES, GP, TPE, Random
  * Model             -> rf, et, ksvc, gb, knn, mlp

That is 3 x 4 x 6 = 72 files, each a 63-task array (3 tasks x 21 seeds).

Budget parity with the TPEC condition
-------------------------------------
``--n_trials`` counts candidates considered, as the EA's POP_SIZE * (GENS + 1) does: a repeated
configuration reuses its stored score instead of being re-fitted. ``--pop_size`` (25, the EA's
POP_SIZE) sets the initial random-search phase (n_startup_trials for TPE/GP) and CatCMA's popsize.
``--n_ei_candidates`` (TPE-only pseudo-offspring) is 20, mirroring the EA's NUM_OFFSPRING=20; it is
passed for every sampler but only the TPE sampler consumes it.

Resources
---------
Optuna evaluates one configuration at a time, so Ray never runs more than its 5 CV folds at once;
5 CPUs per job is enough. TPEC jobs evaluate a whole population in parallel and use 12.

Running this script overwrites any existing .sb files in place. Paths inside the files are the
cluster paths (``/home/hernandezj45/Repos/TPEC/...``), not local repo paths.
"""
import os

# Directory that holds this script (Experiments/HPO/Optuna); files are generated relative to it.
OPTUNA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- factorial dimensions ------------------------------------------------------------

POP_SIZE = 25
N_EI_CANDIDATES = 20

# budget directory -> (--n_trials, short code for the job name)
BUDGET_CONFIGS = {
    "100_Evals": (100, "1h"),
    "200_Evals": (200, "2h"),
    "300_Evals": (300, "3h"),
}

# sampler directory -> (--sampler argument passed to runner.py, short code for the job name)
SAMPLER_CONFIGS = {
    "CMAES": ("cmaes", "cma"),
    "GP": ("gp", "gp"),
    "TPE": ("tpe", "tpe"),
    "Random": ("random", "rnd"),
}

# .sb file stem -> MODEL argument passed to runner.py
MODELS = {
    "rf": "RF",
    "et": "ET",
    "ksvc": "KSVC",
    "gb": "GB",
    "knn": "KNN",
    "mlp": "MLP",
}

# --- constants shared by every file ---------------------------------------------------
DATA_DIRECTORY = "/home/hernandezj45/Repos/TPEC/Data/SO"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/HPO/Optuna/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/Results/HPO/Optuna"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=05:00:00
#SBATCH --mem=50G
#SBATCH --job-name={job_name}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-63

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpe-ea
{sampler_setup}
# standard parameters used for all experiments
TRAIN_P=0.70
MODEL={model}
POP_SIZE={pop_size}
CORES=$SLURM_CPUS_PER_TASK
SAMPLER={sampler}
N_TRIALS={n_trials}
N_EI_CANDIDATES={n_ei_candidates}
DATA_DIRECTORY={data_directory}
# variables that are set by the SLURM job array
# Array of all task IDs from tasks_summary.csv (3 tasks x 21 replicates = 63 jobs)
TASK_IDS=(LOS_extended discharge_Home HOSP_READM_90)

# Calculate which task and seed based on SLURM_ARRAY_TASK_ID (1-63)
# Each task gets 21 replicates (seeds 0-20)
TASK_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 21 ))
SEED=$(( (SLURM_ARRAY_TASK_ID - 1) % 21 ))
# specific task id for the OpenML dataset to be used in this experiment
TASK_ID=${{TASK_IDS[$TASK_INDEX]}}
# Output directory for this specific task and seed
OUTPUT_DIRECTORY={results_root}/{budget_dir}/{sampler_dir}/${{MODEL}}/Task_${{TASK_ID}}/Seed_${{SEED}}/

python {runner} \\
    --seed $SEED \\
    --y_label $TASK_ID \\
    --data_path $DATA_DIRECTORY \\
    --train_p $TRAIN_P \\
    --output_directory $OUTPUT_DIRECTORY \\
    --model $MODEL \\
    --cores $CORES \\
    --sampler $SAMPLER \\
    --n_trials $N_TRIALS \\
    --pop_size $POP_SIZE \\
    --n_ei_candidates $N_EI_CANDIDATES \\
    --task so
"""

# Extra environment setup inserted only into the CMAES files (CatCmawm is loaded via optunahub).
CMAES_SETUP = """
# optunahub imports GitPython, which hard-fails at import without a git
# executable in $PATH ("Bad git executable"). Silence that check; the
# catcmawm module is loaded from the local optunahub cache (see cache_catcmawm.sb).
export GIT_PYTHON_REFRESH=quiet
"""


def main() -> None:
    written = 0
    for budget_dir, (n_trials, budget_short) in BUDGET_CONFIGS.items():
        for sampler_dir, (sampler_arg, samp_short) in SAMPLER_CONFIGS.items():
            out_dir = os.path.join(OPTUNA_DIR, budget_dir, sampler_dir)
            os.makedirs(out_dir, exist_ok=True)
            for stem, model in MODELS.items():
                content = TEMPLATE.format(
                    job_name=f"{budget_short}_{samp_short}_{stem}",
                    sampler_setup=CMAES_SETUP if sampler_arg == "cmaes" else "",
                    model=model,
                    pop_size=POP_SIZE,
                    sampler=sampler_arg,
                    n_trials=n_trials,
                    n_ei_candidates=N_EI_CANDIDATES,
                    data_directory=DATA_DIRECTORY,
                    results_root=RESULTS_ROOT,
                    budget_dir=budget_dir,
                    sampler_dir=sampler_dir,
                    runner=RUNNER,
                )
                with open(os.path.join(out_dir, f"{stem}.sb"), "w") as f:
                    f.write(content)
                written += 1

    n_expected = len(BUDGET_CONFIGS) * len(SAMPLER_CONFIGS) * len(MODELS)
    print(f"Wrote {written} .sb files (expected {n_expected}).")


if __name__ == "__main__":
    main()
