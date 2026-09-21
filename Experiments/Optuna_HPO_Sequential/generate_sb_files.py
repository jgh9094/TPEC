#!/usr/bin/env python3
"""
Generate the full factorial of SLURM batch (.sb) files for the Optuna HPO sweep.

This is the Optuna-baseline counterpart to ``Experiments/HPO/generate_sb_files.py``. It keeps
every experimental knob that carries over from the TPE-EA condition identical (data, task/seed
grid, per-run evaluation budget, SLURM resources) and swaps the EA-specific search dimensions
(mutation probability/variance, tournament, TPE-vs-explore strategy) for the one dimension that
distinguishes Optuna runs: the sampler.

The sweep varies three dimensions, each encoded by a directory level under
``Experiments/Optuna_HPO``:

    Pop{25,50} / <sampler> / <model>.sb

  * Population size -> Pop25 / Pop50   (the ask/tell generation/batch size)
  * Sampler         -> CMAES, GP, TPE, Random
  * Model           -> rf, et, ksvc, gb, knn, mlp

That is 2 x 4 x 6 = 48 files.

Budget parity with the EA condition
-----------------------------------
Every run is standardized to the same total budget of 500 hard evaluations. The Optuna runner
takes the budget directly as ``--n_trials`` (total trials) and ``--pop_size`` as the
generation/batch size; it derives the number of generations as ceil(n_trials / pop_size). This
reproduces the EA's ``POP_SIZE * (GENS + 1) == 500`` structure:

    Pop25  -> 500 trials, batch 25  -> 20 generations
    Pop50  -> 500 trials, batch 50  -> 10 generations

``--n_ei_candidates`` (TPE-only pseudo-offspring) is fixed at 20 to mirror the EA's
NUM_OFFSPRING=20; it is passed for every sampler but only the TPE sampler consumes it. Everything
else (train proportion, task IDs, seeds, cores, memory, wall time) matches the EA condition.

Checkpointing at each generation is handled inside runner.py (it always writes checkpoints.csv +
per-generation snapshots to the output directory); no extra flag is needed.

Running this script overwrites any existing .sb files in place. Paths inside the files are the
cluster paths (``/home/hernandezj45/Repos/TPEC/...``), not local repo paths. Optuna results are
written to a separate ``Results_Optuna`` root so they never intermix with the EA's ``Results`` tree.
"""
import os

# Directory that holds this script (Experiments/Optuna_HPO); files are generated relative to it.
OPTUNA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- factorial dimensions ------------------------------------------------------------

# population size -> POP_SIZE (the ask/tell generation/batch size). N_TRIALS below is the fixed
# 500-evaluation budget, so generations = ceil(N_TRIALS / POP_SIZE) == the EA's GENS + 1.
POP_CONFIGS = {
    "Pop25": 25,
    "Pop50": 50,
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
DATA_DIRECTORY = "/home/hernandezj45/Repos/TPEC/Data/SO/combined.csv"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/Optuna_HPO/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/Results_Optuna"

# total hard-evaluation budget per run (parity with the EA's 500) and TPE pseudo-offspring count
# (mirrors the EA's NUM_OFFSPRING=20).
N_TRIALS = 500
N_EI_CANDIDATES = 20

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=05:00:00
#SBATCH --mem=200G
#SBATCH --job-name={job_name}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-63

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpe-ea

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
OUTPUT_DIRECTORY={results_root}/{pop_dir}/{sampler_dir}/${{MODEL}}/Task_${{TASK_ID}}/Seed_${{SEED}}/

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


def main() -> None:
    written = 0
    for pop_dir, pop_size in POP_CONFIGS.items():
        for sampler_dir, (sampler_arg, samp_short) in SAMPLER_CONFIGS.items():
            out_dir = os.path.join(OPTUNA_DIR, pop_dir, sampler_dir)
            os.makedirs(out_dir, exist_ok=True)
            for stem, model in MODELS.items():
                job_name = f"P{pop_size}_{samp_short}_{stem}"
                content = TEMPLATE.format(
                    job_name=job_name,
                    model=model,
                    pop_size=pop_size,
                    sampler=sampler_arg,
                    n_trials=N_TRIALS,
                    n_ei_candidates=N_EI_CANDIDATES,
                    data_directory=DATA_DIRECTORY,
                    results_root=RESULTS_ROOT,
                    pop_dir=pop_dir,
                    sampler_dir=sampler_dir,
                    runner=RUNNER,
                )
                path = os.path.join(out_dir, f"{stem}.sb")
                with open(path, "w") as f:
                    f.write(content)
                written += 1

    n_expected = len(POP_CONFIGS) * len(SAMPLER_CONFIGS) * len(MODELS)
    print(f"Wrote {written} .sb files (expected {n_expected}).")


if __name__ == "__main__":
    main()
