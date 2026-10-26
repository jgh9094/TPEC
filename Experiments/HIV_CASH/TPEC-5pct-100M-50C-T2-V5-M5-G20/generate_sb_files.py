#!/usr/bin/env python3
"""
Generate the SLURM batch (.sb) files for the TPEC (TPE-guided EA) CASH sweep.

The sweep varies two dimensions, each encoded by a directory level under
``Experiments/HIV_CASH/TPEC-5pct``:

    <budget>_Evals / <strategy> / cash.sb

  * Evaluation budget -> 200_Evals / 300_Evals
  * Search strategy   -> EA, TPE25, TPE50, TPE100

There is no model dimension: CASH evolves the whole pipeline, including the estimator. That is
2 x 4 = 8 files, each a 21-task array (one array task per seed of the single target).

The budget counts candidates considered: the EA's initial population plus one offspring population
per generation, POP_SIZE * (GENS + 1). With POP_SIZE = 25, GENS is derived per budget:

    200_Evals  -> GENS = 7    (25 * 8  = 200)
    300_Evals  -> GENS = 11   (25 * 12 = 300)

The data, tasks, seeds, train proportion and EA settings match the HPO TPEC sweep
(Experiments/HPO/TPEC/generate_sb_files.py); COMPONENT_MUT_PROB is CASH-only.

Running this script overwrites any existing .sb files in place. Paths inside the files are
the cluster paths (``/home/hernandezj45/Repos/TPEC/...``), not local repo paths.
"""
import os

# Directory that holds this script (Experiments/HIV_CASH/TPEC-5pct); files are generated relative to it.
TPEC_DIR = os.path.dirname(os.path.abspath(__file__))

# --- factorial dimensions ------------------------------------------------------------

POP_SIZE = 25

# the single target (y_label) used for every run in this sweep
TASK_ID = "virologic_failure"

# replicates: seeds 0 .. N_SEEDS - 1, one array task per seed
N_SEEDS = 21

# budget directory -> total candidates considered
BUDGET_CONFIGS = {
    "200_Evals": 200,
    "300_Evals": 300,
}

# strategy directory -> (TPE_PROB, short code for the job name)
#   EA    : no TPE -- every offspring resamples its parameters uniformly (unbiased exploration)
#   TPExx : TPE-guided with probability xx%; TPE picks the best of several random candidates,
#           otherwise a single random candidate is used
STRATEGY_CONFIGS = {
    "EA": ("0.0", "ea"),
    "TPE25": ("0.25", "t25"),
    "TPE50": ("0.50", "t50"),
    "TPE100": ("1.0", "t100"),
}

# --- constants shared by every file ---------------------------------------------------
DATA_DIRECTORY = "/home/hernandezj45/Repos/TPEC/Data/HIV/ecoxai_dataset_5pct_no_PHI.csv"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/HIV_CASH/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/All_Results/Results_HIV_5pct_T2_G20/CASH"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --job-name={job_name}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-{n_seeds}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpe-ea

# standard parameters used for all experiments
TRAIN_P=0.80
GENS={gens}
POP_SIZE={pop_size}
CORES=$SLURM_CPUS_PER_TASK
MUT_PROB=0.25
CROSSOVER_PROB=0.5
COMPONENT_MUT_PROB=0.25
TPE_PROB={tpe_prob}
TOURNAMENT_SIZE=2
NUM_OFFSPRING=20
GAMMA=0.20
DATA_DIRECTORY={data_directory}
# variables that are set by the SLURM job array
# Each array task is one seed of the single target ({n_seeds} jobs total)
TASK_ID={task_id}

# SLURM_ARRAY_TASK_ID runs 1-{n_seeds}, one seed per array task (seeds 0-{max_seed})
SEED=$(( SLURM_ARRAY_TASK_ID - 1 ))
# Output directory for this specific task and seed
OUTPUT_DIRECTORY={results_root}/{budget_dir}/{strat_dir}/Task_${{TASK_ID}}/Seed_${{SEED}}/

python {runner} \\
    --seed $SEED \\
    --y_label $TASK_ID \\
    --data_path $DATA_DIRECTORY \\
    --train_p $TRAIN_P \\
    --output_directory $OUTPUT_DIRECTORY \\
    --gens $GENS \\
    --pop_size $POP_SIZE \\
    --cores $CORES \\
    --mut_prob $MUT_PROB \\
    --crossover_prob $CROSSOVER_PROB \\
    --component_mut_prob $COMPONENT_MUT_PROB \\
    --tpe_prob $TPE_PROB \\
    --tournament_size $TOURNAMENT_SIZE \\
    --num_offspring $NUM_OFFSPRING \\
    --gamma $GAMMA
"""


def main() -> None:
    written = 0
    for budget_dir, budget in BUDGET_CONFIGS.items():
        assert budget % POP_SIZE == 0, f"{budget_dir}: budget must be a multiple of POP_SIZE"
        gens = budget // POP_SIZE - 1
        for strat_dir, (tpe_prob, strat_short) in STRATEGY_CONFIGS.items():
            out_dir = os.path.join(TPEC_DIR, budget_dir, strat_dir)
            os.makedirs(out_dir, exist_ok=True)
            content = TEMPLATE.format(
                job_name=f"{budget}_{strat_short}_cash",
                gens=gens,
                pop_size=POP_SIZE,
                tpe_prob=tpe_prob,
                task_id=TASK_ID,
                n_seeds=N_SEEDS,
                max_seed=N_SEEDS - 1,
                data_directory=DATA_DIRECTORY,
                results_root=RESULTS_ROOT,
                budget_dir=budget_dir,
                strat_dir=strat_dir,
                runner=RUNNER,
            )
            with open(os.path.join(out_dir, "cash.sb"), "w") as f:
                f.write(content)
            written += 1

    n_expected = len(BUDGET_CONFIGS) * len(STRATEGY_CONFIGS)
    print(f"Wrote {written} .sb files (expected {n_expected}).")


if __name__ == "__main__":
    main()
