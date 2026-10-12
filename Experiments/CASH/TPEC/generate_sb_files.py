#!/usr/bin/env python3
"""
Generate the SLURM batch (.sb) files for the TPEC (TPE-guided EA) CASH sweep.

The sweep varies two dimensions, each encoded by a directory level under
``Experiments/CASH/TPEC``:

    <budget>_Evals / <strategy> / cash.sb

  * Evaluation budget -> 100_Evals / 200_Evals / 300_Evals
  * Search strategy   -> EA, TPE25, TPE50, TPE100

There is no model dimension: CASH evolves the whole pipeline, including the estimator. That is
3 x 4 = 12 files, each a 63-task array (3 tasks x 21 seeds).

The budget counts candidates considered: the EA's initial population plus one offspring population
per generation, POP_SIZE * (GENS + 1). With POP_SIZE = 25, GENS is derived per budget:

    100_Evals -> GENS = 3    (25 * 4  = 100)
    200_Evals -> GENS = 7    (25 * 8  = 200)
    300_Evals -> GENS = 11   (25 * 12 = 300)

The data, tasks, seeds, train proportion and EA settings match the HPO TPEC sweep
(Experiments/HPO/TPEC/generate_sb_files.py); COMPONENT_MUT_PROB is CASH-only.

Running this script overwrites any existing .sb files in place. Paths inside the files are
the cluster paths (``/home/hernandezj45/Repos/TPEC/...``), not local repo paths.
"""
import os

# Directory that holds this script (Experiments/CASH/TPEC); files are generated relative to it.
TPEC_DIR = os.path.dirname(os.path.abspath(__file__))

# --- factorial dimensions ------------------------------------------------------------

POP_SIZE = 25

# budget directory -> (total candidates considered, short code for the job name)
BUDGET_CONFIGS = {
    "100_Evals": (100, "1h"),
    "200_Evals": (200, "2h"),
    "300_Evals": (300, "3h"),
}

# strategy directory -> (TPE_PROB, short code for the job name)
#   EA    : no TPE -- every offspring resamples its parameters uniformly (unbiased exploration)
#   TPExx : TPE-guided with probability xx%; TPE offspring take a small local shift (mut_var),
#           non-TPE offspring resample uniformly at random
STRATEGY_CONFIGS = {
    "EA": ("0.0", "ea"),
    "TPE25": ("0.25", "t25"),
    "TPE50": ("0.50", "t50"),
    "TPE100": ("1.0", "t100"),
}

# --- constants shared by every file ---------------------------------------------------
DATA_DIRECTORY = "/home/hernandezj45/Repos/TPEC/Data/SO"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/CASH/TPEC/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/Results/CASH/TPEC"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --mem=200G
#SBATCH --job-name={job_name}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-63

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpe-ea

# standard parameters used for all experiments
TRAIN_P=0.70
CLASSIFICATION=true
GENS={gens}
POP_SIZE={pop_size}
CORES=$SLURM_CPUS_PER_TASK
MUT_PROB=0.25
MUT_VAR=0.25
CROSSOVER_PROB=0.5
COMPONENT_MUT_PROB=0.1
TPE_PROB={tpe_prob}
TOURNAMENT_SIZE=2
NUM_OFFSPRING=20
GAMMA=0.3
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
OUTPUT_DIRECTORY={results_root}/{budget_dir}/{strat_dir}/Task_${{TASK_ID}}/Seed_${{SEED}}/

python {runner} \\
    --seed $SEED \\
    --y_label $TASK_ID \\
    --data_path $DATA_DIRECTORY \\
    --train_p $TRAIN_P \\
    --output_directory $OUTPUT_DIRECTORY \\
    --classification $CLASSIFICATION \\
    --gens $GENS \\
    --pop_size $POP_SIZE \\
    --cores $CORES \\
    --mut_prob $MUT_PROB \\
    --mut_var $MUT_VAR \\
    --crossover_prob $CROSSOVER_PROB \\
    --component_mut_prob $COMPONENT_MUT_PROB \\
    --tpe_prob $TPE_PROB \\
    --tournament_size $TOURNAMENT_SIZE \\
    --num_offspring $NUM_OFFSPRING \\
    --gamma $GAMMA \\
    --task so
"""


def main() -> None:
    written = 0
    for budget_dir, (budget, budget_short) in BUDGET_CONFIGS.items():
        assert budget % POP_SIZE == 0, f"{budget_dir}: budget must be a multiple of POP_SIZE"
        gens = budget // POP_SIZE - 1
        for strat_dir, (tpe_prob, strat_short) in STRATEGY_CONFIGS.items():
            out_dir = os.path.join(TPEC_DIR, budget_dir, strat_dir)
            os.makedirs(out_dir, exist_ok=True)
            content = TEMPLATE.format(
                job_name=f"{budget_short}_{strat_short}_cash",
                gens=gens,
                pop_size=POP_SIZE,
                tpe_prob=tpe_prob,
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
