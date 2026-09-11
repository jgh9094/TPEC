#!/usr/bin/env python3
"""
Generate the full factorial of SLURM batch (.sb) files for the HPO sweep.

The sweep varies four dimensions, each encoded by a directory level under
``Experiments/HPO``:

    Pop{25,50,100} / Mut{25,50}_Var{25,50} / <strategy> / <model>.sb

  * Population size  -> Pop25 / Pop50 / Pop100
  * Mutation prob/var -> MutPP_VarVV  (PP, VV in {25, 50} meaning 0.25 / 0.50)
  * Search strategy   -> EA_Exploit, EA_Explore, TPE25, TPE50, TPE75, TPE100
  * Model             -> rf, et, ksvc, gb, knn, mlp

That is 3 x 4 x 6 x 6 = 432 files.

Every run is standardized to a total budget of 500 hard evaluations. With the EA's
convention that total evaluations = POP_SIZE * (GENS + 1) (the initial population plus
one offspring population per generation), GENS is derived per population size:

    Pop25  -> GENS = 19   (25  * 20 = 500)
    Pop50  -> GENS = 9    (50  * 10 = 500)
    Pop100 -> GENS = 4    (100 * 5  = 500)

Checkpointing at each generation is handled inside runner.py / HPO EA (it always writes
checkpoints.csv + per-eval snapshots to the output directory); no extra flag is needed.

Running this script overwrites any existing .sb files in place. Paths inside the files are
the cluster paths (``/home/hernandezj45/Repos/TPEC/...``), not local repo paths.
"""
import os

# Directory that holds this script (Experiments/HPO); files are generated relative to it.
HPO_DIR = os.path.dirname(os.path.abspath(__file__))

# --- factorial dimensions ------------------------------------------------------------

# population size -> (POP_SIZE, GENS) chosen so POP_SIZE * (GENS + 1) == 500
POP_CONFIGS = {
    "Pop25": (25, 19),
    "Pop50": (50, 9),
    "Pop100": (100, 4),
}

# mutation directory -> (MUT_PROB, MUT_VAR, short code for the job name)
MUT_CONFIGS = {
    "Mut25_Var25": ("0.25", "0.25", "M2525"),
    "Mut25_Var50": ("0.25", "0.50", "M2550"),
    "Mut50_Var25": ("0.50", "0.25", "M5025"),
    "Mut50_Var50": ("0.50", "0.50", "M5050"),
}

# strategy directory -> (TPE_PROB, EXPLORE_MUT_SCALE, job-name prefix)
# Values mirror the original Pop25/Mut25_Var25 templates:
#   EA_Exploit : no TPE, small exploration variance (local exploitation)
#   EA_Explore : no TPE, large exploration variance (global exploration)
#   TPExx      : TPE-guided with probability xx%, large exploration variance otherwise
STRATEGY_CONFIGS = {
    "EA_Exploit": ("0.0", "0.50", "explo"),
    "EA_Explore": ("0.0", "2.0", "explr"),
    "TPE25": ("0.25", "2.0", "25"),
    "TPE50": ("0.50", "2.0", "50"),
    "TPE75": ("0.75", "2.0", "75"),
    "TPE100": ("1.0", "2.0", "1h"),
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
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/HPO/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/Results"

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
MODEL={model}
GENS={gens}
POP_SIZE={pop_size}
CORES=$SLURM_CPUS_PER_TASK
MUT_PROB={mut_prob}
MUT_VAR={mut_var}
TPE_PROB={tpe_prob}
TOURNAMENT_SIZE=2
NUM_OFFSPRING=20
GAMMA=0.3
TPE_MUT_SCALE=0.50
EXPLORE_MUT_SCALE={explore_mut_scale}
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
OUTPUT_DIRECTORY={results_root}/{pop_dir}/{mut_dir}/{strat_dir}/${{MODEL}}/Task_${{TASK_ID}}/Seed_${{SEED}}/

python {runner} \\
    --seed $SEED \\
    --y_label $TASK_ID \\
    --data_path $DATA_DIRECTORY \\
    --train_p $TRAIN_P \\
    --output_directory $OUTPUT_DIRECTORY \\
    --model $MODEL \\
    --gens $GENS \\
    --pop_size $POP_SIZE \\
    --cores $CORES \\
    --mut_prob $MUT_PROB \\
    --mut_var $MUT_VAR \\
    --tpe_prob $TPE_PROB \\
    --tournament_size $TOURNAMENT_SIZE \\
    --num_offspring $NUM_OFFSPRING \\
    --gamma $GAMMA \\
    --tpe_mut_scale $TPE_MUT_SCALE \\
    --explore_mut_scale $EXPLORE_MUT_SCALE \\
    --task so
"""


def main() -> None:
    written = 0
    for pop_dir, (pop_size, gens) in POP_CONFIGS.items():
        # sanity: enforce the 500-evaluation budget
        assert pop_size * (gens + 1) == 500, f"{pop_dir}: {pop_size}*({gens}+1) != 500"
        for mut_dir, (mut_prob, mut_var, mut_short) in MUT_CONFIGS.items():
            for strat_dir, (tpe_prob, explore_scale, strat_prefix) in STRATEGY_CONFIGS.items():
                out_dir = os.path.join(HPO_DIR, pop_dir, mut_dir, strat_dir)
                os.makedirs(out_dir, exist_ok=True)
                for stem, model in MODELS.items():
                    job_name = f"P{pop_size}_{mut_short}_{strat_prefix}_{stem}"
                    content = TEMPLATE.format(
                        job_name=job_name,
                        model=model,
                        gens=gens,
                        pop_size=pop_size,
                        mut_prob=mut_prob,
                        mut_var=mut_var,
                        tpe_prob=tpe_prob,
                        explore_mut_scale=explore_scale,
                        data_directory=DATA_DIRECTORY,
                        results_root=RESULTS_ROOT,
                        pop_dir=pop_dir,
                        mut_dir=mut_dir,
                        strat_dir=strat_dir,
                        runner=RUNNER,
                    )
                    path = os.path.join(out_dir, f"{stem}.sb")
                    with open(path, "w") as f:
                        f.write(content)
                    written += 1

    n_expected = len(POP_CONFIGS) * len(MUT_CONFIGS) * len(STRATEGY_CONFIGS) * len(MODELS)
    print(f"Wrote {written} .sb files (expected {n_expected}).")


if __name__ == "__main__":
    main()
