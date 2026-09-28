#!/usr/bin/env python3
"""Generate the three 500-evaluation TPOT SLURM job arrays."""

from pathlib import Path


TPOT_DIRECTORY = Path(__file__).resolve().parent
POP_CONFIGS = {
    "Pop25": (25, 20),
    "Pop50": (50, 10),
    "Pop100": (100, 5),
}

DATA_PATH = "/home/hernandezj45/Repos/TPEC/Data/SO/combined.csv"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/tpot_comparison/tpot_wrapper.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/Results_TPOT"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --mem=200G
#SBATCH --job-name=tpot_P{pop_size}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-63

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpotenv

TRAIN_P=0.70
POP_SIZE={pop_size}
GENERATIONS={generations}
N_EVALUATIONS=500
N_FOLDS=5
MAX_EVAL_TIME_MINS=60
CORES=$SLURM_CPUS_PER_TASK
DATA_PATH={data_path}
RUNNER={runner}
RESULTS_ROOT={results_root}
TASK_IDS=(LOS_extended discharge_Home HOSP_READM_90)

TASK_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 21 ))
SEED=$(( (SLURM_ARRAY_TASK_ID - 1) % 21 ))
TASK_ID=${{TASK_IDS[$TASK_INDEX]}}
OUTPUT_DIRECTORY=${{RESULTS_ROOT}}/Pop${{POP_SIZE}}/TPOT_CASH/Task_${{TASK_ID}}/Seed_${{SEED}}/

python "$RUNNER" \\
    --seed "$SEED" \\
    --task so \\
    --y_label "$TASK_ID" \\
    --data_path "$DATA_PATH" \\
    --train_p "$TRAIN_P" \\
    --output_directory "$OUTPUT_DIRECTORY" \\
    --classification true \\
    --cores "$CORES" \\
    --pop_size "$POP_SIZE" \\
    --generations "$GENERATIONS" \\
    --n_evaluations "$N_EVALUATIONS" \\
    --n_folds "$N_FOLDS" \\
    --max_eval_time_mins "$MAX_EVAL_TIME_MINS"
"""


def main() -> None:
    for pop_directory, (pop_size, generations) in POP_CONFIGS.items():
        # TPOT's generation count includes the initial population.
        assert pop_size * generations == 500
        output_directory = TPOT_DIRECTORY / pop_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "tpot.sb"
        path.write_text(
            TEMPLATE.format(
                pop_size=pop_size,
                generations=generations,
                data_path=DATA_PATH,
                runner=RUNNER,
                results_root=RESULTS_ROOT,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
