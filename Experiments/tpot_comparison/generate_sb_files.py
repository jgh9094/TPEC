#!/usr/bin/env python3
"""Generate linear and graph TPOT arrays with a shared operator space."""

from pathlib import Path


TPOT_DIRECTORY = Path(__file__).resolve().parent
SEARCH_SPACES = {
    "Constrained": "constrained",
    "Graph": "graph",
}

POP_SIZE = 25
EVOLUTION_GENERATIONS = 19
N_EVALUATIONS = POP_SIZE * (EVOLUTION_GENERATIONS + 1)
MAX_EVAL_TIME_MINS = 5
CLUSTER_ROOT = "/home/sainia3/TPEC"
DATA_PATH = f"{CLUSTER_ROOT}/Data-Tools/spinal_surgery"
RUNNER = f"{CLUSTER_ROOT}/Experiments/tpot_comparison/tpot_wrapper.py"
RESULTS_ROOT = f"{CLUSTER_ROOT}/Results_TPOT_SpinalSurgery_CustomCVObjective"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --mem=200G
#SBATCH --job-name=tpot_{job_suffix}
#SBATCH -p defq,moore
#SBATCH --array=1-63
#SBATCH --output={log_directory}/tpot_%A_%a.out

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpotenv

TRAIN_P=0.70
POP_SIZE={pop_size}
GENERATIONS={generations}
N_EVALUATIONS={n_evaluations}
N_FOLDS=5
MAX_EVAL_TIME_MINS={max_eval_time_mins}
CORES=$SLURM_CPUS_PER_TASK
SEARCH_SPACE={search_space}
DATA_PATH={data_path}
RUNNER={runner}
RESULTS_ROOT={results_root}
TASK_IDS=(LOS_extended discharge_Home HOSP_READM_90)

TASK_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 21 ))
SEED=$(( (SLURM_ARRAY_TASK_ID - 1) % 21 + 1 ))
TASK_ID=${{TASK_IDS[$TASK_INDEX]}}
OUTPUT_DIRECTORY=${{RESULTS_ROOT}}/Pop${{POP_SIZE}}/{search_label}/Task_${{TASK_ID}}/Seed_${{SEED}}/

python -c 'import tpot; print("TPOT version:", tpot.__version__)'

python "$RUNNER" \\
    --seed "$SEED" \\
    --task so \\
    --y_label "$TASK_ID" \\
    --data_path "$DATA_PATH" \\
    --train_p "$TRAIN_P" \\
    --output_directory "$OUTPUT_DIRECTORY" \\
    --search_space "$SEARCH_SPACE" \\
    --classification true \\
    --cores "$CORES" \\
    --pop_size "$POP_SIZE" \\
    --generations "$GENERATIONS" \\
    --n_evaluations "$N_EVALUATIONS" \\
    --n_folds "$N_FOLDS" \\
    --max_eval_time_mins "$MAX_EVAL_TIME_MINS"
"""


def main() -> None:
    assert N_EVALUATIONS == 500
    for search_label, search_space in SEARCH_SPACES.items():
        output_directory = TPOT_DIRECTORY / search_label
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "logs").mkdir(parents=True, exist_ok=True)
        path = output_directory / "tpot.sb"
        path.write_text(
            TEMPLATE.format(
                job_suffix=search_space,
                pop_size=POP_SIZE,
                generations=EVOLUTION_GENERATIONS,
                n_evaluations=N_EVALUATIONS,
                max_eval_time_mins=MAX_EVAL_TIME_MINS,
                search_space=search_space,
                search_label=search_label,
                log_directory=(
                    f"{CLUSTER_ROOT}/Experiments/tpot_comparison/"
                    f"{search_label}/logs"
                ),
                data_path=DATA_PATH,
                runner=RUNNER,
                results_root=RESULTS_ROOT,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
