#!/usr/bin/env python3
"""
Generate the SLURM batch (.sb) files for the CASH sweep over the OpenML suite 271 classification tasks.

Why this sweep: the HIV sweeps showed the TPE improves validation AUROC on one dataset, but the test
effect could not be resolved there. Running the same comparison over 35 datasets makes the dataset the
unit of analysis, which gives far more power than more seeds on one dataset.

Conditions (each crossed with every task in tasks_summary.csv):

  * tournament size in TOURNAMENT_SIZES (2 and 5);
  * EA: tpe_offspring = 0. The TPE is never fitted, so gamma cannot affect it; it runs once per
    tournament size (gamma 0.0 is passed and ignored), not once per gamma;
  * TPE: tpe_offspring in {0.5, 1.0} x gamma in GAMMAS (0.20, 0.30).

That is 2 x (1 + 2 x 2) = 10 conditions x 35 tasks = 350 .sb files. Each file is an N_SEEDS-task array
(one array task per seed), so 3,850 runs in total.

Layout (under this folder):

    T<t>/EA/task_<id>.sb                    EA conditions
    T<t>/G<gamma*100>/O<off*100>/task_<id>.sb   TPE conditions, e.g. T2/G20/O50/task_359963.sb

Results (a new root, so nothing overwrites earlier sweeps). The runner adds task_<id>/ itself:

    <RESULTS_ROOT>/<condition>/Seed_<seed>/task_<id>/

Budget: POP_SIZE * (GENS + 1) = 20 * 10 = 200 evaluations. Every run writes checkpoints.csv after each
generation, so the 100-evaluation result comes from the same run (the row with evaluations == 100).

Each file runs ONE task (``--task_ids``), so the large datasets do not hold up the small ones and a
slow or failing task only affects its own jobs. Wall-clock limits are tiered by dataset size
(TIME_LIMITS); adjust them to the cluster's limits.

Running this script overwrites existing .sb files in place. Paths inside the files are cluster paths
(``/home/hernandezj45/Repos/TPEC/...``), not local repo paths.

Usage:
    python Experiments/OpenML-CASH/generate_sb_files.py
"""
import csv
import os

# Directory that holds this script; the .sb tree is written beneath it.
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SWEEP_DIR, os.pardir, os.pardir))
# Where tasks_summary.csv (the task list and row counts) is read from. The HPC copy comes first, because
# the HPC repo has no Experiments/Data-Preprocessing folder; locally the repo copy is used.
SUITE_SOURCE_DIRS = (
    "/home/hernandezj45/Repos/TPEC/Data/OpenML_Suite_271_Classification_CSV",
    os.path.join(REPO_ROOT, "Experiments", "Data-Preprocessing", "OpenML_Suite_271_Classification_CSV"),
)

def find_suite_dir() -> str:
    """The first suite folder that contains tasks_summary.csv."""
    for candidate in SUITE_SOURCE_DIRS:
        if os.path.isfile(os.path.join(candidate, "tasks_summary.csv")):
            return candidate
    raise FileNotFoundError(f"tasks_summary.csv not found in any of: {SUITE_SOURCE_DIRS}")


# --- settings you may edit -------------------------------------------------------------

POP_SIZE = 20
# evaluations per run (candidates considered, duplicates included)
BUDGET = 200
# GENS is derived so that POP_SIZE * (GENS + 1) == BUDGET (20 * 10 = 200)
GENS = BUDGET // POP_SIZE - 1

TRAIN_P = 0.70
N_FOLDS = 5
MUT_PROB = 0.5
CROSSOVER_PROB = 0.5
COMPONENT_MUT_PROB = 0.25
NUM_OFFSPRING = 10
TOURNAMENT_SIZES = (2, 5)
# gamma: fraction of the history treated as "good" by the TPE; used only where the TPE is used
GAMMAS = (0.20, 0.30)
# TPE offspring levels; 0.0 is the EA condition (generated once per tournament size)
TPE_OFFSPRING_LEVELS = (0.0, 0.5, 1.0)
# replicates: seeds 0 .. N_SEEDS - 1, one array task per seed (odd, so each task has a single median run)
N_SEEDS = 11

# SLURM wall-clock limit by dataset size: the first tier whose row limit is >= the task's rows applies.
TIME_LIMITS = (
    (5_000, "12:00:00"),
    (15_000, "24:00:00"),
    (float("inf"), "72:00:00"),
)

# --- constants shared by every file ---------------------------------------------------

DATA_DIRECTORY = "/home/hernandezj45/Repos/TPEC/Data/OpenML_Suite_271_Classification_CSV"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/OpenML-CASH/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/All_Results/Results_OpenML_CASH_Pop20_200Evals/CASH"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time={time_limit}
#SBATCH --mem=200G
#SBATCH --job-name={job_name}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-{n_seeds}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpe-ea

# standard parameters used for all experiments
TRAIN_P={train_p:.2f}
N_FOLDS={n_folds}
GENS={gens}
POP_SIZE={pop_size}
CORES=$SLURM_CPUS_PER_TASK
MUT_PROB={mut_prob}
CROSSOVER_PROB={crossover_prob}
COMPONENT_MUT_PROB={component_mut_prob}
TPE_OFFSPRING={tpe_offspring:.1f}
TOURNAMENT_SIZE={tournament_size}
NUM_OFFSPRING={num_offspring}
GAMMA={gamma:.2f}
DATA_DIRECTORY={data_directory}
# OpenML task ({rows} rows)
TASK_ID={task_id}

# one array task per seed (seeds 0-{max_seed})
SEED=$(( SLURM_ARRAY_TASK_ID - 1 ))
# output directory for this condition and seed; the runner writes into task_<id>/ beneath it
OUTPUT_DIRECTORY={results_root}/{condition}/Seed_${{SEED}}/

python {runner} \\
    --seed $SEED \\
    --data_path $DATA_DIRECTORY \\
    --task_ids $TASK_ID \\
    --train_p $TRAIN_P \\
    --output_directory $OUTPUT_DIRECTORY \\
    --gens $GENS \\
    --pop_size $POP_SIZE \\
    --cores $CORES \\
    --mut_prob $MUT_PROB \\
    --crossover_prob $CROSSOVER_PROB \\
    --component_mut_prob $COMPONENT_MUT_PROB \\
    --tpe_offspring $TPE_OFFSPRING \\
    --tournament_size $TOURNAMENT_SIZE \\
    --num_offspring $NUM_OFFSPRING \\
    --gamma $GAMMA \\
    --n_folds $N_FOLDS
"""


def pct(value: float) -> int:
    """A probability as an integer percentage, for folder and job names (0.5 -> 50)."""
    return int(round(value * 100))


def load_tasks(suite_dir: str = None):
    """(task_id, rows) for every task in the suite's tasks_summary.csv, in file order."""
    suite_dir = suite_dir or find_suite_dir()
    with open(os.path.join(suite_dir, "tasks_summary.csv"), newline="") as f:
        return [(int(row["task_id"]), int(row["rows"])) for row in csv.DictReader(f)]


def conditions():
    """Every condition as (tournament_size, tpe_offspring, gamma); gamma is None for the EA runs."""
    out = []
    for tournament in TOURNAMENT_SIZES:
        for tpe_offspring in TPE_OFFSPRING_LEVELS:
            if tpe_offspring == 0.0:
                out.append((tournament, 0.0, None))
                continue
            for gamma in GAMMAS:
                out.append((tournament, tpe_offspring, gamma))
    return out


def condition_path(tournament: int, tpe_offspring: float, gamma) -> str:
    """Relative folder of one condition: T<t>/EA, or T<t>/G<g>/O<off>."""
    if gamma is None:
        return os.path.join(f"T{tournament}", "EA")
    return os.path.join(f"T{tournament}", f"G{pct(gamma)}", f"O{pct(tpe_offspring)}")


def job_name(tournament: int, tpe_offspring: float, gamma, task_id: int) -> str:
    """A short SLURM job name encoding condition and task, e.g. t2g20o50_359963 or t5ea_2073."""
    if gamma is None:
        return f"t{tournament}ea_{task_id}"
    return f"t{tournament}g{pct(gamma)}o{pct(tpe_offspring)}_{task_id}"


def time_limit(rows: int) -> str:
    """Wall-clock limit for a task of ``rows`` rows (see TIME_LIMITS)."""
    return next(limit for max_rows, limit in TIME_LIMITS if rows <= max_rows)


def generate(output_root: str = SWEEP_DIR, suite_dir: str = None) -> int:
    """Write every (condition, task) .sb file under ``output_root``. Returns the number written."""
    assert BUDGET % POP_SIZE == 0, "BUDGET must be a multiple of POP_SIZE"
    tasks = load_tasks(suite_dir)
    written = 0
    for tournament, tpe_offspring, gamma in conditions():
        rel = condition_path(tournament, tpe_offspring, gamma)
        out_dir = os.path.join(output_root, rel)
        os.makedirs(out_dir, exist_ok=True)
        for task_id, rows in tasks:
            content = TEMPLATE.format(
                time_limit=time_limit(rows),
                job_name=job_name(tournament, tpe_offspring, gamma, task_id),
                n_seeds=N_SEEDS,
                max_seed=N_SEEDS - 1,
                train_p=TRAIN_P,
                n_folds=N_FOLDS,
                gens=GENS,
                pop_size=POP_SIZE,
                mut_prob=MUT_PROB,
                crossover_prob=CROSSOVER_PROB,
                component_mut_prob=COMPONENT_MUT_PROB,
                tpe_offspring=tpe_offspring,
                tournament_size=tournament,
                num_offspring=NUM_OFFSPRING,
                # EA conditions pass gamma 0.0: it is unused there (the TPE is never fitted)
                gamma=0.0 if gamma is None else gamma,
                data_directory=DATA_DIRECTORY,
                rows=rows,
                task_id=task_id,
                results_root=RESULTS_ROOT,
                condition=rel.replace(os.sep, "/"),
                runner=RUNNER,
            )
            with open(os.path.join(out_dir, f"task_{task_id}.sb"), "w") as f:
                f.write(content)
            written += 1
    return written


def main() -> None:
    written = generate()
    n_tasks = len(load_tasks())
    n_conditions = len(conditions())
    print(f"Wrote {written} .sb files ({n_conditions} conditions x {n_tasks} tasks; "
          f"{written * N_SEEDS} runs at {N_SEEDS} seeds each).")


if __name__ == "__main__":
    main()
