"""Aggregate per-generation ``checkpoints.csv`` trajectories into one tidy CSV.

Walks the checkpointing results layout:

    Results/Pop<size>/Mut<mm>_Var<vv>/<strategy>/<model>/Task_<task>/Seed_<n>/checkpoints.csv

and stacks every run's best-so-far trajectory into a single long-format table,
one row per (run, generation) checkpoint. This is the foundation for the
best-so-far-vs-accumulated-``hard_evals`` over-time plots (aggregated across the
21 seeds and compared across pop size, TPE strategy, mutation/variance and
model).

Each ``checkpoints.csv`` provides:

    generation, hard_evals, val_auc, train_auc, test_auc

and we prepend the experimental factors parsed from the directory path:

    pop        - population size (int: 25, 50, ...)
    mut_rate   - per-hyperparameter mutation probability (decimal: 0.25, 0.50)
    variance   - Gaussian mutation variance (decimal: 0.25, 0.50)
    strategy   - search strategy folder (TPE25/50/75/100, EA_Explore, EA_Exploit)
    tpe_prob   - probability of TPE-guided offspring (decimal; 0.0 for pure EA)
    model      - RF / ET / GB / KNN / KSVC / MLP
    task       - HOSP_READM_90 / LOS_extended / discharge_Home
    seed       - replicate seed (int)

The ``Mut``/``Var`` folder integers encode percentages; they are emitted as the
decimals actually used in the experiments (``25`` -> ``0.25``, ``50`` -> ``0.50``).
"""

import re
import csv
from pathlib import Path

# Results is always assumed to live in the repository root directory.
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Results"
OUTPUT_CSV = Path(__file__).resolve().parent / "checkpoint_trajectories.csv"

# directory-name parsers
POP_RE = re.compile(r"^Pop(\d+)$")
MUTVAR_RE = re.compile(r"^Mut(\d+)_Var(\d+)$")

# strategy folder -> tpe_prob (probability of TPE-guided offspring). Pure-EA
# strategies use no TPE and are recorded as 0.0.
STRATEGY_TPE_PROB = {
    "TPE25": 0.25,
    "TPE50": 0.50,
    "TPE75": 0.75,
    "TPE100": 1.0,
    "EA_Explore": 0.0,
    "EA_Exploit": 0.0,
}

# metadata columns prepended, then the raw checkpoints.csv trajectory columns
META_COLS = ["pop", "mut_rate", "variance", "strategy", "tpe_prob", "model", "task", "seed"]
TRAJ_COLS = ["generation", "hard_evals", "val_auc", "train_auc", "test_auc"]
HEADERS = META_COLS + TRAJ_COLS


def parse_run(checkpoints_path):
    """Return the metadata dict for one checkpoints.csv, or None if the path
    does not match the expected layout."""
    # .../Pop<>/Mut<>_Var<>/<strategy>/<model>/Task_<task>/Seed_<n>/checkpoints.csv
    seed_dir = checkpoints_path.parent
    task_dir = seed_dir.parent
    model_dir = task_dir.parent
    strategy_dir = model_dir.parent
    mutvar_dir = strategy_dir.parent
    pop_dir = mutvar_dir.parent

    pop_m = POP_RE.match(pop_dir.name)
    mutvar_m = MUTVAR_RE.match(mutvar_dir.name)
    if pop_m is None or mutvar_m is None:
        return None
    if not (task_dir.name.startswith("Task_") and seed_dir.name.startswith("Seed_")):
        return None

    strategy = strategy_dir.name
    return {
        "pop": int(pop_m.group(1)),
        "mut_rate": int(mutvar_m.group(1)) / 100.0,
        "variance": int(mutvar_m.group(2)) / 100.0,
        "strategy": strategy,
        "tpe_prob": STRATEGY_TPE_PROB.get(strategy, ""),
        "model": model_dir.name,
        "task": task_dir.name.removeprefix("Task_"),
        "seed": int(seed_dir.name.removeprefix("Seed_")),
    }


def collect_rows(results_dir):
    """Yield one output-row dict per generation across every checkpoints.csv."""
    n_runs = 0
    n_skipped = 0
    # Pop*/MutVar/strategy/model/Task_*/Seed_*/checkpoints.csv  -> 6 dir levels
    for ckpt in sorted(results_dir.glob("*/*/*/*/*/*/checkpoints.csv")):
        meta = parse_run(ckpt)
        if meta is None:
            n_skipped += 1
            continue
        try:
            with open(ckpt, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    out = dict(meta)
                    for col in TRAJ_COLS:
                        out[col] = row.get(col)
                    yield out
        except OSError as err:
            print(f"Skipping {ckpt}: {err}")
            n_skipped += 1
            continue
        n_runs += 1
    print(f"Read {n_runs} runs ({n_skipped} skipped).")


def main():
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Results directory not found: {RESULTS_DIR}")

    n_rows = 0
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADERS)
        writer.writeheader()
        for row in collect_rows(RESULTS_DIR):
            writer.writerow(row)
            n_rows += 1

    print(f"Wrote {n_rows} trajectory rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
