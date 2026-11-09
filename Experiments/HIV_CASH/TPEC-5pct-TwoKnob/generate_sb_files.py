#!/usr/bin/env python3
"""
Generate the SLURM batch (.sb) files for the two-knob TPE sweep on the HIV data (70/30 split).

Two knobs are crossed, each probability applied per offspring or per parent tournament:

  * ``tpe_offspring`` (both EAs):  the probability an offspring is chosen by the TPE from
    NUM_OFFSPRING candidate siblings.
  * ``tpe_selection`` (CASH only): the probability a parent tournament compares the candidates'
    TPE scores instead of their validation fitness.

Plus tournament size (T2 and the T1 control) and gamma (the TPE's good-fraction), which applies
only where the TPE is used. Conditions:

  * T2, TPE used: every (tpe_selection, tpe_offspring) in {0, 0.5, 1}^2 except (0, 0), times GAMMAS.
  * T2, EA:       tpe_selection = tpe_offspring = 0. The TPE is never fitted, so gamma cannot affect
                  it, and it is generated ONCE, not once per gamma. Its .sb still passes
                  EA_PLACEHOLDER_GAMMA because the CASH EA always constructs a CASH_TPE, whose
                  constructor requires gamma in (0, 1).
  * T1 control:   tournament size 1, where tpe_selection cannot change the winner, so only
                  tpe_selection = 0 is run; tpe_offspring in {0.5, 1} times GAMMAS, plus one EA run.

That is 24 + 1 + 6 + 1 = 32 .sb files (30 TPE + 2 EA), each a 31-task array (one task per seed).

Layout (under this folder):

    T<t>/G<gamma*100>/S<sel*100>_O<off*100>/cash.sb      (TPE conditions, e.g. T2/G30/S50_O100/cash.sb)
    T<t>/EA/cash.sb                                       (EA conditions)

Results go to a NEW root, so nothing overwrites the earlier sweeps:

    <RESULTS_ROOT>/T<t>/G<gamma*100>/S<sel*100>_O<off*100>/Task_<task>/Seed_<seed>/
    <RESULTS_ROOT>/T<t>/EA/Task_<task>/Seed_<seed>/

Each run writes checkpoints.csv after every generation. The 100/200/300/400 budgets are read from
those rows by Data-Tools/HIV-TwoKnob/aggregator.py, so a single 400-evaluation run per seed suffices.

Running this script overwrites any existing .sb files in place. Paths inside the files are the
cluster paths (``/home/hernandezj45/Repos/TPEC/...``), not local repo paths.

Usage:
    python Experiments/HIV_CASH/TPEC-5pct-TwoKnob/generate_sb_files.py
"""
import os

# Directory that holds this script; the .sb tree is written beneath it.
TPEC_DIR = os.path.dirname(os.path.abspath(__file__))

# --- settings you may edit -------------------------------------------------------------

POP_SIZE = 20
# evaluations of the single max-budget run per seed (the candidates considered, duplicates included)
BUDGET = 400
# GENS is derived so that POP_SIZE * (GENS + 1) == BUDGET (20 * 20 = 400)
GENS = BUDGET // POP_SIZE - 1

TRAIN_P = 0.70
MUT_PROB = 0.5
CROSSOVER_PROB = 0.5
COMPONENT_MUT_PROB = 0.25
NUM_OFFSPRING = 10
# gamma: fraction of the history treated as "good" by the TPE; used only where the TPE is used
GAMMAS = (0.20, 0.30, 0.50)
# gamma written into the EA .sb files. Never used (the TPE is never fitted with both knobs at 0), but
# CASH_TPE is still constructed in load_data_pd and asserts 0 < gamma < 1. Same value the old sweeps used.
EA_PLACEHOLDER_GAMMA = 0.30
# tpe_offspring and tpe_selection levels crossed in the T2 design
TPE_LEVELS = (0.0, 0.5, 1.0)
# tournament sizes: 2 is the main design, 1 is the control (no parent tournament at all)
TOURNAMENT_SIZES = (2, 1)
# replicates: seeds 0 .. N_SEEDS - 1, one array task per seed
N_SEEDS = 31
# the single target (y_label) used for every run in this sweep
TASK_ID = "virologic_failure"

# --- constants shared by every file ---------------------------------------------------

DATA_DIRECTORY = "/home/hernandezj45/Repos/TPEC/Data/HIV/ecoxai_dataset_5pct_no_PHI.csv"
RUNNER = "/home/hernandezj45/Repos/TPEC/Experiments/HIV_CASH/runner.py"
RESULTS_ROOT = "/home/hernandezj45/Repos/TPEC/All_Results/Results_HIV_5pct_TwoKnob_Pop20/CASH"

TEMPLATE = """\
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --job-name={job_name}
#SBATCH -p defq,moore,preemptable
#SBATCH --array=1-{n_seeds}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tpe-ea

# standard parameters used for all experiments
TRAIN_P={train_p:.2f}
GENS={gens}
POP_SIZE={pop_size}
CORES=$SLURM_CPUS_PER_TASK
MUT_PROB={mut_prob}
CROSSOVER_PROB={crossover_prob}
COMPONENT_MUT_PROB={component_mut_prob}
TPE_SELECTION={tpe_selection:.1f}
TPE_OFFSPRING={tpe_offspring:.1f}
TOURNAMENT_SIZE={tournament_size}
NUM_OFFSPRING={num_offspring}
GAMMA={gamma:.2f}
DATA_DIRECTORY={data_directory}
TASK_ID={task_id}

# one array task per seed (seeds 0-{max_seed})
SEED=$(( SLURM_ARRAY_TASK_ID - 1 ))
# output directory for this condition, task and seed
OUTPUT_DIRECTORY={results_root}/{condition}/Task_${{TASK_ID}}/Seed_${{SEED}}/

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
    --tpe_selection $TPE_SELECTION \\
    --tpe_offspring $TPE_OFFSPRING \\
    --tournament_size $TOURNAMENT_SIZE \\
    --num_offspring $NUM_OFFSPRING \\
    --gamma $GAMMA
"""


def pct(value: float) -> int:
    """A probability as an integer percentage, for folder and job names (0.5 -> 50)."""
    return int(round(value * 100))


def conditions():
    """
    Every condition as (tournament_size, tpe_selection, tpe_offspring, gamma), with gamma None for the
    EA conditions (which never fit the TPE, so they run once and carry no gamma).
    """
    out = []
    for tournament in TOURNAMENT_SIZES:
        # tournament size 1 cannot be affected by tpe_selection (one competitor always wins)
        selection_levels = TPE_LEVELS if tournament > 1 else (0.0,)
        for tpe_selection in selection_levels:
            for tpe_offspring in TPE_LEVELS:
                if tpe_selection == 0.0 and tpe_offspring == 0.0:
                    out.append((tournament, 0.0, 0.0, None))
                    continue
                for gamma in GAMMAS:
                    out.append((tournament, tpe_selection, tpe_offspring, gamma))
    return out


def condition_path(tournament: int, tpe_selection: float, tpe_offspring: float, gamma) -> str:
    """Relative folder of one condition: T<t>/EA, or T<t>/G<g>/S<sel>_O<off>."""
    if gamma is None:
        return os.path.join(f"T{tournament}", "EA")
    return os.path.join(f"T{tournament}", f"G{pct(gamma)}", f"S{pct(tpe_selection)}_O{pct(tpe_offspring)}")


def job_name(tournament: int, tpe_selection: float, tpe_offspring: float, gamma) -> str:
    """A short SLURM job name that encodes the condition, e.g. t2g30s50o100 or t2ea."""
    if gamma is None:
        return f"t{tournament}ea"
    return f"t{tournament}g{pct(gamma)}s{pct(tpe_selection)}o{pct(tpe_offspring)}"


def generate(output_root: str = TPEC_DIR) -> int:
    """Write every condition's cash.sb under ``output_root``. Returns the number of files written."""
    written = 0
    for tournament, tpe_selection, tpe_offspring, gamma in conditions():
        rel = condition_path(tournament, tpe_selection, tpe_offspring, gamma)
        out_dir = os.path.join(output_root, rel)
        os.makedirs(out_dir, exist_ok=True)
        content = TEMPLATE.format(
            job_name=job_name(tournament, tpe_selection, tpe_offspring, gamma),
            n_seeds=N_SEEDS,
            max_seed=N_SEEDS - 1,
            train_p=TRAIN_P,
            gens=GENS,
            pop_size=POP_SIZE,
            mut_prob=MUT_PROB,
            crossover_prob=CROSSOVER_PROB,
            component_mut_prob=COMPONENT_MUT_PROB,
            tpe_selection=tpe_selection,
            tpe_offspring=tpe_offspring,
            tournament_size=tournament,
            num_offspring=NUM_OFFSPRING,
            # EA conditions pass a placeholder gamma: unused (the TPE is never fitted), but it must be valid
            gamma=EA_PLACEHOLDER_GAMMA if gamma is None else gamma,
            data_directory=DATA_DIRECTORY,
            task_id=TASK_ID,
            results_root=RESULTS_ROOT,
            condition=rel.replace(os.sep, "/"),
            runner=RUNNER,
        )
        with open(os.path.join(out_dir, "cash.sb"), "w") as f:
            f.write(content)
        written += 1
    return written


def main() -> None:
    written = generate(TPEC_DIR)
    n_ea = sum(1 for condition in conditions() if condition[3] is None)
    print(f"Wrote {written} .sb files ({written - n_ea} TPE + {n_ea} EA).")


if __name__ == "__main__":
    main()
