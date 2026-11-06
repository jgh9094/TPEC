#!/bin/bash
#
# Run generate_sb_files.py in every TPEC-5pct-100M-50C-* folder in this directory.
#
# Loads the tpe-ea conda environment once, then for each folder (in name order) cds into it and
# runs its generate_sb_files.py. Stops at the first failure. Re-running overwrites existing .sb files.
#
# EA files are kept only in the G20 folder of each tournament condition (T1, T2, T5). The EA never
# fits TPE (TPE_PROB=0.0), so gamma cannot change an EA run, and the G25/G30/G40/G50 folders would
# repeat the same EA runs. Their EA/cash.sb files are removed right after generation.
#
# Usage (from any directory):
#   Experiments/HIV_CASH/generate_all_sb_files.sh

# no -u: conda's activation scripts reference variables that may be unset
set -eo pipefail

HIV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda init script not found: $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate tpe-ea
if [[ "${CONDA_DEFAULT_ENV:-}" != "tpe-ea" ]]; then
    echo "ERROR: expected conda env tpe-ea, got '${CONDA_DEFAULT_ENV:-none}'" >&2
    exit 1
fi
echo "Using $(command -v python) ($(python --version 2>&1))"

shopt -s nullglob
folders=("$HIV_DIR"/TPEC-5pct-100M-50C-*/)
shopt -u nullglob
if [[ ${#folders[@]} -eq 0 ]]; then
    echo "ERROR: no TPEC-5pct-100M-50C-* folders found in $HIV_DIR" >&2
    exit 1
fi

# delete the EA cash.sb in every budget folder under $1, and the EA folder once it is empty
remove_ea_files() {
    local ea
    for ea in "$1"/*_Evals/EA; do
        [[ -d "$ea" ]] || continue
        rm -f "$ea/cash.sb"
        rmdir "$ea" 2>/dev/null || echo "   kept $ea (not empty)"
    done
}

for dir in "${folders[@]}"; do
    dir="${dir%/}"
    name="$(basename "$dir")"
    if [[ ! -f "$dir/generate_sb_files.py" ]]; then
        echo "ERROR: $name has no generate_sb_files.py" >&2
        exit 1
    fi
    echo "== $name"
    (
        cd "$dir"
        python generate_sb_files.py
    )
    if [[ "$name" != *-G20 ]]; then
        remove_ea_files "$dir"
    fi
done

echo "Ran generate_sb_files.py in ${#folders[@]} folder(s); EA files kept only in *-G20 folders."
