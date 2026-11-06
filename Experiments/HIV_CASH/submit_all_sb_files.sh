#!/bin/bash
#
# Run ./submit_all.sh in every TPEC-5pct-100M-50C-* folder in this directory.
#
# For each folder (in name order) cds into it and runs its submit_all.sh, which sbatch-submits every
# .sb file under that folder. Arguments are passed through to each submit_all.sh:
#   --dry-run   list what would be submitted, submit nothing
#   BUDGET      only submit .sb files whose path contains BUDGET (e.g. 100_Evals)
# Stops at the first failure. Run generate_all_sb_files.sh first so each folder has .sb files.
#
# Usage (from any directory):
#   Experiments/HIV_CASH/submit_all_sb_files.sh [--dry-run | BUDGET]
#
# With no arguments this submits every .sb file in every folder.

# no -u: keep the script simple and consistent with generate_all_sb_files.sh
set -eo pipefail

HIV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
folders=("$HIV_DIR"/TPEC-5pct-100M-50C-*/)
shopt -u nullglob
if [[ ${#folders[@]} -eq 0 ]]; then
    echo "ERROR: no TPEC-5pct-100M-50C-* folders found in $HIV_DIR" >&2
    exit 1
fi

for dir in "${folders[@]}"; do
    dir="${dir%/}"
    name="$(basename "$dir")"
    if [[ ! -x "$dir/submit_all.sh" ]]; then
        echo "ERROR: $name has no executable submit_all.sh" >&2
        exit 1
    fi
    echo "== $name"
    (
        cd "$dir"
        ./submit_all.sh "$@"
    )
done

echo "Ran submit_all.sh in ${#folders[@]} folder(s)."
