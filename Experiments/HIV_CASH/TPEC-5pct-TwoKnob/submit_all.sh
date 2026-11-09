#!/bin/bash
#
# Submit every SLURM job-array (.sb) found under THIS script's directory, in one call.
#
# Placement determines scope: at Experiments/HIV_CASH/TPEC-5pct-TwoKnob it submits the whole sweep
# (30 TPE + 2 EA = 32 jobs); inside a T<t> or G<g> folder it submits only that subset. Submit a
# subset at a time to avoid flooding the HPC.
# Each .sb is itself a 31-task array (one task per seed).
#
# Usage:
#   ./submit_all.sh             # submit everything
#   ./submit_all.sh --dry-run   # list what would be submitted, submit nothing
#   ./submit_all.sh T2/EA       # only submit .sb files whose path matches "T2/EA"
#
# The optional final argument is a substring filter on the .sb path (e.g. a tournament size,
# a gamma folder such as G30, or a condition such as S50_O100), letting you submit a subset
# without editing files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *)         FILTER="$arg" ;;
    esac
done

if ! command -v sbatch >/dev/null 2>&1 && [[ $DRY_RUN -eq 0 ]]; then
    echo "ERROR: sbatch not found. Run on the SLURM cluster, or use --dry-run to preview." >&2
    exit 1
fi

count=0
while IFS= read -r -d '' sb; do
    if [[ -n "$FILTER" && "$sb" != *"$FILTER"* ]]; then
        continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] would submit: $sb"
    else
        echo "Submitting: $sb"
        sbatch "$sb"
    fi
    count=$((count + 1))
done < <(find "$SCRIPT_DIR" -type f -name "*.sb" -print0 | sort -z)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: $count job(s) would be submitted."
else
    echo "Submitted $count job(s)."
fi
