#!/bin/bash
#
# Submit every SLURM job-array (.sb) found under THIS script's directory, in one call.
#
# Placement determines scope: at Experiments/OpenML-CASH it submits the whole sweep
# (10 conditions x 35 tasks = 350 jobs); inside a T<t>, G<g> or O<off> folder it submits only that
# subset. Submit a subset at a time to avoid flooding the HPC.
# Each .sb is itself an 11-task array (one task per seed), so the full sweep is 3,850 runs.
#
# Usage:
#   ./submit_all.sh                 # submit everything
#   ./submit_all.sh --dry-run       # list what would be submitted, submit nothing
#   ./submit_all.sh T2/EA           # only submit .sb files whose path matches "T2/EA"
#   ./submit_all.sh task_359963     # one dataset across every condition
#
# The optional final argument is a substring filter on the .sb path (a tournament size, a gamma folder
# such as G20, an offspring level such as O50, or a task such as task_2073), letting you submit a
# subset without editing files.
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
