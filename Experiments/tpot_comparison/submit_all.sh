#!/bin/bash
# Submit generated TPOT arrays. Usage: ./submit_all.sh [--dry-run] [Pop25|Pop50|Pop100]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) FILTER="$arg" ;;
    esac
done

if ! command -v sbatch >/dev/null 2>&1 && [[ $DRY_RUN -eq 0 ]]; then
    echo "ERROR: sbatch not found. Run on the SLURM cluster or use --dry-run." >&2
    exit 1
fi

count=0
while IFS= read -r -d '' batch_file; do
    if [[ -n "$FILTER" && "$batch_file" != *"$FILTER"* ]]; then
        continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] would submit: $batch_file"
    else
        echo "Submitting: $batch_file"
        sbatch "$batch_file"
    fi
    count=$((count + 1))
done < <(find "$SCRIPT_DIR" -mindepth 2 -maxdepth 2 -type f -name "tpot.sb" -print0 | sort -z)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: $count job array(s) would be submitted."
else
    echo "Submitted $count job array(s)."
fi
