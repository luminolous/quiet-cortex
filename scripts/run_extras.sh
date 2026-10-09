#!/usr/bin/env bash
# Optional extras (Phase 5, approved by the user): seed runs of the selected model and milder E4 noise levels.
# Run from the repo root in Git Bash:  bash scripts/run_extras.sh
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON=".venv/bin/python"
failed=()
for run in e2_best_seed1 e2_best_seed2; do
    echo "=== $run ($(date +%H:%M)) ==="
    "$PYTHON" -m src.train --config "configs/experiments/$run.yaml" || { failed+=("$run"); echo "!!! $run failed"; }
done
echo "=== noisy test sets 30/20/15 dB ($(date +%H:%M)) ==="
"$PYTHON" -m src.groups --noise-snr 30 20 15 || failed+=("noise")
if [ ${#failed[@]} -gt 0 ]; then echo "Failed: ${failed[*]}"; else echo "All extras finished."; fi
