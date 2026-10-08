#!/usr/bin/env bash
# Train the remaining E1 runs one after another (Linux/macOS/Git Bash). Run from the repo root:
#   bash scripts/run_e1.sh
# A failed run is reported and the script continues with the next one.
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON=".venv/bin/python"
failed=()
for run in e1_lr1e-3 e1_batch32 e1_imgsz512 e1_ep50 e1_adamw; do
    echo "=== $run ($(date +%H:%M)) ==="
    "$PYTHON" -m src.train --config "configs/experiments/$run.yaml" || { failed+=("$run"); echo "!!! $run failed"; }
done
if [ ${#failed[@]} -gt 0 ]; then echo "Failed runs: ${failed[*]}"; else echo "All E1 runs finished."; fi
