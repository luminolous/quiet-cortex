#!/usr/bin/env bash
# Train every reported model, one after another (about 7 h on an RTX 4050 laptop GPU).
# Run from the repository root:  bash scripts/train_all.sh
# A failed run is reported and the script continues. If a DataLoader worker dies (host RAM),
# re-run that config with --workers 1; the worker count does not change the data order or the results.
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON=".venv/bin/python"
RUNS="e1_baseline e1_lr1e-3 e1_batch32 e1_imgsz512 e1_ep50 e1_adamw e2_yolo11s e2_frcnn e3_2class e2_best_seed1 e2_best_seed2"
failed=()
for run in $RUNS; do
    echo "=== $run ($(date +%H:%M)) ==="
    "$PYTHON" -m src.train --config "configs/experiments/$run.yaml" || { failed+=("$run"); echo "!!! $run failed"; }
done
if [ ${#failed[@]} -gt 0 ]; then echo "Failed runs: ${failed[*]}"; else echo "All runs finished."; fi
