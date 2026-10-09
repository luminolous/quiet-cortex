#!/usr/bin/env bash
# Evaluate every trained model and baseline with the common evaluator and rebuild all experiment tables
# (about 30 min). Needs the trained runs (scripts/train_all.sh) and the noisy test sets
# (python -m src.groups --noise-snr 30 20 15 10 5 0). Run from the repository root:  bash scripts/evaluate_all.sh
set -e
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$PYTHON" ] || PYTHON=".venv/bin/python"
for run in e1_baseline e1_lr1e-3 e1_batch32 e1_imgsz512 e1_ep50 e1_adamw e2_yolo11s e2_frcnn e3_2class e2_best_seed1 e2_best_seed2; do
    for split in val test; do "$PYTHON" -m src.evaluate --run "$run" --split "$split" --predict; done
done
for split in val test; do
    "$PYTHON" -m src.baselines threshold --config configs/experiments/threshold.yaml --split "$split"
    "$PYTHON" -m src.evaluate --run threshold --split "$split"
done
"$PYTHON" -m src.baselines threshold --config configs/experiments/threshold_2class.yaml --split test
"$PYTHON" -m src.evaluate --run threshold_2class --split test
"$PYTHON" -m src.baselines csp_lda --config configs/experiments/csp_lda.yaml
"$PYTHON" -m src.evaluate --table e1
"$PYTHON" -m src.evaluate --table e2
"$PYTHON" -m src.evaluate --table e3
"$PYTHON" -m src.evaluate --e4 e1_baseline --table e4
"$PYTHON" -m src.evaluate --table e5
"$PYTHON" -m src.evaluate --table seeds
"$PYTHON" -m src.evaluate --table per_subject
