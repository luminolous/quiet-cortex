# Evaluate every trained model and baseline with the common evaluator and rebuild all experiment tables
# (about 30 min). Needs the trained runs (scripts/train_all.ps1) and the noisy test sets
# (python -m src.groups --noise-snr 30 20 15 10 5 0).
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts/evaluate_all.ps1
$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python.exe"
function Run { & $python @args; if ($LASTEXITCODE -ne 0) { throw "failed: $args" } }
$runs = @("e1_baseline", "e1_lr1e-3", "e1_batch32", "e1_imgsz512", "e1_ep50", "e1_adamw",
          "e2_yolo11s", "e2_frcnn", "e3_2class", "e2_best_seed1", "e2_best_seed2")
foreach ($run in $runs) { foreach ($split in @("val", "test")) { Run -m src.evaluate --run $run --split $split --predict } }
foreach ($split in @("val", "test")) {
    Run -m src.baselines threshold --config configs/experiments/threshold.yaml --split $split
    Run -m src.evaluate --run threshold --split $split
}
Run -m src.baselines threshold --config configs/experiments/threshold_2class.yaml --split test
Run -m src.evaluate --run threshold_2class --split test
Run -m src.baselines csp_lda --config configs/experiments/csp_lda.yaml
foreach ($t in @("e1", "e2", "e3")) { Run -m src.evaluate --table $t }
Run -m src.evaluate --e4 e1_baseline --table e4
foreach ($t in @("e5", "seeds", "per_subject")) { Run -m src.evaluate --table $t }
