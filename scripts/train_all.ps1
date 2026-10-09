# Train every reported model, one after another (about 7 h on an RTX 4050 laptop GPU).
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts/train_all.ps1
# A failed run is reported and the script continues. If a DataLoader worker dies (host RAM),
# re-run that config with --workers 1; the worker count does not change the data order or the results.
$python = ".venv\Scripts\python.exe"
$runs = @("e1_baseline", "e1_lr1e-3", "e1_batch32", "e1_imgsz512", "e1_ep50", "e1_adamw",
          "e2_yolo11s", "e2_frcnn", "e3_2class", "e2_best_seed1", "e2_best_seed2")
$failed = @()
foreach ($run in $runs) {
    Write-Host "=== $run ($(Get-Date -Format 'HH:mm')) ==="
    & $python -m src.train --config "configs/experiments/$run.yaml"
    if ($LASTEXITCODE -ne 0) { $failed += $run; Write-Host "!!! $run failed (exit $LASTEXITCODE)" }
}
if ($failed.Count -gt 0) { Write-Host "Failed runs: $($failed -join ', ')" } else { Write-Host "All runs finished." }
