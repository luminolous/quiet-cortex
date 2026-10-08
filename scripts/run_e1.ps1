# Train the remaining E1 runs one after another (Windows PowerShell). Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/run_e1.ps1
# A failed run is reported and the script continues with the next one.
$runs = @("e1_lr1e-3", "e1_batch32", "e1_imgsz512", "e1_ep50", "e1_adamw")
$python = ".venv\Scripts\python.exe"
$failed = @()
foreach ($run in $runs) {
    Write-Host "=== $run ($(Get-Date -Format 'HH:mm')) ==="
    & $python -m src.train --config "configs/experiments/$run.yaml"
    if ($LASTEXITCODE -ne 0) { $failed += $run; Write-Host "!!! $run failed (exit $LASTEXITCODE)" }
}
if ($failed.Count -gt 0) { Write-Host "Failed runs: $($failed -join ', ')" } else { Write-Host "All E1 runs finished." }
