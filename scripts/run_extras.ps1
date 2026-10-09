# Optional extras (Phase 5, approved by the user): seed runs of the selected model and milder E4 noise levels.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts/run_extras.ps1
$python = ".venv\Scripts\python.exe"
$failed = @()
foreach ($run in @("e2_best_seed1", "e2_best_seed2")) {
    Write-Host "=== $run ($(Get-Date -Format 'HH:mm')) ==="
    & $python -m src.train --config "configs/experiments/$run.yaml"
    if ($LASTEXITCODE -ne 0) { $failed += $run; Write-Host "!!! $run failed (exit $LASTEXITCODE)" }
}
Write-Host "=== noisy test sets 30/20/15 dB ($(Get-Date -Format 'HH:mm')) ==="
& $python -m src.groups --noise-snr 30 20 15
if ($LASTEXITCODE -ne 0) { $failed += "noise" }
if ($failed.Count -gt 0) { Write-Host "Failed: $($failed -join ', ')" } else { Write-Host "All extras finished." }
