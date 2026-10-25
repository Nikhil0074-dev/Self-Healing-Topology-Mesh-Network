$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw 'Python was not found. Install Python 3 or add it to PATH.'
}

$pythonArgs = @('-3')
Write-Host "Using Python: $($python.Source)"
& $python.Source @pythonArgs -m pytest tests -q
& $python.Source @pythonArgs -m gnn.dataset --out experiments/results/dataset.pt `
    --train $(if ($env:DATASET_TRAIN) { $env:DATASET_TRAIN } else { '60' }) `
    --val $(if ($env:DATASET_VAL) { $env:DATASET_VAL } else { '12' }) `
    --test $(if ($env:DATASET_TEST) { $env:DATASET_TEST } else { '12' }) `
    --unseen $(if ($env:DATASET_UNSEEN) { $env:DATASET_UNSEEN } else { '12' }) `
    --steps $(if ($env:DATASET_STEPS) { $env:DATASET_STEPS } else { '300' })
& $python.Source @pythonArgs -m gnn.train --data experiments/results/dataset.pt --out models/link_gnn_v1.pt `
    --epochs $(if ($env:TRAIN_EPOCHS) { $env:TRAIN_EPOCHS } else { '30' })
& $python.Source @pythonArgs -m simulator.experiment_runner --seeds ($env:SEEDS ?? '10') --workers ($env:WORKERS ?? '1') --out experiments/results/results.csv
& $python.Source @pythonArgs -m experiments.analyze experiments/results/results.csv --out experiments/results/report.md
