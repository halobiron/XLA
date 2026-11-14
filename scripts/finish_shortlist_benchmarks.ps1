param(
    [string]$Run = 'outputs/tacs_shortlist_fullft_clean_v3'
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $workspace
$python = Join-Path $workspace '.venv-gpu/Scripts/python.exe'
$trainingProcesses = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'train_tacs_shortlist_cpu_state_snapshot\.py.*--stage train.*--arms no_context full_tacs'
})
$trainingIds = @($trainingProcesses | ForEach-Object { $_.ProcessId })
$mainProcesses = @($trainingProcesses | Where-Object { $_.ParentProcessId -in $trainingIds })
if ($mainProcesses.Count -ne 1) { throw 'Expected one verified live matched-training process' }
$trainingProcess = $mainProcesses[0]
$statusPath = Join-Path $Run 'pipeline_status.json'
while ($true) {
    $counts = @{}
    foreach ($arm in @('no_context', 'full_tacs')) {
        $historyPath = Join-Path $Run "$arm/history.json"
        $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
        $counts[$arm] = $history.Count
    }
    @{ stage = 'waiting_for_matched_training'; epochs = $counts; training_pid = $trainingProcess.ProcessId } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8
    if ($counts.no_context -ge 10 -and $counts.full_tacs -ge 10) { break }
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($trainingProcess.ProcessId)"
    if (-not $current -or $current.CreationDate -ne $trainingProcess.CreationDate) {
        throw 'Training process ended before both fixed budgets completed; inspect checkpoints, do not blindly restart'
    }
    Start-Sleep -Seconds 20
}
# Wait for the verified training process to release GPU resources.
while ($true) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($trainingProcess.ProcessId)"
    if (-not $current -or $current.CreationDate -ne $trainingProcess.CreationDate) { break }
    Start-Sleep -Seconds 2
}
@{ stage = 'evaluating_locked_validation' } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
& $python scripts/evaluate_shortlist_benchmarks.py --stage validation --run $Run
if ($LASTEXITCODE -ne 0) { throw 'Validation evaluation failed' }
$validation = Get-Content (Join-Path $Run 'evaluation/validation_result.json') -Raw | ConvertFrom-Json
if ($validation.positive_familywise_metrics.Count -eq 0) {
    @{ stage = 'validation_not_positive'; test_read = $false } | ConvertTo-Json |
        Set-Content -LiteralPath $statusPath -Encoding utf8
    Write-Output 'No positive predeclared validation comparison. Test remains untouched. Goal requires further research.'
    exit 0
}
@{ stage = 'evaluating_locked_test' } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
& $python scripts/evaluate_shortlist_benchmarks.py --stage test --run $Run
if ($LASTEXITCODE -ne 0) { throw 'Test evaluation failed' }
$test = Get-Content (Join-Path $Run 'evaluation/test_result.json') -Raw | ConvertFrom-Json
@{ stage = 'test_evaluated'; positive_metrics = $test.positive_familywise_metrics; human_completion_audit_required = $true } |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8
Write-Output 'Fixed test evaluation complete. Inspect all metrics, controls, and invariants before claiming the goal achieved.'
