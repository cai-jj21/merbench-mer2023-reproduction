param(
  [Parameter(Mandatory = $true)]
  [int]$TargetProcessId,
  [string]$Root
)

$Root = (Resolve-Path -LiteralPath $Root).Path
$refresh = Join-Path $Root "scripts\refresh_completed_summaries.ps1"
$backfill = Join-Path $Root "scripts\backfill_run_configs.py"
$aggregate = Join-Path $Root "scripts\aggregate_attention_multiseed.py"

$target = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
while ($target -and $target.ProcessName -in @("python", "python3")) {
  Start-Sleep -Seconds 30
  $target = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
}

& powershell -ExecutionPolicy Bypass -File $refresh -Root $Root
if ($LASTEXITCODE -ne 0) { throw "refresh_completed_summaries.ps1 failed" }
& python $backfill --root $Root
if ($LASTEXITCODE -ne 0) { throw "backfill_run_configs.py failed" }
& python $aggregate --root $Root
if ($LASTEXITCODE -ne 0) { Write-Warning "attention multiseed aggregation is pending additional seeds" }
