param(
  [Parameter(Mandatory = $true)]
  [int]$TargetProcessId,
  [Parameter(Mandatory = $true)]
  [string]$Root
)

$Root = (Resolve-Path -LiteralPath $Root).Path
$refresh = Join-Path $Root "scripts\refresh_completed_summaries.ps1"
$multiseed = Join-Path $Root "scripts\run_attention_multiseed.ps1"
$aggregate = Join-Path $Root "scripts\aggregate_attention_multiseed.py"

$target = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
while ($target -and $target.ProcessName -in @("python", "python3")) {
  Start-Sleep -Seconds 30
  $target = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
}

# Allow the basic postprocess watcher to finish its first summary pass.
Start-Sleep -Seconds 120
& powershell -ExecutionPolicy Bypass -File $refresh -Root $Root
if ($LASTEXITCODE -ne 0) { throw "initial summary refresh failed" }

& powershell -ExecutionPolicy Bypass -File $multiseed -Seeds 2027,2028 -Epochs 100 -BatchSize 32
if ($LASTEXITCODE -ne 0) { throw "attention multiseed run failed" }

& powershell -ExecutionPolicy Bypass -File $refresh -Root $Root
if ($LASTEXITCODE -ne 0) { throw "final summary refresh failed" }
& python $aggregate --root $Root
