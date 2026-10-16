param(
  [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)

$Root = (Resolve-Path $Root).Path
$RunsRoot = Join-Path $Root "runs"
$ResultsRoot = Join-Path $Root "results"

foreach ($run in Get-ChildItem -LiteralPath $RunsRoot -Directory | Sort-Object Name) {
  $resultDir = Join-Path $run.FullName "result"
  if (-not (Test-Path -LiteralPath $resultDir)) { continue }
  if (-not (Get-ChildItem -LiteralPath $resultDir -Filter "*.npz" -File -ErrorAction SilentlyContinue)) { continue }

  $prefix = $run.Name
  if ($prefix.EndsWith("-trimodal")) { $prefix = $prefix.Substring(0, $prefix.Length - 9) }
  if ($prefix.EndsWith("-unimodal")) { $prefix = $prefix.Substring(0, $prefix.Length - 9) }
  $outDir = Join-Path $ResultsRoot $prefix
  Write-Host "Summarizing $($run.Name) -> $prefix"
  & python (Join-Path $Root "scripts\summarize_merbench_results.py") `
    --result-dir $resultDir `
    --out-dir $outDir `
    --prefix $prefix
  if ($LASTEXITCODE -ne 0) { throw "summary failed for $($run.Name)" }
}

& python (Join-Path $Root "scripts\build_summary_tables.py") --root $Root
if ($LASTEXITCODE -ne 0) { throw "summary table build failed" }
