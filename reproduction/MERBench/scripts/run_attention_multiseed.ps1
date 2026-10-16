param(
  [string[]]$Seeds = @("2027", "2028"),
  [int]$Epochs = 100,
  [int]$BatchSize = 32,
  [int]$Gpu = 0
)

$SeedList = @(
  foreach ($item in $Seeds) {
    foreach ($value in ($item -split '[,\s]+')) {
      if ($value -match '^\d+$') { [int]$value }
    }
  }
)
if (-not $SeedList) { throw "No valid seeds were provided: $($Seeds -join ',')" }

$WorkspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$RepoRoot = Join-Path $WorkspaceRoot "references\MERTools\MERBench"
$ReproRoot = Join-Path $WorkspaceRoot "reproduction\MERBench"
$DataRoot = Join-Path $ReproRoot "data\mer2023-dataset-process"
$RunsRoot = Join-Path $ReproRoot "runs"
$ResultsRoot = Join-Path $ReproRoot "results"
$HyperPath = Join-Path $ReproRoot "configs\attention_stable.yaml"

$env:MERBENCH_MER2023_ROOT = $DataRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path $RunsRoot, $ResultsRoot | Out-Null

foreach ($Seed in $SeedList) {
  $runName = "attention_high_seed${Seed}_stable"
  $saveRoot = Join-Path $RunsRoot $runName
  $runDir = "${runName}-trimodal"
  $log = Join-Path $RunsRoot "$runName.log"
  $stderr = "$log.stderr"
  if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $stderr)) {
    throw "Refusing to overwrite existing log: $log"
  }
  $args = @(
    "-u", "main-release.py",
    "--model=attention",
    "--feat_type=utt",
    "--dataset=MER2023",
    "--audio_feature=chinese-hubert-large-UTT",
    "--text_feature=Baichuan-13B-Base-UTT",
    "--video_feature=clip-vit-large-patch14-UTT",
    "--gpu=$Gpu",
    "--epochs=$Epochs",
    "--batch_size=$BatchSize",
    "--num_workers=0",
    "--seed=$Seed",
    "--class_weight=none",
    "--save_root=$saveRoot",
    "--hyper_path=$HyperPath"
  )
  Write-Host "python $($args -join ' ')"
  $process = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $RepoRoot -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError $stderr
  if (Test-Path -LiteralPath $stderr) {
    Add-Content -LiteralPath $log -Value "`n--- stderr ---"
    Get-Content -LiteralPath $stderr | Add-Content -LiteralPath $log
    Remove-Item -LiteralPath $stderr -Force
  }
  if ($process.ExitCode -ne 0) {
    throw "Attention seed $Seed failed with exit code $($process.ExitCode). See $log"
  }

  $resultDir = Join-Path $RunsRoot "$runDir\result"
  $outDir = Join-Path $ResultsRoot $runName
  $summaryLog = Join-Path $RunsRoot "$runName.summary.log"
  $summaryArgs = @(
    (Join-Path $ReproRoot "scripts\summarize_merbench_results.py"),
    "--result-dir", $resultDir,
    "--out-dir", $outDir,
    "--prefix", $runName
  )
  & python @summaryArgs 2>&1 | Tee-Object -FilePath $summaryLog
}
