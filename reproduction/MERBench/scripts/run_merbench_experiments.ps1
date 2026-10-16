param(
  [ValidateSet("verify", "attention", "lmf", "misa", "tfn", "mult", "mult_resume", "unimodal", "classweight", "improved", "drop_audio", "drop_text", "drop_visual", "ablation", "summary", "all")]
  [string[]]$Stage = @("verify"),
  [int]$Epochs = 100,
  [int]$BatchSize = 32,
  [int]$Gpu = 0,
  [int]$Seed = 2026
)

$WorkspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$RepoRoot = Join-Path $WorkspaceRoot "references\MERTools\MERBench"
$ReproRoot = Join-Path $WorkspaceRoot "reproduction\MERBench"
$DataRoot = Join-Path $ReproRoot "data\mer2023-dataset-process"
$RunsRoot = Join-Path $ReproRoot "runs"
$ResultsRoot = Join-Path $ReproRoot "results"

$env:MERBENCH_MER2023_ROOT = $DataRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

function Test-Stage {
  param([string]$Name)
  return ($Stage -contains "all") -or ($Stage -contains $Name)
}

function Invoke-LoggedPython {
  param(
    [string]$WorkingDirectory,
    [string[]]$Arguments,
    [string]$LogFile
  )
  $stderrFile = "$LogFile.stderr"
  if ((Test-Path $LogFile) -or (Test-Path $stderrFile)) {
    throw "Refusing to overwrite an existing log: $LogFile"
  }

  Write-Host "python $($Arguments -join ' ')"
  $process = Start-Process `
    -FilePath "python" `
    -ArgumentList $Arguments `
    -WorkingDirectory $WorkingDirectory `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $stderrFile

  if (Test-Path $stderrFile) {
    Add-Content -LiteralPath $LogFile -Value "`n--- stderr ---"
    Get-Content -LiteralPath $stderrFile | Add-Content -LiteralPath $LogFile
    Remove-Item -LiteralPath $stderrFile -Force
  }
  if ($process.ExitCode -ne 0) {
    throw "Command failed with exit code $($process.ExitCode). See log: $LogFile"
  }
}

function Invoke-MerbenchRun {
  param(
    [string]$RunName,
    [string]$Model,
    [string]$AudioFeature,
    [string]$TextFeature,
    [string]$VideoFeature,
    [string]$RunSuffix,
    [string]$ClassWeight = "none",
    [string]$DropModalities = "",
    [string]$FeatType = "utt",
    [string]$HyperPath = "",
    [int]$RunBatchSize = $BatchSize,
    [int]$StartFold = 1,
    [string]$ResumeRoot = ""
  )
  $originalRunName = $RunName
  $existingRunDir = Join-Path $RunsRoot $RunSuffix
  $existingResultDir = Join-Path $ResultsRoot $RunName
  if ((Test-Path $existingRunDir) -or (Test-Path $existingResultDir)) {
    $tag = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "$RunName-$tag"
    $RunSuffix = $RunSuffix.Replace($originalRunName, $RunName)
  }
  $saveRoot = Join-Path $RunsRoot $RunName
  $logFile = Join-Path $RunsRoot "$RunName.log"
  $args = @(
    "-u", "main-release.py",
    "--model=$Model",
    "--feat_type=$FeatType",
    "--dataset=MER2023",
    "--audio_feature=$AudioFeature",
    "--text_feature=$TextFeature",
    "--video_feature=$VideoFeature",
    "--gpu=$Gpu",
    "--epochs=$Epochs",
    "--batch_size=$RunBatchSize",
    "--num_workers=0",
    "--seed=$Seed",
    "--class_weight=$ClassWeight",
    "--save_root=$saveRoot"
  )
  if ($DropModalities -ne "") {
    $args += "--drop_modalities=$DropModalities"
  }
  if ($HyperPath -ne "") {
    $args += "--hyper_path=$HyperPath"
  }
  if ($StartFold -gt 1) {
    $args += "--start_fold=$StartFold"
    $args += "--resume_root=$ResumeRoot"
  }
  Invoke-LoggedPython -WorkingDirectory $RepoRoot -Arguments $args -LogFile $logFile

  $runDir = Join-Path $RunsRoot $RunSuffix
  $resultDir = Join-Path $runDir "result"
  $outDir = Join-Path $ResultsRoot $RunName
  $summaryArgs = @(
    (Join-Path $ReproRoot "scripts\summarize_merbench_results.py"),
    "--result-dir", $resultDir,
    "--out-dir", $outDir,
    "--prefix", $RunName
  )
  Invoke-LoggedPython -WorkingDirectory $WorkspaceRoot -Arguments $summaryArgs -LogFile (Join-Path $RunsRoot "$RunName.summary.log")
}

if (Test-Stage "verify") {
  python (Join-Path $ReproRoot "scripts\verify_mer2023_features.py") `
    --data-root $DataRoot `
    --features chinese-hubert-large-UTT Baichuan-13B-Base-UTT clip-vit-large-patch14-UTT
}

if (Test-Stage "attention") {
  Invoke-MerbenchRun `
    -RunName "attention_high_seed${Seed}_ckpt" `
    -RunSuffix "attention_high_seed${Seed}_ckpt-trimodal" `
    -Model "attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT"
}

if (Test-Stage "lmf") {
  Invoke-MerbenchRun `
    -RunName "lmf_high_seed${Seed}" `
    -RunSuffix "lmf_high_seed${Seed}-trimodal" `
    -Model "lmf" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT"
}

if (Test-Stage "misa") {
  Invoke-MerbenchRun `
    -RunName "misa_high_stable_seed${Seed}" `
    -RunSuffix "misa_high_stable_seed${Seed}-trimodal" `
    -Model "misa" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT" `
    -HyperPath (Join-Path $ReproRoot "configs\hyper_stable.yaml")
}

if (Test-Stage "tfn") {
  Invoke-MerbenchRun `
    -RunName "tfn_high_seed${Seed}" `
    -RunSuffix "tfn_high_seed${Seed}-trimodal" `
    -Model "tfn" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT"
}

if (Test-Stage "mult") {
  $multBatchSize = [Math]::Min($BatchSize, 8)
  Invoke-MerbenchRun `
    -RunName "mult_high_fra_seed${Seed}" `
    -RunSuffix "mult_high_fra_seed${Seed}-trimodal" `
    -Model "mult" `
    -FeatType "frm_align" `
    -RunBatchSize $multBatchSize `
    -AudioFeature "chinese-hubert-large-FRA" `
    -TextFeature "Baichuan-13B-Base-FRA" `
    -VideoFeature "clip-vit-large-patch14-FRA" `
    -HyperPath (Join-Path $ReproRoot "configs\hyper_stable.yaml")
}

if (Test-Stage "mult_resume") {
  $multBatchSize = [Math]::Min($BatchSize, 16)
  Invoke-MerbenchRun `
    -RunName "mult_high_fra_seed${Seed}_resume" `
    -RunSuffix "mult_high_fra_seed${Seed}_resume-trimodal" `
    -Model "mult" `
    -FeatType "frm_align" `
    -RunBatchSize $multBatchSize `
    -AudioFeature "chinese-hubert-large-FRA" `
    -TextFeature "Baichuan-13B-Base-FRA" `
    -VideoFeature "clip-vit-large-patch14-FRA" `
    -HyperPath (Join-Path $ReproRoot "configs\hyper_stable.yaml") `
    -StartFold 2 `
    -ResumeRoot (Join-Path $RunsRoot "mult_high_fra_seed${Seed}-trimodal")
}

if (Test-Stage "unimodal") {
  Invoke-MerbenchRun `
    -RunName "unimodal_audio_attention_seed${Seed}" `
    -RunSuffix "unimodal_audio_attention_seed${Seed}-unimodal" `
    -Model "attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "chinese-hubert-large-UTT" `
    -VideoFeature "chinese-hubert-large-UTT"

  Invoke-MerbenchRun `
    -RunName "unimodal_text_attention_seed${Seed}" `
    -RunSuffix "unimodal_text_attention_seed${Seed}-unimodal" `
    -Model "attention" `
    -AudioFeature "Baichuan-13B-Base-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "Baichuan-13B-Base-UTT"

  Invoke-MerbenchRun `
    -RunName "unimodal_visual_attention_seed${Seed}" `
    -RunSuffix "unimodal_visual_attention_seed${Seed}-unimodal" `
    -Model "attention" `
    -AudioFeature "clip-vit-large-patch14-UTT" `
    -TextFeature "clip-vit-large-patch14-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT"
}

if (Test-Stage "classweight") {
  Invoke-MerbenchRun `
    -RunName "attention_high_classweight_seed${Seed}_ckpt" `
    -RunSuffix "attention_high_classweight_seed${Seed}_ckpt-trimodal" `
    -Model "attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT" `
    -ClassWeight "balanced"
}

if (Test-Stage "improved") {
  Invoke-MerbenchRun `
    -RunName "gated_residual_attention_high_seed${Seed}_formal" `
    -RunSuffix "gated_residual_attention_high_seed${Seed}_formal-trimodal" `
    -Model "gated_attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT" `
    -HyperPath (Join-Path $ReproRoot "configs\hyper_stable.yaml")
}

if ((Test-Stage "drop_audio") -or (Test-Stage "ablation")) {
  Invoke-MerbenchRun `
    -RunName "attention_high_drop_audio_seed${Seed}" `
    -RunSuffix "attention_high_drop_audio_seed${Seed}-trimodal" `
    -Model "attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT" `
    -DropModalities "audio"
}

if ((Test-Stage "drop_text") -or (Test-Stage "ablation")) {
  Invoke-MerbenchRun `
    -RunName "attention_high_drop_text_seed${Seed}" `
    -RunSuffix "attention_high_drop_text_seed${Seed}-trimodal" `
    -Model "attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT" `
    -DropModalities "text"
}

if ((Test-Stage "drop_visual") -or (Test-Stage "ablation")) {
  Invoke-MerbenchRun `
    -RunName "attention_high_drop_visual_seed${Seed}" `
    -RunSuffix "attention_high_drop_visual_seed${Seed}-trimodal" `
    -Model "attention" `
    -AudioFeature "chinese-hubert-large-UTT" `
    -TextFeature "Baichuan-13B-Base-UTT" `
    -VideoFeature "clip-vit-large-patch14-UTT" `
    -DropModalities "visual"
}

if ((Test-Stage "summary") -or (Test-Stage "all")) {
  python (Join-Path $ReproRoot "scripts\build_summary_tables.py") --root $ReproRoot
}
