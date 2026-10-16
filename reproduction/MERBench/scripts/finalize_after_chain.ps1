param(
  [Parameter(Mandatory = $true)]
  [int]$ChainProcessId,
  [Parameter(Mandatory = $true)]
  [string]$Root
)

$Root = (Resolve-Path -LiteralPath $Root).Path
$chain = Get-Process -Id $ChainProcessId -ErrorAction SilentlyContinue
while ($chain -and $chain.ProcessName -in @("powershell", "pwsh")) {
  Start-Sleep -Seconds 30
  $chain = Get-Process -Id $ChainProcessId -ErrorAction SilentlyContinue
}
& python (Join-Path $Root "scripts\finalize_reproduction.py") --root $Root
if ($LASTEXITCODE -ne 0) { throw "finalize_reproduction.py failed" }
