param(
    [string]$Python = "python",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$RunnerArgs = @("-u", "VAE/RUN_JOINT_LVS_VAE_PS.py")
if ($SkipExisting) {
    $RunnerArgs += "--skip-existing"
}

& $Python @RunnerArgs
exit $LASTEXITCODE
