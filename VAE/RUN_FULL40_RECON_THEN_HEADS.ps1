param(
    [string]$Python = "python",
    [switch]$SurvivalOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

& $Python "VAE/TRAIN_RECONSTRUCTION_LVS_VAE.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($SurvivalOnly) {
    & $Python "VAE/TRAIN_PLACING_SURVIVAL_LVS_VAE.py"
} else {
    & $Python "VAE/TRAIN_LVS_VAE_HEADS.py"
}

exit $LASTEXITCODE
