param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

& $Python "VAE/specialized_training_scripts/TRAIN_RECONSTRUCTION_LVS_VAE.py"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python "VAE/specialized_training_scripts/TRAIN_LVS_VAE_HEADS.py"

exit $LASTEXITCODE
