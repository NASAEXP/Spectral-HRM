# Run after build_laptop_hrm_slice.py --preset vast completes.
param(
    [string]$DatasetDir = "data_vast_pilot_slice"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$dataIo = Join-Path $root "data_io"
$shm = Join-Path $root "Spectral-HRM"
$slice = Join-Path $dataIo $DatasetDir

if (-not (Test-Path (Join-Path $slice "metadata.json"))) {
    throw "Missing dataset at $slice - wait for build to finish."
}

Push-Location $shm
python scripts/vast_pilot/verify_dataset.py "..\data_io\$DatasetDir" --batch-size 8192 --peek-batches 2
Pop-Location

& (Join-Path $PSScriptRoot "pack_for_upload.ps1") -DatasetDir $DatasetDir

$tar = Join-Path $dataIo "vast_pilot_data.tar.gz"
$meta = Get-Content (Join-Path $slice "metadata.json") | ConvertFrom-Json
$steps = [int]([math]::Floor($meta.total_length / 8192))

$manifest = @{
    prepared_at = (Get-Date).ToString("o")
    dataset_dir = $slice
    tarball = $tar
    tarball_mb = [math]::Round((Get-Item $tar).Length / 1MB, 2)
    total_tokens = $meta.total_length + 1
    train_steps_batch_8192 = $steps
    upload_to = "/workspace/data_upload.tar"
    unpack_to = "/workspace/data/sampled"
} | ConvertTo-Json -Depth 4

$manifestPath = Join-Path $PSScriptRoot "RENT_READY.json"
$manifest | Set-Content $manifestPath -Encoding utf8
Write-Host "Wrote $manifestPath"
Get-Content $manifestPath
