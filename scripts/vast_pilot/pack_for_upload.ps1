param(
    [string]$DatasetDir = "data_vast_pilot_slice",
    [string]$OutTar = "vast_pilot_data.tar.gz"
)

$ErrorActionPreference = "Stop"
$dataIo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\data_io")
$src = Join-Path $dataIo $DatasetDir
if (-not (Test-Path $src)) {
    throw "Dataset not found: $src. Run: cd data_io; python build_laptop_hrm_slice.py --preset vast"
}

$out = Join-Path $dataIo $OutTar
if (Test-Path $out) { Remove-Item $out -Force }

Push-Location $src
try {
    tar -czf $out *
    $mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
    Write-Host "Created $out (${mb} MB)"
} finally {
    Pop-Location
}

Write-Host "Upload to Vast as /workspace/data_upload.tar, then unpack to /workspace/data/sampled"
