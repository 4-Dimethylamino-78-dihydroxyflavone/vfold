# Assemble a fully portable vfold bundle: runs on a machine with no Rust, no
# CUDA toolkit, no cuDNN install. GPU rung works from bundled DLLs when an
# NVIDIA card is present; CPU rung works everywhere.
#
#   .\package.ps1                  # assemble dist\vfold-win64\
#   .\package.ps1 -Zip             # ... and zip it
#   .\package.ps1 -CudaBin <dir>   # override where CUDA runtime DLLs come from
#
# The CUDA DLL manifest below was determined EMPIRICALLY: bundle assembled,
# PATH stripped to System32 alone, embed run end-to-end on the GPU, and DLLs
# removed until the loader or cuDNN complained. Do not "tidy" it by intuition;
# the loader's import tables, not intuition, decide what is needed.

param(
    [switch]$Zip,
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    [string]$ModelDir = "$env:USERPROFILE\.cache\vfold"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$dist = Join-Path $root "dist\vfold-win64"

# --- build ------------------------------------------------------------------
# cargo narrates progress on stderr; under ErrorActionPreference=Stop that
# chatter would itself kill the script (PS 5.1 wraps native stderr lines in
# ErrorRecords). Judge cargo by its exit code, nothing else.
Write-Host "building release binary..."
Push-Location $root
$ErrorActionPreference = "Continue"
cargo build --release 2>&1 | Out-Null
$buildExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Pop-Location
if ($buildExit -ne 0) { throw "cargo build failed (exit $buildExit)" }

# --- manifest ---------------------------------------------------------------
$rel = Join-Path $root "target\release"

# Core: the binary and the ONNX Runtime provider layer (placed by the build).
$core = @(
    (Join-Path $rel "vfold.exe"),
    (Join-Path $rel "onnxruntime_providers_shared.dll"),
    (Join-Path $rel "onnxruntime_providers_cuda.dll"),
    (Join-Path $rel "DirectML.dll")                    # opt-in rung; tiny, keep
)

# CUDA runtime + cuDNN, from the host toolkit. Empirical minimal set — see the
# header comment. cufft64_11 IS required despite a ViT never running an FFT:
# onnxruntime_providers_cuda.dll import-links it, so LoadLibrary fails without
# it (measured: stripped-PATH run named it in Error 126). Absent after testing:
# cudnn_adv (legacy RNN kernels, loaded lazily) and nvrtc (JIT, unused here).
$cuda = @(
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll"
) | ForEach-Object { Join-Path $CudaBin $_ }

# Model: bundled so the field machine needs no network and no HuggingFace.
$model = @(
    (Join-Path $ModelDir "model.onnx"),
    (Join-Path $ModelDir "model.onnx_data")
)

$docs = @(
    (Join-Path $root "README.md"),
    (Join-Path $root "USAGE.md"),
    (Join-Path $root "AGENTS.md")
)

# --- assemble ---------------------------------------------------------------
if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dist | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $dist "model") | Out-Null

$missing = @()
foreach ($f in ($core + $cuda)) {
    if (Test-Path $f) { Copy-Item $f $dist } else { $missing += $f }
}
foreach ($f in $model) {
    if (Test-Path $f) { Copy-Item $f (Join-Path $dist "model") } else { $missing += $f }
}
foreach ($f in $docs) {
    if (Test-Path $f) { Copy-Item $f $dist } else { $missing += $f }
}

if ($missing) {
    Write-Host "MISSING (bundle incomplete):" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "refusing to emit a bundle that would fail on another machine"
}

# The bundled model lives at .\model\ relative to the exe; tell the user once.
@"
vfold portable bundle
=====================

Run from this folder:

    .\vfold.exe <IMAGE_DIR> -k 64 --model .\model\model.onnx

Everything needed is here: CUDA runtime, cuDNN, ONNX Runtime providers, and
the DINOv3 model. No installs. On a machine without an NVIDIA GPU the device
line will say CPU and the run still completes (~6x slower).

Keep the DLLs in the same folder as vfold.exe -- Windows resolves them from
the exe's directory first, which is what makes this bundle self-contained.
"@ | Set-Content (Join-Path $dist "START_HERE.txt")

# --- report -----------------------------------------------------------------
Write-Host ""
Write-Host "=== bundle contents ==="
Get-ChildItem $dist -Recurse -File | Sort-Object Length -Descending |
    Format-Table @{n='MB';e={[math]::Round($_.Length/1MB,1)};a='right'},
                 @{n='file';e={$_.FullName.Substring($dist.Length+1)}} -AutoSize
$total = (Get-ChildItem $dist -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("total: {0:N2} GB" -f ($total/1GB))

if ($Zip) {
    $zipPath = Join-Path $root "dist\vfold-win64.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path "$dist\*" -DestinationPath $zipPath
    Write-Host ("zip:   {0:N2} GB  ->  {1}" -f ((Get-Item $zipPath).Length/1GB), $zipPath)
}
