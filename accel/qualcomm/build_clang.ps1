# build_clang.ps1 - build loom_accel_qualcomm.dll with the llvm-mingw toolchain.
#
# This is the fallback used on the Windows/ARM64 dev box where CMake + MSVC are
# not installed. It matches the toolchain Go's cgo uses on windows/arm64
# (llvm-mingw), so the plugin DLL and the Go loader share the same ABI. The QNN
# backend DLLs (QnnCpu/QnnHtp/QnnGpu, MSVC-ABI) are loaded at runtime via
# LoadLibrary and only exercised through their C entry points, so the mingw/MSVC
# split is safe.
#
# Prereqs:
#   - QNN_SDK_ROOT set (install_qairt.ps1 downloads the SDK; set it machine-wide
#     so re-logins / other users pick it up:
#       [Environment]::SetEnvironmentVariable('QNN_SDK_ROOT', <path>, 'Machine'))
#   - llvm-mingw at C:\llvm-mingw (override with -LlvmMingw)

param(
    [string]$LlvmMingw = 'C:\llvm-mingw',
    [string]$QnnRoot = $env:QNN_SDK_ROOT
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $QnnRoot) {
    throw "QNN_SDK_ROOT not set. Run install_qairt.ps1, then set QNN_SDK_ROOT (Machine scope for all users)."
}
if (-not (Test-Path (Join-Path $QnnRoot 'include\QNN\QnnInterface.h'))) {
    throw "QnnInterface.h not found under $QnnRoot\include\QNN - is QNN_SDK_ROOT correct?"
}

$clang = Join-Path $LlvmMingw 'bin\aarch64-w64-mingw32-clang++.exe'
if (-not (Test-Path $clang)) {
    throw "aarch64 clang++ not found at $clang (install llvm-mingw or pass -LlvmMingw)."
}

$BuildDir = Join-Path $Root 'build'
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$out = Join-Path $BuildDir 'loom_accel_qualcomm.dll'

$srcs = @(
    (Join-Path $Root 'src\qnn_wrapper.cpp'),
    (Join-Path $Root 'src\loom_accel_qualcomm.cpp'),
    (Join-Path $Root 'src\layer_models.cpp')
)

$args = @(
    '-std=c++17', '-O2', '-DNDEBUG', '-shared',
    '-I', (Join-Path $Root 'include'),
    '-I', (Join-Path $Root 'src'),
    '-I', (Join-Path $QnnRoot 'include\QNN'),
    '-I', (Join-Path $QnnRoot 'include')
) + $srcs + @('-o', $out)

Write-Host "Building $out"
& $clang @args
if ($LASTEXITCODE -ne 0) { throw "clang++ failed (exit $LASTEXITCODE)" }
Write-Host "Built $out"
