# build.ps1 — configure + build loom_accel_qualcomm.dll. Mirrors accel/intel/build.sh.
#   . .\setup_env.ps1   # sets QNN_SDK_ROOT + PATH
#   .\build.ps1
#
# Requires: CMake, and an MSVC/clang-cl toolchain for aarch64-windows-msvc
# (the QNN backend DLLs are MSVC-ABI; the plugin loads them at runtime via
# LoadLibrary, so building the plugin itself with MSVC keeps ABI consistent).

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $env:QNN_SDK_ROOT) {
    Write-Error "QNN_SDK_ROOT not set. Run:  . .\setup_env.ps1  (after install_qairt.ps1)"
}

$BuildDir = Join-Path $Root 'build'
cmake -B $BuildDir -S $Root -A ARM64
cmake --build $BuildDir --config Release

$dll = Join-Path $BuildDir 'Release\loom_accel_qualcomm.dll'
if (-not (Test-Path $dll)) { $dll = Join-Path $BuildDir 'loom_accel_qualcomm.dll' }
if (Test-Path $dll) {
    Copy-Item $dll (Join-Path $BuildDir 'loom_accel_qualcomm.dll') -Force -ErrorAction SilentlyContinue
    Write-Host "Built $dll"
} else {
    Write-Warning "Build finished but loom_accel_qualcomm.dll not found under $BuildDir"
}
