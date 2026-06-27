# Source before building/running CGO apps with webgpu on Windows amd64:
#   . C:\git\chaosglue\loom\scripts\env-windows-amd64.ps1
#
# Requires llvm-mingw at C:\llvm-mingw (see windows_amd64_install_webgpu.ps1).

$ErrorActionPreference = 'Stop'

$llvmRoot = 'C:\llvm-mingw'
$llvmBin = Join-Path $llvmRoot 'bin'
if (-not (Test-Path $llvmBin)) {
    throw "llvm-mingw not found at $llvmBin — run loom\scripts\windows_amd64_install_webgpu.ps1 first."
}

$env:CGO_ENABLED = '1'
$env:GOARCH = 'amd64'
Remove-Item Env:CGO_CFLAGS -ErrorAction SilentlyContinue
Remove-Item Env:CGO_CXXFLAGS -ErrorAction SilentlyContinue

$runtimeBin = Join-Path $llvmRoot 'x86_64-w64-mingw32\bin'
$env:CC = Join-Path $llvmBin 'x86_64-w64-mingw32-gcc.exe'
$env:CXX = Join-Path $llvmBin 'x86_64-w64-mingw32-g++.exe'

$env:Path = "$runtimeBin;$llvmBin;" + (($env:Path -split ';' | Where-Object {
    $_ -and $_ -notmatch 'llvm-mingw'
}) -join ';')

Write-Host 'CGO env: windows/amd64 (webgpu / llvm-mingw)'
Write-Host "  CGO_ENABLED=$env:CGO_ENABLED"
Write-Host "  GOARCH=$env:GOARCH"
Write-Host "  CC=$env:CC"
Write-Host "  runtime PATH=$runtimeBin"
