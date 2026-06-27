@echo off
setlocal
REM Install CGO toolchain for github.com/openfluke/webgpu on Windows amd64.
REM WebGPU native libs ship inside the Go module; this installs llvm-mingw only.
REM
REM Usage:
REM   loom\scripts\windows_amd64_install_webgpu.bat

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows_amd64_install_webgpu.ps1" %*
if errorlevel 1 (
    echo.
    echo Install failed.
    exit /b 1
)

echo.
echo Open a new terminal, then:
echo   . loom\scripts\env-windows-amd64.ps1
echo   cd loom\lucy
echo   go run .
exit /b 0
