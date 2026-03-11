@echo off
echo ==========================================================
echo      Loom NN Docs: Windows Requirements Setup
echo ==========================================================
echo.
echo This script will install Pandoc using Chocolatey.
echo You must run this script as Administrator.
echo.

:: Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Administrative privileges required.
    echo Please right-click and run this script as Administrator.
    echo.
    pause
    exit /b 1
)

:: Check if choco is installed
where choco >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Chocolatey is not installed. Installing Chocolatey first...
    @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
    if %errorLevel% neq 0 (
        echo [ERROR] Failed to install Chocolatey. Please install manually from chocolatey.org
        pause
        exit /b 1
    )
    echo [SUCCESS] Chocolatey installed.
) else (
    echo [INFO] Chocolatey found.
)

:: Install pandoc
echo.
echo [INFO] Installing Pandoc...
choco install pandoc -y

if %errorLevel% equ 0 (
    echo.
    echo ==========================================================
    echo [SUCCESS] Setup complete! You may need to restart your terminal.
    echo Then you can run: python compile_docs.py
    echo ==========================================================
) else (
    echo.
    echo [ERROR] Failed to install Pandoc.
)

pause
