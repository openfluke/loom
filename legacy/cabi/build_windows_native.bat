@echo off
:: LOOM C ABI - Native Windows Build Script
:: Builds libloom.dll + universal_test.exe + quick_talk.exe natively on Windows
:: Requires: Go (CGO-enabled) and gcc in PATH (MSYS2/MinGW or TDM-GCC)
::
:: Usage:
::   build_windows_native.bat              (x86_64, default)
::   build_windows_native.bat arm64        (requires llvm-mingw)
::   build_windows_native.bat x86
::   build_windows_native.bat clean        (cleans output dir first)

setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo =============================================
echo  LOOM C ABI - Windows Native Build
echo =============================================
echo.

:: ── Parse arguments ───────────────────────────────────────────────────────────
set ARCH=x86_64
set CLEAN=0

for %%A in (%*) do (
    if /I "%%A"=="clean" set CLEAN=1
    if /I "%%A"=="x86_64" set ARCH=x86_64
    if /I "%%A"=="amd64"  set ARCH=x86_64
    if /I "%%A"=="arm64"  set ARCH=arm64
    if /I "%%A"=="x86"    set ARCH=x86
    if /I "%%A"=="386"    set ARCH=x86
)

:: ── Map arch to Go/compiler settings ─────────────────────────────────────────
if "%ARCH%"=="x86_64" (
    set GOARCH=amd64
    set CC=gcc
)
if "%ARCH%"=="arm64" (
    set GOARCH=arm64
    set CC=aarch64-w64-mingw32-gcc
)
if "%ARCH%"=="x86" (
    set GOARCH=386
    set CC=i686-w64-mingw32-gcc
)

set OUTPUT_DIR=compiled\windows_%ARCH%
set LIB_NAME=libloom.dll

echo Arch       : %ARCH%  (GOARCH=%GOARCH%)
echo Compiler   : %CC%
echo Output dir : %OUTPUT_DIR%
echo.

:: ── Optional clean ────────────────────────────────────────────────────────────
if "%CLEAN%"=="1" (
    echo Cleaning %OUTPUT_DIR% ...
    if exist "%OUTPUT_DIR%" rd /s /q "%OUTPUT_DIR%"
)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: ── Check gcc ─────────────────────────────────────────────────────────────────
where %CC% >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: %CC% not found in PATH.
    echo.
    echo Install MSYS2 from https://www.msys2.org/ then run:
    echo   pacman -S mingw-w64-x86_64-gcc
    echo Add C:\msys64\mingw64\bin to your PATH and retry.
    exit /b 1
)

:: ── Build Go shared library (DLL) ─────────────────────────────────────────────
echo Building %LIB_NAME% ...
set GOOS=windows
set CGO_ENABLED=1

go build -buildmode=c-shared -o "%OUTPUT_DIR%\%LIB_NAME%" main.go transformer.go
if errorlevel 1 (
    echo ERROR: Go build failed.
    exit /b 1
)
echo   OK %OUTPUT_DIR%\%LIB_NAME%
echo.

:: ── Build universal_test.exe ──────────────────────────────────────────────────
echo Building universal_test.exe ...
%CC% -I"%OUTPUT_DIR%" -o "%OUTPUT_DIR%\universal_test.exe" universal_test.c -L"%OUTPUT_DIR%" -lloom -lm
if errorlevel 1 (
    echo   WARNING: universal_test.exe failed to compile.
) else (
    echo   OK %OUTPUT_DIR%\universal_test.exe
)
echo.

:: ── Build quick_talk.exe ──────────────────────────────────────────────────────
echo Building quick_talk.exe ...
%CC% -I"%OUTPUT_DIR%" -o "%OUTPUT_DIR%\quick_talk.exe" quick_talk.c -L"%OUTPUT_DIR%" -lloom -lm
if errorlevel 1 (
    echo   WARNING: quick_talk.exe failed to compile.
) else (
    echo   OK %OUTPUT_DIR%\quick_talk.exe
)
echo.

:: ── Summary ───────────────────────────────────────────────────────────────────
echo =============================================
echo  Build Complete
echo =============================================
echo.
dir "%OUTPUT_DIR%"
echo.
echo Run:
echo   cd %OUTPUT_DIR% ^&^& universal_test.exe
echo   cd %OUTPUT_DIR% ^&^& quick_talk.exe
echo.
echo Or with a model path:
echo   set LOOM_MODEL_PATH=C:\path\to\model\snapshot
echo   cd %OUTPUT_DIR% ^&^& quick_talk.exe
echo.

endlocal
