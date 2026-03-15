@echo off
setlocal
cd /d "%~dp0"

:: welvet C-ABI Windows build script
::
:: Usage:
::   build_windows.bat                    native amd64
::   build_windows.bat arm64
::   build_windows.bat x86
::   build_windows.bat all                all Windows targets
::   build_windows.bat all --clean
::   build_windows.bat amd64 --test

set TARGET_ARCH=amd64
set EXTRA_FLAGS=

for %%A in (%*) do (
    if "%%A"=="arm64"   set TARGET_ARCH=arm64
    if "%%A"=="x86"     set TARGET_ARCH=x86
    if "%%A"=="amd64"   set TARGET_ARCH=amd64
    if "%%A"=="all"     set TARGET_ARCH=all
    if "%%A"=="--clean" set EXTRA_FLAGS=%EXTRA_FLAGS% -clean
    if "%%A"=="--test"  set EXTRA_FLAGS=%EXTRA_FLAGS% -test
)

if "%TARGET_ARCH%"=="all" (
    echo Building all Windows targets...
    go run builder.go -os windows -arch amd64 %EXTRA_FLAGS%
    go run builder.go -os windows -arch arm64 %EXTRA_FLAGS%
    go run builder.go -os windows -arch 386   %EXTRA_FLAGS%
) else (
    echo Building Windows %TARGET_ARCH%...
    go run builder.go -os windows -arch %TARGET_ARCH% %EXTRA_FLAGS%
)

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build complete. Files are in dist\windows_%TARGET_ARCH%
echo.
endlocal
