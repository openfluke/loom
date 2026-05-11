@echo off
setlocal
cd /d "%~dp0"

:: welvet C-ABI Windows build script (64-bit targets only)
::
:: Usage:
::   build_windows.bat                    native windows/amd64
::   build_windows.bat amd64              windows/amd64
::   build_windows.bat arm64              windows/arm64
::   build_windows.bat android            android/arm64 + android/x86_64
::   build_windows.bat android_all        same as android (64-bit ABIs only)
::   build_windows.bat android_arm64      Android arm64 only
::   build_windows.bat android_x86_64     Android x86_64 only (emulator)
::   build_windows.bat all                windows/amd64 + android arm64 + android x86_64
::   build_windows.bat all --clean
::   build_windows.bat amd64 --test
::
:: Set ANDROID_NDK_HOME or ANDROID_HOME (with ndk\<version>) if Android targets fail.

set TARGET=amd64
set EXTRA_FLAGS=
set BUILD_ERRORS=0

for %%A in (%*) do (
    if "%%A"=="amd64"          set TARGET=amd64
    if "%%A"=="arm64"          set TARGET=arm64
    if "%%A"=="android"        set TARGET=android
    if "%%A"=="android_all"    set TARGET=android_all
    if "%%A"=="android_arm64"  set TARGET=android_arm64
    if "%%A"=="android_x86_64" set TARGET=android_x86_64
    if "%%A"=="all"            set TARGET=all
    if "%%A"=="--clean"        set DO_CLEAN=1
    if "%%A"=="--test"         set EXTRA_FLAGS=%EXTRA_FLAGS% -test
)

:: ── Pre-build Cleanup ────────────────────────────────────────────────────────

if "%DO_CLEAN%"=="1" (
    echo Cleaning dist...
    if exist dist rmdir /s /q dist
)
mkdir dist 2>nul

:: ── Dispatch ──────────────────────────────────────────────────────────────────

if "%TARGET%"=="all" (
    echo Building windows/amd64 + android/arm64 + android/x86_64...
    call :build_one windows amd64
    call :build_android_64
    goto :summary
)

if "%TARGET%"=="android" (
    echo Building android/arm64 + android/x86_64...
    call :build_android_64
    goto :summary
)

if "%TARGET%"=="android_all" (
    echo Building android 64-bit ABIs...
    call :build_android_64
    goto :summary
)

if "%TARGET%"=="android_arm64" (
    call :build_one android arm64
    goto :summary
)
if "%TARGET%"=="android_x86_64" (
    call :build_one android amd64
    goto :summary
)

:: Default: single Windows target (amd64 or arm64)
echo Building windows/%TARGET%...
call :build_one windows %TARGET%
goto :summary

:: ── Subroutines ───────────────────────────────────────────────────────────────

:build_one
go run builder.go -os %1 -arch %2 %EXTRA_FLAGS%
if errorlevel 1 set BUILD_ERRORS=1
goto :eof

:build_android_64
call :build_one android arm64
call :build_one android amd64
goto :eof

:summary
echo.
if "%BUILD_ERRORS%"=="1" (
    echo Some targets failed. Check output above.
    echo Make sure ANDROID_NDK_HOME is set if Android builds failed.
    echo   set ANDROID_NDK_HOME=C:\path\to\ndk
) else (
    echo All targets built successfully.
)
echo.
echo Build artifacts are in dist\
echo.

:: Mirror everything to Python source
echo Mirroring to Python source...
xcopy /s /i /y dist ..\..\..\python\src\welvet
echo.

endlocal
