@echo off
setlocal
cd /d "%~dp0"

:: welvet C-ABI Windows build script
::
:: Usage:
::   build_windows.bat                    native windows/amd64
::   build_windows.bat amd64              windows/amd64
::   build_windows.bat android            android/arm64 + android/x86_64 (64-bit only)
::   build_windows.bat android_all        all 4 Android targets (32-bit ones may fail)
::   build_windows.bat android_arm64      Android arm64 only  (phones)
::   build_windows.bat android_armv7      Android armv7 only  (32-bit, may fail)
::   build_windows.bat android_x86_64     Android x86_64 only (emulator)
::   build_windows.bat android_x86        Android x86 only    (32-bit, may fail)
::   build_windows.bat all                windows/amd64 + android arm64 + android x86_64
::   build_windows.bat all --clean
::   build_windows.bat amd64 --test
::
:: Note: android_armv7 and android_x86 require 32-bit address space support in
::       the codebase. They currently fail due to oversized arrays in training_ext.go.
::       Set ANDROID_NDK_HOME before running Android targets.

set TARGET=amd64
set EXTRA_FLAGS=
set BUILD_ERRORS=0

for %%A in (%*) do (
    if "%%A"=="amd64"          set TARGET=amd64
    if "%%A"=="android"        set TARGET=android
    if "%%A"=="android_all"    set TARGET=android_all
    if "%%A"=="android_arm64"  set TARGET=android_arm64
    if "%%A"=="android_armv7"  set TARGET=android_armv7
    if "%%A"=="android_x86_64" set TARGET=android_x86_64
    if "%%A"=="android_x86"    set TARGET=android_x86
    if "%%A"=="all"            set TARGET=all
    if "%%A"=="--clean"        set EXTRA_FLAGS=%EXTRA_FLAGS% -clean
    if "%%A"=="--test"         set EXTRA_FLAGS=%EXTRA_FLAGS% -test
)

:: ── Dispatch ──────────────────────────────────────────────────────────────────

if "%TARGET%"=="all" (
    echo Building windows/amd64 + android/arm64 + android/x86_64...
    call :build_one windows amd64
    call :build_android_64
    goto :summary
)

if "%TARGET%"=="android" (
    echo Building android/arm64 + android/x86_64 ^(64-bit only^)...
    call :build_android_64
    goto :summary
)

if "%TARGET%"=="android_all" (
    echo Building all 4 Android targets ^(32-bit may fail^)...
    call :build_android_all
    goto :summary
)

if "%TARGET%"=="android_arm64" (
    call :build_one android arm64
    goto :summary
)
if "%TARGET%"=="android_armv7" (
    call :build_one android arm
    goto :summary
)
if "%TARGET%"=="android_x86_64" (
    call :build_one android amd64
    goto :summary
)
if "%TARGET%"=="android_x86" (
    call :build_one android 386
    goto :summary
)

:: Default: single Windows target
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

:build_android_all
call :build_one android arm64
call :build_one android amd64
call :build_one android arm
call :build_one android 386
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

:: Mirror Android .so files to Flutter jniLibs (native/android)
set FLUTTER_NATIVE=..\..\..\..\..\soulglitch\native\android
echo Mirroring Android .so to Flutter jniLibs...
if exist dist\android_arm64\welvet.so (
    if not exist "%FLUTTER_NATIVE%\arm64-v8a" mkdir "%FLUTTER_NATIVE%\arm64-v8a"
    copy /y dist\android_arm64\welvet.so "%FLUTTER_NATIVE%\arm64-v8a\libwelvet.so"
    echo   arm64-v8a ^> libwelvet.so
)
if exist dist\android_x86_64\welvet.so (
    if not exist "%FLUTTER_NATIVE%\x86_64" mkdir "%FLUTTER_NATIVE%\x86_64"
    copy /y dist\android_x86_64\welvet.so "%FLUTTER_NATIVE%\x86_64\libwelvet.so"
    echo   x86_64    ^> libwelvet.so
)
echo.

endlocal
