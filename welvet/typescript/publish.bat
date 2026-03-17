@echo off
REM @openfluke/welvet NPM Publication Script (Batch)

setlocal enabledelayedexpansion

echo === Building and Publishing @openfluke/welvet to NPM ===
echo.

REM Ensure we are in the correct directory
cd /d "%~dp0"

REM 1. Check for WASM binary
if not exist "assets\main.wasm" (
    echo [WARNING] main.wasm missing from assets/.
    echo Attempting to rebuild WASM...
    pushd ..\wasm
    call build.bat
    popd
    if errorlevel 1 (
        echo [ERROR] WASM build failed.
        pause
        exit /b 1
    )
)

REM 2. Build TypeScript package
echo Running full build...
call npm run build
if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [OK] Build complete!
echo.

REM 3. Publish
echo Package info:
for /f "tokens=2 delims=:," %%a in ('findstr /c:\"name\" package.json') do set PKG_NAME=%%~a
for /f "tokens=2 delims=:," %%a in ('findstr /c:\"version\" package.json') do set PKG_VER=%%~a
echo   Name:    %PKG_NAME%
echo   Version: %PKG_VER%
echo.

REM Check login status
call npm whoami >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('npm whoami') do set NPM_USER=%%i
    echo [OK] Logged in as: !NPM_USER!
    echo.
    set /p confirm="Publish @openfluke/welvet to NPM? (y/N): "
    if /i "!confirm!"=="y" (
        call npm publish --access public
    ) else (
        echo Publish cancelled.
    )
) else (
    echo [WARNING] Not logged in to NPM.
    echo Please run "npm login" first.
)

pause
