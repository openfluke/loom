@echo off
REM Build welvet WASM for use with the TypeScript package.
REM Output: welvet/typescript/dist/main.wasm + wasm_exec.js

set SCRIPT_DIR=%~dp0
set DIST_DIR=%SCRIPT_DIR%..\typescript\dist

if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"

echo Building welvet.wasm...
set GOOS=js
set GOARCH=wasm
go build -o "%DIST_DIR%\main.wasm" "%SCRIPT_DIR%main.go"
set GOOS=
set GOARCH=
if errorlevel 1 (
    echo [FAILED] WASM build failed.
    exit /b 1
)

for /f "delims=" %%i in ('go env GOROOT') do set GOROOT=%%i
echo Copying wasm_exec.js...
copy "%GOROOT%\misc\wasm\wasm_exec.js" "%DIST_DIR%\wasm_exec.js" >nul

echo Copying HTML benchmark/verify files...
for %%f in ("%SCRIPT_DIR%*.html") do copy "%%f" "%DIST_DIR%\" >nul

echo.
echo [OK] Build complete:
dir "%DIST_DIR%\main.wasm" "%DIST_DIR%\wasm_exec.js"
