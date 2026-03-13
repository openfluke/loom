@echo off
setlocal
cd /d "%~dp0"

echo Building and Verifying Welvet C-ABI for Windows...
go run builder.go -os windows -arch amd64 -clean -test

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build complete. Files are in welvet/cabi/dist/windows_amd64
echo.
endlocal
