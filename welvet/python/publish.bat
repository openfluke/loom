@echo off
echo === Building and Publishing welvet to PyPI ===
echo.

:: Activate conda base so we get the right python (build + twine live there)
call conda activate base 2>nul || call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base 2>nul

echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%i in (*.egg-info src\*.egg-info) do rmdir /s /q "%%i"

echo Building package...
python -m build
if errorlevel 1 (
    echo.
    echo [!] Build failed. Make sure build is installed:
    echo     pip install build twine
    exit /b 1
)

echo.
echo Build complete!
echo.

python -m twine check dist\*
if errorlevel 1 (
    echo.
    echo [!] Twine check failed. Install with:
    echo     pip install build twine
    exit /b 1
)

echo Package passes twine checks
echo.
echo Files to upload:
dir /b dist\
echo.

set /p CONFIRM="Upload welvet to PyPI? (y/N): "
if /i "%CONFIRM%"=="y" (
    echo Uploading to PyPI...
    python -m twine upload dist\*
    if errorlevel 1 (
        echo.
        echo [!] Upload failed. Check your PyPI credentials in C:\Users\%USERNAME%\.pypirc
        exit /b 1
    )
    echo.
    echo === Published Successfully ===
    echo View at: https://pypi.org/project/welvet/
    echo.
    echo Install with: pip install welvet
) else (
    echo Upload cancelled.
    echo.
    echo To upload manually:
    echo     python -m twine upload dist\*
)
