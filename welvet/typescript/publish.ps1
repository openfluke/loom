# @openfluke/welvet NPM Publication Script (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=== Building and Publishing @openfluke/welvet to NPM ===" -ForegroundColor Cyan
Write-Host ""

# Ensure we are in the correct directory
Set-Location $PSScriptRoot

# 1. Check for WASM binary
if (-not (Test-Path "assets\main.wasm")) {
    Write-Host "⚠️  main.wasm missing from assets/." -ForegroundColor Yellow
    Write-Host "Attempting to rebuild WASM..."
    Push-Location ..\wasm
    .\build.bat
    Pop-Location
}

# 2. Build TypeScript package
Write-Host "Running full build..." -ForegroundColor Cyan
npm run build

Write-Host ""
Write-Host "✓ Build complete!" -ForegroundColor Green
Write-Host ""
# 3. Publish
Write-Host "Package info:"
$packageJson = Get-Content package.json | ConvertFrom-Json
Write-Host "  Name:    $($packageJson.name)"
Write-Host "  Version: $($packageJson.version)"
Write-Host ""

# Check login status
$npmWhoAmI = npm whoami 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Logged in as: $npmWhoAmI" -ForegroundColor Green
    Write-Host ""
    
    $confirmation = Read-Host "Publish @openfluke/welvet to NPM? (y/N)"
    if ($confirmation -eq "y") {
        npm publish --access public
    } else {
        Write-Host "Publish cancelled."
    }
} else {
    Write-Host "⚠️  Not logged in to NPM. Run 'npm login' first." -ForegroundColor Yellow
}
