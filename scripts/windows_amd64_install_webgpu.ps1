# Install CGO toolchain for github.com/openfluke/webgpu on Windows amd64.
#
# WebGPU (wgpu-native) is already bundled inside the Go module
# (github.com/openfluke/webgpu/wgpu/lib/windows/amd64). Go needs a MinGW C
# compiler and CGO_ENABLED=1 to link it — there is no separate WebGPU runtime
# to install.
#
# Usage:
#   .\loom\scripts\windows_amd64_install_webgpu.ps1
#   .\loom\scripts\windows_amd64_install_webgpu.bat
#
# After install, open a new terminal (or dot-source env-windows-amd64.ps1) and:
#   cd loom\lucy
#   go run .

param(
    [string]$InstallRoot = 'C:\llvm-mingw',
    [switch]$SkipDownload
)

$ErrorActionPreference = 'Stop'

$LLVMVersion = '20260602'
$ZipName = "llvm-mingw-$LLVMVersion-ucrt-x86_64.zip"
$ZipUrl = "https://github.com/mstorsjo/llvm-mingw/releases/download/$LLVMVersion/$ZipName"
$ZipSha256 = '3de3eda9377bbaf35f8c9001f190380f63b8ee981fa55d3ae9d7cce7c6ad7c70'

$Gcc = Join-Path $InstallRoot 'bin\x86_64-w64-mingw32-gcc.exe'
$Gpp = Join-Path $InstallRoot 'bin\x86_64-w64-mingw32-g++.exe'
$RuntimeBin = Join-Path $InstallRoot 'x86_64-w64-mingw32\bin'

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-LlvmMingwInstalled {
    (Test-Path $Gcc) -and (Test-Path $Gpp)
}

function Install-LlvmMingwFromZip {
    $tempDir = Join-Path $env:TEMP "llvm-mingw-install-$PID"
    $zipPath = Join-Path $tempDir $ZipName

    try {
        New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

        Write-Step "Downloading llvm-mingw $LLVMVersion (UCRT, x86_64)"
        Write-Host "  $ZipUrl"
        Invoke-WebRequest -Uri $ZipUrl -OutFile $zipPath -UseBasicParsing

        $hash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $ZipSha256) {
            throw "SHA256 mismatch for $zipPath`n  expected: $ZipSha256`n  got:      $hash"
        }

        Write-Step "Extracting to $tempDir"
        Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force

        $extracted = Get-ChildItem -Path $tempDir -Directory |
            Where-Object { $_.Name -like 'llvm-mingw-*' } |
            Select-Object -First 1
        if (-not $extracted) {
            throw "Could not find llvm-mingw folder inside $tempDir"
        }

        if (Test-Path $InstallRoot) {
            Write-Host "Removing existing $InstallRoot"
            Remove-Item -Path $InstallRoot -Recurse -Force
        }

        Write-Step "Installing llvm-mingw -> $InstallRoot"
        Move-Item -Path $extracted.FullName -Destination $InstallRoot
    }
    finally {
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Enable-CgoForUser {
    $current = [Environment]::GetEnvironmentVariable('CGO_ENABLED', 'User')
    if ($current -ne '1') {
        [Environment]::SetEnvironmentVariable('CGO_ENABLED', '1', 'User')
        Write-Host 'Set user CGO_ENABLED=1 (new terminals will pick this up)'
    } else {
        Write-Host 'User CGO_ENABLED already 1'
    }
}

function Show-EnvInstructions {
    Write-Step 'Current-session CGO env (run before go build/run)'
    Write-Host @"
. `"$PSScriptRoot\env-windows-amd64.ps1`"
"@ -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Or manually:'
    Write-Host @"
`$env:CGO_ENABLED = '1'
`$env:GOARCH = 'amd64'
`$env:CC = '$Gcc'
`$env:CXX = '$Gpp'
`$env:Path = '$RuntimeBin;$InstallRoot\bin;' + `$env:Path
"@ -ForegroundColor DarkGray
    Write-Host ''
    Write-Host 'Then build Lucy:'
    Write-Host '  cd loom\lucy' -ForegroundColor Yellow
    Write-Host '  go run .' -ForegroundColor Yellow
}

Write-Host 'windows/amd64 WebGPU CGO toolchain install'
Write-Host '  (installs llvm-mingw; webgpu native lib ships in the Go module)'

if ((go env GOOS) -ne 'windows' -or (go env GOARCH) -ne 'amd64') {
    Write-Warning "Go reports $(go env GOOS)/$(go env GOARCH); this script targets windows/amd64."
}

if (Test-LlvmMingwInstalled) {
    Write-Step "llvm-mingw already installed at $InstallRoot"
} elseif ($SkipDownload) {
    throw "llvm-mingw not found at $InstallRoot and -SkipDownload was set."
} else {
    Install-LlvmMingwFromZip
}

if (-not (Test-LlvmMingwInstalled)) {
    throw "Install failed: missing $Gcc"
}

Write-Step 'Verifying compiler'
& $Gcc --version | Select-Object -First 1

Enable-CgoForUser
Show-EnvInstructions

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
