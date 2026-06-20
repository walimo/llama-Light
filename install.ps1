#!/usr/bin/env pwsh
# -----------------------------------------------------------------------------
# llama-Light Windows Installer — One-Command Setup (Prebuilt Binaries)
# Downloads prebuilt CUDA/CPU binaries from llama.cpp GitHub releases.
# No Visual Studio or CMake required — near-native speed via bundled executables.
# -----------------------------------------------------------------------------
set -euo pipefail

# ── ANSI helpers (works in Windows Terminal, PowerShell 7+, VS Code) ────────
$RED   = [Console]::ConsoleColor::Red
$GREEN = [Console]::ConsoleColor::Green
$YELLOW= [Console]::ConsoleColor::DarkYellow
$BLUE  = [Console]::ConsoleColor::Blue
$CYAN  = [Console]::ConsoleColor::Cyan

function Write-Step  { param($msg) Write-Host "[Step] $msg" -ForegroundColor $BLUE }
function Write-Ok    { param($msg) Write-Host "  [OK] $msg" -ForegroundColor $GREEN }
function Write-Info  { param($msg) Write-Host "  [i]  $msg" -ForegroundColor $CYAN }
function Write-Warn  { param($msg) Write-Host "  [!!] $msg" -ForegroundColor $YELLOW }
function Write-Error { param($msg) Write-Host "  [XX] $msg" -ForegroundColor $RED }

$ErrorActionPreference = "Stop"

# ── Version info ──────────────────────────────────────────────────────────────
$VERSION = "0.2.1"
$LLAMA_CPP_VERSION = "b9738"   # pinned to latest known-good release
$CACHE_ROOT = "$env:USERPROFILE\.cache\llama-light"
$VENV_DIR   = "$env:USERPROFILE\.local\share\llama-light-venv"
$LOG_FILE   = "$env:TEMP\llama-light-install.log"

# ── Banner ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor $BLUE
Write-Host "║  llama-Light v$VERSION — Windows Installer (Prebuilt)     ║" -ForegroundColor $BLUE
Write-Host "║  Lightning-fast LLM server via llama.cpp, zero overhead     ║" -ForegroundColor $BLUE
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor $Blue
Write-Host ""

# ── Helper: exit with error ───────────────────────────────────────────────────
function Die {
    param($msg)
    Write-Error "Error: $msg"
    Write-Host "  Log: $LOG_FILE"
    exit 1
}

# ── Pre-flight: check we're in a repo ────────────────────────────────────────
if (!(Test-Path "pyproject.toml") -and !(Test-Path "setup.py")) {
    Die "Please run this script from the root of the llama-Light source directory."
}

# ── [0/6] Prerequisites ──────────────────────────────────────────────────────
Write-Step "Checking prerequisites..."

# PowerShell 5.1+ is required; modern Windows 10/11 ships with 5.1, Windows 11
# with PowerShell 7 via Microsoft Store or winget.
$PS_VER = $PSVersionTable.PSVersion
if ($PS_VER.Major -lt 5) {
    Die "PowerShell $PS_VER detected — need v5.1 or later. Upgrade via winget install Microsoft.PowerShell."
}

# Python 3.10+ check
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Python not found in PATH." -ForegroundColor $YELLOW
    Write-Host "  Install from https://www.python.org/downloads/ (check 'Add to PATH')" -ForegroundColor $YELLOW
    Write-Host "  Then re-run this script."
    exit 1
}

$PY_VER = & python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>&1
if ($LASTEXITCODE -ne 0) { Die "python invocation failed" }
$PY_MAJOR = $PY_VER.Split('.')[0]
if ([int]$PY_MAJOR -lt 3) { Die "Python 3.x required (found v$PY_VER)" }
if ([int]$PY_MAJOR -eq 3 -and ([int]$PY_VER.Split('.')[1]) -lt 10) {
    Write-Warn "Python 3.10+ recommended (found v$PY_VER)"
}
Write-Ok "Python $PY_VER detected"

# ── [1/6] GPU Detection ──────────────────────────────────────────────────────
Write-Step "Detecting GPU and compute capabilities..."

$GPU_AVAILABLE = $false
$COMPUTE_CAP   = $null
$GPU_NAME      = "N/A"

# Try nvidia-smi first (Windows CUDA toolkit installs it)
$NVIDIA_SMI = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($NVIDIA_SMI) {
    try {
        $SMI_OUT = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($SMI_OUT)) {
            $parts = $SMI_OUT -split ','
            $GPU_NAME   = $parts[0].Trim()
            $COMPUTE_CAP = $parts[1].Trim() -replace '\.'
            if ($COMPUTE_CAP) {
                $GPU_AVAILABLE = $true
                Write-Ok "GPU detected: $GPU_NAME (SM$COMPUTE_CAP)"
            } else {
                $COMPUTE_CAP = "86"  # safe fallback
            }
        }
    } catch {
        Write-Warn "nvidia-smi returned error: $_"
    }
}

if ($GPU_AVAILABLE) {
    # ── [2/6] CUDA Runtime Library Download ──────────────────────────────
    # llama.cpp ships CUDA runtime libs separately (cudart + cublas).
    # We download these to ensure the CUDA binary can find its runtime.
    Write-Step "Downloading CUDA runtime library (cudart)..."

    # Select CUDA version based on compute capability
    $cap_num = [int]$COMPUTE_CAP / 10
    if ($cap_num -ge 12) {
        $CUDA_TAG = "13.3"
    } elseif ($cap_num -ge 8) {
        $CUDA_TAG = "12.4"
    } else {
        $CUDA_TAG = "12.4"  # fallback
    }

    $CUDA_ASSET = "cudart-llama-bin-win-cuda-$CUDA_TAG-x64.zip"
    $CACHE_CUDA = "$CACHE_ROOT\cuda\$CUDA_TAG"

    if (Test-Path "$CACHE_CUDA\cudart64.dll") {
        Write-Ok "CUDA runtime cached (cudart-$CUDA_TAG)"
    } else {
        $RELEASE_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        try {
            $RELEASE = Invoke-RestMethod -Uri $RELEASE_URL -Method Get
            $CUDART_URL = ($RELEASE.assets | Where-Object { $_.name -eq $CUDA_ASSET } | Select-Object -First 1).browser_download_url
            if (-not $CUDART_URL) {
                Write-Warn "No cudart asset '$CUDA_ASSET' found in release $LLAMA_CPP_VERSION"
            } else {
                $DL_DIR = "$CACHE_ROOT\_dl"
                New-Item -ItemType Directory -Force -Path $DL_DIR | Out-Null
                $DL_ZIP = "$DL_DIR\cudart.zip"
                Write-Info "Downloading cudart ($CUDA_TAG)..."
                Invoke-WebRequest -Uri $CUDART_URL -OutFile $DL_ZIP -UseBasicParsing -TimeoutSec 300
                New-Item -ItemType Directory -Force -Path $CACHE_CUDA | Out-Null
                Expand-Archive -Path $DL_ZIP -DestinationPath $CACHE_CUDA -Force
                Remove-Item $DL_ZIP -Force
                Remove-Item $DL_DIR -Recurse -Force
                Write-Ok "CUDA runtime extracted to $CACHE_CUDA"
            }
        } catch {
            Write-Warn "Failed to download cudart: $_"
        }
    }

    # ── [3/6] Download llama-server CUDA binary ──────────────────────────
    Write-Step "Downloading llama.cpp CUDA binary..."

    $BIN_ASSET = "llama-$LLAMA_CPP_VERSION-bin-win-cuda-$CUDA_TAG-x64.zip"
    $CACHE_BIN = "$CACHE_ROOT\llama-cpp\$LLAMA_CPP_VERSION\sm$COMPUTE_CAP"

    if (Test-Path "$CACHE_BIN\llama-server.exe") {
        Write-Ok "CUDA binary cached (sm$COMPUTE_CAP)"
    } else {
        $RELEASE_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        try {
            $RELEASE = Invoke-RestMethod -Uri $RELEASE_URL -Method Get
            $BIN_URL = ($RELEASE.assets | Where-Object { $_.name -eq $BIN_ASSET } | Select-Object -First 1).browser_download_url
            if (-not $BIN_URL) {
                Write-Warn "No CUDA binary asset '$BIN_ASSET' found — falling back to CPU binary"
                $BIN_ASSET = "llama-$LLAMA_CPP_VERSION-bin-win-cpu-x64.zip"
                $BIN_URL   = ($RELEASE.assets | Where-Object { $_.name -eq $BIN_ASSET } | Select-Object -First 1).browser_download_url
            }
            if (-not $BIN_URL) { Die "No matching binary asset found for $LLAMA_CPP_VERSION" }

            $DL_DIR  = "$CACHE_ROOT\_dl"
            New-Item -ItemType Directory -Force -Path $DL_DIR | Out-Null
            $DL_ZIP  = "$DL_DIR\llama-bin.zip"
            Write-Info "Downloading binary ($BIN_ASSET)..."
            Invoke-WebRequest -Uri $BIN_URL -OutFile $DL_ZIP -UseBasicParsing -TimeoutSec 300
            New-Item -ItemType Directory -Force -Path $CACHE_BIN | Out-Null
            Expand-Archive -Path $DL_ZIP -DestinationPath $CACHE_BIN -Force
            Remove-Item $DL_ZIP -Force
            Remove-Item $DL_DIR -Recurse -Force
            Write-Ok "Binary extracted to $CACHE_BIN"
        } catch {
            Die "Failed to download binary: $_"
        }
    }
} else {
    # No GPU — fall back to CPU binary
    Write-Step "No NVIDIA GPU detected — downloading CPU-only binary..."

    $BIN_ASSET = "llama-$LLAMA_CPP_VERSION-bin-win-cpu-x64.zip"
    $CACHE_BIN = "$CACHE_ROOT\llama-cpp\$LLAMA_CPP_VERSION\sm00"

    if (Test-Path "$CACHE_BIN\llama-server.exe") {
        Write-Ok "CPU binary cached"
    } else {
        $RELEASE_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
        try {
            $RELEASE = Invoke-RestMethod -Uri $RELEASE_URL -Method Get
            $BIN_URL = ($RELEASE.assets | Where-Object { $_.name -eq $BIN_ASSET } | Select-Object -First 1).browser_download_url
            if (-not $BIN_URL) { Die "No CPU binary asset '$BIN_ASSET' found" }

            $DL_DIR  = "$CACHE_ROOT\_dl"
            New-Item -ItemType Directory -Force -Path $DL_DIR | Out-Null
            $DL_ZIP  = "$DL_DIR\llama-bin.zip"
            Write-Info "Downloading CPU binary..."
            Invoke-WebRequest -Uri $BIN_URL -OutFile $DL_ZIP -UseBasicParsing -TimeoutSec 300
            New-Item -ItemType Directory -Force -Path $CACHE_BIN | Out-Null
            Expand-Archive -Path $DL_ZIP -DestinationPath $CACHE_BIN -Force
            Remove-Item $DL_ZIP -Force
            Remove-Item $DL_DIR -Recurse -Force
            Write-Ok "CPU binary extracted to $CACHE_BIN"
        } catch {
            Die "Failed to download CPU binary: $_"
        }
    }
}

# ── [4/6] Python virtual environment & pip install ──────────────────────────
Write-Step "Setting up Python virtual environment..."

if (Test-Path $VENV_DIR) {
    Write-Info "Removing stale venv..."
    Remove-Item -Recurse -Force $VENV_DIR
}

python -m venv $VENV_DIR | Out-Null
if ($LASTEXITCODE -ne 0) { Die "Failed to create virtual environment" }

. "$VENV_DIR\Scripts\Activate.ps1"

pip install --upgrade pip -q | Out-Null
pip install . -q
if ($LASTEXITCODE -ne 0) { Die "pip install failed" }

if (!(Test-Path "$VENV_DIR\Scripts\llama.exe")) {
    # Sometimes pip puts it in Scripts/ directly as 'llama'
    if (!(Test-Path "$VENV_DIR\Scripts\llama")) {
        Die "Expected 'llama' binary not found in venv"
    }
}

# Symlink to %USERPROFILE%\.local\bin for easy PATH access
$LOCAL_BIN = "$env:USERPROFILE\.local\bin"
New-Item -ItemType Directory -Force -Path $LOCAL_BIN | Out-Null
$LINK_PATH = "$LOCAL_BIN\llama.ps1"

# Create a small wrapper that activates the venv and calls llama
@"
`$venv = "$VENV_DIR\Scripts\python.exe"
`$cmd = @("$VENV_DIR\Scripts\llama.exe", @args)
& `$venv @cmd
"@ | Set-Content -Path $LINK_PATH -Force

# Also add the venv Scripts dir directly to user PATH if not present
$USER_PATH = [Environment]::GetEnvironmentVariable("PATH", "User")
if (-not $USER_PATH -or ($USER_PATH -split ';' -notcontains "$LOCAL_BIN")) {
    Write-Info "Adding $LOCAL_BIN to user PATH..."
    $NEW_PATH = $LOCAL_BIN
    if ($USER_PATH) { $NEW_PATH = "$USER_PATH;$LOCAL_BIN" }
    [Environment]::SetEnvironmentVariable("PATH", $NEW_PATH, "User")
    # Also update current session
    $env:PATH = "$NEW_PATH;$env:PATH"
}

Write-Ok "Package installed to virtual environment"

# ── [5/6] Verify binary resolution ──────────────────────────────────────────
Write-Step "Verifying binary resolution..."

$LLAMA_PY = "$VENV_DIR\Lib\site-packages\llama_light"
if (Test-Path "$LLAMA_PY\_bincheck.py") {
    # Run a quick import test
    python -c "from llama_light import _bincheck; b = _bincheck.locate_main_bin(); Write-Host '  Binary resolved: ' + b" 2>&1 | Out-Null
} else {
    Write-Info "Binary check module not found — skipping resolution test"
}

# ── [6/6] Windows Service Registration ──────────────────────────────────────
Write-Step "Registering Windows service..."

$SERV_NAME = "llama-light"
$LLAMA_CMD = "$VENV_DIR\Scripts\llama.exe _run"
$WORK_DIR  = $PWD

if (Get-Service -Name $SERV_NAME -ErrorAction SilentlyContinue) {
    Write-Info "Service already registered — updating..."
    sc.exe stop $SERV_NAME 2>&1 | Out-Null
    sc.exe delete $SERV_NAME 2>&1 | Out-Null
}

# Register the service using sc.exe
# Note: sc.exe requires the binary path to be quoted if it contains spaces
sc.exe create "$SERV_NAME" binPath= "\"$LLAMA_CMD\"" start= demand obj= LocalSystem 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "sc.exe create failed — service registration incomplete."
    Write-Info "Manual: sc.exe create `"$SERV_NAME`" binPath= `\"$LLAMA_CMD`\" start= demand"
} else {
    Write-Ok "Windows service registered: $SERV_NAME"
    Write-Info "Start it with: Start-Service $SERV_NAME"
    Write-Info "Stop it with:  Stop-Service $SERV_NAME"
    Write-Info "Logs:          Get-WinEvent -LogName Application -FilterXPath '*[System[Provider[`"$SERV_NAME`"]]]*' -Newest 20"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor $GREEN
Write-Host ""
Write-Host "  To start the server:" -ForegroundColor $Blue
Write-Host "    llama config set default_model <your-model.gguf>"
Write-Host "    llama start"
Write-Host ""
Write-Host "  Or run the service:" -ForegroundColor $Blue
Write-Host "    Start-Service $SERV_NAME"
Write-Host ""
Write-Host "  Full log saved to: $LOG_FILE"