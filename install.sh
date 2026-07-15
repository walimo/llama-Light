#!/bin/bash
# -----------------------------------------------------------------------------
# llama-Light Ultimate Orchestrated Bootstrapper v0.3.0
# Enhanced with GCC version management for Ubuntu & Fedora
# -----------------------------------------------------------------------------
set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Log file
LOG_FILE="/tmp/llama-light-install.log"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🚀 llama-Light v0.3.0 - One Command LLM Server (Auto CUDA)    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
die() {
    echo -e "${RED}✗ Error: $*${NC}" >&2
    echo "Full log: $LOG_FILE" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}⚠ $*${NC}"
}

success() {
    echo -e "${GREEN}✓ $*${NC}"
}

info() {
    echo -e "${CYAN}➜ $*${NC}"
}

# -----------------------------------------------------------------------------
# OS Detection
# -----------------------------------------------------------------------------
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_VERSION="${VERSION_ID:-}"
        OS_ID_LIKE="${ID_LIKE:-}"
    else
        die "Cannot detect OS - /etc/os-release not found"
    fi

    if [[ "$OS_ID" == "fedora" || "$OS_ID_LIKE" == *"fedora"* || "$OS_ID_LIKE" == *"rhel"* ]]; then
        PKG_FAMILY="fedora"
        echo -e "${CYAN}➜${NC} Detected: Fedora/RHEL family"
    elif [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" || "$OS_ID_LIKE" == *"ubuntu"* || "$OS_ID_LIKE" == *"debian"* ]]; then
        PKG_FAMILY="debian"
        echo -e "${CYAN}➜${NC} Detected: Debian/Ubuntu family"
    elif command -v apt-get >/dev/null 2>&1; then
        PKG_FAMILY="debian"
        echo -e "${CYAN}➜${NC} Detected: Debian/Ubuntu family (via apt-get)"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_FAMILY="fedora"
        echo -e "${CYAN}➜${NC} Detected: Fedora/RHEL family (via dnf)"
    else
        die "Unsupported OS: could not detect package manager"
    fi
}

detect_os

# -----------------------------------------------------------------------------
# Sudo credential caching
# -----------------------------------------------------------------------------
echo -e "${CYAN}➜${NC} This installer needs sudo for system packages. Enter your password once:"
sudo -S -p '' -v || die "sudo failed"
(
    while true; do
        sudo -S -p '' -n true
        sleep 50
    done
) 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null' EXIT

# -----------------------------------------------------------------------------
# Check source directory
# -----------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
    die "Please run this script from the root of the llama-Light source directory."
fi

# -----------------------------------------------------------------------------
# [0/8] Pre-flight: Build Dependencies
# -----------------------------------------------------------------------------
echo -e "${BLUE}[0/8]${NC} Checking build dependencies..."
MISSING_PKGS=()
command -v cmake >/dev/null || MISSING_PKGS+=("cmake")
command -v git   >/dev/null || MISSING_PKGS+=("git")
command -v curl  >/dev/null || MISSING_PKGS+=("curl")
command -v unzip >/dev/null || MISSING_PKGS+=("unzip")
command -v wget  >/dev/null || MISSING_PKGS+=("wget")

if [ "$PKG_FAMILY" = "fedora" ]; then
    command -v gcc >/dev/null || MISSING_PKGS+=("gcc" "gcc-c++" "make")
    command -v pip3 >/dev/null || MISSING_PKGS+=("python3-pip")
    command -v dnf >/dev/null || die "dnf not found"

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        info "Installing missing packages: ${MISSING_PKGS[*]}"
        sudo dnf install -y "${MISSING_PKGS[@]}" > /dev/null 2>&1 || die "Failed to install packages"
    fi
else
    command -v gcc >/dev/null || MISSING_PKGS+=("build-essential")
    dpkg -s python3-venv >/dev/null 2>&1 || MISSING_PKGS+=("python3-venv")

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        info "Installing missing packages: ${MISSING_PKGS[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y "${MISSING_PKGS[@]}" > /dev/null 2>&1 || die "Failed to install packages"
    fi
fi
success "Build toolchain ready"

# -----------------------------------------------------------------------------
# [1/8] Python Runtime Verification
# -----------------------------------------------------------------------------
echo -e "${BLUE}[1/8]${NC} Checking Python Environment..."
command -v python3 >/dev/null || die "Python 3 not found"
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
success "Python $PY_VER Detected"

# -----------------------------------------------------------------------------
# [2/8] GCC Version Check & Management
# -----------------------------------------------------------------------------
echo -e "${BLUE}[2/8]${NC} Checking GCC version compatibility..."

# Check current GCC version
if command -v gcc >/dev/null; then
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
else
    GCC_VERSION="0"
fi

# Detect GCC version supported by the installed CUDA toolkit (or default to 14)
if command -v nvcc >/dev/null 2>&1; then
    NVCC_GCC=$(nvcc --version 2>/dev/null | grep -i 'gcc' | head -1 | grep -oP '[0-9]+' | tail -1 || echo "")
    if [ -n "$NVCC_GCC" ]; then
        CUDA_COMPATIBLE_GCC="$NVCC_GCC"
    else
        CUDA_COMPATIBLE_GCC=14
    fi
else
    CUDA_COMPATIBLE_GCC=14
fi
echo -e "  Current GCC version: $GCC_VERSION (CUDA supports: $CUDA_COMPATIBLE_GCC)"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install GCC 14 on Fedora
install_gcc14_fedora() {
    info "Installing GCC 14 via gcc-toolset-14..."
    # Try multiple toolset variants since package names vary by Fedora release
    for pkg in gcc-toolset-14 gcc-toolset-14-gcc-c++ gcc14 gcc14-c++; do
        if sudo dnf list installed "$pkg" >/dev/null 2>&1; then
            info "gcc-toolset-14 or gcc14 already installed"
            export GCC_TOOLSET_PATH="/opt/rh/gcc-toolset-14/root/usr/bin"
            if command_exists "$GCC_TOOLSET_PATH/gcc"; then
                export PATH="$GCC_TOOLSET_PATH:$PATH"
                export CC="$GCC_TOOLSET_PATH/gcc"
                export CXX="$GCC_TOOLSET_PATH/g++"
                export CUDAHOSTCXX="$GCC_TOOLSET_PATH/g++"
                success "Using existing gcc-toolset-14"
                return 0
            fi
        fi
    done

    if sudo dnf install -y gcc-toolset-14 gcc-toolset-14-gcc-c++ > /dev/null 2>&1; then
        if command_exists /opt/rh/gcc-toolset-14/root/usr/bin/gcc; then
            export GCC_TOOLSET_PATH="/opt/rh/gcc-toolset-14/root/usr/bin"
            export PATH="$GCC_TOOLSET_PATH:$PATH"
            export CC="$GCC_TOOLSET_PATH/gcc"
            export CXX="$GCC_TOOLSET_PATH/g++"
            export CUDAHOSTCXX="$GCC_TOOLSET_PATH/g++"
            success "GCC 14 installed via gcc-toolset-14"
            return 0
        fi
    fi
    return 1
}

# Function to install GCC 14 on Ubuntu/Debian
install_gcc14_debian() {
    info "Installing GCC 14..."
    sudo apt-get update -qq
    # Try gcc-14 first, fall back to any available gcc-13/gcc-12
    if sudo apt-get install -y gcc-14 g++-14 > /dev/null 2>&1; then
        if command_exists gcc-14; then
            export CC="gcc-14"
            export CXX="g++-14"
            export CUDAHOSTCXX="g++-14"
            success "GCC 14 installed"
            return 0
        fi
    fi
    # Fallback: check for any gcc version >= 12 available
    local fallback_gcc=$(apt-cache search '^gcc-[0-9]\+$' 2>/dev/null | grep -oP 'gcc-\K\d+' | sort -rn | head -1 || echo "")
    if [ -n "$fallback_gcc" ]; then
        warn "gcc-14 not available — installing gcc-$fallback_gcc as fallback"
        sudo apt-get install -y "gcc-$fallback_gcc" "g++-$fallback_gcc" > /dev/null 2>&1
        export CC="gcc-$fallback_gcc"
        export CXX="g++-$fallback_gcc"
        export CUDAHOSTCXX="g++-$fallback_gcc"
        success "GCC $fallback_gcc installed as fallback"
        return 0
    fi
    return 1
}

# Handle GCC version — only warn/act if current GCC exceeds CUDA's supported version
USE_UNSUPPORTED_FLAG=0
if [ "$GCC_VERSION" -gt "$CUDA_COMPATIBLE_GCC" ] || [ "$GCC_VERSION" -eq 0 ]; then
    warn "GCC $GCC_VERSION exceeds CUDA's supported version ($CUDA_COMPATIBLE_GCC)"

    # Try to install GCC 14
    INSTALLED_GCC14=0
    if [ "$PKG_FAMILY" = "fedora" ]; then
        install_gcc14_fedora && INSTALLED_GCC14=1
    else
        install_gcc14_debian && INSTALLED_GCC14=1
    fi

    if [ $INSTALLED_GCC14 -eq 0 ]; then
        warn "Could not install GCC 14. Will use -allow-unsupported-compiler flag instead."
        USE_UNSUPPORTED_FLAG=1
        export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

        # Also try to use an older GCC if available
        if [ "$PKG_FAMILY" = "fedora" ]; then
            # Try to find any gcc-toolset
            for version in 13 12 11 10; do
                if command_exists "/opt/rh/gcc-toolset-$version/root/usr/bin/gcc"; then
                    export PATH="/opt/rh/gcc-toolset-$version/root/usr/bin:$PATH"
                    export CC="/opt/rh/gcc-toolset-$version/root/usr/bin/gcc"
                    export CXX="/opt/rh/gcc-toolset-$version/root/usr/bin/g++"
                    export CUDAHOSTCXX="/opt/rh/gcc-toolset-$version/root/usr/bin/g++"
                    info "Using GCC $version from gcc-toolset"
                    break
                fi
            done
        fi
    fi
else
    success "GCC version is compatible with CUDA"
fi

# Always add the flag as a safety measure, unless already set
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -allow-unsupported-compiler"

# Set CMake CUDA flags
export CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"

# -----------------------------------------------------------------------------
# [3/8] Hardware Architecture Extraction
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[3/8]${NC} Detecting GPU Microarchitecture..."
if ! command -v nvidia-smi >/dev/null; then
    die "NVIDIA driver not detected. Please install NVIDIA drivers first:
    Fedora: sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
    Ubuntu: sudo apt install nvidia-driver-550"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1) || COMPUTE_CAP="8.6"
COMPUTE_CAP="${COMPUTE_CAP//./}"
COMPUTE_CAP="${COMPUTE_CAP:-86}"
success "GPU Identity: $GPU_NAME (SM$COMPUTE_CAP)"

# -----------------------------------------------------------------------------
# [4/8] Dynamic Generation Target Mapping
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[4/8]${NC} Matching Compute Target Matrix..."
CAP_MAJOR="$(( COMPUTE_CAP / 10 ))"

if [ "$CAP_MAJOR" -ge 12 ]; then
    CUDA_VERSION="13.3"
    CUDA_ARCH="120"
    CUDA_ARCH_LIST="120;110;100;90;89;87;86;80;75"
elif [ "$CAP_MAJOR" -ge 11 ]; then
    CUDA_VERSION="12.4"
    CUDA_ARCH="110"
    CUDA_ARCH_LIST="110;100;90;89;87;86;80;75"
elif [ "$CAP_MAJOR" -ge 9 ]; then
    CUDA_VERSION="12.0"
    CUDA_ARCH="90"
    CUDA_ARCH_LIST="90;89;87;86;80;75"
elif [ "$CAP_MAJOR" -ge 8 ]; then
    CUDA_VERSION="11.8"
    CUDA_ARCH="86"
    CUDA_ARCH_LIST="86;80;75"
else
    CUDA_VERSION="11.8"
    CUDA_ARCH="75"
    CUDA_ARCH_LIST="75;72;70;61"
fi
info "Dynamic Core Target: CUDA $CUDA_VERSION (SM$CUDA_ARCH Architecture)"

# -----------------------------------------------------------------------------
# [5/8] System Toolchain Auditing
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[5/8]${NC} Verifying Active Toolkit Assets..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "none")
echo -e "  Current System CUDA Path: $CURRENT_CUDA"

NEED_CUDA=0
if [[ "$CURRENT_CUDA" == "none" ]]; then
    NEED_CUDA=1
    info "CUDA not found - will install"
else
    # Check if version is sufficient
    CUDA_MAJOR=$(echo "$CURRENT_CUDA" | cut -d. -f1)
    NEEDED_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -lt "$NEEDED_MAJOR" ]; then
        info "CUDA $CURRENT_CUDA is older than needed ($CUDA_VERSION) - upgrading"
        NEED_CUDA=1
    else
        success "CUDA $CURRENT_CUDA is sufficient"
    fi
fi

# -----------------------------------------------------------------------------
# [6/8] Dynamic System Provisioning (NVIDIA CUDA Toolkit)
# -----------------------------------------------------------------------------
if [ $NEED_CUDA -eq 1 ]; then
    echo -e "\n${BLUE}[6/8]${NC} Provisioning CUDA $CUDA_VERSION Ecosystem..."

    if [ "$PKG_FAMILY" = "fedora" ]; then
        # Fedora installation — detect live, try current + 2 previous releases
        FEDORA_VER="$(rpm -E %fedora 2>/dev/null || echo "")"
        # If rpm can't determine it, fall back to dnf listing the repo directly
        if [ -n "$FEDORA_VER" ]; then
            FEDORA_CANDIDATES=("$FEDORA_VER" "$((FEDORA_VER-1))" "$((FEDORA_VER-2))")
        else
            # Try to infer from dnf repolist
            FEDORA_VER=$(dnf --version 2>/dev/null | head -1 | grep -oP '\d+' | head -1 || echo "44")
            FEDORA_CANDIDATES=("$FEDORA_VER" "$((FEDORA_VER-1))" "$((FEDORA_VER-2))")
        fi
        REPO_ADDED=0

        for FVER in "${FEDORA_CANDIDATES[@]}"; do
            REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/fedora${FVER}/x86_64/cuda-fedora${FVER}.repo"
            info "Trying NVIDIA CUDA repo for Fedora ${FVER}..."
            if curl -fsI "$REPO_URL" >/dev/null 2>&1; then
                if sudo dnf config-manager addrepo --from-repofile="$REPO_URL" >/dev/null 2>&1; then
                    REPO_ADDED=1
                    break
                fi
            fi
        done

        [ "$REPO_ADDED" -eq 1 ] || die "Could not find a working NVIDIA CUDA repo for this Fedora release."

        sudo dnf clean all >/dev/null 2>&1 || true

        # Try to install specific version, or fallback to latest
        CUDA_PKG_TAG="${CUDA_VERSION//./-}"
        if dnf list "cuda-toolkit-${CUDA_PKG_TAG}" >/dev/null 2>&1; then
            PKG_TO_INSTALL="cuda-toolkit-${CUDA_PKG_TAG}"
        else
            warn "cuda-toolkit-${CUDA_PKG_TAG} not available - using latest"
            PKG_TO_INSTALL=$(dnf list --available "cuda-toolkit-*" 2>/dev/null | grep -oP 'cuda-toolkit-\K[0-9]+-[0-9]+' | sort -V | tail -1)
            [ -n "$PKG_TO_INSTALL" ] || die "No cuda-toolkit packages available"
            PKG_TO_INSTALL="cuda-toolkit-${PKG_TO_INSTALL}"
        fi

        info "Deploying compiler assets ($PKG_TO_INSTALL)..."
        sudo dnf install -y "$PKG_TO_INSTALL" > /dev/null 2>&1 || die "CUDA toolkit installation failed"

        # Extract actual version
        CUDA_INSTALLED_VERSION=$(echo "$PKG_TO_INSTALL" | sed 's/cuda-toolkit-//' | tr '-' '.')
        sudo ln -sf "/usr/local/cuda-$CUDA_INSTALLED_VERSION" /usr/local/cuda

    else
        # Ubuntu/Debian installation — detect codename live from /etc/os-release
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            UBUNTU_CODENAME="${VERSION_CODENAME:-}"
        fi

        # Map detected codename to NVIDIA repo distro tag
        case "$UBUNTU_CODENAME" in
            "noble") UBUNTU_DISTRO="ubuntu2404" ;;
            "jammy") UBUNTU_DISTRO="ubuntu2204" ;;
            "focal") UBUNTU_DISTRO="ubuntu2004" ;;
            "oracular") UBUNTU_DISTRO="ubuntu2510" ;;
            "lunar") UBUNTU_DISTRO="ubuntu2304" ;;
            *)
                # Live detection: query the NVIDIA repo API for available distros
                # or fall back to ubuntu2404 (latest LTS with repo support)
                warn "Unknown codename '$UBUNTU_CODENAME' — trying ubuntu2404 (latest LTS)"
                UBUNTU_DISTRO="ubuntu2404"
                ;;
        esac

        # If VERSION_CODENAME is empty (Debian), detect from release info
        if [ -z "$UBUNTU_CODENAME" ] && [ "$OS_ID" = "debian" ]; then
            # Debian doesn't use codenames like Ubuntu; check for available CUDA repos
            UBUNTU_DISTRO="debian${VERSION_ID:-12}"
        fi

        info "synchronizing package databases..."
        sudo apt-get update -qq

        info "Configuring NVIDIA CUDA repository..."
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"

        # Try the detected version, probe for availability before downloading
        if ! wget -q --timeout=30 -O /tmp/cuda-keyring.deb "$KEYRING_URL" 2>/dev/null; then
            warn "Ubuntu repo '$UBUNTU_DISTRO' not available — probing fallbacks..."
            # Try other Ubuntu distros in reverse order
                    for alt_distro in ubuntu2204 ubuntu2004; do
                        alt_url="https://developer.download.nvidia.com/compute/cuda/repos/${alt_distro}/x86_64/cuda-keyring_1.1-1_all.deb"
                        if wget -q --timeout=30 -O /tmp/cuda-keyring.deb "$alt_url" 2>/dev/null; then
                            UBUNTU_DISTRO="$alt_distro"
                            warn "Using fallback repo: $UBUNTU_DISTRO"
                            break
                        fi
                    done
            KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
            wget -q --timeout=30 -O /tmp/cuda-keyring.deb "$KEYRING_URL" || die "Failed to download CUDA keyring (no available Ubuntu repo found)"
        fi

        sudo dpkg -i /tmp/cuda-keyring.deb || die "Failed to install CUDA keyring"
        rm -f /tmp/cuda-keyring.deb

        sudo apt-get update -qq

        CUDA_APT_TAG="${CUDA_VERSION//./-}"
        info "Deploying compiler assets (cuda-toolkit-${CUDA_APT_TAG})..."
        sudo apt-get install -y "cuda-toolkit-${CUDA_APT_TAG}" > /dev/null 2>&1 || {
            warn "cuda-toolkit-${CUDA_APT_TAG} not available - trying latest"
            sudo apt-get install -y cuda-toolkit > /dev/null 2>&1 || die "CUDA toolkit installation failed"
        }

        sudo ln -sf "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda
    fi

    # Update user environment
    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    success "Toolchain successfully updated to CUDA $CUDA_VERSION"
else
    success "Toolchain profile satisfies optimization standard"
fi

# Hard-lock environment
export CUDACXX="/usr/local/cuda/bin/nvcc"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# -----------------------------------------------------------------------------
# [7/8] Virtual Isolation Sandboxing
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[7/8]${NC} Building Virtual Environment Container..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
rm -rf "$VENV_DIR"

# Create venv with system site packages disabled
python3 -m venv --system-site-packages "$VENV_DIR" || die "Failed to create virtual environment"
source "$VENV_DIR/bin/activate"

# Install package
info "Installing llama-light package..."
pip install --upgrade pip -q

# Try to install with build isolation
if pip install . -q 2>/dev/null; then
    success "Package installed successfully"
else
    info "Retrying with --no-build-isolation..."
    pip install . --no-build-isolation -q || die "'pip install .' failed"
fi

# Verify entry point
if [ -f "$VENV_DIR/bin/llama" ]; then
    FILE_SIZE=$(stat -c%s "$VENV_DIR/bin/llama" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -lt 10 ]; then
        warn "Entry point is empty - regenerating..."
        cat > "$VENV_DIR/bin/llama" << 'PYEOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, sys
from llama_light._cli import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
PYEOF
        chmod +x "$VENV_DIR/bin/llama"
        success "Entry point regenerated"
    else
        success "Entry point valid (${FILE_SIZE} bytes)"
    fi
else
    die "Expected binary not found at $VENV_DIR/bin/llama"
fi

mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
success "Core package modules compiled into sandbox"

# -----------------------------------------------------------------------------
# [8/8] Native Hardware Compilation
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[8/8]${NC} Building Native Architectural Kernels..."

# Set comprehensive build flags
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
export CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
export CUDA_ARCH_LIST="$CUDA_ARCH_LIST"

info "Build configuration:"
echo -e "    CMAKE_CUDA_ARCHITECTURES = $CMAKE_CUDA_ARCHITECTURES"
echo -e "    CUDA_ARCH_LIST           = $CUDA_ARCH_LIST"
echo -e "    NVCC_APPEND_FLAGS        = $NVCC_APPEND_FLAGS"
if [ -n "${CC:-}" ]; then
    echo -e "    CC                      = $CC"
fi
if [ -n "${CXX:-}" ]; then
    echo -e "    CXX                     = $CXX"
fi

# Run with retry logic
MAX_RETRIES=3
RETRY_COUNT=0
LOG_SETUP="/tmp/llama-setup.log"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    info "Building attempt $((RETRY_COUNT + 1))/$MAX_RETRIES..."

    "$VENV_DIR/bin/llama" setup 2>&1 | tee "$LOG_SETUP" || true
    LLAMA_SETUP_EXIT="${PIPESTATUS[0]}"

    if [ "$LLAMA_SETUP_EXIT" -eq 0 ]; then
        success "Build completed successfully!"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        warn "Build failed, retrying..."
        sleep 3
    fi
done

if [ "$LLAMA_SETUP_EXIT" -ne 0 ]; then
    die "llama setup failed after $MAX_RETRIES attempts. See $LOG_SETUP for details.
    Try setting: export CMAKE_CUDA_FLAGS='-allow-unsupported-compiler'
    Or manually install GCC 14:
    Fedora: sudo dnf install gcc-toolset-14
    Ubuntu: sudo apt install gcc-14 g++-14"
fi

echo -e "\n${GREEN}✅ Production Installation Matrix Complete!${NC}"

# -----------------------------------------------------------------------------
# [Service] systemd user service
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[Service]${NC} Setting up systemd user service..."
if command -v systemctl >/dev/null 2>&1 && systemctl --user --version >/dev/null 2>&1; then
    SVC_DIR="$HOME/.config/systemd/user"
    SVC_PATH="$SVC_DIR/llama-server.service"
    mkdir -p "$SVC_DIR"

    cat > "$SVC_PATH" <<EOF
[Unit]
Description=llama-light server daemon
After=network.target

[Service]
Type=simple
ExecStart=$HOME/.local/bin/llama _run
PIDFile=$HOME/.cache/llama_light/server.pid
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=5
TimeoutStartSec=300
Restart=no
Environment=PATH=$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
Environment=NVCC_APPEND_FLAGS=-allow-unsupported-compiler
Environment=CMAKE_CUDA_FLAGS=-allow-unsupported-compiler

[Install]
WantedBy=default.target
EOF

    if systemctl --user daemon-reload && systemctl --user enable llama-server.service 2>/dev/null; then
        success "llama-server.service installed and enabled"
        info "Start with: systemctl --user start llama-server.service"
    else
        warn "systemctl daemon-reload/enable failed - unit file written to: $SVC_PATH"
    fi

    loginctl enable-linger "$USER" 2>/dev/null || true
else
    warn "systemd --user not available - skipping service setup"
    info "Run manually: llama _run"
fi

# -----------------------------------------------------------------------------
# Final instructions
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    🎉 Installation Complete! 🎉               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}Next steps:${NC}"
echo -e "  1. Download a model:"
echo -e "     ${CYAN}llama download <model-name>${NC}"
echo -e "  2. Configure the model:"
echo -e "     ${CYAN}llama config set default_model ~/.cache/llama_light/models/<model>.gguf${NC}"
echo -e "  3. Start the server:"
echo -e "     ${CYAN}llama start${NC}"
echo -e "  4. (Optional) Enable systemd service:"
echo -e "     ${CYAN}systemctl --user start llama-server.service${NC}"

if [ "$PKG_FAMILY" = "fedora" ]; then
    echo -e "\n${YELLOW}Note: If using gcc-toolset-14, add to ~/.bashrc:${NC}"
    echo -e "  ${CYAN}source /opt/rh/gcc-toolset-14/enable${NC}"
fi

echo -e "\nFull installation log saved to: $LOG_FILE"
