#!/bin/bash
# -----------------------------------------------------------------------------
# llama-Light Ultimate Orchestrated Bootstrapper
# Robust, Cross-Generation Hardware Detection & Dynamic Dependency Provisioner
# -----------------------------------------------------------------------------
set -euo pipefail

# Precise ANSI Color Formatter Profiles
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🚀 llama-Light - One Command LLM Server (Auto CUDA)    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Sudo Credential Cache — prompt once, run everything unattended
# -----------------------------------------------------------------------------
echo -e "${CYAN}➜${NC} This installer needs sudo for system packages. Enter your password once:"
sudo -v
# Keep sudo timestamp alive in background for the duration of the script
( while true; do sudo -n true; sleep 50; done ) &
SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null' EXIT

# -----------------------------------------------------------------------------
# [0/7] Pre-flight: Build Dependencies
# -----------------------------------------------------------------------------
echo -e "${BLUE}[0/7]${NC} Checking build dependencies..."
MISSING_PKGS=()
command -v cmake >/dev/null || MISSING_PKGS+=("cmake")
command -v gcc   >/dev/null || MISSING_PKGS+=("build-essential")
command -v git   >/dev/null || MISSING_PKGS+=("git")
command -v curl  >/dev/null || MISSING_PKGS+=("curl")
command -v unzip >/dev/null || MISSING_PKGS+=("unzip")
    dpkg -s python3-venv >/dev/null 2>&1 || MISSING_PKGS+=("python3-venv")
if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "  ${CYAN}➜${NC} Installing missing packages: ${MISSING_PKGS[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING_PKGS[@]}" > /dev/null
fi
echo -e "  ${GREEN}✓${NC} Build toolchain ready"

# -----------------------------------------------------------------------------
# [1/7] Python Runtime Verification
# -----------------------------------------------------------------------------
echo -e "${BLUE}[1/7]${NC} Checking Python Environment..."
command -v python3 >/dev/null || { echo -e "${RED}✗ Python 3 not found on system paths.${NC}"; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "  ${GREEN}✓${NC} Python $PY_VER Detected"

# -----------------------------------------------------------------------------
# [2/7] Hardware Architecture Extraction
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[2/7]${NC} Detecting GPU Microarchitecture..."
if ! command -v nvidia-smi >/dev/null; then
    echo -e "  ${YELLOW}⚠${NC} NVIDIA driver not detected. Installing automatically..."
    sudo apt-get update -qq
    sudo apt-get install -y ubuntu-drivers-common > /dev/null
    sudo ubuntu-drivers autoinstall
    echo -e "\n${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  NVIDIA drivers installed — a REBOOT is required.        ║${NC}"
    echo -e "${YELLOW}║  After rebooting, run this command again to continue:    ║${NC}"
    echo -e "${YELLOW}║    curl -sSL https://raw.githubusercontent.com/walimo/llama-Light/main/install.sh | bash  ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
    exit 0
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "  ${GREEN}✓${NC} GPU Identity: $GPU_NAME (SM$COMPUTE_CAP)"

# -----------------------------------------------------------------------------
# [3/7] Dynamic Generation Target Mapping
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[3/7]${NC} Matching Compute Target Matrix..."
CAP_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [ "$CAP_MAJOR" -ge 12 ]; then
    CUDA_VERSION="13.3"   # Target native Blackwell instruction features
    CUDA_ARCH="120"
elif [ "$CAP_MAJOR" -ge 11 ]; then
    CUDA_VERSION="12.4"   # Stable targeting for Ada Lovelace / Hopper
    CUDA_ARCH="110"
else
    CUDA_VERSION="11.8"   # Universal fallback target for legacy frameworks
    CUDA_ARCH="86"
fi
echo -e "  ${GREEN}✓${NC} Dynamic Core Target: CUDA $CUDA_VERSION (SM$CUDA_ARCH Architecture)"

# -----------------------------------------------------------------------------
# [4/7] System Toolchain Auditing
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[4/7]${NC} Verifying Active Toolkit Assets..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "none")
echo -e "  Current System CUDA Path: $CURRENT_CUDA"

NEED_CUDA=0
if [[ "$CURRENT_CUDA" != "$CUDA_VERSION" ]]; then
    NEED_CUDA=1
fi

# -----------------------------------------------------------------------------
# [5/7] Dynamic System Provisioning (NVIDIA Repositories)
# -----------------------------------------------------------------------------
if [ $NEED_CUDA -eq 1 ]; then
    echo -e "\n${BLUE}[5/7]${NC} Provisioning CUDA $CUDA_VERSION Ecosystem..."
    
    # Intelligently query host platform release variables to avoid system target pollution
    UBUNTU_DISTRO="ubuntu2204"
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "${UBUNTU_CODENAME:-}" == "noble" || "${VERSION_ID:-}" == "24.04" ]]; then
            UBUNTU_DISTRO="ubuntu2404"
        fi
    fi

    echo -e "  ${CYAN}➜${NC} synchronizing package databases..."
    sudo apt-get update -qq
    
    # Safe disk-buffered file download structure preventing curl execution blockades
    echo -e "  ${CYAN}➜${NC} Configuring target microarchitecture keyring ($UBUNTU_DISTRO)..."
    wget -q -O /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
    
    sudo apt-get update -qq

    # Translate target configuration semantic dots into system manager syntax (e.g. 13.3 -> 13-3)
    CUDA_APT_TAG="${CUDA_VERSION//./-}"
    echo -e "  ${CYAN}➜${NC} Deploying compiler assets (cuda-toolkit-${CUDA_APT_TAG})..."
    sudo apt-get install -y "cuda-toolkit-${CUDA_APT_TAG}" > /dev/null

    # Establish canonical linkage layers
    sudo ln -sf "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda 2>/dev/null || true

    # Persist environments onto disk for user convenience profiles
    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    echo -e "  ${GREEN}✓${NC} Toolchain successfully updated to CUDA $CUDA_VERSION"
else
    echo -e "  ${GREEN}✓${NC} Toolchain profile satisfies optimization standard (CUDA up to date)"
fi

# Inject CUDA paths into current script session
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Persist ~/.local/bin to /etc/environment so it survives curl|bash subshells and reboots
if ! grep -q "$HOME/.local/bin" /etc/environment 2>/dev/null; then
    # Append to PATH line in /etc/environment or create it
    if grep -q "^PATH=" /etc/environment 2>/dev/null; then
        sudo sed -i "s|^PATH="\(.*\)"|PATH="$HOME/.local/bin:\1"|" /etc/environment
    else
        echo "PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"" | sudo tee -a /etc/environment > /dev/null
    fi
fi

# -----------------------------------------------------------------------------
# [6/7] Virtual Isolation Sandboxing
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[6/7]${NC} Building Virtual Environment Container..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
pip install git+https://github.com/walimo/llama-Light.git -q

mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
ln -sf "$VENV_DIR/bin/hermes" "$HOME/.local/bin/hermes"
ln -sf "$VENV_DIR/bin/claude" "$HOME/.local/bin/claude"
deactivate
echo -e "  ${GREEN}✓${NC} Core package modules compiled into sandbox"

# -----------------------------------------------------------------------------
# [7/7] Native Hardware Compilation
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[7/7]${NC} Building Native Architectural Kernels..."
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

# Pipe parsing correctly formats output streams using standard stream tracking flags
"$VENV_DIR/bin/llama" setup 2>&1 | while IFS= read -r line; do 
    echo -e "  ${CYAN}➜${NC} $line"
done; LLAMA_SETUP_EXIT="${PIPESTATUS[0]}"
if [ "${LLAMA_SETUP_EXIT}" != "0" ]; then
    echo -e "  ${RED}✗${NC} llama setup failed (exit $LLAMA_SETUP_EXIT)"
    exit 1
fi

echo -e "\n${GREEN}✅ Production Installation Matrix Complete!${NC}"
echo "Run: llama config set default_model <your-model.gguf>"