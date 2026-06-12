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
command -v nvidia-smi >/dev/null || { echo -e "${RED}✗ NVIDIA Management Library (nvidia-smi) missing. Installation aborted.${NC}"; exit 1; }
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

# CRITICAL STEP: Inject paths immediately to current active script memory buffer
# This guarantees step [7/7] has immediate kernel compiler privileges
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

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
deactivate
echo -e "  ${GREEN}✓${NC} Core package modules compiled into sandbox"

# -----------------------------------------------------------------------------
# [7/7] Native Hardware Compilation
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[7/7]${NC} Building Native Architectural Kernels..."
export PATH="$HOME/.local/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

# Pipe parsing correctly formats output streams using standard stream tracking flags
"$VENV_DIR/bin/llama" setup 2>&1 | while IFS= read -r line; do 
    echo -e "  ${CYAN}➜${NC} $line"
done

echo -e "\n${GREEN}✅ Production Installation Matrix Complete!${NC}"
echo "Run: llama config set default_model <your-model.gguf>"