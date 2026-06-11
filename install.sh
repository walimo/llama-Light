#!/bin/bash
set -euo pipefail

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

# Step 1: Python
echo -e "${BLUE}[1/7]${NC} Checking Python..."
command -v python3 >/dev/null || { echo -e "${RED}✗ Python 3 not found${NC}"; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "  ${GREEN}✓${NC} Python $PY_VER"

# Step 2: GPU
echo -e "\n${BLUE}[2/7]${NC} Detecting GPU..."
command -v nvidia-smi >/dev/null || { echo -e "${RED}✗ NVIDIA GPU not found${NC}"; exit 1; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (SM$COMPUTE_CAP)"

# Step 3: CUDA version logic
echo -e "\n${BLUE}[3/7]${NC} Determining CUDA version..."
CAP_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
if [ "$CAP_MAJOR" -ge 12 ]; then
    CUDA_VERSION="13.0"
    CUDA_ARCH="120"
elif [ "$CAP_MAJOR" -ge 11 ]; then
    CUDA_VERSION="12.4"
    CUDA_ARCH="110"
else
    CUDA_VERSION="11.8"
    CUDA_ARCH="86"
fi
echo -e "  ${GREEN}✓${NC} Target: CUDA $CUDA_VERSION (SM$CUDA_ARCH)"

# Step 4: Check current CUDA
echo -e "\n${BLUE}[4/7]${NC} Checking current CUDA..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "none")

if [[ "$CURRENT_CUDA" == "$CUDA_VERSION" || "$CURRENT_CUDA" == "13.0" && "$CUDA_VERSION" == "13.0" ]]; then
    echo -e "  ${GREEN}✓${NC} CUDA $CURRENT_CUDA already installed"
    NEED_CUDA=0
else
    echo -e "  ${YELLOW}→${NC} Current: $CURRENT_CUDA → Need $CUDA_VERSION"
    NEED_CUDA=1
fi

# Step 5: Install CUDA via apt (recommended for Pop!_OS/Ubuntu)
if [ $NEED_CUDA -eq 1 ]; then
    echo -e "\n${BLUE}[5/7]${NC} Installing CUDA $CUDA_VERSION via NVIDIA apt repository..."
    echo -e "${YELLOW}This will use official packages (~1-2 GB total).${NC}"

    # Add NVIDIA repo
    wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb | sudo dpkg -i -
    sudo apt-get update -qq

    echo -e "  ${CYAN}➜${NC} Installing CUDA toolkit..."
    sudo apt-get install -y cuda-toolkit-$CUDA_VERSION 2>&1 | tee /tmp/cuda-install.log | while IFS= read -r line; do
        echo "    $line"
    done

    # Symlink
    sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda 2>/dev/null || true

    # Environment
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    echo -e "  ${GREEN}✓${NC} CUDA $CUDA_VERSION installed"
fi

# Step 6: llama-light
echo -e "\n${BLUE}[6/7]${NC} Installing llama-light..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
echo -e "  ${CYAN}➜${NC} Installing package..."
pip install git+https://github.com/walimo/llama-Light.git -q
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
echo -e "  ${GREEN}✓${NC} llama-light installed"

# Step 7: Build
echo -e "\n${BLUE}[7/7]${NC} Building CUDA kernels for SM$CUDA_ARCH..."
echo -e "  ${YELLOW}This may take 5-15 minutes${NC}"
export PATH="$HOME/.local/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

llama setup 2>&1 | while IFS= read -r line; do
    echo "  ${CYAN}➜${NC} $line"
done

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Quick start:"
echo "  llama config set default_model /path/to/model.gguf"
echo "  llama start"
echo ""
