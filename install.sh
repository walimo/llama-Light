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

# Step 1-4 same as before...
echo -e "${BLUE}[1/7]${NC} Checking Python..."
command -v python3 >/dev/null || { echo -e "${RED}✗ Python 3 not found${NC}"; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "  ${GREEN}✓${NC} Python $PY_VER"

echo -e "\n${BLUE}[2/7]${NC} Detecting GPU..."
command -v nvidia-smi >/dev/null || { echo -e "${RED}✗ NVIDIA GPU not found${NC}"; exit 1; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (SM$COMPUTE_CAP)"

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

echo -e "\n${BLUE}[4/7]${NC} Checking current CUDA..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "none")
echo -e "  Current: $CURRENT_CUDA"

NEED_CUDA=0
if [[ "$CURRENT_CUDA" != "$CUDA_VERSION" ]]; then
    NEED_CUDA=1
fi

# Step 5: CUDA via apt
if [ $NEED_CUDA -eq 1 ]; then
    echo -e "\n${BLUE}[5/7]${NC} Installing CUDA $CUDA_VERSION via NVIDIA apt..."
    
    # Add repo (works on Pop!_OS)
    sudo apt-get update -qq
    wget -q -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb
    sudo apt-get update -qq

    echo -e "  ${CYAN}➜${NC} Installing cuda-toolkit..."
    sudo apt-get install -y cuda-toolkit-$CUDA_VERSION | tee /tmp/cuda.log

    sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda 2>/dev/null || true

    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    echo -e "  ${GREEN}✓${NC} CUDA $CUDA_VERSION installed"
else
    echo -e "  ${GREEN}✓${NC} CUDA already up to date"
fi

# Step 6 & 7
echo -e "\n${BLUE}[6/7]${NC} Installing llama-light..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install git+https://github.com/walimo/llama-Light.git -q
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
echo -e "  ${GREEN}✓${NC} llama-light installed"

echo -e "\n${BLUE}[7/7]${NC} Building CUDA kernels..."
export PATH="$HOME/.local/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
llama setup 2>&1 | while IFS= read -r line; do echo "  ${CYAN}➜${NC} $line"; done

echo -e "\n${GREEN}✅ Installation complete!${NC}"
echo "Run: llama config set default_model <your-model.gguf>"
