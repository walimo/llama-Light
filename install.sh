#!/bin/bash
set -e

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

# Step 1: Python check
echo -e "${BLUE}[1/7]${NC} Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.8+"
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "  ${GREEN}✓${NC} Python $PY_VER"

# Step 2: GPU detection
echo -e "\n${BLUE}[2/7]${NC} Detecting GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA GPU not found${NC}"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (SM$COMPUTE_CAP)"

# Step 3: Determine required CUDA version
echo -e "\n${BLUE}[3/7]${NC} Determining CUDA requirements..."
CAP_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
if [ "$CAP_MAJOR" -ge 12 ]; then
    REQUIRED_CUDA="13.0"
    CUDA_ARCH="120"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_550.54.15_linux.run"
elif [ "$CAP_MAJOR" -ge 11 ]; then
    REQUIRED_CUDA="12.4"
    CUDA_ARCH="110"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run"
else
    REQUIRED_CUDA="11.8"
    CUDA_ARCH="86"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
fi
echo -e "  ${GREEN}✓${NC} Required CUDA: $REQUIRED_CUDA"
echo -e "  ${GREEN}✓${NC} Build architecture: SM$CUDA_ARCH"

# Step 4: Check current CUDA
echo -e "\n${BLUE}[4/7]${NC} Checking current CUDA..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")
if [ -n "$CURRENT_CUDA" ]; then
    echo -e "  ${GREEN}✓${NC} Found CUDA $CURRENT_CUDA"
fi

# Step 5: Install correct CUDA if needed
NEED_INSTALL=0
if [ -z "$CURRENT_CUDA" ]; then
    NEED_INSTALL=1
elif [ "$(printf '%s\n' "$REQUIRED_CUDA" "$CURRENT_CUDA" | sort -V | head -1)" != "$REQUIRED_CUDA" ]; then
    NEED_INSTALL=1
fi

if [ $NEED_INSTALL -eq 1 ]; then
    echo -e "\n${BLUE}[5/7]${NC} Installing CUDA $REQUIRED_CUDA..."
    echo -e "  ${YELLOW}⚠${NC} This downloads ~4GB and takes 5-10 minutes"
    
    # Download
    echo -e "  ${CYAN}➜${NC} Downloading..."
    wget --show-progress -q "$CUDA_URL" -O /tmp/cuda.run
    
    # Install
    echo -e "  ${CYAN}➜${NC} Installing..."
    sudo sh /tmp/cuda.run --toolkit --silent --override
    
    # Setup symlink
    sudo ln -sf "/usr/local/cuda-$REQUIRED_CUDA" /usr/local/cuda
    
    # Update PATH
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Cleanup
    rm -f /tmp/cuda.run
    echo -e "  ${GREEN}✓${NC} CUDA $REQUIRED_CUDA installed"
    
    # Rehash to find nvcc
    hash -r
fi

# Step 6: Install llama-light
echo -e "\n${BLUE}[6/7]${NC} Installing llama-light..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install git+https://github.com/walimo/llama-Light.git -q
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
echo -e "  ${GREEN}✓${NC} llama-light installed"

# Step 7: Build CUDA binary
echo -e "\n${BLUE}[7/7]${NC} Building CUDA binary..."
export PATH="$HOME/.local/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
llama setup

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Quick start:"
echo "  llama config set default_model /path/to/model.gguf"
echo "  llama start"
echo "  llama chat"
echo ""
