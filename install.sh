#!/bin/bash
set -euo pipefail

# Colors
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

# Helper functions
progress_bar() {
    local percent=$1
    local width=50
    local filled=$((percent * width / 100))
    local empty=$((width - filled))
    printf "\r  ${CYAN}➜${NC} ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%%" "$percent"
}

spinner() {
    local spin='-\|/'
    local i=0
    while kill -0 "$1" 2>/dev/null; do
        printf "\r  ${CYAN}➜${NC} %s " "${spin:i++%4:1}"
        sleep 0.2
    done
}

confirm() {
    echo -e "${YELLOW}⚠${NC} $1"
    read -r -p "Continue? [y/N] " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted.${NC}"
        exit 1
    fi
}

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

# Step 3: CUDA version
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
echo -e "  ${GREEN}✓${NC} Required CUDA: $REQUIRED_CUDA (SM$CUDA_ARCH)"

# Step 4: Current CUDA
echo -e "\n${BLUE}[4/7]${NC} Checking current CUDA..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")

# Step 5: Install CUDA if needed
NEED_INSTALL=0
if [ -z "$CURRENT_CUDA" ] || [ "$(printf '%s\n' "$REQUIRED_CUDA" "$CURRENT_CUDA" | sort -V | head -1)" != "$REQUIRED_CUDA" ]; then
    NEED_INSTALL=1
fi

if [ $NEED_INSTALL -eq 1 ]; then
    confirm "This will download ~4-8GB and install CUDA $REQUIRED_CUDA. Continue?"
    
    echo -e "\n${BLUE}[5/7]${NC} Installing CUDA $REQUIRED_CUDA..."
    
    # Install prerequisites
    if command -v apt-get >/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y wget curl build-essential
    elif command -v dnf >/dev/null; then
        sudo dnf install -y wget curl gcc-c++ make
    fi

    echo -e "  ${CYAN}➜${NC} Downloading CUDA installer (~4GB)..."
    if command -v curl >/dev/null; then
        curl -L --progress-bar -o /tmp/cuda.run "$CUDA_URL" &
        SPID=$!
        spinner $SPID
        wait $SPID
    else
        wget --show-progress -O /tmp/cuda.run "$CUDA_URL"
    fi
    echo -e "\n  ${GREEN}✓${NC} Download complete"

    echo -e "  ${CYAN}➜${NC} Running CUDA installer (this may take several minutes)..."
    sudo sh /tmp/cuda.run --toolkit --silent --override || {
        echo -e "${RED}CUDA installer failed. Check logs or try NVIDIA package manager method.${NC}"
        exit 1
    }

    sudo ln -sf "/usr/local/cuda-$REQUIRED_CUDA" /usr/local/cuda 2>/dev/null || true
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    rm -f /tmp/cuda.run
    hash -r
    echo -e "  ${GREEN}✓${NC} CUDA $REQUIRED_CUDA installed"
fi

# Step 6: llama-light
echo -e "\n${BLUE}[6/7]${NC} Installing llama-light..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
echo -e "  ${CYAN}➜${NC} Installing Python package..."
pip install git+https://github.com/walimo/llama-Light.git -q
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
echo -e "  ${GREEN}✓${NC} llama-light installed"

# Step 7: Build
echo -e "\n${BLUE}[7/7]${NC} Building CUDA binary for SM$CUDA_ARCH..."
echo -e "  ${YELLOW}⚠${NC} This can take 5-20 minutes"
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
echo "Next steps:"
echo "  llama config set default_model /path/to/your/model.gguf"
echo "  llama start"
echo "  llama chat"
echo ""
echo "Also try: llama pull --repo TheBloke/Llama-2-7B-Chat-GGUF --file llama-2-7b-chat.Q4_K_M.gguf"
