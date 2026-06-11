#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🚀 llama-Light - One Command LLM Server           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Python check
echo -e "${BLUE}[1/6]${NC} Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.8+ from your package manager"
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "  ${GREEN}✓${NC} Python $PY_VER"

# Step 2: GPU detection
echo -e "\n${BLUE}[2/6]${NC} Detecting GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA GPU not found${NC}"
    echo "  This tool requires an NVIDIA GPU with CUDA support"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (SM$COMPUTE_CAP)"

# Step 3: Determine required CUDA version
echo -e "\n${BLUE}[3/6]${NC} Checking CUDA requirements..."
CAP_NUM=$(echo "$COMPUTE_CAP" | cut -d. -f1)
if [ "$CAP_NUM" -ge 12 ]; then
    REQUIRED_CUDA="13.0"
    CUDA_ARCH="120"
elif [ "$CAP_NUM" -ge 11 ]; then
    REQUIRED_CUDA="12.4"
    CUDA_ARCH="110"
else
    REQUIRED_CUDA="11.8"
    CUDA_ARCH="86"
fi
echo -e "  ${GREEN}✓${NC} Required CUDA: $REQUIRED_CUDA"
echo -e "  ${GREEN}✓${NC} Build architecture: SM$CUDA_ARCH"

# Step 4: Check/Install CUDA
echo -e "\n${BLUE}[4/6]${NC} Setting up CUDA..."
CURRENT_CUDA=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")
if [ -n "$CURRENT_CUDA" ]; then
    echo -e "  ${GREEN}✓${NC} CUDA $CURRENT_CUDA found"
    if [ "$(printf '%s\n' "$REQUIRED_CUDA" "$CURRENT_CUDA" | sort -V | head -1)" != "$REQUIRED_CUDA" ]; then
        echo -e "  ${YELLOW}⚠${NC} CUDA $CURRENT_CUDA may be too old for optimal performance"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} CUDA not found, will build without GPU acceleration"
fi

# Step 5: Install llama-light
echo -e "\n${BLUE}[5/6]${NC} Installing llama-light..."
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

# Step 6: First run (download/build)
echo -e "\n${BLUE}[6/6]${NC} Building CUDA binary (first run)..."
export PATH="$HOME/.local/bin:$PATH"
llama start --setup-only 2>&1 | tee /tmp/llama-setup.log

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Quick start:"
echo "  llama start          # Start the server"
echo "  llama chat           # Start chatting"
echo "  llama ps             # Monitor GPU usage"
echo ""
echo "To download a model:"
echo "  llama pull --repo TheBloke/Llama-2-7B-Chat-GGUF --file llama-2-7b-chat.Q4_K_M.gguf"
echo ""
echo "Add to PATH (if needed):"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
