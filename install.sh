#!/bin/bash
set -e

echo "🚀 llama-Light Portable Installer (Auto CUDA)"
echo "=============================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. Python check
echo -e "${BLUE}[1/7]${NC} Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3.8+ not found${NC}"
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "${GREEN}✅ Python $PY_VER${NC}"
echo ""

# 2. Detect GPU
echo -e "${BLUE}[2/7]${NC} Detecting GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected.${NC}"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "${GREEN}✅ GPU: $GPU_NAME${NC}"
echo -e "${GREEN}✅ Compute Capability: $COMPUTE_CAP${NC}"
echo ""

# 3. Determine required CUDA version
echo -e "${BLUE}[3/7]${NC} Determining required CUDA version..."
case "$COMPUTE_CAP" in
    12.*)   REQUIRED_CUDA="13.0" ; CUDA_ARCH="120" ;;
    11.*)   REQUIRED_CUDA="12.4" ; CUDA_ARCH="110" ;;
    10.*)   REQUIRED_CUDA="12.4" ; CUDA_ARCH="100" ;;
    9.*)    REQUIRED_CUDA="12.4" ; CUDA_ARCH="90" ;;
    8.*)    REQUIRED_CUDA="12.4" ; CUDA_ARCH="89" ;;
    7.*)    REQUIRED_CUDA="12.4" ; CUDA_ARCH="75" ;;
    6.*)    REQUIRED_CUDA="12.4" ; CUDA_ARCH="61" ;;
    *)      REQUIRED_CUDA="12.4" ; CUDA_ARCH="89" ;;
esac
echo -e "${GREEN}✅ Required CUDA: $REQUIRED_CUDA${NC}"
echo -e "${GREEN}✅ Build Architecture: SM$CUDA_ARCH${NC}"
echo ""

# 4. Check current CUDA
echo -e "${BLUE}[4/7]${NC} Checking current CUDA installation..."
CURRENT_CUDA=""
if command -v nvcc &> /dev/null; then
    CURRENT_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo -e "${GREEN}✅ Current CUDA: $CURRENT_CUDA${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not found${NC}"
fi
echo ""

# 5. Install/upgrade CUDA if needed
NEED_CUDA=0
if [ -z "$CURRENT_CUDA" ]; then
    echo -e "${YELLOW}⚠️  CUDA Toolkit not found.${NC}"
    NEED_CUDA=1
elif [ "$(printf '%s\n' "$REQUIRED_CUDA" "$CURRENT_CUDA" | sort -V | head -n1)" != "$REQUIRED_CUDA" ]; then
    echo -e "${YELLOW}⚠️  CUDA $CURRENT_CUDA is too old (need $REQUIRED_CUDA+)${NC}"
    NEED_CUDA=1
fi

if [ $NEED_CUDA -eq 1 ]; then
    echo -e "${BLUE}[5/7]${NC} Installing CUDA $REQUIRED_CUDA..."
    echo -e "${YELLOW}   This may take 5-10 minutes. Please wait...${NC}"
    echo ""
    
    CUDA_MAJOR=$(echo $REQUIRED_CUDA | cut -d. -f1)
    CUDA_MINOR=$(echo $REQUIRED_CUDA | cut -d. -f2)
    
    # Download with progress
    echo "   Downloading CUDA installer..."
    wget --progress=bar:force "https://developer.download.nvidia.com/compute/cuda/${REQUIRED_CUDA}/local_installers/cuda_${REQUIRED_CUDA}_*_linux.run" -O /tmp/cuda.run 2>&1
    
    echo "   Running CUDA installer..."
    sudo sh /tmp/cuda.run --toolkit --silent --override
    
    echo "   Setting up CUDA environment..."
    sudo ln -sf "/usr/local/cuda-$REQUIRED_CUDA" /usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    # Add to bashrc if not already there
    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    rm -f /tmp/cuda.run
    echo -e "${GREEN}✅ CUDA $REQUIRED_CUDA installed successfully${NC}"
else
    echo -e "${GREEN}✅ CUDA $CURRENT_CUDA is sufficient${NC}"
fi
echo ""

# 6. Set environment for build
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

# 7. Install llama-light
echo -e "${BLUE}[6/7]${NC} Setting up Python environment..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
python3 -m venv --clear "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "   Upgrading pip..."
pip install --upgrade pip -q
echo "   Installing llama-light..."
pip install git+https://github.com/walimo/llama-Light.git -q
echo "   Creating symlink..."
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
echo -e "${GREEN}✅ llama-light installed${NC}"
echo ""

# 8. Add to PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

# 9. First-time build
echo -e "${BLUE}[7/7]${NC} Building llama.cpp for SM$CUDA_ARCH..."
echo -e "${YELLOW}   This may take 10-15 minutes. Please wait...${NC}"
echo ""
"$HOME/.local/bin/llama" start --setup-only 2>&1 | tee /tmp/llama-setup.log

# 10. Verify
if curl -s http://127.0.0.1:8080/health &> /dev/null; then
    echo ""
    echo -e "${GREEN}✅ Installation successful! Server is running with GPU acceleration.${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Server not running. Check logs: llama logs${NC}"
fi

echo ""
echo "🎉 You can now use:"
echo "   llama start          # start server"
echo "   llama chat           # interactive chat"
echo "   llama ps             # show GPU usage"
