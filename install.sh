#!/bin/bash
set -e

echo "🚀 llama-Light Portable Installer (Auto CUDA)"
echo "=============================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Python check
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3.8+ not found${NC}"
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "${GREEN}✅ Python $PY_VER${NC}"

# 2. Detect NVIDIA GPU and compute capability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected. This tool requires an NVIDIA GPU.${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo -e "${GREEN}✅ GPU detected: $GPU_NAME (Compute Capability $COMPUTE_CAP)${NC}"

# 3. Determine required CUDA version based on compute capability
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

echo -e "${YELLOW}📦 Required CUDA Toolkit: $REQUIRED_CUDA (for SM $CUDA_ARCH)${NC}"

# 4. Check current CUDA version
CURRENT_CUDA=""
if command -v nvcc &> /dev/null; then
    CURRENT_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo -e "${GREEN}✅ Current CUDA: $CURRENT_CUDA${NC}"
fi

# 5. Install/upgrade CUDA if needed
NEED_CUDA=0
if [ -z "$CURRENT_CUDA" ]; then
    echo -e "${YELLOW}⚠️  CUDA Toolkit not found. Installing CUDA $REQUIRED_CUDA...${NC}"
    NEED_CUDA=1
elif [ "$(printf '%s\n' "$REQUIRED_CUDA" "$CURRENT_CUDA" | sort -V | head -n1)" != "$REQUIRED_CUDA" ]; then
    echo -e "${YELLOW}⚠️  CUDA $CURRENT_CUDA is too old. Need $REQUIRED_CUDA or newer.${NC}"
    NEED_CUDA=1
fi

if [ $NEED_CUDA -eq 1 ]; then
    echo -e "${GREEN}🔧 Installing CUDA $REQUIRED_CUDA...${NC}"
    CUDA_MAJOR=$(echo $REQUIRED_CUDA | cut -d. -f1)
    wget -q "https://developer.download.nvidia.com/compute/cuda/${REQUIRED_CUDA}/local_installers/cuda_${REQUIRED_CUDA}_*_linux.run" -O /tmp/cuda.run
    sudo sh /tmp/cuda.run --toolkit --silent --override
    sudo ln -sf "/usr/local/cuda-$REQUIRED_CUDA" /usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    rm -f /tmp/cuda.run
    echo -e "${GREEN}✅ CUDA $REQUIRED_CUDA installed${NC}"
fi

# 6. Set CUDA architecture for build
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

# 7. Create virtual environment and install llama-light
VENV_DIR="$HOME/.local/share/llama-light-venv"
echo -e "${GREEN}📦 Setting up Python environment...${NC}"
python3 -m venv --clear "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install git+https://github.com/walimo/llama-Light.git

# 8. Symlink binary
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate

# 9. Add to PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

# 10. First-time setup (builds llama.cpp)
echo ""
echo -e "${GREEN}🔧 Building llama.cpp for SM$CUDA_ARCH...${NC}"
"$HOME/.local/bin/llama" start --setup-only 2>&1 | tee /tmp/llama-setup.log

# 11. Verify
if curl -s http://127.0.0.1:8080/health &> /dev/null; then
    echo -e "${GREEN}✅ Installation successful! Server is running with GPU acceleration.${NC}"
else
    echo -e "${YELLOW}⚠️  Server not running. Check logs: llama logs${NC}"
fi

echo ""
echo "🎉 You can now use:"
echo "   llama start          # start server"
echo "   llama chat           # interactive chat"
echo "   llama ps             # show GPU usage"
