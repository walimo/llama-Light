#!/bin/bash
set -e

echo "🚀 llama-Light Portable Installer (CUDA required)"
echo "=================================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# 1. Python check
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3.8+ not found${NC}"
    echo "Install Python from your package manager."
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "${GREEN}✅ Python $PY_VER${NC}"

# 2. CUDA check
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU with CUDA required${NC}"
    exit 1
fi
echo -e "${GREEN}✅ CUDA detected:${NC}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

# 3. Create a virtual environment (bypasses system Python restrictions)
VENV_DIR="$HOME/.local/share/llama-light-venv"
echo "📦 Setting up isolated Python environment in $VENV_DIR ..."
python3 -m venv --clear "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install git+https://github.com/walimo/llama-Light.git

# 4. Symlink the binary to ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate

# 5. Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

# 6. First-time setup (downloads/builds CUDA binary)
echo ""
echo "🔧 Running first-time setup – this may take a while..."
"$HOME/.local/bin/llama" start --setup-only 2>&1 | tee /tmp/llama-setup.log

# 7. Verification
if curl -s http://127.0.0.1:8080/health &> /dev/null; then
    echo -e "${GREEN}✅ Installation successful! Server is running with GPU acceleration.${NC}"
else
    echo -e "${RED}⚠️  Server not running. Check logs: llama logs${NC}"
fi

echo ""
echo "🎉 You can now use:"
echo "   llama start          # start server"
echo "   llama chat           # interactive chat"
echo "   llama ps             # show GPU usage"
