#!/bin/bash
set -e
echo "🚀 llama-Light Portable Installer (CUDA required)"
echo "=================================================="
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
if ! command -v python3 &> /dev/null; then echo -e "${RED}❌ Python 3.8+ not found${NC}"; exit 1; fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "${GREEN}✅ Python $PY_VER${NC}"
if ! command -v nvidia-smi &> /dev/null; then echo -e "${RED}❌ NVIDIA GPU with CUDA required${NC}"; exit 1; fi
echo -e "${GREEN}✅ CUDA detected:${NC}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
if command -v pipx &> /dev/null; then
    pipx install git+https://github.com/walimo/llama-Light.git
else
    python3 -m venv ~/.local/share/llama-light-venv
    source ~/.local/share/llama-light-venv/bin/activate
    pip install --upgrade pip
    pip install git+https://github.com/walimo/llama-Light.git
    mkdir -p ~/.local/bin
    ln -sf ~/.local/share/llama-light-venv/bin/llama ~/.local/bin/llama
    deactivate
fi
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ""; echo "🔧 Running first-time setup – this may take a while..."
llama start --setup-only 2>&1 | tee /tmp/llama-setup.log
if curl -s http://127.0.0.1:8080/health &> /dev/null; then
    echo -e "${GREEN}✅ Installation successful! Server is running with GPU acceleration.${NC}"
else
    echo -e "${RED}⚠️  Server not running. Check logs: llama logs${NC}"
fi
echo ""; echo "🎉 You can now use: llama start, llama chat, llama ps"
