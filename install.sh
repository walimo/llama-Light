#!/bin/bash
# -----------------------------------------------------------------------------
# llama-Light Ultimate Orchestrated Bootstrapper
# Robust, Cross-Generation Hardware Detection & Dynamic Dependency Provisioner
# -----------------------------------------------------------------------------
set -euo pipefail

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Log file for build output (captures stderr only)
LOG_FILE="/tmp/llama-light-install.log"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🚀 llama-Light v0.2.1 - One Command LLM Server (Auto CUDA)    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Helper: exit with error message
# -----------------------------------------------------------------------------
die() {
    echo -e "${RED}✗ Error: $*${NC}" >&2
    echo "Full log: $LOG_FILE" >&2
    exit 1
}

# -----------------------------------------------------------------------------
# Sudo credential caching – keep alive during the whole script
# -----------------------------------------------------------------------------
echo -e "${CYAN}➜${NC} This installer needs sudo for system packages. Enter your password once:"
sudo -S -p '' -v || die "sudo failed"
(
    while true; do
        sudo -S -p '' -n true
        sleep 50
    done
) 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null' EXIT

# -----------------------------------------------------------------------------
# Ensure we are inside the llama-Light source directory
# -----------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
    die "Please run this script from the root of the llama-Light source directory."
fi

# -----------------------------------------------------------------------------
# [0/7] Pre-flight: Build Dependencies
# -----------------------------------------------------------------------------
echo -e "${BLUE}[0/7]${NC} Checking build dependencies..."
MISSING_PKGS=()
command -v cmake >/dev/null || MISSING_PKGS+=("cmake")
command -v gcc   >/dev/null || MISSING_PKGS+=("build-essential")
command -v git   >/dev/null || MISSING_PKGS+=("git")
command -v curl  >/dev/null || MISSING_PKGS+=("curl")
command -v unzip >/dev/null || MISSING_PKGS+=("unzip")
dpkg -s python3-venv >/dev/null 2>&1 || MISSING_PKGS+=("python3-venv")
if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "  ${CYAN}➜${NC} Installing missing packages: ${MISSING_PKGS[*]}"
    sudo -S -p '' apt-get update -qq
    sudo -S -p '' apt-get install -y "${MISSING_PKGS[@]}" > /dev/null
fi
echo -e "  ${GREEN}✓${NC} Build toolchain ready"

# -----------------------------------------------------------------------------
# [1/7] Python Runtime Verification
# -----------------------------------------------------------------------------
echo -e "${BLUE}[1/7]${NC} Checking Python Environment..."
command -v python3 >/dev/null || die "Python 3 not found"
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo -e "  ${GREEN}✓${NC} Python $PY_VER Detected"

# -----------------------------------------------------------------------------
# [2/7] Hardware Architecture Extraction
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[2/7]${NC} Detecting GPU Microarchitecture..."
if ! command -v nvidia-smi >/dev/null; then
    echo -e "  ${RED}✗${NC} NVIDIA driver not detected."
    echo -e "  Install drivers and rerun:"
    echo -e "    sudo apt install nvidia-driver-550"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1) || COMPUTE_CAP="8.6"
# Strip dot and normalise (e.g. "8.6" → "86")
COMPUTE_CAP="${COMPUTE_CAP//./}"
COMPUTE_CAP="${COMPUTE_CAP:-86}"
echo -e "  ${GREEN}✓${NC} GPU Identity: $GPU_NAME (SM$COMPUTE_CAP)"

# -----------------------------------------------------------------------------
# [3/7] Dynamic Generation Target Mapping
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[3/7]${NC} Matching Compute Target Matrix..."
CAP_MAJOR="$(( COMPUTE_CAP / 10 ))"

if [ "$CAP_MAJOR" -ge 12 ]; then
    CUDA_VERSION="13.3"
    CUDA_ARCH="120"      # Blackwell SM12.0 native profile
elif [ "$CAP_MAJOR" -ge 11 ]; then
    CUDA_VERSION="12.4"
    CUDA_ARCH="110"
else
    CUDA_VERSION="11.8"
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

    # Start with ubuntu2404; fall back to ubuntu2204 if the repo doesn't exist
    UBUNTU_DISTRO="ubuntu2404"

    echo -e "  ${CYAN}➜${NC} synchronizing package databases..."
    sudo -S -p '' apt-get update -qq

    echo -e "  ${CYAN}➜${NC} Configuring target microarchitecture keyring ($UBUNTU_DISTRO)..."
    if ! wget -q --timeout=30 -O /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"; then
        echo -e "  ${YELLOW}⚠${NC} ubuntu2404 CUDA repo not available — falling back to ubuntu2204."
        UBUNTU_DISTRO="ubuntu2204"
        wget -q --timeout=30 -O /tmp/cuda-keyring.deb \
            "https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
            || die "Failed to download CUDA keyring (ubuntu2204 repo also unavailable)"
    fi
    sudo -S -p '' dpkg -i /tmp/cuda-keyring.deb || die "Failed to install CUDA keyring"
    rm -f /tmp/cuda-keyring.deb

    sudo -S -p '' apt-get update -qq

    CUDA_APT_TAG="${CUDA_VERSION//./-}"
    echo -e "  ${CYAN}➜${NC} Deploying compiler assets (cuda-toolkit-${CUDA_APT_TAG})..."
    sudo -S -p '' apt-get install -y "cuda-toolkit-${CUDA_APT_TAG}" > /dev/null || die "CUDA toolkit installation failed"

    sudo -S -p '' ln -sf "/usr/local/cuda-$CUDA_VERSION" /usr/local/cuda

    # Update user environment (bashrc)
    grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    echo -e "  ${GREEN}✓${NC} Toolchain successfully updated to CUDA $CUDA_VERSION"
else
    echo -e "  ${GREEN}✓${NC} Toolchain profile satisfies optimization standard (CUDA up to date)"
fi

# Hard‑lock environment for the rest of the script
export CUDACXX="/usr/local/cuda/bin/nvcc"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# (Optional) System‑wide PATH – commented out because ~/.bashrc is enough
# if ! grep -q "$HOME/.local/bin" /etc/environment 2>/dev/null; then
#     if grep -q "^PATH=" /etc/environment 2>/dev/null; then
#         sudo sed -i "s|^PATH=\"\(.*\)\"|PATH=\"$HOME/.local/bin:\1\"|" /etc/environment
#     else
#         echo "PATH=\"$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\"" | sudo tee -a /etc/environment > /dev/null
#     fi
# fi

# -----------------------------------------------------------------------------
# [6/7] Virtual Isolation Sandboxing
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[6/7]${NC} Building Virtual Environment Container..."
VENV_DIR="$HOME/.local/share/llama-light-venv"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR" || die "Failed to create virtual environment"
source "$VENV_DIR/bin/activate"

# Install the package from current directory (no global npm modifications)
pip install --upgrade pip -q
pip install . --break-system-packages -q || die "'pip install .' failed"

# Verify the installed binary exists before creating the symlink
if ! [ -f "$VENV_DIR/bin/llama" ]; then
    die "Expected binary not found at $VENV_DIR/bin/llama — pip install may have failed silently"
fi

mkdir -p "$HOME/.local/bin"
ln -sf "$VENV_DIR/bin/llama" "$HOME/.local/bin/llama"
deactivate
echo -e "  ${GREEN}✓${NC} Core package modules compiled into sandbox"

# -----------------------------------------------------------------------------
# [7/7] Native Hardware Compilation
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[7/7]${NC} Building Native Architectural Kernels..."
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

LOG_SETUP="/tmp/llama-setup.log"
# Use '|| true' to prevent set -e from killing the script when pipefail makes
# the pipeline fail; then read PIPESTATUS to get the actual exit code.
"$VENV_DIR/bin/llama" setup 2>&1 | tee "$LOG_SETUP" || true
LLAMA_SETUP_EXIT="${PIPESTATUS[0]}"
if [ "$LLAMA_SETUP_EXIT" -ne 0 ]; then
    echo -e "  ${RED}✗${NC} llama setup failed (exit $LLAMA_SETUP_EXIT). See $LOG_SETUP for details."
    exit 1
fi

echo -e "\n${GREEN}✅ Production Installation Matrix Complete!${NC}"

# -----------------------------------------------------------------------------
# [Service] systemd user service — written directly by install.sh
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}[Service]${NC} Setting up systemd user service..."
if command -v systemctl >/dev/null 2>&1 && systemctl --user --version >/dev/null 2>&1; then
    SVC_DIR="$HOME/.config/systemd/user"
    SVC_PATH="$SVC_DIR/llama-server.service"
    mkdir -p "$SVC_DIR"

    cat > "$SVC_PATH" <<EOF
[Unit]
Description=llama-light server daemon
After=network.target

[Service]
Type=simple
ExecStart=$HOME/.local/bin/llama _run
PIDFile=$HOME/.cache/llama_light/server.pid
KillMode=control-group
KillSignal=SIGKILL
TimeoutStopSec=5
TimeoutStartSec=300
Restart=no
Environment=PATH=$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

    if systemctl --user daemon-reload && systemctl --user enable llama-server.service; then
        echo -e "  ${GREEN}✓${NC} llama-server.service installed and enabled (not started)"
        echo -e "      unit: $SVC_PATH"
    else
        echo -e "  ${YELLOW}⚠${NC} systemctl daemon-reload/enable failed — unit file was written to:"
        echo -e "      $SVC_PATH"
        echo -e "      Retry manually: systemctl --user daemon-reload && systemctl --user enable llama-server.service"
    fi

    # Allow the service to keep running after logout / terminal close
    loginctl enable-linger "$USER" 2>/dev/null || true
else
    echo -e "  ${YELLOW}⚠${NC} systemd --user not available — skipping service setup."
    echo -e "      Run 'llama _run' manually instead."
fi

echo -e "  ${CYAN}➜${NC} You may now run: llama config set default_model <your-model.gguf>"
echo -e "  ${CYAN}➜${NC} Then:        llama start"

# -----------------------------------------------------------------------------
# [8/7] Ultimate MCP Server Integration
# -----------------------------------------------------------------------------
echo -e "\\n${BLUE}[8/7]${NC} Setting up Ultimate MCP Server..."

MCP_DIR="$HOME/.cache/llama-light/ultimate-mcp"
MCP_SOURCE="$PWD/llama_light/ultimate_mcp_server.py"
MCP_DEST="$MCP_DIR/server.py"

if [ -d "$MCP_DIR" ]; then
    rm -rf "$MCP_DIR"
fi
mkdir -p "$MCP_DIR"

if [ -f "$MCP_SOURCE" ]; then
    cp "$MCP_SOURCE" "$MCP_DEST"
    echo -e "  ${GREEN}✓${NC} MCP server copied to $MCP_DIR"
else
    echo -e "  ${YELLOW}⚠${NC} MCP server source not found — skipping server copy"
fi

# Install MCP dependencies in the venv
echo -e "  ${CYAN}➜${NC} Installing MCP dependencies (ddgs, beautifulsoup4)..."
pip install ddgs beautifulsoup4 -q 2>&1 | grep -v "already satisfied"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "  ${YELLOW}⚠${NC} Failed to install MCP dependencies"
else
    echo -e "  ${GREEN}✓${NC} MCP dependencies installed"
fi

# Install Playwright Chromium browser
echo -e "  ${CYAN}➜${NC} Installing Playwright Chromium browser..."
python -m playwright install chromium 2>&1 | tail -1
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} Playwright Chromium installed"
else
    echo -e "  ${YELLOW}⚠${NC} Playwright Chromium installation failed"
fi

# Create MCP config file for MCP clients
MCP_CONFIG_PATH="$HOME/.mcp_config.json"
cat > "$MCP_CONFIG_PATH" <<EOF
{
    "mcpServers": {
        "ultimate-mcp": {
            "command": "python",
            "args": [
                "$MCP_DEST"
            ],
            "env": {}
        }
    }
}
EOF
echo -e "  ${GREEN}✓${NC} MCP config written to $MCP_CONFIG_PATH"

# Create launcher script
LAUNCHER="$PWD/llama-mcp"
cat > "$LAUNCHER" <<'EOF'
#!/bin/bash
# Ultimate MCP Server launcher
echo "Starting Ultimate MCP Server..."
echo "API docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
python "$HOME/.cache/llama-light/ultimate-mcp/server.py"
EOF
chmod +x "$LAUNCHER"
echo -e "  ${GREEN}✓${NC} Launcher created: $LAUNCHER"

# Create systemd service for MCP server
if command -v systemctl >/dev/null 2>&1 && systemctl --user --version >/dev/null 2>&1; then
    SVC_DIR="$HOME/.config/systemd/user"
    SVC_PATH="$SVC_DIR/ultimate-mcp.service"
    mkdir -p "$SVC_DIR"

    cat > "$SVC_PATH" <<EOF
[Unit]
Description=Ultimate MCP Server for llama-light
After=network.target

[Service]
Type=simple
ExecStart=python $MCP_DEST
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

    if systemctl --user daemon-reload && systemctl --user enable ultimate-mcp.service 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} MCP server systemd service installed (not started)"
        echo -e "      Start: systemctl --user start ultimate-mcp.service"
    fi
fi

echo -e "  ${CYAN}➜${NC} MCP Server Info:"
echo -e "      API docs:  http://localhost:8000/docs"
echo -e "      Start:     $PWD/llama-mcp"
echo -e "      Stop:      Ctrl+C or kill the process"
echo -e "  ${CYAN}➜${NC} To start the server:"
echo -e "      llama config set default_model <your-model.gguf>"
echo -e "      llama start"
echo -e "  ${CYAN}➜${NC} Then run the MCP server:"
echo -e "      $PWD/llama-mcp"
echo -e "\\nFull installation log saved to: $LOG_FILE"