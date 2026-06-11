#!/usr/bin/env python3
# llama_light/_python_check.py
"""Python version detection and installation helper."""

import os
import sys
import platform
import subprocess
from typing import Tuple, Optional

MIN_PYTHON_VERSION = (3, 8)
RECOMMENDED_PYTHON_VERSION = (3, 10)

def check_python() -> Tuple[bool, Optional[str]]:
    """
    Check if Python version meets requirements.
    
    Returns:
        (is_valid, error_message)
    """
    current = sys.version_info[:2]
    
    if current < MIN_PYTHON_VERSION:
        return False, f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required (found {current[0]}.{current[1]})"
    
    return True, None

def get_python_install_instructions() -> str:
    """Return OS-specific Python installation instructions."""
    system = platform.system().lower()
    
    instructions = {
        "linux": """
╔══════════════════════════════════════════════════════════════════╗
║  🐍 PYTHON INSTALLATION INSTRUCTIONS                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Ubuntu/Debian:                                                  ║
║    sudo apt update                                               ║
║    sudo apt install python3.10 python3-pip python3-venv          ║
║                                                                   ║
║  Fedora/RHEL:                                                    ║
║    sudo dnf install python3.10 python3-pip                       ║
║                                                                   ║
║  Arch Linux:                                                     ║
║    sudo pacman -S python python-pip                              ║
║                                                                   ║
║  After installation, re-run the llama-light installer:           ║
║    curl -sSL https://install.llama-light.sh | bash               ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""",
        "darwin": """
╔══════════════════════════════════════════════════════════════════╗
║  🐍 PYTHON INSTALLATION INSTRUCTIONS (macOS)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Using Homebrew:                                                 ║
║    brew install python@3.10                                      ║
║                                                                   ║
║  Using official installer:                                       ║
║    https://www.python.org/downloads/                             ║
║                                                                   ║
║  After installation, re-run the llama-light installer:           ║
║    curl -sSL https://install.llama-light.sh | bash               ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""",
        "windows": """
╔══════════════════════════════════════════════════════════════════╗
║  🐍 PYTHON INSTALLATION INSTRUCTIONS (Windows)                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. Download Python 3.10+ from:                                  ║
║     https://www.python.org/downloads/                            ║
║                                                                   ║
║  2. During installation, check:                                  ║
║     ✓ "Add Python to PATH"                                       ║
║     ✓ "Install pip"                                              ║
║                                                                   ║
║  3. Open PowerShell as Administrator and re-run:                 ║
║     curl -sSL https://install.llama-light.sh | bash              ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
    }
    
    return instructions.get(system, instructions["linux"])

def ensure_python() -> bool:
    """Ensure Python is available. Exit with instructions if not."""
    valid, error = check_python()
    
    if valid:
        print(f"✅ Python {sys.version_info[0]}.{sys.version_info[1]} detected")
        return True
    
    print(f"❌ {error}")
    print(get_python_install_instructions())
    return False