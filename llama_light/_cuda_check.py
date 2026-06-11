# llama_light/_cuda_check.py
"""CUDA detection and verification."""

import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

def detect_cuda() -> Dict:
    """
    Detect CUDA availability and return detailed information.
    
    Returns:
        {
            "available": bool,
            "driver_version": str,
            "cuda_version": str,
            "compute_capability": str,
            "toolkit_installed": bool,
            "gpu_name": str,
            "vram_gb": float,
            "error": str
        }
    """
    result = {
        "available": False,
        "driver_version": None,
        "cuda_version": None,
        "compute_capability": None,
        "toolkit_installed": False,
        "gpu_name": None,
        "vram_gb": 0.0,
        "error": None
    }
    
    # Check for nvidia-smi
    if not shutil.which("nvidia-smi"):
        result["error"] = "nvidia-smi not found - NVIDIA drivers not installed"
        return result
    
    try:
        # Get GPU info
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(", ")]
            result["gpu_name"] = parts[0] if len(parts) > 0 else None
            result["driver_version"] = parts[1] if len(parts) > 1 else None
            result["compute_capability"] = parts[2] if len(parts) > 2 else None
            if len(parts) > 3:
                try:
                    result["vram_gb"] = int(parts[3].split()[0]) / 1024
                except:
                    pass
            result["available"] = True
    
    except Exception as e:
        result["error"] = str(e)
        return result
    
    # Check for CUDA toolkit
    result["toolkit_installed"] = shutil.which("nvcc") is not None
    
    # Get CUDA version from nvcc if available
    if result["toolkit_installed"]:
        try:
            r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            for line in r.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.3"
                    import re
                    match = re.search(r'release (\d+\.\d+)', line)
                    if match:
                        result["cuda_version"] = match.group(1)
                    break
        except:
            pass
    
    return result

def print_cuda_error(cuda_info: Dict) -> None:
    """Display formatted CUDA error with recommendations."""
    
    error_msg = f"""
╔══════════════════════════════════════════════════════════════════╗
║  ❌ CUDA GPU REQUIRED BUT NOT AVAILABLE                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  This version of llama-light REQUIRES an NVIDIA GPU with CUDA.   ║
║  No CPU fallback is provided.                                    ║
║                                                                   ║
"""
    
    if cuda_info.get("error"):
        error_msg += f"║  Detected issue: {cuda_info['error'][:50]}                    ║\n"
    
    error_msg += """
╠══════════════════════════════════════════════════════════════════╣
║  📋 INSTALLATION STEPS:                                           ║
║                                                                   ║
║  1. Install NVIDIA drivers:                                      ║
║     Ubuntu/Debian:  sudo apt install nvidia-driver-550          ║
║     Fedora:         sudo dnf install akmod-nvidia               ║
║     Arch:           sudo pacman -S nvidia                       ║
║                                                                   ║
║  2. Install CUDA Toolkit (required for auto-build):              ║
║     https://developer.nvidia.com/cuda-downloads                  ║
║                                                                   ║
║  3. Reboot after installation:                                   ║
║     sudo reboot                                                  ║
║                                                                   ║
║  4. Verify installation:                                         ║
║     nvidia-smi                                                   ║
║     nvcc --version                                               ║
║                                                                   ║
║  5. Re-run llama-light:                                          ║
║     llama start                                                  ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    print(error_msg)

def ensure_cuda() -> bool:
    """Ensure CUDA is available. Exit with instructions if not."""
    cuda_info = detect_cuda()
    
    if cuda_info["available"]:
        print(f"✅ CUDA detected:")
        print(f"   GPU: {cuda_info['gpu_name']}")
        print(f"   VRAM: {cuda_info['vram_gb']:.1f}GB")
        print(f"   Compute Capability: {cuda_info['compute_capability']}")
        print(f"   Driver: {cuda_info['driver_version']}")
        if cuda_info["toolkit_installed"]:
            print(f"   CUDA Toolkit: {cuda_info['cuda_version']} ✅")
        else:
            print(f"   ⚠️  CUDA Toolkit not installed (auto-build will fail)")
        return True
    
    print_cuda_error(cuda_info)
    return False