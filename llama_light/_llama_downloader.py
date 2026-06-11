#!/usr/bin/env python3
"""Auto-detect GPU and build llama.cpp with correct CUDA architecture."""
import os, shutil, subprocess, sys, tempfile, time, re, platform
from typing import Optional, Tuple, Dict

CACHE_ROOT = os.path.expanduser("~/.cache/llama_light/llama-cpp")

# ANSI colors
GREEN = "\033[0;32m"; YELLOW = "\033[1;33m"; RED = "\033[0;31m"
BLUE = "\033[0;34m"; CYAN = "\033[0;36m"; NC = "\033[0m"

def print_step(msg: str, step: int, total: int):
    print(f"\n{BLUE}[{step}/{total}]{NC} {msg}")

def print_ok(msg: str):    print(f"  {GREEN}✓{NC} {msg}")
def print_info(msg: str):  print(f"  {CYAN}➜{NC} {msg}")
def print_warn(msg: str):  print(f"  {YELLOW}⚠{NC} {msg}")
def print_error(msg: str): print(f"  {RED}✗{NC} {msg}")

class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.current = 0
        self.start_time = time.time()
    def step(self, msg: str):
        self.current += 1
        print_step(msg, self.current, self.total)
    def done(self):
        elapsed = time.time() - self.start_time
        print(f"\n{GREEN}✅ Complete!{NC} Total time: {elapsed:.1f}s\n")

def detect_gpu() -> Dict:
    result = {"available": False, "compute_cap": None, "gpu_name": None}
    print_info("Checking for NVIDIA GPU...")
    if not shutil.which("nvidia-smi"):
        print_warn("nvidia-smi not found")
        return result
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(", ")]
            result["gpu_name"] = parts[0]
            result["compute_cap"] = parts[1] if len(parts) > 1 else None
            result["available"] = True
            print_ok(f"GPU detected: {result['gpu_name']} (SM{result['compute_cap']})")
    except Exception as e:
        print_error(f"GPU detection failed: {e}")
    return result

def detect_cuda_version() -> Optional[str]:
    print_info("Checking CUDA Toolkit...")
    if not shutil.which("nvcc"):
        print_warn("nvcc not found")
        return None
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if "release" in line:
                match = re.search(r'release (\d+\.\d+)', line)
                if match:
                    ver = match.group(1)
                    print_ok(f"CUDA Toolkit: {ver}")
                    return ver
    except Exception as e:
        print_error(f"CUDA detection failed: {e}")
    return None

def get_required_cuda_version(compute_cap: str) -> str:
    cap = float(compute_cap)
    if cap >= 12.0:   return "13.3"   # RTX 50-series
    elif cap >= 11.0: return "12.4"
    elif cap >= 9.0:  return "11.8"
    elif cap >= 8.0:  return "11.4"
    elif cap >= 7.0:  return "10.2"
    return "10.0"

def cache_bin_dir(version: str, compute_cap: str) -> str:
    return os.path.join(CACHE_ROOT, version, f"sm{compute_cap.replace('.', '')}")

def binary_path(version: str, compute_cap: str) -> Optional[str]:
    d = cache_bin_dir(version, compute_cap)
    for cand in (os.path.join(d, "llama-server"), os.path.join(d, "bin", "llama-server")):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None

def run_with_progress(cmd, cwd, description: str) -> bool:
    print_info(f"{description}...")
    try:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_update = time.time()
        while process.poll() is None:
            if time.time() - last_update > 5:
                print(f"     Still working... ({int(time.time() - last_update)}s)", end="\r")
                last_update = time.time()
            time.sleep(0.5)
        print(" " * 50, end="\r")
        if process.returncode == 0:
            print_ok(f"{description} complete")
            return True
        else:
            print_error(f"{description} failed (code {process.returncode})")
            return False
    except Exception as e:
        print_error(f"{description} failed: {e}")
        return False

def build_from_source(version: str, cache_dir: str, compute_cap: str) -> bool:
    sm = compute_cap.replace(".", "")
    print_info(f"Building for SM{sm} (5-15 minutes)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "llama.cpp")
        build_dir = os.path.join(repo_dir, "build")

        # Clone
        print_info("Cloning llama.cpp...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", version,
                            "https://github.com/ggml-org/llama.cpp.git", repo_dir],
                           check=True, capture_output=True)
            print_ok("Repository cloned")
        except:
            print_info("Trying latest main...")
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", repo_dir], check=True)
            print_ok("Cloned latest")

        os.makedirs(build_dir, exist_ok=True)

        if not run_with_progress(["cmake", "..", "-DGGML_CUDA=ON",
                                  f"-DCMAKE_CUDA_ARCHITECTURES={sm}",
                                  "-DCMAKE_BUILD_TYPE=Release"], build_dir, "CMake configuration"):
            return False

        if not run_with_progress(["cmake", "--build", ".", "--config", "Release",
                                  "-j", str(os.cpu_count() or 4)], build_dir,
                                 f"Building with {os.cpu_count() or 4} threads"):
            return False

        # Copy binaries & libs
        print_info("Installing to cache...")
        for src_dir_name in ("bin", "lib"):
            src_dir = os.path.join(build_dir, src_dir_name)
            if os.path.isdir(src_dir):
                for item in os.listdir(src_dir):
                    src = os.path.join(src_dir, item)
                    dst = os.path.join(cache_dir, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        print(f"     Copied: {item}")

        server_path = os.path.join(cache_dir, "llama-server")
        if os.path.exists(server_path):
            print_ok(f"Binary ready: {server_path}")
            return True
        return False

def ensure_binaries(version: str) -> Tuple[Optional[str], Optional[str]]:
    progress = ProgressTracker(4)

    progress.step("Detecting GPU")
    gpu = detect_gpu()
    if not gpu["available"]:
        print_error("No NVIDIA GPU detected. NVIDIA GPU + CUDA required.")
        return None, None

    compute_cap = gpu["compute_cap"] or "89"
    sm = compute_cap.replace(".", "")
    cache_dir = cache_bin_dir(version, compute_cap)
    server = binary_path(version, compute_cap)

    progress.step("Checking cache")
    if server:
        print_ok(f"Using cached binary for SM{sm}")
        return cache_dir, server
    print_info("No cached binary → will build")

    progress.step("Checking CUDA")
    cuda_version = detect_cuda_version()
    required = get_required_cuda_version(compute_cap)

    print_info(f"GPU (SM{sm}) requires CUDA {required}+")

    if not cuda_version:
        print_error(f"CUDA Toolkit not found. Install CUDA {required}+")
        print_info("→ https://developer.nvidia.com/cuda-downloads")
        return None, None

    try:
        from packaging import version
        if version.parse(cuda_version) < version.parse(required):
            print_error(f"CUDA {cuda_version} too old (need {required}+)")
            print_info("→ Upgrade: https://developer.nvidia.com/cuda-downloads")
            return None, None
        print_ok(f"CUDA {cuda_version} is sufficient")
    except ImportError:
        print_warn("packaging module missing — assuming version OK")

    progress.step("Building from source")
    os.makedirs(cache_dir, exist_ok=True)
    if build_from_source(version, cache_dir, compute_cap):
        server = binary_path(version, compute_cap)
        if server:
            progress.done()
            return cache_dir, server

    print_error("Build failed. Check CUDA installation and try again.")
    return None, None

# Required stubs for _cli.py
def check_version(version: str) -> str:
    return "up_to_date"

def upgrade() -> bool:
    return False

def bundled_tool_binaries() -> dict:
    return {}

def check_with_subcmd(name: str, subcmd: str, install_hint: str = "") -> str:
    path = shutil.which(name)
    if not path:
        print(f"Error: {name} not found. {install_hint}")
        sys.exit(1)
    return path

def find_bin(name: str):
    return shutil.which(name)
