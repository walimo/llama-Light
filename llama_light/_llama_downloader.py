#!/usr/bin/env python3
"""Auto-detect GPU and build llama.cpp with correct CUDA architecture.

This module handles:
- GPU and CUDA detection
- Binary caching
- CUDA version validation
- Source building with proper error handling
- Progress tracking
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import re
import platform
from typing import Optional, Tuple, Dict

CACHE_ROOT = os.path.expanduser("~/.cache/llama_light/llama-cpp")

# ANSI colors for pretty output
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color


# ─────────────────────────────────────────────────────────────────────────────
# Logging Functions
# ─────────────────────────────────────────────────────────────────────────────

def print_step(msg: str, step: int, total: int):
    """Print a numbered step message."""
    print(f"\n{BLUE}[{step}/{total}]{NC} {msg}")


def print_ok(msg: str):
    """Print a success message."""
    print(f"  {GREEN}✓{NC} {msg}")


def print_info(msg: str):
    """Print an informational message."""
    print(f"  {CYAN}➜{NC} {msg}")


def print_warn(msg: str):
    """Print a warning message."""
    print(f"  {YELLOW}⚠{NC} {msg}")


def print_error(msg: str):
    """Print an error message."""
    print(f"  {RED}✗{NC} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Version Comparison (with fallback if packaging not available)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_version(v: str) -> Tuple[int, ...]:
    """Fallback version parser: '12.4.5' → (12, 4, 5).
    
    This is used when the packaging module is not available.
    Returns a tuple of integers that can be compared directly.
    """
    try:
        parts = v.split(".")
        return tuple(int(p) for p in parts[:3])
    except (ValueError, IndexError, AttributeError):
        return (0, 0, 0)


def _version_sufficient(current: str, required: str) -> bool:
    """Check if current version >= required version.
    
    Tries to use packaging.version for accuracy, falls back to
    simple tuple comparison if packaging is not available.
    """
    try:
        from packaging import version
        return version.parse(current) >= version.parse(required)
    except ImportError:
        # Fallback: simple tuple comparison
        curr_tuple = _parse_version(current)
        req_tuple = _parse_version(required)
        return curr_tuple >= req_tuple


# ─────────────────────────────────────────────────────────────────────────────
# GPU and CUDA Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_gpu() -> Dict:
    """Detect NVIDIA GPU and compute capability.
    
    Returns:
        Dict with keys:
            - available (bool): GPU detected
            - compute_cap (str): Compute capability (e.g., "12.0")
            - gpu_name (str): GPU model name
    """
    result = {"available": False, "compute_cap": None, "gpu_name": None}
    print_info("Checking for NVIDIA GPU...")
    
    if not shutil.which("nvidia-smi"):
        print_warn("nvidia-smi not found")
        return result
    
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(", ")]
            result["gpu_name"] = parts[0]
            result["compute_cap"] = parts[1] if len(parts) > 1 else None
            result["available"] = True
            print_ok(f"GPU detected: {result['gpu_name']} (SM{result['compute_cap']})")
    except subprocess.TimeoutExpired:
        print_error("GPU detection timed out (timeout=5s)")
    except Exception as e:
        print_error(f"GPU detection failed: {e}")
    
    return result


def detect_cuda_version() -> Optional[str]:
    """Detect CUDA Toolkit version.
    
    Returns:
        Version string (e.g., "12.4") or None if not found/error.
    """
    print_info("Checking CUDA Toolkit...")
    
    if not shutil.which("nvcc"):
        print_warn("nvcc not found (CUDA Toolkit not installed)")
        return None
    
    try:
        r = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        for line in r.stdout.read().splitlines():
            if "release" in line.lower():
                match = re.search(r'release (\d+\.\d+)', line)
                if match:
                    version_str = match.group(1)
                    print_ok(f"CUDA Toolkit: {version_str}")
                    return version_str
    except subprocess.TimeoutExpired:
        print_error("CUDA detection timed out (timeout=5s)")
    except Exception as e:
        print_error(f"CUDA detection failed: {e}")
    
    return None


def get_required_cuda_version(compute_cap: str) -> str:
    """Get minimum required CUDA version for a given compute capability.
    
    Args:
        compute_cap: Compute capability string (e.g., "12.0")
    
    Returns:
        Minimum required CUDA version string (e.g., "13.3")
    """
    try:
        cap = float(compute_cap)
    except (ValueError, TypeError):
        cap = 8.9  # Fallback default
    
    # Map compute capability to minimum required CUDA version
    if cap >= 12.0:
        return "13.3"   # RTX 50-series (Ada+)
    elif cap >= 11.0:
        return "12.4"   # RTX 30-series (Ampere)
    elif cap >= 9.0:
        return "11.8"   # RTX 20-series and Ampere compute only
    elif cap >= 8.0:
        return "11.4"   # RTX 20-series (Turing)
    elif cap >= 7.0:
        return "10.2"   # RTX 10-series (Volta/Turing)
    else:
        return "10.0"   # Older GPUs


# ─────────────────────────────────────────────────────────────────────────────
# Caching and Binary Resolution
# ─────────────────────────────────────────────────────────────────────────────

def cache_bin_dir(version: str, compute_cap: str) -> str:
    """Get the cache directory path for binaries for a specific GPU.
    
    Args:
        version: llama.cpp version string
        compute_cap: Compute capability (e.g., "12.0")
    
    Returns:
        Full path to cache directory
    """
    sm = compute_cap.replace(".", "")
    return os.path.join(CACHE_ROOT, version, f"sm{sm}")


def binary_path(version: str, compute_cap: str) -> Optional[str]:
    """Find cached llama-server binary for a specific compute capability.
    
    Returns:
        Full path to executable, or None if not found
    """
    cache_dir = cache_bin_dir(version, compute_cap)
    
    # Check multiple possible locations
    candidates = [
        os.path.join(cache_dir, "llama-server"),
        os.path.join(cache_dir, "bin", "llama-server"),
    ]
    
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Build Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_with_progress(cmd, cwd: str, description: str, timeout: Optional[int] = None) -> bool:
    """Run a command with progress indication.
    
    Args:
        cmd: Command list to execute
        cwd: Working directory
        description: Human-readable description of what's running
        timeout: Optional timeout in seconds
    
    Returns:
        True if successful (returncode 0), False otherwise
    """
    print_info(f"{description}...")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        last_update = time.time()
        while process.poll() is None:
            # Show progress every 5 seconds for long-running operations
            if time.time() - last_update > 5:
                elapsed = int(time.time() - last_update)
                print(f"     Still working... ({elapsed}s)", end="\r")
                last_update = time.time()
            time.sleep(0.5)
        
        # Clear progress line
        print(" " * 50, end="\r")
        
        if process.returncode == 0:
            print_ok(f"{description} complete")
            return True
        else:
            print_error(f"{description} failed (exit code {process.returncode})")
            # Show last few lines of output for debugging
            if process.stdout:
                lines = process.stdout.read().splitlines()[-3:]
                for line in lines:
                    print(f"     {line}")
            return False
    
    except subprocess.TimeoutExpired:
        process.kill()
        print_error(f"{description} timed out (timeout={timeout}s)")
        return False
    except Exception as e:
        print_error(f"{description} failed: {e}")
        return False


def build_from_source(version: str, cache_dir: str, compute_cap: str) -> bool:
    """Clone and build llama.cpp for a specific GPU architecture.
    
    Args:
        version: llama.cpp version/tag to build
        cache_dir: Where to store built binaries
        compute_cap: Compute capability (e.g., "12.0")
    
    Returns:
        True if build succeeded, False otherwise
    """
    sm = compute_cap.replace(".", "")
    print_info(f"Building for SM{sm} (5-15 minutes)...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "llama.cpp")
        build_dir = os.path.join(repo_dir, "build")
        
        # ── Clone repository ──────────────────────────────────────────────
        print_info("Cloning llama.cpp repository...")
        clone_success = False
        clone_error = None
        
        # Try specific version tag first
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", version,
                 "https://github.com/ggml-org/llama.cpp.git", repo_dir],
                check=True,
                capture_output=True,
                timeout=300
            )
            print_ok("Repository cloned (tagged version)")
            clone_success = True
        except subprocess.CalledProcessError as e:
            clone_error = f"Version tag '{version}' not found"
            print_info(f"Version tag not found, trying latest main...")
        except subprocess.TimeoutExpired:
            clone_error = "Clone timed out (timeout=300s)"
        except Exception as e:
            clone_error = str(e)
        
        # Fallback to latest main branch
        if not clone_success and not clone_error:
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/ggml-org/llama.cpp.git", repo_dir],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
                print_ok("Repository cloned (latest main)")
                clone_success = True
            except subprocess.CalledProcessError:
                print_error("Clone failed (no network or git issue)")
                return False
            except subprocess.TimeoutExpired:
                print_error("Clone timed out (timeout=300s)")
                return False
            except Exception as e:
                print_error(f"Clone failed: {e}")
                return False
        elif not clone_success:
            print_error(f"Clone failed: {clone_error}")
            return False
        
        # ── Prepare build directory ───────────────────────────────────────
        os.makedirs(build_dir, exist_ok=True)
        
        # ── CMake configuration ───────────────────────────────────────────
        if not run_with_progress(
            ["cmake", "..", "-DGGML_CUDA=ON",
             f"-DCMAKE_CUDA_ARCHITECTURES={sm}",
             "-DCMAKE_BUILD_TYPE=Release"],
            build_dir,
            "CMake configuration"
        ):
            return False
        
        # ── Build ─────────────────────────────────────────────────────────
        num_threads = os.cpu_count() or 4
        if not run_with_progress(
            ["cmake", "--build", ".", "--config", "Release",
             "-j", str(num_threads)],
            build_dir,
            f"Building with {num_threads} threads"
        ):
            return False
        
        # ── Copy binaries and libraries ───────────────────────────────────
        print_info("Installing binaries and libraries to cache...")
        files_copied = 0
        
        for src_dir_name in ("bin", "lib"):
            src_dir = os.path.join(build_dir, src_dir_name)
            if not os.path.isdir(src_dir):
                continue
            
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                if not os.path.isfile(src):
                    continue
                
                dst = os.path.join(cache_dir, item)
                try:
                    shutil.copy2(src, dst)
                    files_copied += 1
                    print(f"     Copied: {item}")
                except Exception as e:
                    print_warn(f"Failed to copy {item}: {e}")
        
        if files_copied == 0:
            print_error("No files were copied from build output")
            return False
        
        # ── Verify main binary exists ─────────────────────────────────────
        server_path = os.path.join(cache_dir, "llama-server")
        if os.path.exists(server_path) and os.access(server_path, os.X_OK):
            print_ok(f"Binary ready: {server_path}")
            return True
        else:
            print_error(f"Binary not found after build: {server_path}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Progress Tracking
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """Track multi-step progress with timing."""
    
    def __init__(self, total_steps: int):
        self.total = total_steps
        self.current = 0
        self.start_time = time.time()
    
    def step(self, msg: str):
        """Move to next step."""
        self.current += 1
        print_step(msg, self.current, self.total)
    
    def done(self):
        """Mark as complete and show total time."""
        elapsed = time.time() - self.start_time
        print(f"\n{GREEN}✅ Complete!{NC} Total time: {elapsed:.1f}s\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prebuilt Binary Fallback (CPU-only, last resort)
# ─────────────────────────────────────────────────────────────────────────────

def download_prebuilt_binary(version: str, compute_cap: str, cache_dir: str) -> Optional[str]:
    """Download a prebuilt CPU-only llama-server binary as a last-resort fallback.

    llama.cpp's official GitHub releases do not ship CUDA-enabled binaries
    (CUDA builds require runtime libraries matched to the host's CUDA
    install, which varies per machine). When a source build fails, this
    downloads the CPU-only release so the server can still run rather than
    leaving the user with nothing — performance will be reduced, and this
    is clearly surfaced to the user.

    Args:
        version: llama.cpp version/tag (e.g. "b9596")
        compute_cap: Compute capability (e.g. "12.0"), used only for cache pathing
        cache_dir: Where to extract the binary

    Returns:
        Path to llama-server binary, or None if download/extract failed
    """
    print_warn("Falling back to CPU-only prebuilt binary (reduced performance)")

    system = platform.system().lower()
    if system == "linux":
        asset_substr = "ubuntu-x64.zip"
    elif system == "darwin":
        asset_substr = "macos-arm64.zip" if platform.machine() == "arm64" else "macos-x64.zip"
    else:
        print_error(f"No prebuilt binary available for platform '{system}'")
        return None

    try:
        # Resolve the release tag to query
        tag = version if version.startswith("b") else "latest"
        api_url = (
            f"https://api.github.com/repos/ggml-org/llama.cpp/releases/{'latest' if tag == 'latest' else f'tags/{tag}'}"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "release.json")
            print_info(f"Querying llama.cpp releases ({tag})...")
            r = subprocess.run(
                ["curl", "-sSL", api_url, "-o", meta_path],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode != 0:
                print_error("Failed to query GitHub releases API")
                return None

            import json as _json
            with open(meta_path) as f:
                release = _json.load(f)

            asset_url = None
            for asset in release.get("assets", []):
                if asset_substr in asset.get("name", ""):
                    asset_url = asset.get("browser_download_url")
                    break

            if not asset_url:
                print_error(f"No matching prebuilt asset found ({asset_substr})")
                return None

            zip_path = os.path.join(tmpdir, "release.zip")
            print_info("Downloading prebuilt binary...")
            r = subprocess.run(
                ["curl", "-sSL", asset_url, "-o", zip_path],
                capture_output=True, text=True, timeout=300
            )
            if r.returncode != 0:
                print_error("Failed to download prebuilt binary")
                return None

            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            r = subprocess.run(
                ["unzip", "-o", "-q", zip_path, "-d", extract_dir],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode != 0:
                print_error("Failed to extract prebuilt binary archive")
                return None

            # Locate llama-server in extracted tree and copy alongside its libs
            os.makedirs(cache_dir, exist_ok=True)
            files_copied = 0
            for root, _, files in os.walk(extract_dir):
                for fname in files:
                    src = os.path.join(root, fname)
                    dst = os.path.join(cache_dir, fname)
                    try:
                        shutil.copy2(src, dst)
                        if fname in ("llama-server", "llama-server.exe"):
                            os.chmod(dst, 0o755)
                        files_copied += 1
                    except Exception:
                        pass

            if files_copied == 0:
                print_error("No files found in prebuilt archive")
                return None

            server_path = os.path.join(cache_dir, "llama-server")
            if os.path.isfile(server_path) and os.access(server_path, os.X_OK):
                print_ok(f"CPU-only binary ready: {server_path}")
                print_warn("This is a CPU-only build. GPU acceleration is unavailable.")
                print_info("To enable GPU acceleration, fix the build environment and run:")
                print_info("  llama setup --rebuild")
                return server_path

            print_error("llama-server not found in downloaded archive")
            return None

    except subprocess.TimeoutExpired:
        print_error("Prebuilt binary download timed out")
        return None
    except Exception as e:
        print_error(f"Prebuilt binary fallback failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def ensure_binaries(version: str) -> Tuple[Optional[str], Optional[str]]:
    """Ensure llama-server binaries are available.
    
    Orchestrates the full process:
    1. Detect GPU and compute capability
    2. Check cache for existing binary
    3. Validate CUDA Toolkit is installed and compatible
    4. Build from source if needed
    
    Args:
        version: llama.cpp version to target (e.g., "b9596")
    
    Returns:
        Tuple of (cache_dir, binary_path) or (None, None) if failed
    """
    progress = ProgressTracker(4)
    
    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Detect GPU
    # ──────────────────────────────────────────────────────────────────────
    progress.step("Detecting GPU")
    gpu = detect_gpu()
    
    if not gpu["available"]:
        print_error("No NVIDIA GPU detected. NVIDIA GPU + CUDA required.")
        return None, None
    
    compute_cap = gpu["compute_cap"] or "89"
    sm = compute_cap.replace(".", "")
    cache_dir = cache_bin_dir(version, compute_cap)
    
    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Check cache
    # ──────────────────────────────────────────────────────────────────────
    progress.step("Checking cache")
    server = binary_path(version, compute_cap)
    
    if server:
        print_ok(f"Using cached binary for SM{sm}")
        return cache_dir, server
    
    print_info("No cached binary found → will build from source")
    
    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Check CUDA Toolkit
    # ──────────────────────────────────────────────────────────────────────
    progress.step("Checking CUDA Toolkit")
    cuda_version = detect_cuda_version()
    required = get_required_cuda_version(compute_cap)
    
    print_info(f"GPU (SM{sm}) requires CUDA {required}+")
    
    if not cuda_version:
        print_error(f"CUDA Toolkit not found. Install CUDA {required}+")
        print_info("→ Download: https://developer.nvidia.com/cuda-downloads")
        return None, None
    
    # Validate version compatibility
    if not _version_sufficient(cuda_version, required):
        print_error(f"CUDA {cuda_version} is too old (need {required}+)")
        print_info("→ Upgrade: https://developer.nvidia.com/cuda-downloads")
        return None, None
    
    print_ok(f"CUDA {cuda_version} is sufficient for SM{sm}")
    
    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Build from source
    # ──────────────────────────────────────────────────────────────────────
    progress.step("Building from source")
    os.makedirs(cache_dir, exist_ok=True)
    
    if build_from_source(version, cache_dir, compute_cap):
        server = binary_path(version, compute_cap)
        if server:
            progress.done()
            return cache_dir, server
    
    print_error("Build failed. Check CUDA installation and try again.")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Required stubs for _cli.py imports
# ─────────────────────────────────────────────────────────────────────────────

def check_version(version: str) -> str:
    """Stub: Check if version is up to date."""
    return "up_to_date"


def upgrade() -> bool:
    """Stub: Upgrade to latest version."""
    return False


def bundled_tool_binaries() -> dict:
    """Stub: Return bundled tool binaries."""
    return {}


def check_with_subcmd(name: str, subcmd: str, install_hint: str = "") -> str:
    """Stub: Check binary with subcommand support."""
    path = shutil.which(name)
    if not path:
        print(f"Error: {name} not found. {install_hint}")
        sys.exit(1)
    return path


def find_bin(name: str) -> Optional[str]:
    """Stub: Find binary in PATH."""
    return shutil.which(name)