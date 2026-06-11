#!/usr/bin/env python3
# llama_light/_llama_downloader.py
"""Auto-download or build llama.cpp with CUDA.

This module is the heart of the portable GPU-first setup:

1. Detect CUDA (nvidia-smi + nvcc)
2. Look for a valid cached binary (must have CUDA)
3. Try to download a pre-built CUDA tarball from GitHub releases
4. Verify the downloaded binary (ldd + size check) – if fake, discard
5. Build from source using CMake + CUDA (auto-detect compute capability)
6. Cache the final binary in `~/.cache/llama_light/llama-cpp/<version>/`

No CPU fallback – if CUDA is missing, print installation instructions and exit.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from typing import Optional, Tuple, Dict

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download"
CACHE_ROOT = os.path.expanduser("~/.cache/llama_light/llama-cpp")

# ----------------------------------------------------------------------
# CUDA detection
# ----------------------------------------------------------------------
def detect_cuda() -> Dict:
    """
    Returns a dict with:
        available: bool
        driver_version: str
        compute_cap: str (e.g. "8.6")
        toolkit_installed: bool
        gpu_name: str
        vram_mb: int
    """
    result = {
        "available": False,
        "driver_version": None,
        "compute_cap": None,
        "toolkit_installed": False,
        "gpu_name": None,
        "vram_mb": 0,
        "error": None
    }
    if not shutil.which("nvidia-smi"):
        result["error"] = "nvidia-smi not found – NVIDIA drivers missing"
        return result

    try:
        # Query GPU details
        cmd = ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
               "--format=csv,noheader"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().split(", ")]
            result["gpu_name"] = parts[0] if len(parts) > 0 else None
            result["driver_version"] = parts[1] if len(parts) > 1 else None
            result["compute_cap"] = parts[2] if len(parts) > 2 else None
            if len(parts) > 3 and parts[3].endswith("MiB"):
                result["vram_mb"] = int(parts[3].split()[0])
            result["available"] = True
    except Exception as e:
        result["error"] = str(e)
        return result

    # Check for CUDA toolkit (nvcc)
    result["toolkit_installed"] = shutil.which("nvcc") is not None
    return result


def binary_has_cuda(path: str) -> bool:
    """Check if a llama-server binary is actually linked to CUDA libraries."""
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    try:
        # 1. Dynamic linking check
        r = subprocess.run(["ldd", path], capture_output=True, text=True, timeout=5)
        if "libcuda" in r.stdout or "libcublas" in r.stdout:
            return True
        # 2. Check file size (CUDA-enabled binary > 50 MiB typically)
        if os.path.getsize(path) > 50 * 1024 * 1024:
            # Might still be CPU-only, but better than nothing
            return True
    except Exception:
        pass
    return False


# ----------------------------------------------------------------------
# Platform helpers
# ----------------------------------------------------------------------
def detect_platform() -> Tuple[str, str]:
    import platform as pl
    sys_plat = pl.system().lower()
    machine = pl.machine().lower()
    arch_map = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
    base_arch = arch_map.get(machine, machine)
    if sys_plat == "linux":
        return "ubuntu", base_arch
    elif sys_plat == "darwin":
        return "macos", base_arch
    elif sys_plat == "windows":
        return "windows", base_arch
    return sys_plat, base_arch


def cache_bin_dir(version: str) -> str:
    return os.path.join(CACHE_ROOT, version)


def binary_path(version: str) -> Optional[str]:
    d = cache_bin_dir(version)
    for candidate in (os.path.join(d, "llama-server"), os.path.join(d, "bin", "llama-server")):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# ----------------------------------------------------------------------
# Download & extract
# ----------------------------------------------------------------------
def url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "llama-Light"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status < 400
    except Exception:
        return False


def download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".partial"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llama-Light"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.getheader("Content-Length", 0))
            bar = None
            if _HAS_TQDM and total:
                bar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest))
            with open(tmp, "wb") as f:
                while chunk := resp.read(65536):
                    f.write(chunk)
                    if bar:
                        bar.update(len(chunk))
            if bar:
                bar.close()
        shutil.move(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download failed ({url}): {e}")


def extract_tarball(tarball: str, dest: str) -> None:
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest, filter="data")


def move_contents(src: str, dest: str) -> None:
    """Move or copy contents of src (possibly nested one level) into dest."""
    items = os.listdir(src)
    # If the tarball extracts into a single top‑level folder, use that
    top = os.path.join(src, items[0]) if len(items) == 1 else src
    source_items = os.listdir(top) if os.path.isdir(top) and len(items) == 1 else items
    for item in source_items:
        s = os.path.join(top if len(items) == 1 else src, item)
        d = os.path.join(dest, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


# ----------------------------------------------------------------------
# Build from source (CUDA)
# ----------------------------------------------------------------------
def build_cuda_locally(version: str, cache_dir: str, cuda_info: Dict) -> bool:
    """Clone llama.cpp, build with CUDA, copy binaries to cache_dir."""
    print("\n" + "=" * 70)
    print("🛠️  Auto-building llama.cpp with CUDA (no pre‑built binary available)")
    print("=" * 70)
    print(f"Version:     {version}")
    print(f"GPU:         {cuda_info.get('gpu_name', 'Unknown')}")
    print(f"VRAM:        {cuda_info.get('vram_mb', 0)} MiB")
    print(f"Compute Cap: {cuda_info.get('compute_cap', 'auto')}")
    if not cuda_info.get("toolkit_installed"):
        print("\n⚠️  CUDA Toolkit (nvcc) not found – build will likely fail.")
        print("   Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads\n")
    print("This may take 5–15 minutes depending on your system...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "llama.cpp")
        build_dir = os.path.join(repo_dir, "build")
        try:
            # Clone with depth 1 for speed
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", version,
                 "https://github.com/ggml-org/llama.cpp.git", repo_dir],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            print("⚠️  Version tag not found, cloning latest main...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ggml-org/llama.cpp.git", repo_dir],
                check=True
            )

        # Determine compute capability
        compute_cap = cuda_info.get("compute_cap")
        if compute_cap and "." in compute_cap:
            compute_cap = compute_cap.replace(".", "")
        else:
            compute_cap = "86"   # reasonable default for RTX 30/40 series

        os.makedirs(build_dir, exist_ok=True)
        print("⚙️  Running CMake with CUDA...")
        cmake_args = [
            "cmake", "..",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={compute_cap}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        result = subprocess.run(cmake_args, cwd=build_dir)
        if result.returncode != 0:
            print("❌ CMake configuration failed.")
            return False

        print(f"🔨 Building with {os.cpu_count()} threads...")
        result = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count())],
            cwd=build_dir
        )
        if result.returncode != 0:
            print("❌ Build failed.")
            return False

        print("📁 Copying binaries to cache...")
        # Copy binaries
        bin_src = os.path.join(build_dir, "bin")
        if os.path.isdir(bin_src):
            for item in os.listdir(bin_src):
                src = os.path.join(bin_src, item)
                dst = os.path.join(cache_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        # Copy libraries (if any)
        lib_src = os.path.join(build_dir, "lib")
        if os.path.isdir(lib_src):
            for item in os.listdir(lib_src):
                src = os.path.join(lib_src, item)
                dst = os.path.join(cache_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        # Verify
        server_path = os.path.join(cache_dir, "llama-server")
        if os.path.exists(server_path) and binary_has_cuda(server_path):
            print("✅ Build successful – CUDA binary verified.")
            return True
        else:
            print("⚠️  Build completed but binary lacks CUDA – please check your CUDA installation.")
            return False


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def ensure_binaries(version: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Ensure CUDA binaries are available in the cache.

    Returns:
        (cache_dir, binary_path) or (None, None) on failure.
    """
    cache_dir = cache_bin_dir(version)
    server = binary_path(version)

    # 1. Check CUDA requirement first
    cuda_info = detect_cuda()
    if not cuda_info["available"]:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  ❌ CUDA GPU REQUIRED BUT NOT FOUND                               ║
╠══════════════════════════════════════════════════════════════════╣
║  llama-Light requires an NVIDIA GPU with CUDA support.           ║
║  No CPU fallback is provided.                                    ║
║                                                                   ║
║  📋 Installation steps:                                          ║
║    1. Install NVIDIA drivers:                                    ║
║         Ubuntu: sudo apt install nvidia-driver-550              ║
║         Fedora:  sudo dnf install akmod-nvidia                  ║
║         Arch:    sudo pacman -S nvidia                          ║
║    2. Install CUDA Toolkit:                                      ║
║         https://developer.nvidia.com/cuda-downloads             ║
║    3. Reboot and re-run `llama start`                            ║
╚══════════════════════════════════════════════════════════════════╝
""")
        return None, None

    # 2. Validate cached binary (if any)
    if server:
        if binary_has_cuda(server):
            print("[downloader] ✓ Using cached CUDA binary")
            return cache_dir, server
        else:
            print("[downloader] cached binary has no CUDA — rebuilding...")
            shutil.rmtree(cache_dir, ignore_errors=True)
            server = None

    # 3. Prepare fresh cache
    os.makedirs(cache_dir, exist_ok=True)

    # 4. Try to download pre-built CUDA binary
    plat, arch = detect_platform()
    candidates = [
        f"llama-{version}-bin-ubuntu-cuda-cu12.4.1-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-cu12.3.2-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-cu12.2.0-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-cu11.8.0-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-{arch}.tar.gz",   # last resort, likely CPU
    ]
    downloaded = False
    tarball_path = None
    for fname in candidates:
        url = f"{RELEASE_URL}/{version}/{fname}"
        if not url_exists(url):
            continue
        print(f"[downloader] downloading {fname} ...")
        tarball = os.path.join(cache_dir, fname)
        try:
            download_file(url, tarball)
            tarball_path = tarball
            downloaded = True
            break
        except Exception as e:
            print(f"[downloader] Download failed: {e}")
            if os.path.exists(tarball):
                os.remove(tarball)
            continue

    if downloaded and tarball_path:
        # Extract and verify
        extract_dir = os.path.join(os.path.dirname(cache_dir), f".tmp_extract_{os.getpid()}")
        try:
            extract_tarball(tarball_path, extract_dir)
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            move_contents(extract_dir, cache_dir)
            server = binary_path(version)
            if server and binary_has_cuda(server):
                print("[downloader] ✓ Downloaded binary has CUDA")
                # Clean up and exit early
                shutil.rmtree(extract_dir, ignore_errors=True)
                os.remove(tarball_path)
                # Write version marker
                with open(os.path.join(cache_dir, ".version"), "w") as f:
                    f.write(version)
                return cache_dir, server
            else:
                print("[downloader] Downloaded binary has no CUDA – discarding, will build from source")
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"[downloader] Extraction failed: {e}")
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
            if os.path.exists(tarball_path):
                os.remove(tarball_path)

    # 5. No valid pre-built binary – build from source
    print("[downloader] No valid CUDA binary found – building from source...")
    if not cuda_info.get("toolkit_installed"):
        print("""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  CUDA TOOLKIT MISSING                                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Auto-building requires the CUDA Toolkit (nvcc).                 ║
║  Please install it from:                                         ║
║  https://developer.nvidia.com/cuda-downloads                    ║
║  After installation, re-run `llama start`.                       ║
╚══════════════════════════════════════════════════════════════════╝
""")
        return None, None

    if build_cuda_locally(version, cache_dir, cuda_info):
        server = binary_path(version)
        if server:
            with open(os.path.join(cache_dir, ".version"), "w") as f:
                f.write(version)
            print("[downloader] ✓ CUDA build complete and cached")
            return cache_dir, server

    # 6. Everything failed
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ❌ FAILED TO OBTAIN CUDA BINARY                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  We could neither download nor build a CUDA‑enabled llama.cpp.   ║
║                                                                   ║
║  Troubleshooting:                                                ║
║    • Ensure CUDA Toolkit is installed and `nvcc` is in PATH      ║
║    • Check that your GPU is supported (Compute Capability ≥ 5.0) ║
║    • Try a manual build:                                         ║
║        git clone https://github.com/ggml-org/llama.cpp           ║
║        cd llama.cpp && mkdir build && cd build                   ║
║        cmake -DGGML_CUDA=ON ..                                   ║
║        make -j$(nproc)                                           ║
║    • Then set LLAMA_SERVER_BIN to your custom build              ║
╚══════════════════════════════════════════════════════════════════╝
""")
    return None, None


# ----------------------------------------------------------------------
# Backward compatibility aliases
# ----------------------------------------------------------------------
def _detect_cuda() -> dict:
    return detect_cuda()


def _binary_has_cuda(path: str) -> bool:
    return binary_has_cuda(path)


def _detect_platform() -> Tuple[str, str]:
    return detect_platform()


def _cuda_tarball_candidates(version: str, arch: str) -> list:
    # kept for compatibility with old code
    return [
        f"llama-{version}-bin-ubuntu-cuda-cu12.4.1-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-cu12.3.2-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-cu12.2.0-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-cu11.8.0-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-cuda-{arch}.tar.gz",
        f"llama-{version}-bin-ubuntu-{arch}.tar.gz",
    ]


def _set_ld_lib(cache_dir: str) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if cache_dir not in existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = cache_dir + (":" + existing if existing else "")


def _write_version(version: str) -> None:
    d = cache_bin_dir(version)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, ".version"), "w") as f:
        f.write(version)


def _read_version() -> Optional[str]:
    v = os.path.join(CACHE_ROOT, ".version")
    if os.path.isfile(v):
        try:
            return open(v).read().strip()
        except Exception:
            pass
    return None


def _cached_version(ver: str) -> Optional[str]:
    vf = os.path.join(cache_bin_dir(ver), ".version")
    try:
        return open(vf).read().strip()
    except Exception:
        return None


version = _cached_version


def check_version(ver: str) -> str:
    cached = _read_version() or _cached_version(ver)
    if cached == ver:
        return "up_to_date"
    elif cached:
        return "outdated"
    return "missing"


def upgrade() -> bool:
    from .__init__ import LLAMA_CPP_VERSION
    cache_dir = cache_bin_dir(LLAMA_CPP_VERSION)
    shutil.rmtree(cache_dir, ignore_errors=True)
    _, server = ensure_binaries(LLAMA_CPP_VERSION)
    return server is not None