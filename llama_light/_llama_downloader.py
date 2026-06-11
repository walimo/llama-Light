#!/usr/bin/env python3
# llama_light/_llama_downloader.py – compute‑capability‑first, auto‑build fallback
import os, shutil, subprocess, sys, tarfile, tempfile, urllib.request
from typing import Optional, Tuple, Dict
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

RELEASE_URL = "https://github.com/ai-dock/llama.cpp-cuda/releases/download"
CACHE_ROOT = os.path.expanduser("~/.cache/llama_light/llama-cpp")

def detect_cuda() -> Dict:
    result = {"available": False, "toolkit": False, "driver_version": None,
              "compute_cap": None, "gpu_name": None, "vram_mb": 0, "error": None}
    if not shutil.which("nvidia-smi"):
        result["error"] = "nvidia-smi not found"
        return result
    try:
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
    result["toolkit"] = shutil.which("nvcc") is not None
    return result

def binary_has_cuda(path: str) -> bool:
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    try:
        r = subprocess.run(["ldd", path], capture_output=True, text=True, timeout=5)
        return "libcuda" in r.stdout or "libcublas" in r.stdout
    except:
        return False

def cache_bin_dir(version: str, compute_cap: str = None) -> str:
    if compute_cap:
        return os.path.join(CACHE_ROOT, version, f"cuda-{compute_cap}")
    return os.path.join(CACHE_ROOT, version)

def binary_path(version: str, compute_cap: str = None) -> Optional[str]:
    d = cache_bin_dir(version, compute_cap)
    for candidate in (os.path.join(d, "llama-server"), os.path.join(d, "bin", "llama-server")):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None

def url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "llama-Light"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status < 400
    except:
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
                    if bar: bar.update(len(chunk))
            if bar: bar.close()
        shutil.move(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp): os.remove(tmp)
        raise RuntimeError(f"Download failed ({url}): {e}")

def extract_tarball(tarball: str, dest: str) -> None:
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest, filter="data")

def move_contents(src: str, dest: str) -> None:
    items = os.listdir(src)
    top = os.path.join(src, items[0]) if len(items) == 1 else src
    source_items = os.listdir(top) if os.path.isdir(top) and len(items) == 1 else items
    for item in source_items:
        s = os.path.join(top if len(items) == 1 else src, item)
        d = os.path.join(dest, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def build_cuda_locally(version: str, cache_dir: str, cuda_info: Dict) -> bool:
    print("\n" + "=" * 70)
    print("🛠️  Auto-building llama.cpp with CUDA (no pre‑built binary available)")
    print("=" * 70)
    if not cuda_info.get("toolkit"):
        print("❌ CUDA Toolkit not found. Cannot build from source.")
        return False
    compute_cap = cuda_info.get("compute_cap", "86")
    if compute_cap and "." in compute_cap:
        compute_cap = compute_cap.replace(".", "")
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "llama.cpp")
        build_dir = os.path.join(repo_dir, "build")
        subprocess.run(["git", "clone", "--depth", "1", "--branch", version,
                        "https://github.com/ggml-org/llama.cpp.git", repo_dir], check=True)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", "..", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={compute_cap}",
                        "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j", str(os.cpu_count())],
                       cwd=build_dir, check=True)
        bin_src = os.path.join(build_dir, "bin")
        if os.path.isdir(bin_src):
            for item in os.listdir(bin_src):
                src = os.path.join(bin_src, item)
                dst = os.path.join(cache_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        lib_src = os.path.join(build_dir, "lib")
        if os.path.isdir(lib_src):
            for item in os.listdir(lib_src):
                src = os.path.join(lib_src, item)
                dst = os.path.join(cache_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        server_path = os.path.join(cache_dir, "llama-server")
        return os.path.exists(server_path) and binary_has_cuda(server_path)

def ensure_binaries(version: str) -> Tuple[Optional[str], Optional[str]]:
    cuda_info = detect_cuda()
    if not cuda_info["available"]:
        print("❌ No NVIDIA GPU detected. This tool requires CUDA.")
        return None, None

    compute_cap = cuda_info.get("compute_cap", "")
    if not compute_cap:
        print("⚠️  Could not detect compute capability. Will try generic build.")
        compute_cap = "89"  # safe fallback

    cache_dir = cache_bin_dir(version, compute_cap.replace(".", ""))
    server = binary_path(version, compute_cap.replace(".", ""))

    if server and binary_has_cuda(server):
        print(f"[downloader] ✓ Using cached CUDA binary for SM{compute_cap}")
        return cache_dir, server

    # Try to download pre‑built binary for this compute capability
    sm = compute_cap.replace(".", "")
    candidates = [
        f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/{version}/llama-server-sm{sm}.tar.gz",
        f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/latest/llama-server-sm{sm}.tar.gz",
        f"https://github.com/ai-dock/llama.cpp-cuda/releases/download/{version}/llama-server-sm89.tar.gz",
    ]
    downloaded = False
    tarball = None
    for url in candidates:
        if url_exists(url):
            print(f"[downloader] Downloading pre‑built CUDA binary from {url}")
            tarball = os.path.join(cache_dir, os.path.basename(url))
            try:
                download_file(url, tarball)
                downloaded = True
                break
            except Exception as e:
                print(f"Download failed: {e}")
                if tarball and os.path.exists(tarball):
                    os.remove(tarball)
    if downloaded and tarball:
        extract_dir = os.path.join(os.path.dirname(cache_dir), f".tmp_extract_{os.getpid()}")
        try:
            extract_tarball(tarball, extract_dir)
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            move_contents(extract_dir, cache_dir)
            server = binary_path(version, compute_cap.replace(".", ""))
            if server and binary_has_cuda(server):
                print(f"[downloader] ✓ Pre‑built CUDA binary for SM{sm} installed.")
                with open(os.path.join(cache_dir, ".version"), "w") as f:
                    f.write(version)
                return cache_dir, server
        except Exception as e:
            print(f"Extraction failed: {e}")
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
            if tarball and os.path.exists(tarball):
                os.remove(tarball)

    # Fallback to local build
    print("[downloader] No pre‑built binary found. Attempting local build...")
    os.makedirs(cache_dir, exist_ok=True)
    if build_cuda_locally(version, cache_dir, cuda_info):
        with open(os.path.join(cache_dir, ".version"), "w") as f:
            f.write(version)
        print("[downloader] ✓ Build successful.")
        return cache_dir, binary_path(version, compute_cap.replace(".", ""))

    print("❌ Could not obtain a CUDA binary. Please ensure CUDA Toolkit is installed or try a different version.")
    return None, None
