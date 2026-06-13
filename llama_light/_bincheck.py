import os
import sys
import shutil
import subprocess
from typing import Optional

LLAMA_BIN_LOCATIONS = [
    os.path.join(sys.prefix, "bin"),
    os.path.expanduser("~/.local/bin"),
    os.path.expanduser("~/bin"),
    "/usr/local/bin",
    "/usr/bin",
    "/snap/bin",
    os.path.expanduser("~/.npm-global/bin"),
    os.path.expanduser("~/miniconda3/bin"),
    os.path.expanduser("~/anaconda3/bin"),
    os.path.expanduser("~/.pyenv/shims"),
]

def detect_cuda() -> dict:
    result = {"available": False, "toolkit": False, "driver_version": None, "compute_cap": None}
    if not shutil.which("nvidia-smi"):
        return result
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",")
            result["available"] = True
            result["driver_version"] = parts[0].strip() if parts else None
            result["compute_cap"] = parts[1].strip() if len(parts) > 1 else None
    except Exception:
        return result
    result["toolkit"] = bool(shutil.which("nvcc"))
    return result

def _binary_has_cuda(path: str) -> bool:
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    try:
        r = subprocess.run(["ldd", path], capture_output=True, text=True, timeout=5)
        return "libcuda" in r.stdout or "libcublas" in r.stdout
    except Exception:
        return False

def _set_ld_library_path(lib_dir: str) -> None:
    if not lib_dir or not os.path.isdir(lib_dir):
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if not (existing and lib_dir in existing.split(":")):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + existing if existing else "")

def locate_self() -> Optional[str]:
    candidate = os.path.abspath(sys.argv[0])
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        if "llama" in os.path.basename(candidate):
            return candidate
    path = shutil.which("llama")
    if path:
        return os.path.abspath(path)
    for dir_path in LLAMA_BIN_LOCATIONS:
        if not os.path.isdir(dir_path):
            continue
        full = os.path.join(dir_path, "llama")
        if _is_executable(full):
            return full
    return None

def locate_main_bin() -> Optional[str]:
    from .config import LLAMA_SERVER_BIN
    if LLAMA_SERVER_BIN and _is_executable(LLAMA_SERVER_BIN):
        return LLAMA_SERVER_BIN
    local_build = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
    if _is_executable(local_build):
        return local_build
    try:
        from .__init__ import LLAMA_CPP_VERSION
        from ._llama_downloader import ensure_binaries
        cache_dir, cache_bin = ensure_binaries(LLAMA_CPP_VERSION)
        if cache_bin and _is_executable(cache_bin):
            _set_ld_library_path(os.path.dirname(cache_bin))
            return cache_bin
    except Exception:
        pass
    try:
        import llama_light
        src_dir = os.path.dirname(llama_light.__file__)
        rel = os.path.abspath(os.path.join(src_dir, "..", "build", "bin", "llama-server"))
        if _is_executable(rel):
            return rel
        bundled = os.path.abspath(os.path.join(src_dir, "..", "bin", "llama-server"))
        if _is_executable(bundled):
            _set_ld_library_path(os.path.dirname(bundled))
            return bundled
    except (ImportError, ModuleNotFoundError):
        pass
    for dir_path in LLAMA_BIN_LOCATIONS:
        if not os.path.isdir(dir_path):
            continue
        full = os.path.join(dir_path, "llama-server")
        if _is_executable(full):
            return full
    return shutil.which("llama-server")

def find_bin(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path and _is_executable(path):
        return path
    for dir_path in LLAMA_BIN_LOCATIONS:
        if not os.path.isdir(dir_path):
            continue
        for ext in ("", ".exe"):
            full = os.path.join(dir_path, name) + ext
            if _is_executable(full):
                return full
    return None

def check(name: str, install_hint: str = "") -> str:
    path = find_bin(name)
    if path:
        return path
    msg = f"[{name}] error: '{name}' is not installed or not in PATH."
    if install_hint:
        msg += f"\n  {install_hint}"
    print(msg, file=sys.stderr)
    sys.exit(1)

def status() -> dict:
    result = {}
    main_bin = locate_main_bin()
    result["llama"] = locate_self()
    result["llama-server"] = main_bin
    cuda = detect_cuda()
    result["_cuda_available"] = cuda["available"]
    result["_cuda_toolkit"] = cuda["toolkit"]
    if main_bin:
        result["_llama_server_has_cuda"] = _binary_has_cuda(main_bin)
    for name in ("hermes", "claude"):
        result[name] = find_bin(name)
    return result

def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)

def check_with_subcmd(name: str, subcmd: str, install_hint: str = "") -> str:
    """Check binary and verify subcommand support."""
    path = find_bin(name)
    if not path:
        msg = f"[{name}] error: '{name}' is not installed or not in PATH."
        if install_hint:
            msg += f"\n  {install_hint}"
        print(msg, file=sys.stderr)
        sys.exit(1)
    try:
        r = subprocess.run(
            [path, subcmd, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode in (0, 1, 2) and not (
            r.stderr and "unknown command" in r.stderr.lower()
        ):
            return path
    except Exception:
        pass
    msg = f"[{name}] error: '{name} {subcmd}' is not available (binary at {path})."
    if install_hint:
        msg += f"\n  {install_hint}"
    print(msg, file=sys.stderr)
    sys.exit(1)

def bundled_tool_binaries() -> dict:
    """Return paths to bundled llama.cpp tools."""
    from .config import LLAMA_SERVER_BIN
    bin_dir = None
    if LLAMA_SERVER_BIN:
        bin_dir = os.path.dirname(LLAMA_SERVER_BIN)
    else:
        bin_dir = os.path.expanduser("~/llama.cpp/build/bin")
    if not bin_dir or not os.path.isdir(bin_dir):
        return {}
    tools = [
        "llama-quantize", "llama-bench", "llama-perplexity", "llama-cli",
        "llama-gguf-split", "llama-tokenize", "llama-imatrix",
        "llama-embedding", "llama-parallel", "llama-speculative",
    ]
    return {
        t: os.path.join(bin_dir, t)
        for t in tools
        if os.path.isfile(os.path.join(bin_dir, t)) and os.access(os.path.join(bin_dir, t), os.X_OK)
    }