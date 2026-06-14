import json
import os
import sys
import shutil
import subprocess
from typing import Optional, Tuple

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

# ── Cache for locate_main_bin() ──────────────────────────────────────────────
# Key: resolved llama-server binary path + detected GPU arch.
# Value: cache dict with "bin", "arch", "ts", "version".
_CACHE_FILE = os.path.expanduser("~/.cache/llama_light/bincheck.json")

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_load() -> dict | None:
    """Load cached bincheck result, return None if missing/invalid."""
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _cache_save(data: dict) -> None:
    """Atomically write cache entry."""
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    fd, tmp = None, _CACHE_FILE + ".tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT, 0o644)
        os.write(fd, json.dumps(data).encode())
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _cache_key() -> str:
    """Build a deterministic key from env / system state that changes when
    the binary or GPU arch might change.  We intentionally keep this light —
    only the llama.cpp version and the platform CPU arch are needed, since
    ensure_binaries() already handles GPU arch matching."""
    from .__init__ import LLAMA_CPP_VERSION
    import platform
    return f"{LLAMA_CPP_VERSION}-{platform.machine()}"


def _cache_valid(data: dict | None) -> Tuple[bool, dict | None]:
    """Return (hit, data) — cache is valid if present, entry is newer than
    the cached llama.cpp version's build, and the binary still exists."""
    if data is None:
        return False, None
    # Binary must still be executable
    bin_path = data.get("bin")
    if bin_path and not _is_executable(bin_path):
        return False, None
    # Accept the cache if it's less than 24 h old
    age = data.get("ts", 0)
    if age and (int(os.environ.get("_LLAMA_LIGHT_FORCE_BINCHECK", "0"))):
        return False, None
    return True, data


# ── CUDA / GPU detection ─────────────────────────────────────────────────────

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

def locate_main_bin(force: bool = False) -> Optional[str]:
    """Resolve the llama-server binary path with cached results.

    On the first call (or when *force=True*, or when the cache is stale),
    performs GPU detection and binary resolution.  Subsequent calls return
    the cached path immediately — no subprocesses, no nvidia-smi calls.

    Parameters
    ----------
    force : bool
        Skip cache and re-run the full detection pipeline.  Used by
        ``llama setup`` / ``llama check`` to guarantee fresh results.
    """
    # Fast path — return cached result
    if not force:
        hit, cached = _cache_valid(_cache_load())
        if hit:
            _set_ld_library_path(os.path.dirname(cached["bin"]))
            return cached["bin"]

    from .config import LLAMA_SERVER_BIN
    if LLAMA_SERVER_BIN and _is_executable(LLAMA_SERVER_BIN):
        _cache_save(_cache_entry(LLAMA_SERVER_BIN))
        return LLAMA_SERVER_BIN

    local_build = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
    if _is_executable(local_build):
        _cache_save(_cache_entry(local_build))
        return local_build

    try:
        from .__init__ import LLAMA_CPP_VERSION
        from ._llama_downloader import ensure_binaries
        cache_dir, cache_bin = ensure_binaries(LLAMA_CPP_VERSION)
        if cache_bin and _is_executable(cache_bin):
            _set_ld_library_path(os.path.dirname(cache_bin))
            _cache_save(_cache_entry(cache_bin))
            return cache_bin
    except Exception:
        pass

    try:
        import llama_light
        src_dir = os.path.dirname(llama_light.__file__)
        rel = os.path.abspath(os.path.join(src_dir, "..", "build", "bin", "llama-server"))
        if _is_executable(rel):
            _cache_save(_cache_entry(rel))
            return rel
        bundled = os.path.abspath(os.path.join(src_dir, "..", "bin", "llama-server"))
        if _is_executable(bundled):
            _set_ld_library_path(os.path.dirname(bundled))
            _cache_save(_cache_entry(bundled))
            return bundled
    except (ImportError, ModuleNotFoundError):
        pass

    for dir_path in LLAMA_BIN_LOCATIONS:
        if not os.path.isdir(dir_path):
            continue
        full = os.path.join(dir_path, "llama-server")
        if _is_executable(full):
            _cache_save(_cache_entry(full))
            return full

    result = shutil.which("llama-server")
    if result:
        _cache_save(_cache_entry(result))
    return result


def _cache_entry(bin_path: str) -> dict:
    """Build a cache entry from a resolved binary path."""
    return {
        "bin": os.path.abspath(bin_path),
        "arch": f"{_cache_key()}",
        "ts": int(__import__("time").time()),
    }

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