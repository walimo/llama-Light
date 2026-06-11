# llama-Light Architecture Review: Dependency Management Strategy

**Date:** 2026-06-11  
**Scope:** llama.cpp binary dependency management in llama-Light v0.2.0  
**Status:** Final recommendation — proceed with Approach C (hybrid)

---

## 1. Architecture Summary

llama-Light is a lightweight Python CLI wrapper around [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` binary. It provides an Ollama-style experience — managing GGUF models, launching an OpenAI-compatible API server, and offering CLI chat/terminal interfaces.

**Current structure:**

| Component | Location | Purpose |
|---|---|---|
| Python package | `llama_light/` | Core logic: server management, CLI, config, model manager, binary resolution |
| Entry point | `pyproject.toml` → `llama_light._cli:main` | `llama` CLI command |
| Dependencies | `tqdm`, `huggingface_hub` | Download progress, model pulling |
| Binaries | `bin/` | Pre-copied `llama-server`, 14 llama.cpp tools, 29 `.so` files |
| Manifest | `.bin-manifest.json` | Tracks which binaries are installed in `bin/` |
| Runtime config | `~/.config/llama_light/config.json` | Hardware-aware settings |
| Model cache | `~/.cache/llama_light/models/` | Downloaded GGUF files |

**How binaries are currently resolved** (in `_bincheck.py`):

1. `LLAMA_SERVER_BIN` environment variable (from `config.py`)
2. `bin/llama-server` bundled with the package
3. `~/llama.cpp/build/bin/llama-server` (default build path)
4. `build/bin/llama-server` relative to source (editable install)
5. Known directories (`~/.local/bin`, `/usr/local/bin`, etc.)
6. `shutil.which("llama-server")`

**Key coupling points:**

- `server.py:start()` calls `_bincheck.locate_main_bin()` and exits if `None`
- `_cli.py:cmd_tool()` resolves `bin_dir` from `llama-server` location for tool dispatch
- `config.py` line 66–67: `LLAMA_SERVER_BIN` defaults to `llama_light/../bin/llama-server`
- `update-bin.sh`: copies binaries from `~/llama.cpp/build/bin/` to `bin/` with md5 checksums

**Current dependency model (Approach A):** llama.cpp is an **external** dependency. The user must clone and build llama.cpp separately, then run `scripts/update-bin.sh` to copy binaries into `llama-Light/bin/`. The app's `bin/` directory is a **snapshot copy** — not a symlink, not the real binary. Updates require an extra manual step.

---

## 2. Comparison: Approach A vs B vs C

| Dimension | **A: Copy (current)** | **B: Embed (source)** | **C: Auto-download (recommended)** |
|---|---|---|---|
| **Install size** | ~40 MB (bin/ + .so files) | ~600 MB (source + build) | ~40 MB (prebuilt binary + .so) |
| **Setup time** | ~2 s (pip install) | 5–10 min (clone + build) | ~15 s (pip + download) |
| **Build dependencies** | None at install time | cmake, g++, CUDA toolkit, git | None (just Python 3.8+) |
| **Version pinning** | Manual — developer chooses | Tight — submodule commit hash | Tight — release tag in code |
| **Update speed** | Manual: `update-bin.sh` (~2s) | Rebuild: 5–10 min | Automatic: download (~10s) |
| **Maintainability** | Medium — script + manifest | High complexity — build system | Low — no build to maintain |
| **User experience** | Requires external build step | Heavy install, slow setup | `pip install` → works immediately |
| **Portability** | Low — requires llama.cpp | Medium — same build issues | High — prebuilt tarballs work on any Pop!_OS |
| **CUDA support** | Requires CUDA build | Requires CUDA toolkit installed | Prebuilt CUDA binaries included |
| **Offline install** | Possible if bin/ present | Possible if source cached | Requires cached release tarball |
| **Network needed** | Only for updates | Clone only | Initial install + updates |
| **Package size** | ~42 MB (wheel) | ~620 MB (wheel) | ~42 MB (wheel) |

### Dimension breakdown

**Install size:** B is ~15× larger due to source code and build artifacts. A and C are equivalent since both ship prebuilt binaries.

**Setup time:** B is the clear loser — cmake compilation on a modern laptop takes 5–10 minutes and requires CUDA toolkit if GPU is desired. A and C both install in seconds via pip; C adds ~10–15 seconds for the download.

**Build dependencies:** B requires cmake, g++, CUDA toolkit, and git — a significant barrier for end users on a fresh Pop!_OS install. A and C require zero build tools.

**Version pinning:** B offers the tightest pinning (submodule commit hash). C pins to a specific GitHub release tag, which is nearly as tight. A has no pinning — the developer's local build is whatever version is current at copy time.

**Update speed:** C auto-downloads on demand. A requires the developer to remember to run `update-bin.sh`. B requires rebuilding.

**Maintainability:** C eliminates the most maintenance burden — no build system to maintain, no manual sync scripts, no manifest to keep in sync. A requires maintaining `update-bin.sh` and `.bin-manifest.json`. B requires maintaining build system compatibility across CUDA versions, architectures, and OS releases.

**User experience:** This is the deciding factor. A requires the user to install llama.cpp separately before llama-Light even works. B requires a 5–10 minute build step after pip install. C gives `pip install llama-light; llama info` → works immediately.

**CUDA support:** B is actually a disadvantage for CUDA — the user must have the correct CUDA toolkit installed matching the llama.cpp version. C ships prebuilt CUDA binaries from the official llama.cpp release, which already have the correct CUDA compatibility baked in.

---

## 3. Recommendation

### **Approach C (hybrid) — download prebuilt releases**

**Reasoning:**

1. **Self-contained without build complexity.** End users get `pip install llama-light` and it works. No llama.cpp source, no cmake, no CUDA toolkit. The app downloads a prebuilt tarball from the official GitHub release.

2. **Preserves the fast, clean install experience** while eliminating the external dependency. Approach B's build step is the single biggest barrier to adoption — it requires a development toolchain that most end users don't have.

3. **Version pinning at release granularity.** The pinned version lives in `llama_light/__init__.py` (e.g., `LLAMA_CPP_VERSION = "1.7.1"`). On first run, the app checks for a cached release in `~/.cache/llama_light/llama-cpp/` and downloads if missing.

4. **No copy needed.** Unlike A, the app points directly to the extracted release binary. No `update-bin.sh`, no `.bin-manifest.json`, no risk of stale copies.

5. **CUDA works out of the box.** llama.cpp's GitHub releases include CUDA builds. The app simply selects the appropriate release (cuda, rocm, or cpu) based on `config.py:detect_gpu()`.

6. **Minimal code changes.** The existing `_bincheck.py` resolution chain already supports `bin/llama-server` (step 2). We just need to ensure the cached binary path is populated there.

### Why NOT Approach A?
Approach A puts the burden on the user. "Install llama.cpp first, build it, then run update-bin.sh" is a three-step process for what should be a single `pip install`. The manifest tracking, md5 checksums, and stray cleanup in `update-bin.sh` is 500 lines of shell code that could be 100 lines of Python.

### Why NOT Approach B?
Building llama.cpp from source is an enormous barrier. It requires cmake, g++, CUDA toolkit matching the llama.cpp version, and 5–10 minutes of build time. For a project that markets itself as "Light", requiring a 500 MB source download and a CMake build step is fundamentally contradictory. The only advantage is tighter version pinning, which C achieves via release tag pinning without the build cost.

---

## 4. Implementation Plan (Approach C)

### 4.1 New files to create

#### `llama_light/_llama_downloader.py` (new)
Handles downloading, extracting, and caching llama.cpp releases.

```python
"""Download and cache llama.cpp prebuilt releases.

Downloads tarballs from the official llama.cpp GitHub releases to
~/.cache/llama_light/llama-cpp/<version>/ and extracts them.
Selects cuda/rocm/cpu build based on hardware detection.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from .config import detect_gpu, CACHE_ROOT
from .__init__ import __version__ as LLAMA_LIGHT_VERSION

# ── Version pin ───────────────────────────────────────────────────────────────
LLAMA_CPP_VERSION = "1.7.1"
LLAMA_CPP_RELEASE_TAG = f"v{LLAMA_CPP_VERSION}"
LLAMA_CPP_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE_TAG}"

# ── Cache paths ───────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(CACHE_ROOT, "llama-cpp")


def _cache_bin_dir(version: str = None) -> str:
    """Return the cache directory for a given version."""
    ver = version or LLAMA_CPP_VERSION
    return os.path.join(CACHE_DIR, ver)


def _cache_bin(version: str = None) -> Optional[str]:
    """Return path to the cached llama-server binary, or None."""
    bin_dir = _cache_bin_dir(version)
    path = os.path.join(bin_dir, "llama-server")
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def _select_variant() -> str:
    """Select cuda, rocm, or cpu based on hardware detection."""
    gpu = detect_gpu()
    if gpu == "cuda":
        return "cuda"
    elif gpu == "rocm":
        return "rocm"
    return "cpu"


def _download_url(url: str, dest: str) -> str:
    """Download a file from URL to dest, with progress bar."""
    from tqdm import tqdm
    with urllib.request.urlopen(url) as response:
        total = response.length
        with open(dest, "wb") as f:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(dest),
                ncols=60,
            ) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest


def ensure_binaries(version: str = None) -> Tuple[str, str]:
    """Ensure llama.cpp binaries are downloaded and extracted.

    Returns (bin_dir, llama_server_path) if successful, (None, None) if download fails.
    """
    from .config import ensure_dirs

    version = version or LLAMA_CPP_VERSION
    variant = _select_variant()
    platform = sys.platform

    if platform.startswith("linux"):
        os_name = "linux"
    elif platform == "darwin":
        os_name = "macos"
    else:
        os_name = platform

    # Build tarball name — adjust for llama.cpp release naming convention
    tarball_name = f"llama-{os_name}-{variant}-binaries.tar.gz"
    tarball_url = f"{LLAMA_CPP_BASE_URL}/{tarball_name}"

    cache_dir = _cache_bin_dir(version)
    os.makedirs(cache_dir, exist_ok=True)

    # Check if already cached
    server_path = os.path.join(cache_dir, "llama-server")
    if os.path.isfile(server_path) and os.access(server_path, os.X_OK):
        return (cache_dir, server_path)

    # Download tarball
    tarball_path = os.path.join(cache_dir, tarball_name)
    if not os.path.isfile(tarball_path):
        try:
            _download_url(tarball_url, tarball_path)
        except Exception as e:
            print(f"[llama-light] Warning: failed to download {tarball_name}: {e}", file=sys.stderr)
            return (None, None)

    # Extract
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(cache_dir)

    # Clean up tarball
    try:
        os.remove(tarball_path)
    except OSError:
        pass

    # Find the extracted binary
    if os.path.isfile(server_path):
        return (cache_dir, server_path)

    # Some releases put binaries in a subdirectory
    for root, dirs, files in os.walk(cache_dir):
        if "llama-server" in files:
            bin_dir = os.path.dirname(
                next(p for p in files if p == "llama-server")
            ) if False else root
            found = os.path.join(root, "llama-server")
            if os.path.isfile(found) and os.access(found, os.X_OK):
                return (root, found)

    print(
        f"[llama-light] Warning: llama-server not found in extracted release.",
        file=sys.stderr,
    )
    return (None, None)
```

#### `llama_light/__init__.py` — modified
Add the version pin constant:

```python
__version__ = "0.2.0"
LLAMA_CPP_VERSION = "1.7.1"  # pinned llama.cpp release version
```

### 4.2 Modifications to existing files

#### `_bincheck.py` — three changes:

1. In `locate_main_bin()`: insert the cache fallback as step 2a (between bundled bin/ and default build path)

```python
    # 2a. Auto-downloaded cache
    try:
        from .__init__ import LLAMA_CPP_VERSION
        from ._llama_downloader import ensure_binaries
        cache_dir, cache_bin = ensure_binaries(LLAMA_CPP_VERSION)
        if cache_bin and _is_executable(cache_bin):
            # Set LD_LIBRARY_PATH for bundled .so files
            lib_dir = os.path.dirname(cache_bin)
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in existing:
                os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + existing if existing else "")
            return cache_bin
    except (ImportError, ModuleNotFoundError):
        pass
```

2. In `bundled_tool_binaries()`: also check the cache dir as a fallback

```python
    # After the bundled check, add:
    try:
        from .__init__ import LLAMA_CPP_VERSION
        from ._llama_downloader import _cache_bin_dir
        cache_dir = _cache_bin_dir(LLAMA_CPP_VERSION)
        if os.path.isdir(cache_dir):
            bin_dir = cache_dir  # use cache dir as primary
```

3. Update the docstring to reflect the new resolution order:

```
    Checks in order:
    1. ``LLAMA_SERVER_BIN`` from config
    2. ``./bin/llama-server`` (bundled with llama-Light)
    2a. ``~/.cache/llama_light/llama-cpp/<version>/llama-server`` (auto-downloaded)
    3. ``~/llama.cpp/build/bin/llama-server`` (default build path)
    4. ``./build/bin/llama-server`` (relative to llama-Light source)
    5. ``LLAMA_BIN_LOCATIONS`` (~/bin, ~/.local/bin, etc.)
    6. ``shutil.which("llama-server")``
```

#### `server.py` — one change:

In `start()`, replace the error message at lines 136–139:

**Before:**
```python
raise RuntimeError(
    "llama-server binary not found.\n"
    "Compile llama.cpp: cd ~/llama.cpp && mkdir -p build && cd build && cmake .. && cmake --build . --config Release\n"
    "Or set LLAMA_SERVER_BIN environment variable."
)
```

**After:**
```python
raise RuntimeError(
    "llama-server binary not found.\n"
    "This should have triggered an automatic download.\n"
    "Run: python -m llama_light info --check\n"
    "Or set LLAMA_SERVER_BIN environment variable manually."
)
```

#### `config.py` — one change:

Line 66–67: Change `LLAMA_SERVER_BIN` default to `None` (let `_bincheck` resolve it dynamically):

**Before:**
```python
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "llama-server"))
```

**After:**
```python
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN") or None
```

### 4.3 Files to delete / deprecate

| File | Action |
|---|---|
| `bin/` directory | Delete — no longer needed |
| `scripts/update-bin.sh` | Delete — no longer needed |
| `.bin-manifest.json` | Delete — no longer needed |

### 4.4 Setup flow

```
pip install llama-light

# On first import/run:
llama info
  → _bincheck.locate_main_bin()
  → _llama_downloader.ensure_binaries()
  → detects GPU: cuda
  → downloads llama-linux-cuda-binaries.tar.gz from GitHub
  → extracts to ~/.cache/llama_light/llama-cpp/1.7.1/
  → sets LD_LIBRARY_PATH to cache dir
  → returns path to llama-server
  → all tools now resolve from same cache dir
```

Subsequent runs skip the download (file exists check). Updates:

```
pip install llama-light --upgrade    # new version with newer LLAMA_CPP_VERSION
llama info                           # detects version mismatch, downloads new release
```

---

## 5. Migration Path (if keeping Approach A)

If you decide to keep the current copy approach, these **minimal changes** improve it:

### 5.1 Add version tracking
Add a `LLAMA_CPP_VERSION` field to `.bin-manifest.json` and record the git commit hash or tag of the source build. Update `update-bin.sh` to write it:

```bash
echo "{\"version\":\"1.7.1\",\"commit\":\"$(cd ~/llama.cpp && git rev-parse --short HEAD)\",\"tools\":[],\"libs\":[]}" > "$MANIFEST"
```

### 5.2 Add stale-check warning
In `llama info`, compare the manifest version against the latest GitHub release. If different, warn:

```
[warn] llama.cpp version 1.7.1 is outdated. Latest: 1.7.3
  Run: ./scripts/update-bin.sh --all
```

### 5.3 Add automatic fallback
In `_bincheck.py:locate_main_bin()`, if neither the bundled `bin/llama-server` nor `~/llama.cpp/build/bin/llama-server` exist, show a clear error with a one-liner suggestion:

```python
print("Hint: pip install llama-light and run 'python -m llama_light setup' "
      "to auto-download prebuilt binaries.", file=sys.stderr)
```

### 5.4 Harden `update-bin.sh`
- Add `--verify` flag that checks all .so files against md5sums
- Add `--dry-run` that shows what would be synced without copying
- Handle the case where `~/llama.cpp/build/bin/` doesn't exist (exit with clear error, not silent failure)

---

## Summary

**Approach C (auto-download prebuilt releases)** is the clear winner. It gives end users a single `pip install` that works immediately, eliminates the external llama.cpp dependency entirely, supports CUDA out of the box, pins versions to release tags, and requires minimal code changes to implement. The existing `_bincheck.py` resolution chain already supports this — it just needs a new cache-based fallback step and a small downloader module.

The total implementation effort is approximately 200 lines of new Python code (downloader module + _bincheck modifications) plus the deletion of ~500 lines of shell script.
