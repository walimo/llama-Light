# llama_light/config.py
"""Central configuration module — hardware detection, persistent JSON config,
type-safe key validation, and dynamic server binary resolution.

All paths live under ``~/.config/llama_light`` and ``~/.cache/llama_light``.
Config files are auto-created on first run with hardware-aware defaults.
"""

import json
import os
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

# ── Known config keys with their expected types ──────────────────────────────

# Keys that must be int when set
_INT_KEYS = {
    "ctx", "batch_size", "ubatch_size", "parallel", "ngl", "threads",
    "threads_batch", "max_tokens", "top_k", "predict", "keep", "yarn_orig_ctx",
    "n_cpu_moe", "reasoning_budget",
}
# Keys that must be float when set
_FLOAT_KEYS = {"min_p", "temperature", "top_p", "frequency_penalty",
               "presence_penalty", "rope_freq_base", "rope_scale", "rope_freq_scale",
               "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_slow", "yarn_beta_fast"}
# Keys that must be bool when set
_BOOL_KEYS = {"mlock", "mmap", "direct_io", "no_host", "kv_offload",
              "repack", "swa_full", "perf", "escape", "cpu_moe", "tool_calling",
              "reasoning"}
# Keys where "none" is a valid string (not a null sentinel)
_STRING_NONE_KEYS = {
    "reasoning_format", "flash_attn", "split_mode", "numa",
    "rope_scaling", "cache_type_k", "cache_type_v",
}
# All valid config keys
_VALID_KEYS = set(_INT_KEYS) | set(_FLOAT_KEYS) | set(_BOOL_KEYS) | _STRING_NONE_KEYS | {
    "host", "port", "default_model", "last_model", "device", "numa",
    "override_tensor", "ui_mcp_proxy", "tools",
    "reasoning", "active_profile",
}


# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_ROOT       = os.path.expanduser("~/.cache/llama_light/models")
LOG_DIR          = os.path.expanduser("~/.cache/llama_light/logs")
PID_FILE         = os.path.expanduser("~/.cache/llama_light/server.pid")
STATE_FILE       = os.path.expanduser("~/.cache/llama_light/state.json")
REGISTRY_FILE    = os.path.expanduser("~/.config/llama_light/registry.json")
HF_CACHE_DIR     = os.path.expanduser("~/.cache/huggingface/hub")

# Directory-based config layout
CONFIG_DIR       = os.path.expanduser("~/.config/llama_light")
GLOBAL_CONFIG    = os.path.join(CONFIG_DIR, "config.json")

# ── Env overrides ─────────────────────────────────────────────────────────────
DEFAULT_HOST       = "127.0.0.1"
DEFAULT_PORT       = 8080
DEFAULT_CTX        = 200000
DEFAULT_GPU_LAYERS = 99
_LLAMA_HOST_val = os.environ.get("LLAMA_HOST")
LLAMA_HOST   = _LLAMA_HOST_val if _LLAMA_HOST_val else DEFAULT_HOST
_LLAMA_PORT_str = os.environ.get("LLAMA_PORT")
LLAMA_PORT   = int(_LLAMA_PORT_str) if _LLAMA_PORT_str and _LLAMA_PORT_str.isdigit() else DEFAULT_PORT
LLAMA_MODELS    = os.environ.get("LLAMA_MODELS", CACHE_ROOT)
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN") or None


def ensure_dirs() -> None:
    """Create all runtime directories."""
    for d in (
        CACHE_ROOT,
        LOG_DIR,
        os.path.dirname(PID_FILE),
        os.path.dirname(STATE_FILE),
        CONFIG_DIR,
        os.path.dirname(REGISTRY_FILE),
    ):
        os.makedirs(d, exist_ok=True)


# ── Hardware detection ────────────────────────────────────────────────────────

def detect_gpu() -> str:
    """Detect which GPU backend is available: cuda, metal, rocm, or cpu."""
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return "cuda"
        except Exception:
            pass
    if platform.system() == "Darwin" and shutil.which("sysctl"):
        try:
            r = subprocess.run(["sysctl", "hw.optional.metal"],
                               capture_output=True, text=True, timeout=5)
            if "1" in r.stdout:
                return "metal"
        except Exception:
            pass
    if shutil.which("rocminfo"):
        return "rocm"
    return "cpu"


def get_gpu_vram_gb() -> Optional[float]:
    """Query NVIDIA GPU VRAM via nvidia-smi, or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return round(int(r.stdout.strip().split("\n")[0]) / 1024, 2)
    except Exception:
        pass
    return None



def get_defaults(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Hardware-aware defaults — dynamically sized for your GPU/CPU."""
    gpu       = detect_gpu()
    cpu_cores = os.cpu_count() or 4

    # Threads — don't oversubscribe; 1 thread per physical core, cap at 16
    if cpu_cores <= 4:
        threads = cpu_cores
    else:
        threads = min(cpu_cores, 16)

    return {
        # Server
        "host":             "127.0.0.1",
        "port":             8080,
        # Model
        "default_model":    None,
        "last_model":       None,
        # Context / batching
        "ctx":              200000,
        "batch_size":       512,
        "ubatch_size":      512,
        "parallel":         1,
        # GPU
        "ngl":              99 if gpu != "cpu" else 0,
        "split_mode":       "layer",
        "device":           None,
        "kv_offload":       True,
        "repack":           True,
        # Attention / KV
        "flash_attn":       "on",
        "cache_type_k":     "q8_0",
        "cache_type_v":     "q8_0",
        # Threading
        "threads":          threads,
        "threads_batch":    threads,
        # Generation
        "temperature":      0.7,
        "top_k":            40,
        "top_p":            0.9,
        "min_p":            0.05,
        "max_tokens":       16384,
        "predict":          -1,
        "keep":             0,
        # Penalty
        "frequency_penalty": 0.1,
        "presence_penalty":  0.5,
        # Memory
        "mlock":            False,
        "mmap":             True,
        "direct_io":        False,
        "no_host":          False,
        # RoPE
        "rope_scaling":     None,
        "rope_freq_base":   None,
        "rope_scale":       None,
        "rope_freq_scale":  None,
        # YaRN
        "yarn_orig_ctx":    0,
        "yarn_ext_factor":  -1.0,
        "yarn_attn_factor": -1.0,
        "yarn_beta_slow":   -1.0,
        "yarn_beta_fast":   -1.0,
        # MoE
        "cpu_moe":          False,
        "n_cpu_moe":        None,
        # NUMA
        "numa":             None,
        # Misc flags
        "override_tensor":  None,
        "swa_full":         False,
        "perf":             False,
        "escape":           True,
        "ui_mcp_proxy":     "on",
        "tools":            "all",
        # Reasoning
        "reasoning":        False,
        "reasoning_format": "deepseek",
        "reasoning_budget": 256,
        # Profile
        "active_profile":   "default",
        # Tool calling
        "tool_calling":     True,
    }


# ── Config class ──────────────────────────────────────────────────────────────

class Config:
    """Persistent, type-safe JSON configuration singleton.

    Keys are validated against ``_VALID_KEYS`` on write.
    Strings are auto-coerced to int/float/bool.
    """

    def __init__(self):
        ensure_dirs()
        self._data: Dict[str, Any] = get_defaults()
        self._load()

    def _load(self) -> None:
        if os.path.exists(GLOBAL_CONFIG):
            try:
                with open(GLOBAL_CONFIG) as f:
                    self._data.update(json.load(f))
            except Exception:
                pass

    def save(self) -> None:
        ensure_dirs()
        fd, tmp_path = tempfile.mkstemp(dir=CONFIG_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, GLOBAL_CONFIG)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set a config key with type validation and coercion.

        Raises
        ------
        ValueError
            If *key* is not a known configuration key.
        """
        # Validate key
        if key not in _VALID_KEYS:
            known = sorted(_VALID_KEYS)
            raise ValueError(
                f"Unknown config key '{key}'. "
                f"Valid keys: {known[:10]}... ({len(known)} total)"
            )

        # Coerce from string — only for typed keys
        if isinstance(value, str):
            if key in _BOOL_KEYS and value.lower() in (
                "true", "false", "yes", "no", "1", "0", "on", "off",
            ):
                value = value.lower() in ("true", "yes", "1", "on")
            elif key in _INT_KEYS:
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f"Key '{key}' requires integer, got '{value}'")
            elif key in _FLOAT_KEYS:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Key '{key}' requires float, got '{value}'")
            elif key not in _STRING_NONE_KEYS and value.lower() in ("null", "none"):
                value = None
            elif key in _STRING_NONE_KEYS and value.lower() == "none":
                value = None
            elif key in _BOOL_KEYS:
                raise ValueError(
                    f"Key '{key}' requires boolean value "
                    f"(true/false/yes/no/1/0/on/off), got '{value}'"
                )

        self._data[key] = value
        if persist:
            self.save()

    def all(self) -> Dict[str, Any]:
        """Return a shallow copy of all config keys."""
        return dict(self._data)

    def reset(self) -> None:
        """Reset to hardware-detected defaults and persist."""
        self._data = get_defaults()
        self.save()

    # Convenience properties
    @property
    def host(self)          -> str:           return self.get("host") or DEFAULT_HOST
    @property
    def port(self)          -> int:           return int(self.get("port") or DEFAULT_PORT)
    @property
    def ctx(self)           -> int:           return int(self.get("ctx") or DEFAULT_CTX)
    @property
    def ngl(self)           -> int:
        v = self.get("ngl")
        return int(v) if v not in (None, 0, "") else DEFAULT_GPU_LAYERS
    @property
    def flash_attn(self)    -> str:           return str(self.get("flash_attn") or "on")
    @property
    def batch_size(self)    -> int:           return int(self.get("batch_size") or 512)
    @property
    def ubatch_size(self)   -> int:           return int(self.get("ubatch_size") or 512)
    @property
    def parallel(self)      -> int:           return int(self.get("parallel") or 1)
    @property
    def threads(self)       -> int:           return int(self.get("threads") or 0)
    @property
    def default_model(self) -> Optional[str]: return self.get("default_model")
    @property
    def last_model(self)    -> Optional[str]: return self.get("last_model")


# Singleton
_cfg: Optional[Config] = None

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg