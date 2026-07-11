# llama_light/config.py
"""Central configuration module — hardware detection, persistent JSON config,
type-safe key validation, and dynamic server binary resolution.

All paths live under ``~/.config/llama_light`` and ``~/.cache/llama_light``.
Config files are auto-created on first run with hardware-aware defaults.

This module maps 1:1 to llama.cpp llama-server flags. Every flag that makes
sense for a general-purpose server is exposed here so users can tune everything
via ``llama config set <key> <value>``.
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
    "threads_batch", "max_tokens", "top_k", "predict", "keep",
    "yarn_orig_ctx",
    "n_cpu_moe",
    "reasoning_budget",
    "mirostat", "mirostat_lr", "mirostat_ent",
    "top_n_sigma", "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "adaptive_target", "adaptive_decay",
    "dynatemp_range", "dynatemp_exp",
    "spec_draft_n_max", "spec_draft_n_min",
    "spec_draft_p_split", "spec_draft_p_min",
    "poll", "poll_batch", "prio", "prio_batch",
    "threads_http", "sse_ping_interval",
    "cache_reuse", "fit_ctx",
    "image_min_tokens", "image_max_tokens",
    "embd_normalize",
    "log_verbosity",
    "port",
}

# Keys that must be float when set
_FLOAT_KEYS = {
    "min_p", "temperature", "top_p", "frequency_penalty",
    "presence_penalty", "rope_freq_base", "rope_scale", "rope_freq_scale",
    "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_slow", "yarn_beta_fast",
    "typical_p",
}

# Keys that must be bool when set
_BOOL_KEYS = {
    "mlock", "mmap", "direct_io", "no_host", "kv_offload",
    "repack", "swa_full", "perf", "escape", "cpu_moe", "tool_calling",
    "reasoning",
    "cache_prompt", "slots", "metrics", "props",
    "offline", "log_prefix", "log_timestamps",
    "skip_chat_parsing", "prefill_assistant",
    "models_autoload",
    "warmup", "spm_infill", "mmproj_auto", "mmproj_offload",
    "cache_idle_slots", "context_shift",
    "cpu_strict", "cpu_strict_batch",
}

# Keys where "none" is a valid string (not a null sentinel)
_STRING_NONE_KEYS = {
    "reasoning_format", "flash_attn", "split_mode", "numa",
    "rope_scaling", "cache_type_k", "cache_type_v",
    "log_colors", "spec_type", "pooling",
    # String config keys not covered above
    "host", "default_model", "last_model", "device",
    "override_tensor", "ui_mcp_proxy", "tools",
}

# All valid config keys — the single source of truth for `llama config set`
_VALID_KEYS = set(_INT_KEYS) | set(_FLOAT_KEYS) | set(_BOOL_KEYS) | _STRING_NONE_KEYS | {
    "host", "port", "default_model", "last_model", "device",
    "override_tensor", "ui_mcp_proxy", "tools",
    "reasoning_budget_message",
    "chat_template_kwargs",
    "samplers", "sampler_seq",
    "api_key", "api_key_file", "ssl_key_file", "ssl_cert_file",
    "ui_config", "ui", "path", "api_prefix",
    "log_disable", "log_file",
    "seed", "ignore_eos",
    "grammar", "grammar_file", "json_schema", "json_schema_file",
    "lora", "lora_scaled", "control_vector", "control_vector_scaled",
    "control_vector_layer_range",
    "embedding", "rerank",
    "slot_save_path", "media_path",
    "tags",
    "active_profile",
}

# ── Section headings and key descriptions ────────────────────────────────────
# Format: { section_header: [(key, description), ...], ... }

_CONFIG_SECTIONS: Dict[str, list] = {
    "── SERVER ─────────────────────────────────────": [
        ("host", "IP address the server listens on"),
        ("port", "Port number for the API endpoint"),
        ("default_model", "Path to the default GGUF model file"),
        ("last_model", "Last model loaded (auto-set)"),
        ("device", "GPU device(s) to use, comma-separated"),
        ("override_tensor", "Override tensor buffer type pattern"),
        ("ui_mcp_proxy", "Enable MCP CORS proxy for the WebUI"),
        ("tools", "Built-in tools for agents: 'all', 'read_file', etc."),
    ],
    "── CONTEXT / BATCHING ─────────────────────────": [
        ("ctx", "Context window size in tokens"),
        ("batch_size", "Logical maximum batch size for prompt processing"),
        ("ubatch_size", "Physical maximum batch size (usually 512)"),
        ("parallel", "Number of server slots / concurrent requests"),
    ],
    "── GPU ────────────────────────────────────────": [
        ("ngl", "Number of layers to offload to GPU (99 = all)"),
        ("split_mode", "How to split across GPUs: layer, row, tensor, none"),
        ("kv_offload", "Offload KV cache to CPU if GPU memory is full"),
        ("repack", "Repack weights for faster GPU inference"),
    ],
    "── ATTENTION / KV ─────────────────────────────": [
        ("flash_attn", "Use FlashAttention: on, off, auto"),
        ("cache_type_k", "KV cache type for K tensor: f32, f16, q8_0, etc."),
        ("cache_type_v", "KV cache type for V tensor: f32, f16, q8_0, etc."),
    ],
    "── THREADING / CPU ───────────────────────────": [
        ("threads", "Number of CPU threads for generation"),
        ("threads_batch", "Number of CPU threads for batch/prompt processing"),
        ("poll", "Polling level (0 = none, 100 = aggressive)"),
        ("poll_batch", "Polling level for batch processing"),
        ("prio", "Process priority: -1=low, 0=normal, 1=medium, 2=high, 3=realtime"),
        ("prio_batch", "Priority for batch processing threads"),
        ("cpu_strict", "Use strict CPU placement (avoid oversubscription)"),
        ("cpu_strict_batch", "Strict placement for batch threads"),
    ],
    "── GENERATION ─────────────────────────────────": [
        ("max_tokens", "Maximum tokens to generate (-1 = unlimited)"),
        ("predict", "Alias for max_tokens, same effect"),
        ("keep", "Keep N tokens from the initial prompt in context"),
    ],
    "── SAMPLING ───────────────────────────────────": [
        ("temperature", "How creative the output is (0=deterministic, 1=random)"),
        ("top_k", "Limit sampling to top K tokens (0=disabled)"),
        ("top_p", "Nucleus sampling: only consider tokens with cumulative prob >= top_p"),
        ("min_p", "Minimum probability threshold for sampling"),
        ("typical_p", "Locally typical sampling parameter p"),
        ("top_n_sigma", "Top-n-sigma sampling (-1=disabled)"),
        ("xtc_probability", "XTC sampling probability (0=disabled)"),
        ("xtc_threshold", "XTC sampling threshold"),
        ("repeat_penalty", "Penalize repeated sequences of tokens"),
        ("repeat_last_n", "How many recent tokens to check for repetition (0=disabled, -1=all)"),
        ("presence_penalty", "Encourage the model to talk about new topics"),
        ("frequency_penalty", "Discourage the model from repeating itself"),
    ],
    "── DRY SAMPLING ───────────────────────────────": [
        ("dry_multiplier", "DRY multiplier (0=disabled)"),
        ("dry_base", "DRY base value"),
        ("dry_allowed_length", "Allowed length for DRY sampling"),
        ("dry_penalty_last_n", "DRY penalty for last N tokens (-1=all)"),
        ("dry_sequence_breaker", "Sequence breaker chars for DRY (use 'none' to disable)"),
    ],
    "── ADAPTIVE / DYNAMIC ────────────────────────": [
        ("adaptive_target", "Select tokens near this probability (0=disabled)"),
        ("adaptive_decay", "Decay rate for target adaptation (lower=reactive, higher=stable)"),
        ("dynatemp_range", "Dynamic temperature range (0=disabled)"),
        ("dynatemp_exp", "Dynamic temperature exponent"),
    ],
    "── MIROSTAT ──────────────────────────────────": [
        ("mirostat", "Mirostat algorithm version (0=disabled, 1=Mirostat, 2=Mirostat 2.0)"),
        ("mirostat_lr", "Mirostat learning rate (eta)"),
        ("mirostat_ent", "Mirostat target entropy (tau)"),
    ],
    "── MEMORY / KV ───────────────────────────────": [
        ("mlock", "Force model into RAM (prevent swapping)"),
        ("mmap", "Memory-map the model file"),
        ("direct_io", "Use DirectIO if available"),
        ("no_host", "Bypass host buffer (saves memory)"),
        ("cache_prompt", "Enable prompt caching for faster repeated prompts"),
        ("cache_reuse", "Min chunk size to reuse from cache via KV shifting (0=disabled)"),
        ("cache_idle_slots", "Save idle slots to prompt cache"),
        ("context_shift", "Use context shift on infinite generation"),
        ("slots", "Expose slot monitoring endpoint"),
        ("slot_save_path", "Path to save slot KV cache"),
    ],
    "── ROPE ──────────────────────────────────────": [
        ("rope_scaling", "RoPE frequency scaling: none, linear, yarn"),
        ("rope_scale", "RoPE context scaling factor"),
        ("rope_freq_base", "RoPE base frequency for NTK-aware scaling"),
        ("rope_freq_scale", "RoPE frequency scaling factor (1/N expands context)"),
    ],
    "── YARN ──────────────────────────────────────": [
        ("yarn_orig_ctx", "YaRN: original context size of model"),
        ("yarn_ext_factor", "YaRN: extrapolation mix factor (-1=auto)"),
        ("yarn_attn_factor", "YaRN: scale sqrt(t) or attention magnitude (-1=auto)"),
        ("yarn_beta_slow", "YaRN: high correction dim or alpha (-1=auto)"),
        ("yarn_beta_fast", "YaRN: low correction dim or beta (-1=auto)"),
    ],
    "── MOE ───────────────────────────────────────": [
        ("cpu_moe", "Keep all MoE weights in CPU memory"),
        ("n_cpu_moe", "Keep first N MoE layers in CPU memory"),
    ],
    "── NUMA ──────────────────────────────────────": [
        ("numa", "NUMA optimization: distribute, isolate, numactl"),
    ],
    "── REASONING / THINKING ──────────────────────": [
        ("reasoning", "Enable reasoning/thinking output"),
        ("reasoning_format", "How to format reasoning: deepseek, deepseek-legacy, none"),
        ("reasoning_budget", "Token budget for thinking (-1=unlimited, 0=immediate end)"),
        ("reasoning_budget_message", "Message injected when reasoning budget is exhausted"),
        ("chat_template_kwargs", "Additional JSON params for the template parser"),
        ("skip_chat_parsing", "Force pure content parser (include reasoning/tool calls in content)"),
        ("prefill_assistant", "Prefill assistant response if last message is from assistant"),
    ],
    "── SPECULATIVE DECODING ──────────────────────": [
        ("spec_type", "Speculative decoding type: none, draft-simple, ngram-mod, etc."),
        ("spec_draft_n_max", "Max tokens to draft for speculative decoding"),
        ("spec_draft_n_min", "Min draft tokens for speculative decoding"),
        ("spec_draft_p_split", "Speculative decoding split probability"),
        ("spec_draft_p_min", "Minimum speculative decoding probability (greedy)"),
        ("models_autoload", "Automatically load models in router mode"),
    ],
    "── SERVER FEATURES ───────────────────────────": [
        ("ui", "Enable the WebUI"),
        ("embedding", "Restrict to embedding-only mode"),
        ("rerank", "Enable reranking endpoint"),
        ("metrics", "Enable Prometheus-compatible metrics endpoint"),
        ("props", "Enable changing global properties via POST /props"),
        ("sse_ping_interval", "Server SSE ping interval in seconds (-1=disabled)"),
        ("threads_http", "Threads for HTTP request processing (-1=auto)"),
        ("fit", "Adjust unset args to fit device memory"),
        ("fit_target", "Target margin per device in MiB"),
        ("fit_ctx", "Minimum ctx size for --fit option"),
    ],
    "── UI / WEB ──────────────────────────────────": [
        ("ui_config", "JSON string for default WebUI settings"),
        ("path", "Directory to serve static files from"),
        ("api_prefix", "Prefix path the API serves from"),
    ],
    "── LOGGING ───────────────────────────────────": [
        ("log_disable", "Disable all logging"),
        ("log_file", "File path to write logs to"),
        ("log_colors", "Colored logging: on, off, auto"),
        ("offline", "Force offline mode (use cache only)"),
        ("log_prefix", "Enable prefix in log messages"),
        ("log_timestamps", "Enable timestamps in log messages"),
        ("log_verbosity", "Verbosity threshold: 0=generic, 1=error, 2=warning, 3=info, 4=trace, 5=debug"),
    ],
    "── AUTH / SSL ────────────────────────────────": [
        ("api_key", "API key for authentication"),
        ("api_key_file", "File containing API keys"),
        ("ssl_key_file", "Path to PEM-encoded SSL private key"),
        ("ssl_cert_file", "Path to PEM-encoded SSL certificate"),
    ],
    "── SAMPLING ADVANCED ────────────────────────": [
        ("samplers", "Comma-separated list of samplers in order (e.g. penalties;top_k;top_p)"),
        ("sampler_seq", "Simplified sequence for samplers (e.g. edskypmxt)"),
    ],
    "── GRAMMAR / CONSTRAINTS ─────────────────────": [
        ("grammar", "BNF-like grammar string to constrain generation"),
        ("grammar_file", "Path to a grammar file"),
        ("json_schema", "JSON schema string to constrain generation"),
        ("json_schema_file", "Path to a JSON schema file"),
    ],
    "── MODEL / LORA ─────────────────────────────": [
        ("lora", "Path to LoRA adapter file (comma-sep for multiple)"),
        ("lora_scaled", "LoRA adapter with scaling: path:scale"),
        ("control_vector", "Path to a control vector file"),
        ("control_vector_scaled", "Control vector with scaling: path:scale"),
        ("control_vector_layer_range", "Layer range for control vectors: START END"),
    ],
    "── MEDIA / MULTIMODAL ───────────────────────": [
        ("media_path", "Directory for loading local media files"),
        ("mmproj_auto", "Auto-download multimodal projector file"),
        ("mmproj_offload", "Offload multimodal projector to GPU"),
        ("image_min_tokens", "Minimum tokens each image can take"),
        ("image_max_tokens", "Maximum tokens each image can take"),
    ],
    "── POOLING / EMBEDDINGS ─────────────────────": [
        ("pooling", "Pooling type: none, mean, cls, last, rank"),
        ("embd_normalize", "Embedding normalization: -1=none, 0=max int16, 1=taxicab, 2=euclidean"),
    ],
    "── IDENTITY ──────────────────────────────────": [
        ("tags", "Comma-separated model tags (informational)"),
    ],
    "── PROFILE ───────────────────────────────────": [
        ("active_profile", "Active configuration profile name"),
    ],
    "── TOOL CALLING ──────────────────────────────": [
        ("tool_calling", "Enable automatic tool calling in chat responses"),
    ],
    "── MISCELLANEOUS ─────────────────────────────": [
        ("seed", "RNG seed (-1=random)"),
        ("ignore_eos", "Ignore end-of-sequence token and keep generating"),
        ("swa_full", "Use full-size SWA (sliding window attention) cache"),
        ("perf", "Enable internal libllama performance timings"),
        ("escape", "Process escape sequences like \\n, \\r, \\t"),
    ],
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

    threads = min(cpu_cores, 8) if cpu_cores <= 4 else (min(cpu_cores, 16) if cpu_cores <= 8 else min(cpu_cores, 32))

    return {
        "host":             "127.0.0.1",
        "port":             8080,
        "default_model":    None,
        "last_model":       None,
        "ctx":              200000,
        "batch_size":       512,
        "ubatch_size":      512,
        "parallel":         1,
        "ngl":              99 if gpu != "cpu" else 0,
        "split_mode":       "layer",
        "device":           None,
        "override_tensor":  None,
        "kv_offload":       True,
        "repack":           True,
        "flash_attn":       "on",
        "cache_type_k":     "q8_0",
        "cache_type_v":     "q8_0",
        "threads":          threads,
        "threads_batch":    threads,
        "poll":             0,
        "poll_batch":       0,
        "prio":             2,
        "prio_batch":       0,
        "cpu_strict":       False,
        "cpu_strict_batch": False,
        "temperature":      0.7,
        "top_k":            40,
        "top_p":            0.9,
        "min_p":            0.05,
        "max_tokens":       16384,
        "predict":          -1,
        "keep":             0,
        "typical_p":        1.0,
        "top_n_sigma":      -1.0,
        "xtc_probability":  0.0,
        "xtc_threshold":    0.1,
        "repeat_last_n":    64,
        "repeat_penalty":   1.0,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.1,
        "dry_multiplier":   0.0,
        "dry_base":         1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": -1,
        "dry_sequence_breaker": None,
        "adaptive_target":  -1.0,
        "adaptive_decay":   0.9,
        "dynatemp_range":   0.0,
        "dynatemp_exp":     1.0,
        "mirostat":         0,
        "mirostat_lr":      0.1,
        "mirostat_ent":     5.0,
        "mlock":            False,
        "mmap":             True,
        "direct_io":        False,
        "no_host":          False,
        "cache_prompt":     True,
        "cache_reuse":      1,
        "cache_idle_slots": True,
        "context_shift":    False,
        "slots":            True,
        "slot_save_path":   None,
        "metrics":          False,
        "props":            False,
        "sse_ping_interval": 30,
        "threads_http":     -1,
        "fit":              True,
        "fit_target":       1024,
        "fit_ctx":          4096,
        "rope_scaling":     None,
        "rope_freq_base":   None,
        "rope_scale":       None,
        "rope_freq_scale":  None,
        "yarn_orig_ctx":    0,
        "yarn_ext_factor":  -1.0,
        "yarn_attn_factor": -1.0,
        "yarn_beta_slow":   -1.0,
        "yarn_beta_fast":   -1.0,
        "cpu_moe":          False,
        "n_cpu_moe":        None,
        "numa":             None,
        "swa_full":         False,
        "perf":             False,
        "escape":           True,
        "ui_mcp_proxy":     "on",
        "tools":            "all",
        "reasoning":        False,
        "reasoning_format": "deepseek",
        "reasoning_budget": 256,
        "reasoning_budget_message": None,
        "chat_template_kwargs": None,
        "skip_chat_parsing":  False,
        "prefill_assistant":  True,
        "spec_type":          None,
        "spec_draft_n_max":   3,
        "spec_draft_n_min":   0,
        "spec_draft_p_split": 0.1,
        "spec_draft_p_min":   0.0,
        "models_autoload":    True,
        "ui":               True,
        "embedding":        False,
        "rerank":           False,
        "log_disable":      False,
        "log_file":         None,
        "log_colors":       "auto",
        "offline":          False,
        "log_prefix":       False,
        "log_timestamps":   False,
        "log_verbosity":    3,
        "api_key":          None,
        "api_key_file":     None,
        "ssl_key_file":     None,
        "ssl_cert_file":    None,
        "seed":             -1,
        "ignore_eos":       False,
        "samplers":         None,
        "sampler_seq":      None,
        "grammar":          None,
        "grammar_file":     None,
        "json_schema":      None,
        "json_schema_file": None,
        "lora":             None,
        "lora_scaled":      None,
        "control_vector":   None,
        "control_vector_scaled": None,
        "control_vector_layer_range": None,
        "media_path":       None,
        "mmproj_auto":      True,
        "mmproj_offload":   True,
        "image_min_tokens": None,
        "image_max_tokens": None,
        "pooling":          None,
        "embd_normalize":   2,
        "tags":             None,
        "active_profile":   "default",
        "tool_calling":     True,
    }


class Config:
    """Persistent, type-safe JSON configuration singleton."""

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
        """Set a config key with type validation and coercion."""
        if key not in _VALID_KEYS:
            known = sorted(_VALID_KEYS)
            raise ValueError(
                f"Unknown config key '{key}'. "
                f"Valid keys: {known[:10]}... ({len(known)} total)"
            )

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


_cfg: Optional[Config] = None

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
