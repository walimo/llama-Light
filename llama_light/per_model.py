"""Per-model configuration — one JSON file per model, auto-detected settings.

Each model gets its own config file at::

    ~/.config/llama_light/models/<model_name>.json

The model name is derived from the filename (basename without extension).
When starting a model, llama merges::

    defaults  >  global config  >  model config  >  CLI args

So model configs only need to set what differs from the global default.

Model config fields (generation + server tuning):

    temperature           float      Sampling temperature
    top_k                 int        Top-K sampling
    top_p                 float      Top-P (nucleus) sampling
    min_p                 float      Min-P sampling
    max_tokens            int        Max output tokens (-1 = unlimited)
    frequency_penalty     float      Penalize repetition by token frequency
    presence_penalty      float      Penalize repetition by token presence
    reasoning             bool       Disable thinking tags
    reasoning_budget      int        Token budget for reasoning (0=off)
    ctx                     int        Context window size
    ngl                   int        GPU layers
    threads               int        CPU threads
    batch_size            int        Batch size
    flash_attn            str        "auto"/"on"/"off"
    keep                  int        Keep tokens from prompt in cache
    predict               int        Predict tokens (-1 = default)

All fields are optional — missing keys fall back to the global config,
which falls back to hardware-detected defaults.
"""

import json
import os
import re
from typing import Any, Dict

from .config import CONFIG_DIR, ensure_dirs

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(CONFIG_DIR, "models")


def _ensure_models_dirs() -> None:
    """Create the models config directory."""
    ensure_dirs()
    os.makedirs(MODELS_DIR, exist_ok=True)


# ── Model name resolution ────────────────────────────────────────────────────

def _model_name_from_path(path: str) -> str:
    """Extract a clean model name from a path or filename.

    ``/models/qwen2-7b.Q4_K_M.gguf`` → ``qwen2-7b-Q4-K-M``
    """
    base = os.path.basename(path)
    # Strip common extensions
    name = re.sub(r"\.(gguf|bin|pth)$", "", base, flags=re.IGNORECASE)
    # Normalize: underscores, spaces, dots → hyphens
    name = re.sub(r"[_\. ]", "-", name).strip("-")
    return name or "unknown"


def _model_config_path(model_name: str) -> str:
    """Return the path to a model's config JSON."""
    return os.path.join(MODELS_DIR, f"{model_name}.json")


# ── Read / Write ──────────────────────────────────────────────────────────────

def load_model_config(model_name: str) -> Dict[str, Any]:
    """Load a model's saved config, or empty dict if not found."""
    path = _model_config_path(model_name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_model_config(model_name: str, settings: Dict[str, Any]) -> None:
    """Persist a model's config to its JSON file."""
    _ensure_models_dirs()
    path = _model_config_path(model_name)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)


def update_model_config(model_name: str, key: str, value: Any) -> None:
    """Update a single key in a model's config, preserving other keys."""
    cfg = load_model_config(model_name)
    cfg[key] = value
    save_model_config(model_name, cfg)


# ── Auto-detection ────────────────────────────────────────────────────────────

# ── Family base configs ──────────────────────────────────────────────────────
# Each family inherits from "default" and overrides only what differs.
# This eliminates duplication between opus/claude/codellama.

_FAMILY_BASE = {
    # Opus / high-quality reasoning models
    "opus": {
        "temperature": 0.1, "top_k": 1, "top_p": 0.1, "min_p": 0.05,
        "frequency_penalty": 0.2, "presence_penalty": 0.1,
        "reasoning": False, "max_tokens": 8192,
    },
    # Claude-like / instruction-following
    "claude": {
        "temperature": 0.1, "top_k": 1, "top_p": 0.1, "min_p": 0.05,
        "frequency_penalty": 0.3, "presence_penalty": 0.2,
        "reasoning": False, "max_tokens": 8192,
    },
    # General-purpose / coding
    "codellama": {
        "temperature": 0.1, "top_k": 1, "top_p": 0.1, "min_p": 0.05,
        "frequency_penalty": 0.3, "presence_penalty": 0.2,
        "reasoning": False, "max_tokens": 4096,
    },
    # Qwen / general
    "qwen": {
        "temperature": 0.1, "top_k": 1, "top_p": 0.1, "min_p": 0.05,
        "frequency_penalty": 0.2, "presence_penalty": 0.1,
        "reasoning": False, "max_tokens": 8192,
    },
    # Catch-all
    "default": {
        "temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
        "max_tokens": 2048,
    },
}

# All families the auto-detector can match — must all exist in _FAMILY_BASE
_FAMILY_NAMES = ("opus", "claude", "codellama", "qwen")


def auto_detect_model_family(model_name: str) -> str:
    """Guess a model family from the model name for heuristic defaults.

    Matches substrings case-insensitively.  First match wins.
    """
    lower = model_name.lower()
    parts = lower.replace("-", "_").split("_")
    for family in _FAMILY_NAMES:
        if family in parts:
            return family
    return "default"


def auto_detect_config(model_name: str) -> Dict[str, Any]:
    """Return auto-detected settings based on model family.

    Starts from the family base, then merges the generic ``default``
    settings so that any keys the family doesn't override fall back
    to sensible generic values.
    """
    family = auto_detect_model_family(model_name)
    base = dict(_FAMILY_BASE.get(family, _FAMILY_BASE["default"]))
    return base


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get a model's effective config, merged from:

    1. Auto-detected defaults (based on model family)
    2. Saved model config (user overrides, persisted)

    User's saved settings always win — they override auto-detection.
    """
    auto = auto_detect_config(model_name)
    saved = load_model_config(model_name)
    auto.update(saved)
    return auto
