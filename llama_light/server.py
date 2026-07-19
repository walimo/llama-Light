import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, Optional

from .config import (
    DEFAULT_CTX, DEFAULT_GPU_LAYERS, DEFAULT_HOST, DEFAULT_PORT,
    LLAMA_HOST, LLAMA_PORT, LOG_DIR, PID_FILE,
    STATE_FILE, ensure_dirs,
)
from ._bincheck import locate_main_bin


def _find_mmproj(model_path: str) -> Optional[str]:
    """Look for a co-located mmproj-*.gguf file next to the model."""
    model_dir = os.path.dirname(model_path)
    if not os.path.isdir(model_dir):
        return None
    candidates = [
        f for f in os.listdir(model_dir)
        if f.lower().startswith("mmproj") and f.lower().endswith(".gguf")
    ]
    if not candidates:
        return None

    def _rank(name: str) -> int:
        n = name.lower()
        if "f16" in n and "bf16" not in n:
            return 0
        if "bf16" in n:
            return 1
        if "f32" in n:
            return 2
        return 3

    candidates.sort(key=_rank)
    return os.path.join(model_dir, candidates[0])


# ── State ─────────────────────────────────────────────────────────────────────

def _read_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _write_state(state: Dict[str, Any]) -> None:
    ensure_dirs()
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def _clear_state() -> None:
    for p in (STATE_FILE, PID_FILE):
        if os.path.exists(p):
            os.remove(p)

def _pid() -> Optional[int]:
    return _read_state().get("pid")

def _is_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False

def _base_url() -> str:
    state = _read_state()
    return f"http://{state.get('host', LLAMA_HOST)}:{state.get('port', LLAMA_PORT)}"


# ── Health ────────────────────────────────────────────────────────────────────

def _health(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check single /health endpoint (2s timeout)."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _is_healthy(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Return True only if both /health and /props return successfully (8s each)."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=8) as r:
            if r.status != 200:
                return False
        with urllib.request.urlopen(f"http://{host}:{port}/props", timeout=8) as r:
            if r.status != 200:
                return False
        return True
    except Exception:
        return False


# ── Port helpers ──────────────────────────────────────────────────────────────

def _detect_port_in_use(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> bool:
    """Check whether *port* is currently bound and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


# ── systemd helper ────────────────────────────────────────────────────────────

def _systemd_active() -> bool:
    """True if llama-server.service is currently active under systemd --user."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "llama-server.service"],
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def _systemd_unit_exists() -> bool:
    return os.path.exists(_service_path())


# ── Start ─────────────────────────────────────────────────────────────────────

def start(
    model_path: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    flash_attn: Optional[bool] = None,
    extra_args: Optional[list] = None,
) -> int:
    """Launch llama-server, reading defaults from config.json."""
    ensure_dirs()

    bin_path = locate_main_bin()
    if not bin_path:
        raise RuntimeError(
            "llama-server binary not found.\n"
            "This should have triggered an automatic download.\n"
            "Run: python -m llama_light info --check\n"
            "Or set LLAMA_SERVER_BIN environment variable manually."
        )
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    from .config import get_config
    cfg = get_config()
    host = host if host is not None else cfg.host
    port = port if port is not None else cfg.port
    if flash_attn is None:
        flash_attn = str(cfg.flash_attn).lower() not in ("off", "false", "0")

    pid = _pid()
    if _is_running(pid):
        state = _read_state()
        raise RuntimeError(
            f"Server already running (pid {pid}) with "
            f"{state.get('model_filename','?')} on port {state.get('port', port)}.\n"
            "Run 'llama stop' first."
        )

    if _detect_port_in_use(host=host, port=port):
        raise RuntimeError(
            f"Port {port} is already in use. "
            "Run 'llama stop' first, or specify a different port with --port."
        )

    log_file = os.path.join(LOG_DIR, "server.log")

    from .per_model import get_model_config, _model_name_from_path
    model_name = _model_name_from_path(model_path)
    model_cfg = get_model_config(model_name)

    m_ctx      = model_cfg.get("ctx",    cfg.get("ctx"))
    m_ngl      = model_cfg.get("ngl",    cfg.get("ngl"))
    m_threads  = model_cfg.get("threads", cfg.get("threads"))
    m_flash    = model_cfg.get("flash_attn", cfg.flash_attn)
    m_keep     = model_cfg.get("keep",     cfg.get("keep"))
    m_predict  = model_cfg.get("predict",  cfg.get("predict"))
    m_batch    = model_cfg.get("batch_size", cfg.batch_size)
    m_temp     = model_cfg.get("temperature", cfg.get("temperature"))
    m_top_p    = model_cfg.get("top_p", cfg.get("top_p"))
    m_min_p    = model_cfg.get("min_p", cfg.get("min_p"))
    m_repeat_p = model_cfg.get("repeat_penalty", cfg.get("repeat_penalty"))
    m_repeat_n = model_cfg.get("repeat_last_n", cfg.get("repeat_last_n"))
    m_pres_p   = model_cfg.get("presence_penalty", cfg.get("presence_penalty"))
    m_freq_p   = model_cfg.get("frequency_penalty", cfg.get("frequency_penalty"))
    m_top_k    = model_cfg.get("top_k", cfg.get("top_k"))

    args = [
        bin_path,
        "-m",            model_path,
        "--host",        host,
        "--port",        str(port),
        "-c",            str(m_ctx),
        "-ngl",          str(m_ngl),
        "--parallel",    str(cfg.parallel),
        "--flash-attn",  "on" if str(m_flash).lower() not in ("off", "false", "0") else "off",
        "--tools",       str(cfg.get("tools")),
        "-b",            str(m_batch),
        "--ubatch-size", str(cfg.ubatch_size),
        "--threads",     str(m_threads),
        "--threads-batch", str(cfg.get("threads_batch")),
        "--cache-type-k", str(cfg.get("cache_type_k")),
        "--cache-type-v", str(cfg.get("cache_type_v")),
    ]

    if not cfg.get("kv_offload"):
        args.append("--no-kv-offload")

    if not cfg.get("mmap"):
        args.append("--no-mmap")
    if cfg.get("mlock"):
        args.append("--mlock")

    if cfg.get("split_mode"):
        args += ["--split-mode", str(cfg.get("split_mode"))]

    if cfg.get("device"):
        args += ["--device", str(cfg.get("device"))]

    if cfg.get("numa"):
        args += ["--numa", str(cfg.get("numa"))]

    if cfg.get("rope_scaling"):
        args += ["--rope-scaling", str(cfg.get("rope_scaling"))]
    if cfg.get("rope_freq_base"):
        args += ["--rope-freq-base", str(cfg.get("rope_freq_base"))]
    if cfg.get("rope_scale"):
        args += ["--rope-scale", str(cfg.get("rope_scale"))]
    if cfg.get("rope_freq_scale"):
        args += ["--rope-freq-scale", str(cfg.get("rope_freq_scale"))]

    if cfg.get("yarn_orig_ctx", 0):
        args += ["--yarn-orig-ctx", str(cfg.get("yarn_orig_ctx"))]
    for key, flag in [
        ("yarn_ext_factor",  "--yarn-ext-factor"),
        ("yarn_attn_factor", "--yarn-attn-factor"),
        ("yarn_beta_slow",   "--yarn-beta-slow"),
        ("yarn_beta_fast",   "--yarn-beta-fast"),
    ]:
        val = cfg.get(key, -1.0)
        if val != -1.0:
            args += [flag, str(val)]

    if cfg.get("cpu_moe"):
        args.append("--cpu-moe")
    if cfg.get("n_cpu_moe"):
        args += ["--n-cpu-moe", str(cfg.get("n_cpu_moe"))]

    # reasoning config
    m_reasoning    = model_cfg.get("reasoning", cfg.get("reasoning"))
    reasoning_on   = m_reasoning is not False and str(m_reasoning).lower() not in ("false", "off", "0")
    m_reason_budget = model_cfg.get("reasoning_budget", cfg.get("reasoning_budget"))
    if not reasoning_on:
        args += ["--reasoning", "off"]
    elif reasoning_on:
        rf = model_cfg.get("reasoning_format", cfg.get("reasoning_format"))
        if rf and str(rf).lower() not in ("none", "null", ""):
            args += ["--reasoning-format", str(rf)]
        if m_reason_budget:
            args += ["--reasoning-budget", str(m_reason_budget)]
        else:
            args += ["--reasoning-budget", "0"]

    if cfg.get("swa_full"):
        args.append("--swa-full")
    if cfg.get("swa_decay") is not None and float(cfg.get("swa_decay")) != 0.9:
        args += ["--swa-decay", str(cfg.get("swa_decay"))]
    if cfg.get("swa_ctx"):
        args += ["--swa-ctx", str(cfg.get("swa_ctx"))]
    if cfg.get("swa_target"):
        args += ["--swa-target", str(cfg.get("swa_target"))]
    if cfg.get("perf"):
        args.append("--perf")
    if not cfg.get("escape"):
        args.append("--no-escape")
    if cfg.get("override_tensor"):
        args += ["--override-tensor", str(cfg.get("override_tensor"))]
    if cfg.get("direct_io"):
        args.append("--direct-io")
    if cfg.get("no_host"):
        args.append("--no-host")
    if not cfg.get("repack"):
        args.append("--no-repack")
    if cfg.get("ui_mcp_proxy") == "on":
        args.append("--ui-mcp-proxy")
    elif cfg.get("ui_mcp_proxy"):
        args += ["--ui-mcp-proxy", str(cfg.get("ui_mcp_proxy"))]

    if m_predict != -1:
        args += ["-n", str(m_predict)]
    if m_keep != 0:
        args += ["--keep", str(m_keep)]

    if m_temp is not None:
        args += ["--temp", str(m_temp)]
    if m_top_p is not None:
        args += ["--top-p", str(m_top_p)]
    if m_min_p is not None:
        args += ["--min-p", str(m_min_p)]
    if m_repeat_p is not None:
        args += ["--repeat-penalty", str(m_repeat_p)]
    if m_repeat_n is not None:
        args += ["--repeat-last-n", str(m_repeat_n)]
    if m_pres_p is not None:
        args += ["--presence-penalty", str(m_pres_p)]
    if m_freq_p is not None:
        args += ["--frequency-penalty", str(m_freq_p)]
    if m_top_k is not None:
        args += ["--top-k", str(m_top_k)]

    typical_p = cfg.get("typical_p")
    if typical_p is not None and float(typical_p) != 1.0:
        args += ["--typical-p", str(typical_p)]
    top_n_sigma = cfg.get("top_n_sigma")
    if top_n_sigma is not None and float(top_n_sigma) != -1.0:
        args += ["--top-n-sigma", str(top_n_sigma)]
    xtc_prob = cfg.get("xtc_probability")
    if xtc_prob is not None and float(xtc_prob) > 0:
        args += ["--xtc-probability", str(xtc_prob)]
    xtc_thresh = cfg.get("xtc_threshold")
    if xtc_thresh is not None and float(xtc_thresh) > 0:
        args += ["--xtc-threshold", str(xtc_thresh)]

    dry_mult = cfg.get("dry_multiplier")
    if dry_mult is not None and float(dry_mult) > 0:
        args += ["--dry-multiplier", str(dry_mult)]
    dry_base = cfg.get("dry_base")
    if dry_base is not None and float(dry_base) != 1.75:
        args += ["--dry-base", str(dry_base)]
    dry_len = cfg.get("dry_allowed_length")
    if dry_len is not None and int(dry_len) != 2:
        args += ["--dry-allowed-length", str(dry_len)]
    dry_pen = cfg.get("dry_penalty_last_n")
    if dry_pen is not None and int(dry_pen) != -1:
        args += ["--dry-penalty-last-n", str(dry_pen)]
    dry_seq = cfg.get("dry_sequence_breaker")
    if dry_seq:
        args += ["--dry-sequence-breaker", str(dry_seq)]

    adapt_tgt = cfg.get("adaptive_target")
    if adapt_tgt is not None and float(adapt_tgt) >= 0:
        args += ["--adaptive-target", str(adapt_tgt)]
    adapt_dec = cfg.get("adaptive_decay")
    if adapt_dec is not None and float(adapt_dec) > 0:
        args += ["--adaptive-decay", str(adapt_dec)]
    dyn_range = cfg.get("dynatemp_range")
    if dyn_range is not None and float(dyn_range) > 0:
        args += ["--dynatemp-range", str(dyn_range)]
    dyn_exp = cfg.get("dynatemp_exp")
    if dyn_exp is not None and float(dyn_exp) != 1.0:
        args += ["--dynatemp-exp", str(dyn_exp)]

    mirostat = cfg.get("mirostat")
    if mirostat is not None and int(mirostat) > 0:
        args += ["--mirostat", str(mirostat)]
        args += ["--mirostat-lr", str(cfg.get("mirostat_lr"))]
        args += ["--mirostat-ent", str(cfg.get("mirostat_ent"))]

    spec_type = cfg.get("spec_type")
    if spec_type:
        args += ["--spec-type", str(spec_type)]
    spec_n_max = cfg.get("spec_draft_n_max")
    if spec_n_max is not None and int(spec_n_max) > 0:
        args += ["--spec-draft-n-max", str(spec_n_max)]
    spec_n_min = cfg.get("spec_draft_n_min")
    if spec_n_min is not None and int(spec_n_min) > 0:
        args += ["--spec-draft-n-min", str(spec_n_min)]
    spec_p_split = cfg.get("spec_draft_p_split")
    if spec_p_split is not None and float(spec_p_split) > 0:
        args += ["--spec-draft-p-split", str(spec_p_split)]
    spec_p_min = cfg.get("spec_draft_p_min")
    if spec_p_min is not None and float(spec_p_min) > 0:
        args += ["--spec-draft-p-min", str(spec_p_min)]

    # Chat / template (Consolidated Logic Block)
    chat_kwargs = cfg.get("chat_template_kwargs")
    if not reasoning_on:
        if isinstance(chat_kwargs, dict):
            chat_kwargs = {**chat_kwargs, "thinking": False}
        elif isinstance(chat_kwargs, str) and chat_kwargs.strip():
            try:
                parsed = json.loads(chat_kwargs)
                if isinstance(parsed, dict):
                    parsed["thinking"] = False
                    chat_kwargs = parsed
            except Exception:
                pass
        elif not chat_kwargs:
            chat_kwargs = {"thinking": False}

    if chat_kwargs:
        serialized_kwargs = json.dumps(chat_kwargs) if isinstance(chat_kwargs, (dict, list)) else str(chat_kwargs)
        args += ["--chat-template-kwargs", serialized_kwargs]

    if cfg.get("skip_chat_parsing"):
        args.append("--skip-chat-parsing")
    if not cfg.get("prefill_assistant", True):
        args.append("--no-prefill-assistant")

    if not cfg.get("ui", True):
        args.append("--no-ui")
    if cfg.get("embedding"):
        args.append("--embedding")
    if cfg.get("rerank"):
        args.append("--rerank")
    if not cfg.get("cache_prompt", True):
        args.append("--no-cache-prompt")
    cache_reuse = cfg.get("cache_reuse")
    if cache_reuse is not None and int(cache_reuse) > 0:
        args += ["--cache-reuse", str(cache_reuse)]
    cache_ram = cfg.get("cache_ram")
    if cache_ram is not None and int(cache_ram) > 0:
        args += ["--cache-ram", str(cache_ram)]
    if cfg.get("kv_unified"):
        args.append("--kv-unified")
    else:
        args.append("--no-kv-unified")
    defrag = cfg.get("defrag_thold")
    if defrag is not None and int(defrag) > 0:
        args += ["--defrag-thold", str(defrag)]
    if cfg.get("cache_idle_slots"):
        args.append("--cache-idle-slots")
    if cfg.get("context_shift"):
        args.append("--context-shift")
    if not cfg.get("slots", True):
        args.append("--no-slots")
    slot_save = cfg.get("slot_save_path")
    if slot_save:
        args += ["--slot-save-path", os.path.expanduser(str(slot_save))]
    if cfg.get("metrics"):
        args.append("--metrics")
    if cfg.get("props"):
        args.append("--props")
    sse_ping = cfg.get("sse_ping_interval")
    if sse_ping is not None and int(sse_ping) > 0:
        args += ["--sse-ping-interval", str(sse_ping)]
    threads_http = cfg.get("threads_http")
    if threads_http is not None and int(threads_http) > 0:
        args += ["--threads-http", str(threads_http)]

    if not cfg.get("fit", True):
        args.append("--no-fit")
    fit_target = cfg.get("fit_target")
    if fit_target is not None and int(fit_target) != 1024:
        args += ["--fit-target", str(fit_target)]
    fit_ctx = cfg.get("fit_ctx")
    if fit_ctx is not None and int(fit_ctx) != 4096:
        args += ["--fit-ctx", str(fit_ctx)]

    # UI / web (Safely Serialized)
    ui_config = cfg.get("ui_config")
    if ui_config:
        serialized_ui = json.dumps(ui_config) if isinstance(ui_config, (dict, list)) else str(ui_config)
        args += ["--ui-config", serialized_ui]
    path = cfg.get("path")
    if path:
        args += ["--path", str(path)]
    api_prefix = cfg.get("api_prefix")
    if api_prefix:
        args += ["--api-prefix", str(api_prefix)]

    if cfg.get("log_disable"):
        args.append("--log-disable")
    log_file_cli = cfg.get("log_file")
    if log_file_cli:
        args += ["--log-file", os.path.expanduser(str(log_file_cli))]
    log_colors = cfg.get("log_colors")
    if log_colors and str(log_colors).lower() not in ("auto", "none", ""):
        args += ["--log-colors", str(log_colors)]
    if cfg.get("offline"):
        args.append("--offline")
    if cfg.get("log_prefix"):
        args.append("--log-prefix")
    if cfg.get("log_timestamps"):
        args.append("--log-timestamps")
    log_verbosity = cfg.get("log_verbosity")
    if log_verbosity is not None and int(log_verbosity) != 3:
        args += ["--log-verbosity", str(log_verbosity)]

    api_key = cfg.get("api_key")
    if api_key:
        args += ["--api-key", str(api_key)]
    api_key_file = cfg.get("api_key_file")
    if api_key_file:
        args += ["--api-key-file", str(api_key_file)]
    ssl_key = cfg.get("ssl_key_file")
    if ssl_key:
        args += ["--ssl-key-file", str(ssl_key)]
    ssl_cert = cfg.get("ssl_cert_file")
    if ssl_cert:
        args += ["--ssl-cert-file", str(ssl_cert)]

    seed = cfg.get("seed")
    if seed is not None and int(seed) >= 0:
        args += ["--seed", str(seed)]
    if cfg.get("ignore_eos"):
        args.append("--ignore-eos")

    samplers = cfg.get("samplers")
    if samplers:
        args += ["--samplers", str(samplers)]
    sampler_seq = cfg.get("sampler_seq")
    if sampler_seq:
        args += ["--sampler-seq", str(sampler_seq)]
    grammar = cfg.get("grammar")
    if grammar:
        args += ["--grammar", str(grammar)]
    grammar_file = cfg.get("grammar_file")
    if grammar_file:
        args += ["--grammar-file", os.path.expanduser(str(grammar_file))]
    json_schema = cfg.get("json_schema")
    if json_schema:
        serialized_schema = json.dumps(json_schema) if isinstance(json_schema, (dict, list)) else str(json_schema)
        args += ["--json-schema", serialized_schema]
    json_schema_file = cfg.get("json_schema_file")
    if json_schema_file:
        args += ["--json-schema-file", os.path.expanduser(str(json_schema_file))]

    lora = cfg.get("lora")
    if lora:
        args += ["--lora", os.path.expanduser(str(lora))]
    lora_scaled = cfg.get("lora_scaled")
    if lora_scaled:
        args += ["--lora-scaled", os.path.expanduser(str(lora_scaled))]
    cv = cfg.get("control_vector")
    if cv:
        args += ["--control-vector", os.path.expanduser(str(cv))]
    cv_scaled = cfg.get("control_vector_scaled")
    if cv_scaled:
        args += ["--control-vector-scaled", os.path.expanduser(str(cv_scaled))]
    cv_range = cfg.get("control_vector_layer_range")
    if cv_range:
        args += ["--control-vector-layer-range", str(cv_range)]

    media_path = cfg.get("media_path")
    if media_path:
        args += ["--media-path", os.path.expanduser(str(media_path))]
    mmproj_path = cfg.get("mmproj_path")
    if not mmproj_path and cfg.get("mmproj_auto", True):
        mmproj_path = _find_mmproj(model_path)
    if mmproj_path:
        args += ["--mmproj", os.path.expanduser(str(mmproj_path))]
        if not cfg.get("mmproj_offload", True):
            args.append("--no-mmproj-offload")
    img_min = cfg.get("image_min_tokens")
    if img_min is not None:
        args += ["--image-min-tokens", str(img_min)]
    img_max = cfg.get("image_max_tokens")
    if img_max is not None:
        args += ["--image-max-tokens", str(img_max)]

    pooling = cfg.get("pooling")
    if pooling:
        args += ["--pooling", str(pooling)]
    embd_norm = cfg.get("embd_normalize")
    if embd_norm is not None and int(embd_norm) != 2:
        args += ["--embd-normalize", str(embd_norm)]

    tags = cfg.get("tags")
    if tags:
        args += ["--tags", str(tags)]

    if not cfg.get("warmup", True):
        args.append("--no-warmup")
    elif cfg.get("no_warmup"):
        args.append("--no-warmup")
    if not cfg.get("cont_batching", True):
        args.append("--no-cont-batching")
    if cfg.get("spm_infill"):
        args.append("--spm-infill")

    if cfg.get("cpu_strict"):
        args.append("--cpu-strict")
    if cfg.get("cpu_strict_batch"):
        args.append("--cpu-strict-batch")
    if cfg.get("prio", 0) != 0:
        args += ["--prio", str(cfg.get("prio"))]
    if cfg.get("prio_batch", 0) != 0:
        args += ["--prio-batch", str(cfg.get("prio_batch"))]
    if cfg.get("poll", 50) != 50:
        args += ["--poll", str(cfg.get("poll"))]
    if cfg.get("poll_batch", 50) != 50:
        args += ["--poll-batch", str(cfg.get("poll_batch"))]

    rb_msg = model_cfg.get("reasoning_budget_message", cfg.get("reasoning_budget_message"))
    if rb_msg:
        args += ["--reasoning-budget-message", str(rb_msg)]

    if extra_args:
        args += extra_args

    print(f"[start] launching llama-server")
    print(f"  model : {model_path}")
    print(f"  host  : {host}:{port}")
    print(f"  ctx   : {m_ctx}")
    print(f"  ngl   : {m_ngl}")
    print(f"  log   : {log_file}")

    proc_ref = [None]

    def _shutdown(sig, frame):
        print(f"\n[start] shutting down (signal {sig})", end="", flush=True)
        try:
            os.kill(proc_ref[0].pid, signal.SIGTERM)
        except (OSError, AttributeError):
            pass
        _clear_state()
        print(" done")
        raise SystemExit(0)

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            args,
            stdout=log, stderr=log,
            stdin=subprocess.DEVNULL,
        )
        proc_ref[0] = proc

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def _cleanup():
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    _write_state({
        "pid":            proc.pid,
        "host":           host,
        "port":           port,
        "ctx":            m_ctx,
        "ngl":            m_ngl,
        "model_path":     model_path,
        "model_filename": os.path.basename(model_path),
        "model_name":     model_name,
        "started_at":     time.time(),
        "log":            log_file,
    })

    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    print(f"[start] pid {proc.pid} — waiting for server", end="", flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(1)
        if not _is_running(proc.pid):
            _cleanup()
            print(" FAILED")
            print(f"[start] process died — check {log_file}")
            _clear_state()
            raise RuntimeError("process died — check the log")
        if _health(host, port):
            print(" ready ✓")
            _cleanup()
            return proc.pid
        print(".", end="", flush=True)

    print(" timeout")
    print(f"[start] did not become healthy — check {log_file}")
    _cleanup()
    return proc.pid


# ── Stop / Kill / Restart ─────────────────────────────────────────────────────

def stop() -> None:
    """Stop the server gracefully via systemd."""
    if not _systemd_unit_exists():
        print("[stop] systemd service not installed — server may still be running")
        return
    result = subprocess.run(
        ["systemctl", "--user", "stop", "llama-server.service"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[stop] systemctl failed: {result.stderr.strip()}")
        return
    _clear_state()
    print("[stop] server stopped ✓")


def kill() -> None:
    """Force-kill the server via systemd (SIGKILL)."""
    if not _systemd_unit_exists():
        print("[kill] systemd service not installed — server may still be running")
        return
    result = subprocess.run(
        ["systemctl", "--user", "kill", "-s", "SIGKILL", "llama-server.service"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[kill] systemctl failed: {result.stderr.strip()}")
        return
    _clear_state()
    print("[kill] server killed ✓")


def restart(*_args, **_kwargs) -> None:
    """Restart the server via systemd."""
    if not _systemd_unit_exists():
        print("[restart] systemd service not installed — cannot restart")
        return

    from .config import get_config
    cfg = get_config()

    subprocess.run(["systemctl", "--user", "reset-failed", "llama-server.service"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    print("[restart] restarting server", end="", flush=True)
    result = subprocess.run(
        ["systemctl", "--user", "restart", "llama-server.service"],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        print(f"\n[restart] failed: {result.stderr.strip()}")
        return

    deadline = time.time() + 180
    while time.time() < deadline:
        if _health(cfg.host, cfg.port):
            print(" ready ✓")
            status()
            return
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" timeout")
    print("[restart] check: journalctl --user -u llama-server -f")


# ── ps ────────────────────────────────────────────────────────────────────────

def ps() -> None:
    state = _read_state()
    pid   = state.get("pid")
    if not state or not _is_running(pid):
        print("No server running.")
        return

    props = {}
    try:
        with urllib.request.urlopen(f"{_base_url()}/props", timeout=2) as r:
            props = json.load(r)
    except Exception:
        pass

    from .config import get_config
    cfg = get_config()

    model_name = state.get("model_filename", "?")
    port = state.get("port", "?")
    ctx_val = props.get("context_length", state.get("ctx", cfg.ctx))
    ngl_val = props.get("n_gpu_layers", state.get("ngl", cfg.ngl))
    batch_val = props.get("batch_size", state.get("batch_size", cfg.batch_size))
    threads_val = props.get("n_threads", state.get("threads", cfg.threads))
    flash_val = props.get("flash_attn", "yes" if cfg.flash_attn != "off" else "no")

    gpu_data = {"used_mi": 0, "total_mi": 0, "temp": 0, "pwr": 0, "util": 0}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            vals = [float(x.strip()) for x in r.stdout.strip().split(",") if x.strip()]
            if len(vals) == 5:
                gpu_data = {
                    "used_mi": vals[0],
                    "total_mi": vals[1],
                    "temp": int(vals[2]),
                    "pwr": round(vals[3], 1),
                    "util": int(vals[4])
                }
    except Exception: pass

    try:
        load_1 = round(os.getloadavg()[0], 2)
    except Exception:
        load_1 = 0.0

    total_layers = props.get("n_layers", None)
    ngl_int = int(ngl_val) if str(ngl_val) not in ("?", "-") else 0
    if total_layers:
        cpu_layers = max(0, int(total_layers) - ngl_int)
        offload_str = f"{cpu_layers}L" if cpu_layers > 0 else "0L"
    else:
        offload_str = "?L"

    mem_used_gb = round(gpu_data['used_mi'] / 1024, 1)
    mem_total_gb = round(gpu_data['total_mi'] / 1024, 1)
    gpu_mem_str = f"{mem_used_gb}/{mem_total_gb}GB"

    uptime = int(time.time() - state.get("started_at", time.time()))
    h, rem = divmod(uptime, 3600)
    m, s = divmod(rem, 60)

    print("llama ps")
    print("=" * 60)
    print("[Server]")
    print(f"  Model          : {model_name}")
    print(f"  PID            : {pid}")
    print(f"  Port           : {port}")
    print(f"  Uptime         : {h:02d}:{m:02d}:{s:02d}")
    print("-" * 60)
    print("[Model Config]")
    print(f"  Context        : {ctx_val}")
    print(f"  GPU Layers     : {ngl_val}")
    print(f"  Batch Size     : {batch_val}")
    print(f"  Threads        : {threads_val}")
    print(f"  Flash Attention: {flash_val}")
    print(f"  CPU Offload    : {offload_str}")
    print("-" * 60)
    print("[Hardware Status]")
    print(f"  GPU VRAM       : {gpu_mem_str}")
    print(f"  GPU Util       : {gpu_data['util']}%")
    print(f"  GPU Temp       : {gpu_data['temp']}C")
    print(f"  GPU Power      : {gpu_data['pwr']}W")
    print(f"  CPU Load (1m)  : {load_1}")
    print("-" * 60)

def status() -> None:
    state = _read_state()
    pid   = state.get("pid")
    alive = _is_running(pid)

    if alive:
        host = state.get("host", LLAMA_HOST)
        port = state.get("port", LLAMA_PORT)
        ok   = _is_healthy(host, port)
    else:
        ok = False

    if alive:
        print(f"Server PID   : {pid} (running, {'healthy' if ok else 'unhealthy'})")
    else:
        print(f"Server PID   : {pid or 'none'} (stopped)")

    if not alive:
        return

    print(f"Health       : {'OK' if ok else 'UNREACHABLE'}")
    print(f"Address      : {state.get('host', LLAMA_HOST)}:{state.get('port', LLAMA_PORT)}")
    print(f"Model        : {state.get('model_path', '?')}")
    print(f"Context      : {state.get('ctx', '?')}")
    print(f"GPU layers   : {state.get('ngl', '?')}")
    print(f"Log          : {state.get('log', '?')}")

    uptime = int(time.time() - state.get("started_at", time.time()))
    h, rem = divmod(uptime, 3600)
    m, s   = divmod(rem, 60)
    print(f"Uptime       : {h:02d}:{m:02d}:{s:02d}")


# ── logs ──────────────────────────────────────────────────────────────────────

def logs(n: int = 40) -> None:
    # Always read from the canonical log path, not the state file (which can be stale)
    log = os.path.join(LOG_DIR, "server.log")
    if not os.path.exists(log):
        print(f"[logs] not found: {log}")
        print("  Run 'llama start' to generate the log file, or use:")
        print(f"  journalctl --user -u llama-server.service -n {n}")
        return
    with open(log) as f:
        lines = f.readlines()[-n:]
    for line in lines:
        print(line, end="")


# ── chat_messages ─────────────────────────────────────────────────────────────

_CHAT_NOT_SET = object()

def chat_messages(
    messages: list,
    temperature: float = _CHAT_NOT_SET,
    top_k: int = _CHAT_NOT_SET,
    max_tokens: int = _CHAT_NOT_SET,
    stream: bool = True,
    top_p: float = _CHAT_NOT_SET,
    min_p: float = _CHAT_NOT_SET,
    frequency_penalty: float = _CHAT_NOT_SET,
    presence_penalty: float = _CHAT_NOT_SET,
) -> "Iterator[str]":
    """POST /v1/chat/completions with a proper messages array."""
    from .config import get_config
    state = _read_state()
    pid   = state.get("pid")
    if not _is_running(pid):
        raise RuntimeError("Server not running. Run \'llama start\'")

    host = state.get("host", LLAMA_HOST)
    port = state.get("port", LLAMA_PORT)

    cfg = get_config()
    if temperature is _CHAT_NOT_SET:
        temperature = cfg.get("temperature")
    if top_k is _CHAT_NOT_SET:
        top_k = cfg.get("top_k")
    if max_tokens is _CHAT_NOT_SET:
        max_tokens = cfg.get("max_tokens")
    if top_p is _CHAT_NOT_SET:
        top_p = cfg.get("top_p")
    if min_p is _CHAT_NOT_SET:
        min_p = cfg.get("min_p")
    if frequency_penalty is _CHAT_NOT_SET:
        frequency_penalty = cfg.get("frequency_penalty")
    if presence_penalty is _CHAT_NOT_SET:
        presence_penalty = cfg.get("presence_penalty")

    payload = json.dumps({
        "messages":            messages,
        "temperature":         temperature,
        "top_k":               top_k,
        "max_tokens":          max_tokens,
        "top_p":               top_p,
        "min_p":               min_p,
        "frequency_penalty":   frequency_penalty,
        "presence_penalty":    presence_penalty,
        "stream":              stream,
        "stream_options":      {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data  = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                    if data.get("choices", [{}])[0].get("finish_reason"):
                        break
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach server: {e}")


# ── systemd service ───────────────────────────────────────────────────────────

def _service_path() -> str:
    return os.path.expanduser("~/.config/systemd/user/llama-server.service")

def install_service() -> None:
    """Install a systemd user service that runs the server."""
    import shutil as _shutil
    llama_bin = _shutil.which("llama") or f"{sys.executable} -m llama_light"

    cuda_path = "/usr/local/cuda/lib64"
    if not os.path.exists(cuda_path):
        for alt in ["/usr/local/cuda-12/lib64", "/usr/local/cuda-11/lib64"]:
            if os.path.exists(alt):
                cuda_path = alt
                break

    svc = "\n".join([
        "[Unit]",
        "Description=llama-light server daemon",
        "After=network.target",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={llama_bin} _run",
        "KillMode=control-group",
        "KillSignal=SIGTERM",
        "TimeoutStopSec=5",
        "TimeoutStartSec=300",
        f"Environment=PATH={os.path.expanduser('~/.local/bin')}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        f"Environment=LD_LIBRARY_PATH={cuda_path}",
        f"StandardOutput=append:{LOG_DIR}/server.log",
        f"StandardError=append:{LOG_DIR}/server.log",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ])

    svc_path = _service_path()
    os.makedirs(os.path.dirname(svc_path), exist_ok=True)
    with open(svc_path, "w") as f:
        f.write(svc)

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "llama-server.service"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"systemctl command failed: {e}") from e

    try:
        subprocess.run(
            ["loginctl", "enable-linger", os.environ.get("USER", "")],
            check=False, capture_output=True,
        )
    except Exception:
        pass

    print("[service] installed and enabled (not started)")
    print(f"  unit : {svc_path}")
    from .config import get_config
    if not get_config().default_model:
        print("  Set a model first : llama config set default_model <path>")
    print("  Start it          : llama start   (or: systemctl --user start llama-server)")
    print("  Logs              : journalctl --user -u llama-server -f")

def uninstall_service() -> None:
    try:
        subprocess.run(["systemctl", "--user", "stop",    "llama-server.service"], check=False)
        subprocess.run(["systemctl", "--user", "disable", "llama-server.service"], check=False)
        svc_path = _service_path()
        if os.path.exists(svc_path):
            os.remove(svc_path)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[service] warning: systemctl command failed: {e}")
    print("[service] removed")
