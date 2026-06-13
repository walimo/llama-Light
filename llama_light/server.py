# llama_light/server.py
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, Optional, Tuple

from .config import (
    DEFAULT_CTX, DEFAULT_GPU_LAYERS, DEFAULT_HOST, DEFAULT_PORT,
    LLAMA_HOST, LLAMA_PORT, LOG_DIR, PID_FILE,
    STATE_FILE, ensure_dirs,
)
from ._bincheck import locate_main_bin


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
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

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
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _is_healthy(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Return True only if both /health and /props return successfully."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as r:
            if r.status != 200:
                return False
        with urllib.request.urlopen(f"http://{host}:{port}/props", timeout=3) as r:
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


# ── Crash watcher ─────────────────────────────────────────────────────────────

def _watch_and_restart(
    pid: int, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Tuple[threading.Thread, int]:
    """Run a daemon thread that monitors *pid* and returns the crashed PID."""

    def _loop() -> int:
        while True:
            time.sleep(10)
            if not _is_running(pid):
                # Process has exited — the watcher has done its job
                break
            if not _is_healthy(host, port):
                print(f"[crash] detected, server unresponsive (pid {pid})")
                _clear_state()
                return pid
        return pid

    t = threading.Thread(target=_loop, daemon=True, name="llama-watcher")
    t.start()
    return t, pid


# ── Start ─────────────────────────────────────────────────────────────────────

def start(
    model_path: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ctx: int = DEFAULT_CTX,
    gpu_layers: int = DEFAULT_GPU_LAYERS,
    flash_attn: bool = True,
    extra_args: Optional[list] = None,
) -> Tuple[int, threading.Thread]:
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

    pid = _pid()
    if _is_running(pid):
        state = _read_state()
        raise RuntimeError(
            f"Server already running (pid {pid}) with "
            f"{state.get('model_filename','?')} on port {state.get('port', port)}.\n"
            "Run 'llama stop' first."
        )

    log_file = os.path.join(LOG_DIR, "server.log")

    # pull full config so all keys are wired through
    from .config import get_config
    cfg = get_config()

    # per-model settings — auto-detected + user-saved overrides
    from .per_model import get_model_config, _model_name_from_path
    model_name = _model_name_from_path(model_path)
    model_cfg = get_model_config(model_name)

    # Merge: global config → per-model overrides (model wins selectively)
    m_ctx      = model_cfg.get("ctx",         ctx)
    m_ngl      = model_cfg.get("ngl",         gpu_layers)
    m_threads  = model_cfg.get("threads",     cfg.threads)
    m_flash    = model_cfg.get("flash_attn",  cfg.flash_attn)
    m_keep     = model_cfg.get("keep",        cfg.get("keep", 0))
    m_predict  = model_cfg.get("predict",     -1)
    m_batch    = model_cfg.get("batch_size",  cfg.batch_size)

    args = [
        bin_path,
        "-m",            model_path,
        "--host",        host,
        "--port",        str(port),
        "-c",            str(m_ctx),
        "-ngl",          str(m_ngl),
        "--parallel",    str(cfg.parallel),
        "--flash-attn",  "on" if str(m_flash).lower() not in ("off", "false", "0") else "off",
        "--tools",       "all",
        "-b",            str(m_batch),
        "--ubatch-size", str(cfg.ubatch_size),
        "--threads",     str(m_threads),
        "--threads-batch", str(cfg.get("threads_batch", m_threads)),
        "--cache-type-k", str(cfg.get("cache_type_k", "f16")),
        "--cache-type-v", str(cfg.get("cache_type_v", "f16")),
    ]

    # KV offload — default on; flag only needed to disable
    if not cfg.get("kv_offload", True):
        args.append("--no-kv-offload")

    # mmap / mlock
    if not cfg.get("mmap", True):
        args.append("--no-mmap")
    if cfg.get("mlock", False):
        args.append("--mlock")

    # split mode
    if cfg.get("split_mode"):
        args += ["--split-mode", str(cfg.get("split_mode"))]

    # specific device
    if cfg.get("device"):
        args += ["--device", str(cfg.get("device"))]

    # NUMA
    if cfg.get("numa"):
        args += ["--numa", str(cfg.get("numa"))]

    # RoPE
    if cfg.get("rope_scaling"):
        args += ["--rope-scaling", str(cfg.get("rope_scaling"))]
    if cfg.get("rope_freq_base"):
        args += ["--rope-freq-base", str(cfg.get("rope_freq_base"))]
    if cfg.get("rope_scale"):
        args += ["--rope-scale", str(cfg.get("rope_scale"))]
    if cfg.get("rope_freq_scale"):
        args += ["--rope-freq-scale", str(cfg.get("rope_freq_scale"))]

    # YaRN (only pass if non-default)
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

    # MoE
    if cfg.get("cpu_moe", False):
        args.append("--cpu-moe")
    if cfg.get("n_cpu_moe"):
        args += ["--n-cpu-moe", str(cfg.get("n_cpu_moe"))]

    # reasoning — model-level setting overrides global
    m_reasoning    = model_cfg.get("reasoning",  cfg.get("reasoning",  False))
    reasoning_on   = str(m_reasoning).lower() not in ("false", "off", "0") and m_reasoning is not False
    m_reason_budget = model_cfg.get("reasoning_budget", cfg.get("reasoning_budget", 0))
    if not reasoning_on:
        args += ["--reasoning", "off"]
        args += ["--chat-template-kwargs", '{"thinking":false}']
    elif m_reason_budget:
        rf = cfg.get("reasoning_format", "none")
        if rf and str(rf).lower() not in ("none", "null", ""):
            args += ["--reasoning-format", str(rf)]
        args += ["--reasoning-budget", str(m_reason_budget)]

    # misc flags
    if cfg.get("swa_full", False):
        args.append("--swa-full")
    if cfg.get("perf", False):
        args.append("--perf")
    if not cfg.get("escape", True):
        args.append("--no-escape")
    if cfg.get("override_tensor"):
        args += ["--override-tensor", str(cfg.get("override_tensor"))]
    if cfg.get("direct_io", False):
        args.append("--direct-io")
    if cfg.get("no_host", False):
        args.append("--no-host")
    if not cfg.get("repack", True):
        args.append("--no-repack")
    if cfg.get("ui_mcp_proxy", "on") != "on":
        args += ["--ui-mcp-proxy", str(cfg.get("ui_mcp_proxy"))]
    if cfg.get("tools", "all") != "all":
        args += ["--tools", str(cfg.get("tools"))]

    # generation defaults wired as server-side caps (use model values when set)
    if m_predict != -1:
        args += ["-n", str(m_predict)]
    if m_keep != 0:
        args += ["--keep", str(m_keep)]

    if extra_args:
        args += extra_args

    print(f"[start] launching llama-server")
    print(f"  model : {model_path}")
    print(f"  host  : {host}:{port}")
    print(f"  ctx   : {m_ctx}")
    print(f"  ngl   : {m_ngl}")
    print(f"  log   : {log_file}")

   # Graceful shutdown — clean up on SIGTERM/SIGINT
    proc_ref = [None]  # mutable container so _shutdown can access proc after Popen
    def _shutdown(sig, frame):
        print(f"\n[start] shutting down (signal {sig})", end="", flush=True)
        try:
            os.kill(proc_ref[0].pid, signal.SIGTERM)
        except (OSError, AttributeError):
            pass
        _clear_state()
        print(" done")
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    with open(log_file, "a") as log:
        proc = subprocess.Popen(
            args,
            stdout=log, stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        proc_ref[0] = proc

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

    # Start crash watcher thread
    watcher_thread, _watcher_pid = _watch_and_restart(
        proc.pid, host=host, port=port
    )

    print(f"[start] pid {proc.pid} — waiting for server ...", end="", flush=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(1)
        if not _is_running(proc.pid):
            print(" FAILED")
            print(f"[start] process died — check {log_file}")
            _clear_state()
            raise RuntimeError("process died — check the log")
        if _health(host, port):
            print(" ready ✓")
            return proc.pid, watcher_thread
        print(".", end="", flush=True)

    print(" timeout")
    print(f"[start] did not become healthy — check {log_file}")
    return proc.pid, watcher_thread


# ── Stop / Kill / Restart ─────────────────────────────────────────────────────

def _send_signal(sig: int, label: str) -> None:
    pid = _pid()
    if not _is_running(pid):
        print(f"[{label}] no server running")
        _clear_state()
        return
    os.kill(pid, sig)
    print(f"[{label}] signal sent to pid {pid} ...", end="", flush=True)
    for _ in range(10):
        time.sleep(0.5)
        if not _is_running(pid):
            break
    _clear_state()
    print(" done")

def stop() -> None:  _send_signal(signal.SIGTERM, "stop")
def kill() -> None:  _send_signal(signal.SIGKILL, "kill")

def restart(model_path: Optional[str] = None, **kwargs) -> None:
    mp = model_path or _read_state().get("model_path")
    if not mp:
        raise RuntimeError("No model known — pass --model")
    # Save port before stopping (state may be cleared by stop)
    port = kwargs.get("port", _read_state().get("port", DEFAULT_PORT))
    stop()
    # Wait for port to free up (TCP TIME_WAIT handling)
    state = _read_state()
    port = kwargs.get("port", state.get("port", port))
    deadline = time.time() + 30
    while _detect_port_in_use(port=port):
        if time.time() > deadline:
            print("[restart] WARNING: port still bound after 30 s, proceeding anyway")
            break
        time.sleep(1)
    start(mp, **kwargs)


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

    # Static Values
    model_name = state.get("model_filename", "?")
    port = state.get("port", "?")
    ctx_val = props.get("context_length", state.get("ctx", cfg.ctx))
    ngl_val = props.get("n_gpu_layers", state.get("ngl", cfg.ngl))
    batch_val = props.get("batch_size", state.get("batch_size", cfg.batch_size))
    threads_val = props.get("n_threads", state.get("threads", cfg.threads))
    flash_val = props.get("flash_attn", "yes" if cfg.flash_attn != "off" else "no")

    # Helper: Fetch Live Hardware Stats

    # Get live stats once
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
    log = _read_state().get("log", os.path.join(LOG_DIR, "server.log"))
    if not os.path.exists(log):
        print(f"[logs] not found: {log}")
        return
    with open(log) as f:
        lines = f.readlines()
    for line in lines[-n:]:
        print(line, end="")




# ── chat_messages (OpenAI-compatible, disables thinking via message format) ───

def chat_messages(
    messages: list,
    temperature: float = 0.7,
    top_k: int = 40,
    max_tokens: int = 2048,
    stream: bool = True,
    # Token-efficient params — pass through to llama-server
    top_p: float = 0.95,
    min_p: float = 0.05,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> "Iterator[str]":
    """
    POST /v1/chat/completions with a proper messages array.

    Token-efficient defaults (low temp, tight top-p, penalties) reduce
    wasted tokens on repetition and verbosity, keeping costs down.
    """
    state = _read_state()
    pid   = state.get("pid")
    if not _is_running(pid):
        raise RuntimeError("Server not running. Run \'llama start\'")

    host = state.get("host", LLAMA_HOST)
    port = state.get("port", LLAMA_PORT)

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
        with urllib.request.urlopen(req, timeout=120) as resp:
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
    """Install a minimal systemd user service that runs `llama start`.
    All server config is read from ~/.config/llama_light/config.json at startup.
    To change settings: llama config set <key> <value>, then llama restart.
    """
    import shutil as _shutil
    llama_bin = _shutil.which("llama") or f"{sys.executable} -m llama_light"

    svc = "\n".join([
        "[Unit]",
        "Description=llama-light server daemon",
        "After=network.target",
        "",
        "[Service]",
        "Type=forking",
        f"ExecStart={llama_bin} start",
        "KillMode=none",
        "Restart=on-failure",
        "RestartSec=10",
        "Environment=PATH=/home/%u/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
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
        subprocess.run(["systemctl", "--user", "start",  "llama-server.service"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"systemctl command failed: {e}") from e
    print(f"[service] installed and started")
    print(f"  unit : {svc_path}")
    print(f"  logs : journalctl --user -u llama-server -f")

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