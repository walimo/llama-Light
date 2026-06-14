# llama_light/_cli.py
import argparse
import os
import subprocess
import sys
import time
import webbrowser
from typing import Dict, Optional

from .server import (
    chat_messages, kill, logs, ps, restart, start, status, stop,
    install_service, uninstall_service,
    _read_state, _is_running, _pid, _base_url, _is_healthy,
)
from .model_manager import pull, ls, rm
from .config import (
    CACHE_ROOT, LLAMA_SERVER_BIN, LOG_DIR, HF_CACHE_DIR,
    get_config,
)
from .registry import find, scan_hf_cache
from ._bincheck import check, status as _bincheck_status
from ._backup import backup, restore, list_backups
from ._llama_downloader import ensure_binaries, check_version

VERSION = "0.2.0"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model_arg(args: argparse.Namespace) -> Optional[str]:
    m = getattr(args, "model", None)
    if not m:
        return m
    if os.path.isabs(m) and os.path.exists(m):
        args._model_path = m  # for _resolve_gen_args to pick up
        return m
    # triple-layer: registry + HF cache
    scan_hf_cache()
    info = find(m)
    if info:
        args._model_path = info["local_path"]
        return info["local_path"]
    # walk llama_light cache
    for root, _, files in os.walk(CACHE_ROOT):
        for f in files:
            if f == m or f.endswith(m):
                args._model_path = os.path.join(root, f)
                return os.path.join(root, f)
    args._model_path = m  # let server.start() give the error
    return m  # let server.start() give the error


def _ensure_server_or_start(args: argparse.Namespace) -> None:
    pid = _pid()
    if _is_running(pid):
        return

    from .server import _systemd_unit_exists
    cfg = get_config()

    if _systemd_unit_exists():
        print("[info] server not running — starting llama-server.service")
        subprocess.run(["systemctl", "--user", "start", "llama-server.service"])
        deadline = time.time() + 180
        while time.time() < deadline:
            if _is_healthy(cfg.host, cfg.port):
                return
            time.sleep(1)
        print("[warn] server did not become healthy in time — "
              "check: journalctl --user -u llama-server -f")
        return

    # No systemd unit installed — direct start (dev / no-systemd fallback)
    model = getattr(args, "model", None) or cfg.default_model or cfg.last_model
    if not model:
        print("[error] server not running and no --model / default_model set")
        sys.exit(1)
    args.model = model
    model_path = _resolve_model_arg(args)
    start(model_path=model_path, **_resolve_server_args(args))


def _server_env() -> Dict[str, str]:
    """
    Build an env dict that tells any OpenAI-compatible client (Hermes TUI,
    open-webui, shell scripts, etc.) how to reach the running llama-server.
    Also sets LLAMA_BASE_URL as a convenience alias.
    """
    state    = _read_state()
    base_url = _base_url()                        # e.g. http://127.0.0.1:8080
    model    = state.get("model_filename", state.get("model_path", "local-model"))

    env = os.environ.copy()
    env["OPENAI_BASE_URL"]    = f"{base_url}/v1"
    env["OPENAI_API_KEY"]     = "sk-local"        # llama-server ignores the key
    env["LLAMA_BASE_URL"]     = base_url
    env["LLAMA_MODEL"]        = model
    env["LLAMA_HOST"]         = state.get("host", "127.0.0.1")
    env["LLAMA_PORT"]         = str(state.get("port", 8080))
    env["LLAMA_CTX"]          = str(state.get("ctx",  4096))
    env["LLAMA_NGL"]          = str(state.get("ngl",  99))
    return env


def _print_server_env(env: Dict[str, str]) -> None:
    """Print server environment info for the user."""
    print(f"  OPENAI_BASE_URL = {env.get('OPENAI_BASE_URL', '?')}")
    print(f"  LLAMA_MODEL     = {env.get('LLAMA_MODEL', '?')}")
    print(f"  LLAMA_HOST      = {env.get('LLAMA_HOST', '?')}")
    print(f"  LLAMA_PORT      = {env.get('LLAMA_PORT', '?')}")
    print(f"  LLAMA_CTX       = {env.get('LLAMA_CTX', '?')}")
    print(f"  LLAMA_NGL       = {env.get('LLAMA_NGL', '?')}")


# ── Command handlers ──────────────────────────────────────────────────────────

# ── Server Lifecycle ─────────────────────────────────────────────────────────

def cmd_start(_args):
    """Start llama-server via systemd (reads all settings from config.json)."""
    from .server import _systemd_unit_exists
    if not _systemd_unit_exists():
        print("[error] systemd service not installed.")
        print("  Re-run install.sh, or: llama service --install")
        sys.exit(1)

    subprocess.run(["systemctl", "--user", "start", "llama-server.service"])

    cfg = get_config()
    print("[start] waiting for server", end="", flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        if _is_healthy(cfg.host, cfg.port):
            print(" ready ✓")
            return
        print(".", end="", flush=True)
        time.sleep(1)
    print(" timeout")
    print("[start] check: journalctl --user -u llama-server -f")


def cmd_run_server(args):
    """Internal systemd launcher (llama _run). Resolves model, starts server."""
    cfg = get_config()
    model = cfg.default_model or cfg.last_model
    if not model:
        print("[error] no default_model set — run: llama config set default_model <path>")
        sys.exit(1)
    args.model = model
    model_path = _resolve_model_arg(args)
    start(model_path=model_path, **_resolve_server_args(args))
    cfg.set("last_model", model_path)


def cmd_stop(_args):
    """Stop the running server."""
    stop()


def cmd_kill(_args):
    """Force-kill the server (SIGKILL)."""
    kill()


def cmd_restart(_args):
    """Restart server and reload config.json."""
    restart()


def cmd_run(args):
    _ensure_server_or_start(args)
    msgs   = []
    system = getattr(args, "system", None)
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": args.prompt})
    gen = _resolve_gen_args(args)
    for tok in chat_messages(msgs, **gen):
        print(tok, end="", flush=True)
    print()


def cmd_chat(args):
    _ensure_server_or_start(args)
    system   = getattr(args, "system", None)
    if system is None:
        system = _CHAT_SYSTEM
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    gen = _resolve_gen_args(args)
    print("llama chat — /clear to reset, /exit to quit")
    print("-" * 50)
    max_turns = 100
    turn_count = 0
    while True:
        if turn_count >= max_turns:
            print("[chat] conversation limit reached (100 turns) — type /exit to quit")
            break
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[bye]")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "/exit", "quit"):
            break
        if user_input == "/clear":
            messages = [m for m in messages if m["role"] == "system"]
            print("[cleared]")
            continue
        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="", flush=True)
        resp = []
        try:
            for tok in chat_messages(messages, **gen):
                print(tok, end="", flush=True)
                resp.append(tok)
        except RuntimeError as e:
            print(f"\n[error] {e}")
            break
        print()
        messages.append({"role": "assistant", "content": "".join(resp)})
        turn_count += 1


# ── Hermes / Claude persona helpers ──────────────────────────────────────────

_HERMES_SYSTEM = (
    "You are Hermes, a highly capable AI assistant created by NousResearch. "
    "You are precise, analytical, and thorough. Always reason step-by-step."
)

_CLAUDE_SYSTEM = (
    "You are Claude, an AI assistant by Anthropic. "
    "You are helpful, harmless, and honest. Think carefully before answering."
)

_CHAT_SYSTEM = (
    "You are a concise, capable AI coding assistant. "
    "Be direct, precise, and terse. No fluff."
)

_WEBUI_SYSTEM = (
    "You are a concise, capable AI coding assistant. "
    "Be direct, precise, and terse. No fluff."
)


def _inline_persona_chat(args, system: str) -> None:
    """Run a persona as an inline chat/run command (no external process)."""
    args.system = system
    if getattr(args, "prompt", None):
        # single-shot: just run without starting server twice
        _ensure_server_or_start(args)
        msgs = [{"role": "system", "content": system},
                {"role": "user",   "content": args.prompt}]
        for tok in chat_messages(msgs, **_resolve_gen_args(args)):
            print(tok, end="", flush=True)
        print()
    else:
        cmd_chat(args)


def cmd_hermes(args):
    """
    llama hermes — Launch the Hermes TUI wired to the local llama-server.

    Exits with an error if the ``hermes`` binary is not installed.
    --prompt : single-shot reply (no TUI).
    """
    _ensure_server_or_start(args)

    if getattr(args, "prompt", None):
        _inline_persona_chat(args, _HERMES_SYSTEM)
        return

    hermes_bin = check("hermes",
        'Install: pip install hermes-tui  (or the NousResearch Hermes CLI)')

    env = _server_env()
    cfg = get_config()
    gen = _resolve_gen_args(args)
    env["HERMES_TEMPERATURE"] = str(gen["temperature"])
    env["HERMES_TOP_K"] = str(gen["top_k"])
    env["HERMES_MAX_TOKENS"] = str(gen["max_tokens"])
    if getattr(args, "system", None):
        env["HERMES_SYSTEM"] = args.system

    state = _read_state()
    print(f"[hermes] launching {hermes_bin}")
    print(f"  base-url : {env['OPENAI_BASE_URL']}")
    print(f"  model    : {env['LLAMA_MODEL']}")
    _print_server_env(env)

    try:
        subprocess.run([hermes_bin], env=env, stdin=None, stdout=None, stderr=None)
    except KeyboardInterrupt:
        print("\n[hermes] exited")


def cmd_hermes_desktop(args):
    """
    llama hermes-desktop — Launch the Hermes Electron desktop app.

    Exits with an error if ``hermes`` is not installed.
    """
    import shutil

    _ensure_server_or_start(args)
    env = _server_env()
    cfg = get_config()
    host = _read_state().get("host", cfg.host)
    port = _read_state().get("port", cfg.port)

    hermes_bin = check("hermes",
        'Install: pip install hermes-tui  (or the NousResearch Hermes CLI)')

    print(f"[hermes-desktop] launching {hermes_bin} desktop")
    print(f"  Model    : {env['LLAMA_MODEL']}")
    print(f"  Port     : {port}")
    _print_server_env(env)

    try:
        subprocess.run(
            [hermes_bin, "desktop", "--source"],
            env=env,
            stdin=None,
            stdout=None,
            stderr=None,
        )
    except KeyboardInterrupt:
        print("\n[hermes-desktop] exited")


def cmd_claude(args):
    """
    llama claude — Launch the real Claude Code CLI wired to the local llama-server.

    Exits with an error if the ``claude`` binary is not installed.
    ANTHROPIC_API_KEY is set to a dummy value so Claude Code skips
    interactive login and uses the local OpenAI-compatible endpoint.
    """
    _ensure_server_or_start(args)
    env = _server_env()
    cfg = get_config()

    claude_bin = check("claude",
        "Install: npm install -g @anthropic-ai/claude-code")

    # Suppress Claude Code's interactive login — it uses the dummy key
    # and routes to OPENAI_BASE_URL (our local llama-server).
    env["ANTHROPIC_API_KEY"] = "sk-local"

    gen = _resolve_gen_args(args)
    env["LLAMA_TEMPERATURE"] = str(gen["temperature"])
    env["LLAMA_TOP_K"] = str(gen["top_k"])
    env["LLAMA_MAX_TOKENS"] = str(gen["max_tokens"])

    print(f"[claude] launching {claude_bin}")
    print(f"  Model    : {env['LLAMA_MODEL']}")
    print(f"  Base URL : {env['OPENAI_BASE_URL']}")
    _print_server_env(env)

    try:
        subprocess.run([claude_bin, "--dangerously-skip-permissions"], env=env, cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n[claude] Exiting...")



def cmd_tool(args):
    """Run a llama.cpp tool via the bundled binary."""
    from ._bincheck import locate_main_bin
    import shlex

    main_bin = locate_main_bin()
    if not main_bin:
        print("[error] llama-server binary not found")
        sys.exit(1)

    bin_dir = os.path.join(os.path.dirname(main_bin), "..", "bin")
    bin_dir = os.path.abspath(bin_dir)
    if not os.path.isdir(bin_dir):
        print(f"[error] bin directory not found: {bin_dir}")
        sys.exit(1)

    # Map subcommand to the actual llama.cpp binary
    tool_map = {
        "quantize":   "llama-quantize",
        "bench":      "llama-bench",
        "perplexity": "llama-perplexity",
        "cli":        "llama-cli",
        "gguf-split": "llama-gguf-split",
        "tokenize":   "llama-tokenize",
        "gguf":       "llama-gguf",
        "export-lora":"llama-export-lora",
        "imatrix":    "llama-imatrix",
        "embedding":  "llama-embedding",
    }

    tool = tool_map.get(args.command)
    if not tool:
        print(f"[error] unknown tool: {args.command}")
        sys.exit(1)

    tool_path = os.path.join(bin_dir, tool)
    if not os.path.exists(tool_path):
        print(f"[error] {tool} not found in {bin_dir}")
        sys.exit(1)

    cmd = [tool_path]
    # Pass through any positional args to the tool
    if hasattr(args, "args") and args.args:
        cmd += args.args

    print(f"[{args.command}] running {shlex.join(cmd)}")
    env = {**os.environ, "LD_LIBRARY_PATH": bin_dir}
    subprocess.run(cmd, env=env, cwd=os.getcwd())


def cmd_pull(args):
    path = pull(repo_id=args.repo, filename=args.file,
                model_id=getattr(args, "model_id", None) or None)
    print(f"[pull] complete → {path}")


def cmd_ls(_args):
    scan_hf_cache()
    seen   = set()
    models = []
    for search_dir in (HF_CACHE_DIR, CACHE_ROOT):
        if not os.path.isdir(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".gguf"):
                    path = os.path.realpath(os.path.join(root, f))
                    if path not in seen:
                        seen.add(path)
                        models.append((f, os.path.getsize(path) / 1024 ** 3, path))
    if not models:
        print("No models found.")
        return
    print(f"{'MODEL':<55} {'SIZE':>8}")
    print("-" * 65)
    for name, size, _ in sorted(models):
        print(f"{name:<55} {size:>7.2f}G")


def cmd_rm(args):
    from .registry import delete_model as reg_delete
    try:
        rm(args.model_id, getattr(args, "file", None))
    except FileNotFoundError as e:
        print(f"[rm] {e}")
        sys.exit(1)
    # Clean up registry entry (file may have been auto-detected)
    reg_delete(args.model_id)


# ── Misc ──────────────────────────────────────────────────────────────────────

def cmd_ps(_args):
    ps()


def cmd_status(_args):
    status()


def cmd_logs(args):
    logs(n=args.lines)


def cmd_version(_args):
    print(f"llama-Light {VERSION}")


def cmd_info(_args):
    import platform
    cfg   = get_config()
    state = _read_state()
    pid   = state.get("pid")
    print(f"llama-Light : {VERSION}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.platform()}")
    # Binary locations (from robust _bincheck algorithm)
    b = _bincheck_status()
    print(f"[Binaries]")
    for name, path in b.items():
        status_str = str(path) if path else "✗ not found"
        print(f"  {name:<15} {status_str}")
    print(f"Cache root  : {CACHE_ROOT}")
    print(f"Log dir     : {LOG_DIR}")
    print(f"Server PID  : {pid or 'none'} ({'running' if _is_running(pid) else 'stopped'})")
    print(f"Default mdl : {cfg.default_model or 'not set'}")
    print(f"Last model  : {cfg.last_model or 'not set'}")
    # Per-model active settings
    model_name = state.get("model_name", "")
    if model_name:
        from .per_model import get_model_config
        mc = get_model_config(model_name)
        if mc:
            print(f"[Model: {model_name} settings]")
            for k, v in sorted(mc.items()):
                print(f"  {k:<25} {v}")


def cmd_config_show(_args):
    cfg = get_config()
    print(f"{'KEY':<25} VALUE")
    print("-" * 55)
    for k, v in sorted(cfg.all().items()):
        print(f"{k:<25} {v}")


def cmd_config_set(args):
    cfg = get_config()
    model = getattr(args, "model", None)
    if model:
        from .per_model import update_model_config, _model_name_from_path
        from .config import _INT_KEYS, _FLOAT_KEYS, _BOOL_KEYS
        resolved = _resolve_model_arg(args) if model != "auto" else None
        name = _model_name_from_path(resolved) if resolved else model
        value = args.value
        if args.key in _INT_KEYS:
            value = int(value)
        elif args.key in _FLOAT_KEYS:
            value = float(value)
        elif args.key in _BOOL_KEYS:
            value = value.lower() in ("true", "1", "yes", "on")
        update_model_config(name, args.key, value)
        print(f"[config] [{name}] {args.key} = {value}")
    else:
        cfg.set(args.key, args.value)
        print(f"[config] {args.key} = {cfg.get(args.key)}")


def cmd_config_backup(args):
    path = backup()
    if path:
        print(f"[backup] saved → {path}")
    else:
        print("[backup] no config file to backup")


def cmd_config_restore(args):
    if getattr(args, "latest", False):
        restore(None)
    elif getattr(args, "path", None):
        restore(args.path)
    else:
        restore(None)
    print("[restore] complete — restart server to apply")



def cmd_webui(args):
    """Open llama.cpp's built-in web UI in browser."""
    _ensure_server_or_start(args)
    state = _read_state()
    port = state.get("port", 8080)
    host = state.get("host", "127.0.0.1")
    url = f"http://{host}:{port}"
    sys_param = _WEBUI_SYSTEM.replace(" ", "%20")
    url += "?system=" + sys_param
    print(f"[webui] opening {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[webui] could not open browser: {e}")
        print(f"       Open manually: {url}")


def cmd_service(args):
    """Manage the llama-server systemd service.

    Usage:
        llama service              → show unit file, systemd status, model config
        llama service install      → create unit file, enable (backward compat)
        llama service stop         → stop the service
        llama service remove       → stop, disable, uninstall
    """
    from .server import (
        _systemd_active, _systemd_unit_exists, _service_path,
        install_service, uninstall_service,
    )
    from .config import get_config
    import os

    sub = getattr(args, "service_cmd", None)

    # ── Read-only status (bare) ────────────────────────────────────────────────

    if sub is None:
        svc_path = _service_path()
        print(f"[service] systemd unit : {svc_path}")
        print(f"  exists              : {'yes' if os.path.exists(svc_path) else 'no'}")

        if os.path.exists(svc_path):
            with open(svc_path) as f:
                content = f.read()
            indented = "\n".join("    " + line for line in content.splitlines())
            print(f"  content             :\n{indented}")

        try:
            r = subprocess.run(
                ["systemctl", "--user", "status", "llama-server.service"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                print(f"[service] status : active (running)")
            elif r.returncode == 4:
                print("[service] status : inactive (dead)")
            else:
                print(f"[service] status : {r.stdout.strip() or r.stderr.strip()}")
        except Exception:
            print("[service] status : could not query systemctl")

        cfg = get_config()
        print(f"[service] model           : {cfg.default_model or '(not set — run: llama config set default_model <path)'}")
        print(f"  ctx               : {cfg.ctx}")
        print(f"  ngl               : {cfg.ngl}")
        print(f"  threads           : {cfg.threads}")
        return

    # ── install (backward compat) ──────────────────────────────────────────────

    if sub == "install":
        install_service()
        return

    # ── stop ───────────────────────────────────────────────────────────────────

    if sub == "stop":
        if _systemd_active():
            subprocess.run(
                ["systemctl", "--user", "stop", "llama-server.service"],
                check=False,
            )
            print("[service] stopped")
        else:
            print("[service] service not running")
        return

    # ── remove (backward compat) ───────────────────────────────────────────────

    if sub == "remove":
        uninstall_service()
        return


# ── Argument helpers ──────────────────────────────────────────────────────────

def _server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model",         default=None)
    p.add_argument("--host",          default=None)
    p.add_argument("--port",          type=int, default=None)
    p.add_argument("--ctx",           type=int, default=None)
    p.add_argument("--ngl",           type=int, default=None)
    p.add_argument("--no-flash-attn", action="store_true")


def _resolve_server_args(args) -> dict:
    """Merge CLI args with config; CLI wins when explicitly set."""
    cfg = get_config()

    # flash_attn: config stores "auto"/"on"/"off"; treat "auto"/"on" as True
    cfg_flash_raw = cfg.flash_attn                         # str from config
    cfg_flash     = str(cfg_flash_raw).lower() not in ("off", "false", "0")

    return {
        "host":       args.host  if args.host is not None else cfg.host,
        "port":       args.port  if args.port is not None else cfg.port,
        "ctx":        args.ctx   if args.ctx  is not None else cfg.ctx,
        "gpu_layers": args.ngl   if args.ngl  is not None else cfg.ngl,
        # CLI --no-flash-attn overrides config; otherwise use config value
        "flash_attn": False if getattr(args, "no_flash_attn", False) else cfg_flash,
    }


def _gen_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k",       type=int,   default=None, dest="top_k")
    p.add_argument("--max-tokens",  type=int,   default=None, dest="max_tokens")
    p.add_argument("--top-p",       type=float, default=None, dest="top_p")
    p.add_argument("--min-p",       type=float, default=None, dest="min_p")
    p.add_argument("--freq-penalty", type=float, default=None, dest="frequency_penalty")
    p.add_argument("--presence-penalty", type=float, default=None, dest="presence_penalty")
    p.add_argument("--system",      default=None)


def _resolve_gen_args(args) -> dict:
    """Merge CLI gen args with per-model config; CLI > model > global > defaults."""
    from .per_model import get_model_config, _model_name_from_path
    from .config import get_config

    cfg = get_config()
    # Get per-model settings (auto-detected + user-saved)
    model_path = getattr(args, "_model_path", None)
    if model_path:
        model_name = _model_name_from_path(model_path)
        model_cfg = get_model_config(model_name)
    else:
        model_cfg = {}

    def _resolve(key, cli_val, model_default, global_default):
        """Priority: CLI > model config > global config."""
        if cli_val is not None:
            return cli_val
        if key in model_cfg:
            return model_cfg[key]
        return global_default

    return {
        "temperature":         _resolve("temperature", args.temperature, model_cfg.get("temperature"), cfg.get("temperature", 0.7)),
        "top_k":               _resolve("top_k", args.top_k, model_cfg.get("top_k"), cfg.get("top_k", 40)),
        "max_tokens":          _resolve("max_tokens", args.max_tokens, model_cfg.get("max_tokens"), cfg.get("max_tokens", 2048)),
        "top_p":               _resolve("top_p", args.top_p, model_cfg.get("top_p"), 0.95),
        "min_p":               _resolve("min_p", args.min_p, model_cfg.get("min_p"), 0.05),
        "frequency_penalty":   _resolve("frequency_penalty", args.frequency_penalty, model_cfg.get("frequency_penalty"), 0.0),
        "presence_penalty":    _resolve("presence_penalty", args.presence_penalty, model_cfg.get("presence_penalty"), 0.0),
    }



def cmd_config_list_backups(args):
    backups = list_backups()
    if not backups:
        print('No backups found.')
        return
    print(f"{'PATH':<60} {'SIZE':>8} {'MODIFIED':>14}")
    print("-" * 90)
    for path, size, mtime in backups:
        from datetime import datetime
        dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        print(f"{path:<60} {size:>7.1f}KB {dt:>14}")


def cmd_setup(args):
    '''Download/verify llama.cpp binaries.'''
    from .__init__ import LLAMA_CPP_VERSION
    status = check_version(LLAMA_CPP_VERSION)
    if status == 'up_to_date':
        print(f'[setup] llama.cpp {LLAMA_CPP_VERSION} binaries already installed')
        return
    print(f'[setup] installing llama.cpp {LLAMA_CPP_VERSION}...')
    cache_dir, server = ensure_binaries(LLAMA_CPP_VERSION)
    if server:
        print(f'[setup] installed → {server}')
        print(f'  cache: {cache_dir}')
    else:
        print('[setup] ERROR: failed to install binaries', file=sys.stderr)
        sys.exit(1)


def cmd_check(args):
    '''Check if llama.cpp binaries are up to date.'''
    from .__init__ import LLAMA_CPP_VERSION
    status = check_version(LLAMA_CPP_VERSION)
    if status == 'up_to_date':
        print(f'[check] llama.cpp {LLAMA_CPP_VERSION} — up to date')
    elif status == 'missing':
        print(f'[check] llama.cpp {LLAMA_CPP_VERSION} — not installed')
    elif status == 'outdated':
        print(f'[check] llama.cpp {LLAMA_CPP_VERSION} — cached version is outdated')


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='llama',
        description='llama-Light — Ollama-style llama.cpp wrapper',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # start / restart
    sub.add_parser('start',   help='Start llama-server (systemd)').set_defaults(func=cmd_start)
    sub.add_parser('restart', help='Restart — reloads config.json').set_defaults(func=cmd_restart)

    # hidden internal launcher — invoked only by systemd's ExecStart
    p_run = sub.add_parser('_run', help=argparse.SUPPRESS)
    _server_args(p_run)
    p_run.set_defaults(func=cmd_run_server)

    # stop / kill
    sub.add_parser('stop', help='Stop (systemd)').set_defaults(func=cmd_stop)
    sub.add_parser('kill', help='Force-kill (systemd, SIGKILL)').set_defaults(func=cmd_kill)

    # setup / check
    sub.add_parser('setup', help='Download/verify llama.cpp binaries').set_defaults(func=cmd_setup)
    sub.add_parser('check', help='Check binary version status').set_defaults(func=cmd_check)

    # run — single-shot prompt
    p = sub.add_parser('run', help='One-shot prompt')
    _server_args(p)
    _gen_args(p)
    p.add_argument('--prompt', required=True)
    p.set_defaults(func=cmd_run)

    # chat — interactive loop
    p = sub.add_parser('chat', help='Interactive chat loop')
    _server_args(p)
    _gen_args(p)
    p.set_defaults(func=cmd_chat)

    # hermes
    p = sub.add_parser('hermes', help='Launch Hermes TUI wired to local model')
    _server_args(p)
    _gen_args(p)
    p.add_argument('--prompt', default=None)
    p.set_defaults(func=cmd_hermes)

    # hermes-desktop
    p = sub.add_parser('hermes-desktop', help='Launch Hermes Electron desktop app')
    _server_args(p)
    p.set_defaults(func=cmd_hermes_desktop)

    # claude
    p = sub.add_parser('claude', help='Launch Claude Code CLI with local model')
    _server_args(p)
    _gen_args(p)
    p.add_argument('--prompt', default=None)
    p.set_defaults(func=cmd_claude)

    # pull / ls / rm
    p = sub.add_parser('pull', help='Download a GGUF from Hugging Face')
    p.add_argument('--repo',     required=True)
    p.add_argument('--file',     required=True)
    p.add_argument('--model-id', default=None, dest='model_id')
    p.set_defaults(func=cmd_pull)

    sub.add_parser('ls', help='List downloaded models').set_defaults(func=cmd_ls)

    p = sub.add_parser('rm', help='Remove a model')
    p.add_argument('model_id')
    p.add_argument('--file', default=None)
    p.set_defaults(func=cmd_rm)

    # server info
    sub.add_parser('ps',     help='Show running server table').set_defaults(func=cmd_ps)
    sub.add_parser('status', help='Show server status').set_defaults(func=cmd_status)

    p = sub.add_parser('logs', help='Tail server log')
    p.add_argument('--lines', '-n', type=int, default=40)
    p.set_defaults(func=cmd_logs)

    sub.add_parser('version').set_defaults(func=cmd_version)
    sub.add_parser('info',    help='Show llama-Light environment info').set_defaults(func=cmd_info)

    # webui
    sub.add_parser('webui', help='Open llama.cpp web UI in browser').set_defaults(func=cmd_webui)

    # config
    p_cfg = sub.add_parser('config', help='Get/set configuration')
    cfg_sub = p_cfg.add_subparsers(dest='config_cmd', required=True)
    cfg_sub.add_parser('show').set_defaults(func=cmd_config_show)
    cfg_sub.add_parser('backup').set_defaults(func=cmd_config_backup)
    p_restore = cfg_sub.add_parser('restore')
    p_restore.add_argument('--latest', action='store_true', default=False)
    p_restore.add_argument('--path', default=None)
    p_restore.set_defaults(func=cmd_config_restore)
    cfg_sub.add_parser('list-backups').set_defaults(func=cmd_config_list_backups)
    p_set = cfg_sub.add_parser('set')
    p_set.add_argument('key')
    p_set.add_argument('value')
    p_set.add_argument('--model', default=None)
    p_set.set_defaults(func=cmd_config_set)

    # tools — llama.cpp CLI wrappers
    for cmd, desc in [
        ('quantize',   'Quantize model'),
        ('bench',      'Benchmark throughput'),
        ('perplexity', 'Perplexity test'),
        ('cli',        'Interactive CLI'),
        ('gguf-split', 'Split model into shards'),
        ('tokenize',   'Tokenize text'),
        ('gguf',       'Inspect GGUF header'),
        ('export-lora','Export LoRA adapter'),
        ('imatrix',    'Compute importance matrix'),
        ('embedding',  'Run embedding model'),
    ]:
        p = sub.add_parser(cmd, help=desc)
        p.add_argument('args', nargs=argparse.REMAINDER, help='Pass-through to llama.cpp tool')
        p.set_defaults(func=cmd_tool)

    # ── Service Management ───────────────────────────────────────────────────

    p_svc = sub.add_parser('service', help='Manage systemd user service')
    svc_sub = p_svc.add_subparsers(dest='service_cmd', required=False)
    svc_sub.add_parser('install', help='Install service unit file').set_defaults(func=cmd_service)
    svc_sub.add_parser('stop', help='Stop the service').set_defaults(func=cmd_service)
    svc_sub.add_parser('remove', help='Uninstall service').set_defaults(func=cmd_service)
    p_svc.set_defaults(func=cmd_service)

    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except RuntimeError as e:
        print(f'[error] {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\n[interrupted]')
        sys.exit(0)