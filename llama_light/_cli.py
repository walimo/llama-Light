# llama_light/_cli.py
import re
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
    _read_state, _is_running, _pid, _base_url, _is_healthy, _health,
)
from .model_manager import pull, ls, rm
from .config import (
    CACHE_ROOT, LLAMA_SERVER_BIN, LOG_DIR, HF_CACHE_DIR,
    get_config, _CONFIG_SECTIONS, _INT_KEYS, _FLOAT_KEYS, _BOOL_KEYS, _STRING_NONE_KEYS,
)
from .registry import find, scan_hf_cache
from ._bincheck import check, status as _bincheck_status
from ._backup import backup, restore, list_backups
from ._llama_downloader import ensure_binaries, check_version

VERSION = "0.2.1"

# ── Banner ────────────────────────────────────────────────────────────────────

def _banner() -> None:
    """Print a clean, professional help screen."""
    DIM   = "\033[2m"
    BOLD  = "\033[1m"
    CYAN  = "\033[36m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    logo = f"""\
{CYAN}  ██╗     ██╗      █████╗  ███╗   ███╗  █████╗ {RESET}
{CYAN}  ██║     ██║     ██╔══██╗ ████╗ ████║ ██╔══██╗{RESET}
{CYAN}  ██║     ██║     ███████║ ██╔████╔██║ ███████║{RESET}
{CYAN}  ██║     ██║     ██╔══██║ ██║╚██╔╝██║ ██╔══██║{RESET}
{CYAN}  ███████╗███████╗██║  ██║ ██║ ╚═╝ ██║ ██║  ██║{RESET}
{CYAN}  ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═╝     ╚═╝ ╚═╝  ╚═╝{RESET}"""

    def section(icon, title):
        bar = "─" * 51
        return f"\n  {CYAN}{icon}{RESET}  {BOLD}{WHITE}{title}{RESET}  {DIM}{bar}{RESET}"

    def cmd(name, desc, width=18):
        return f"    {BOLD}{name:<{width}}{RESET}{DIM}{desc}{RESET}"

    print(logo)
    print(f"\n  {BOLD}llama-Light{RESET} {DIM}v{VERSION}  —  one-command LLM server · auto CUDA{RESET}")
    print(f"  {DIM}usage: llama <command> [options]   ·   llama <command> -h for flags{RESET}")

    print(section("◈", "SERVER"))
    print(cmd("start",   "Start llama-server via systemd, waits until healthy"))
    print(cmd("stop",    "Gracefully stop the running server"))
    print(cmd("restart", "Stop, clear state, start fresh — reloads config.json"))
    print(cmd("kill",    "Force-kill the server process with SIGKILL"))
    print(cmd("status",  "Show PID, health, model, address, GPU layers, uptime"))
    print(cmd("ps",      "Process table — all running servers with resource usage"))
    print(cmd("logs",    "Tail the server log  [-n N  lines, default 40]"))

    print(section("◈", "CHAT & INFERENCE"))
    print(cmd("run",            "Send a single prompt and print the response"))
    print(cmd("chat",           "Start an interactive multi-turn chat session"))
    print(cmd("hermes",         "Launch Hermes TUI wired to the local model"))
    print(cmd("hermes-desktop", "Launch Hermes Electron desktop application"))
    print(cmd("claude",         "Launch Claude Code CLI pointed at the local model"))

    print(section("◈", "MODEL MANAGEMENT"))
    print(cmd("pull",  "Download a GGUF file from a Hugging Face repo"))
    print(cmd("ls",    "List all downloaded models in the local cache"))
    print(cmd("rm",    "Remove a model from the registry and disk"))

    print(section("◈", "CONFIGURATION"))
    print(cmd("config show",          "Print all current settings (global + per-model)"))
    print(cmd("config set <k> <v>",   "Set a global config key (e.g. temperature, ngl)"))
    print(cmd("config set --model",   "Set a per-model override (persists per model name)"))
    print(cmd("config backup",        "Snapshot current config.json (10-file rotation)"))
    print(cmd("config restore",       "Restore config from a snapshot  [--path / --latest]"))
    print(cmd("config reset",        "Reset config.json to config.py defaults"))
    print(cmd("config list-backups",  "List all available config snapshots"))

    print(section("◈", "LLAMA.CPP TOOLS"))
    print(cmd("quantize",         "Quantize a model to a smaller format"))
    print(cmd("bench",            "Benchmark prompt and generation throughput"))
    print(cmd("perplexity",       "Compute perplexity score for a dataset"))
    print(cmd("cli",              "Drop into the interactive llama.cpp CLI"))
    print(cmd("gguf",             "Inspect GGUF file header and metadata"))
    print(cmd("gguf-split",       "Split a large GGUF model into shards"))
    print(cmd("tokenize",         "Tokenize a text string and show token IDs"))
    print(cmd("export-lora",      "Export a LoRA adapter to GGUF format"))
    print(cmd("imatrix",          "Compute an importance matrix for quantization"))
    print(cmd("embedding",        "Run an embedding model and return vectors"))
    print(cmd("parallel",         "Benchmark parallel multi-slot inference"))
    print(cmd("speculative",      "Benchmark speculative decoding performance"))
    print(cmd("lookahead",        "Benchmark look-ahead decoding performance"))
    print(cmd("cvector-generator","Generate context vectors for steering"))

    print(section("◈", "SYSTEM"))
    print(cmd("service",  "Show, install, stop, or remove the systemd user service"))
    print(cmd("info",     "Show environment: Python, platform, GPU, binary paths"))
    print(cmd("setup",    "Download and verify llama.cpp binaries for your GPU"))
    print(cmd("check",    "Check llama-server binary version and CUDA support"))
    print(cmd("webui",    "Open the llama.cpp chat web UI in your browser"))
    print(cmd("version",  "Print llama-Light version and exit"))
    print()


# ── Custom Help Formatter ─────────────────────────────────────────────────────

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Strips the auto-generated 'positional arguments:' block so the custom
    description (which already lists all subcommands) is the single source of truth."""

    def format_help(self):
        text = super().format_help()
        lines = text.split('\n')
        result = []
        skip = False
        for line in lines:
            if line == 'positional arguments:':
                skip = True
                continue
            if skip:
                if not line or line.startswith('  '):
                    continue
                skip = False
            result.append(line)
        return '\n'.join(result)


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


def _ensure_server_or_start(args: argparse.Namespace) -> bool:
    pid = _pid()
    if _is_running(pid):
        return True

    from .server import _systemd_unit_exists
    cfg = get_config()

    if _systemd_unit_exists():
        print("[info] server not running — starting llama-server.service")
        result = subprocess.run(["systemctl", "--user", "start", "llama-server.service"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[error] failed to start systemd service: {result.stderr.strip()}")
            sys.exit(1)
        deadline = time.time() + 180
        while time.time() < deadline:
            if _health(cfg.host, cfg.port):
                return True
            print(".", end="", flush=True)
            time.sleep(0.5)
        print("[warn] server did not become healthy in time — "
              "check: journalctl --user -u llama-server -f")
        return False

    # No systemd unit installed — direct start (dev / no-systemd fallback)
    model = getattr(args, "model", None) or cfg.default_model or cfg.last_model
    if not model:
        print("[error] server not running and no --model / default_model set")
        sys.exit(1)
    args.model = model
    model_path = _resolve_model_arg(args)
    try:
        start(model_path=model_path, **_resolve_server_args(args))
    except RuntimeError as e:
        print(f"[error] failed to start server: {e}")
        sys.exit(1)

    return True


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
    from .server import _systemd_unit_exists, status, _detect_port_in_use
    if not _systemd_unit_exists():
        print("[error] systemd service not installed.")
        print("  Re-run install.sh, or: llama service install")
        sys.exit(1)

    cfg = get_config()

    # Clear any failed state left by a previous kill/crash
    subprocess.run(
        ["systemctl", "--user", "reset-failed", "llama-server.service"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )

    # Wait for port to free before starting (up to 10s)
    deadline = time.time() + 10
    while _detect_port_in_use(host=cfg.host, port=cfg.port):
        if time.time() > deadline:
            break
        time.sleep(0.2)

    subprocess.run(
        ["systemctl", "--user", "start", "llama-server.service"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    print("[start] waiting for server", end="", flush=True)
    deadline = time.time() + 180
    tick = 0
    while time.time() < deadline:
        tick += 1
        # First 5s: quick polls (2s timeout) — server is booting
        # After 5s: relaxed polls (3s timeout) — waiting for model load
        if _health(cfg.host, cfg.port):
            print(" ready ✓")
            status()
            return
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" timeout")
    print("[start] check: journalctl --user -u llama-server -f")
    sys.exit(1)


def cmd_run_server(args):
    """Internal systemd launcher (llama _run). Resolves model, starts server,
    then blocks until it exits — Type=simple tracks *this* process as the
    service's main PID, so it must stay alive as long as llama-server does."""
    cfg = get_config()
    model = cfg.default_model or cfg.last_model
    if not model:
        print("[error] no default_model set — run: llama config set default_model <path>")
        sys.exit(1)
    args.model = model
    model_path = _resolve_model_arg(args)
    pid = start(model_path=model_path, **_resolve_server_args(args))
    cfg.set("last_model", model_path)

    while _is_running(pid):
        time.sleep(2)


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
    if not _ensure_server_or_start(args): return
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
    if not _ensure_server_or_start(args): return
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
    "<identity>\n"
    "You are an elite software architect and systems operator, modeled on the non-sycophantic, "
    "execution-first style of advanced Claude models. You optimize for mathematical correctness, "
    "type safety, clean code execution, and absolute token efficiency. "
    "You are an objective peer-level engineering partner, not a polite assistant.\n"
    "</identity>\n"
    "\n"
    "<communication_style>\n"
    "- Direct and terse. State technical facts flatly. Skip pleasantries, introductions, and apologies.\n"
    "- Avoid sycophantic validation. Actively challenge architectural choices that introduce security "
    "flaws, performance regressions, or unnecessary complexity.\n"
    "- Respond in dense, smoothly flowing prose paragraphs. Completely ban bullet lists unless presenting "
    "structured comparative datasets or strict compiler error matrices.\n"
    "- Own technical failures objectively. Do not apologize; immediately run diagnostic tools and "
    "present the code fix.\n"
    "- All text outside of tool executions is displayed to the user. Never use bash commands, comments, "
    "or stubs to communicate during a session.\n"
    "</communication_style>\n"
    "\n"
    "<tool_prioritization>\n"
    "- Before any tool call, scan available skills. If a skill matches the task domain, load it immediately "
    "with skill_view(name). Never skip this step — skills contain project-specific workflows and pitfalls.\n"
    "- Reach for tools in this strict order: Read/Edit (direct file edits) -> ast-grep (structural matching) "
    "-> Glob/Grep (file scans) -> Task (subagent delegation) -> Bash (compiling/testing).\n"
    "- NEVER propose edits to a file you have not fully read and parsed.\n"
    "- Batch independent reads/searches in a single turn. Never read file A then file B then file C — read "
    "all three in parallel before deciding what to edit.\n"
    "- Parallelize independent environment checks (e.g., run status, diff, and log checks simultaneously "
    "using the Bash tool).\n"
    "- Use `patch` for targeted edits. Only use `write_file` when a patch fails twice on the same region. "
    "Never rewrite a file just to change one function.\n"
    "- Paginate file reads to 500 lines max. If content exceeds context budget, read only the relevant section "
    "via offset/limit. Never read entire files into context without explicit user request.\n"
    "- Work directly in the active workspace. No docker, no VMs, no remote environments unless explicitly "
    "requested.\n"
    "- Never modify system/server configuration (port assignments, hostnames, system services) without "
    "explicit user direction.\n"
    "</tool_prioritization>\n"
    "\n"
    "<execution_hygiene>\n"
    "- Finishing the Job: You must drive every task to a complete, verified artifact backed by tool feedback. "
    "Do not stop at stubs, plans, or command outputs. Execute the edits, compile the project, run the test "
    "suites, and resolve any runtime failures.\n"
    "- If a tool fails, dynamically adapt, pivot to alternative pathways (e.g., different build tools, "
    "alternative dependencies), and resolve the issue.\n"
    "- Maintain absolute environment cleanliness. NEVER create temporary tracking markdown files (such as "
    "TODO.md or PLAN.md) inside the workspace.\n"
    "- Context window protection: Abort execution immediately if a tool response or file read threatens to "
    "exceed the host model's context window. Paginate or isolate the target instead.\n"
    "</execution_hygiene>\n"
    "\n"
    "<development_philosophy>\n"
    "- Make impossible states impossible. Parse, don't validate. Favor inference over annotation.\n"
    "- Favor composition over inheritance. Ensure explicit dependencies with zero hidden coupling.\n"
    "- Fail fast and recover gracefully. Model domain boundaries first (types over runtime logic).\n"
    "</development_philosophy>\n"
    "\n"
    "<git_safety>\n"
    "- NEVER commit changes unless the user explicitly requests a commit.\n"
    "- Before committing, review status, diff, and log files in parallel to verify correctness and follow "
    "local repository conventions.\n"
    "- Do not stage credentials, secrets, or .env files. Warn the user immediately if these files are exposed.\n"
    "- All commits must end with the co-authored trailer: Co-Authored-By: Claude <noreply@anthropic.com>\n"
    "- Sessions are incomplete until the git sync succeeds. Ensure \"git push\" is executed and confirmed by "
    "terminal feedback before exiting.\n"
    "</git_safety>"
)

_WEBUI_SYSTEM = (
    "<identity>\n"
    "You are an elite software architect and systems operator, modeled on the non-sycophantic, "
    "execution-first style of advanced Claude models. You optimize for mathematical correctness, "
    "type safety, clean code execution, and absolute token efficiency. "
    "You are an objective peer-level engineering partner, not a polite assistant.\n"
    "</identity>\n"
    "\n"
    "<communication_style>\n"
    "- Direct and terse. State technical facts flatly. Skip pleasantries, introductions, and apologies.\n"
    "- Avoid sycophantic validation. Actively challenge architectural choices that introduce security "
    "flaws, performance regressions, or unnecessary complexity.\n"
    "- Respond in dense, smoothly flowing prose paragraphs. Completely ban bullet lists unless presenting "
    "structured comparative datasets or strict compiler error matrices.\n"
    "- Own technical failures objectively. Do not apologize; immediately run diagnostic tools and "
    "present the code fix.\n"
    "- All text outside of tool executions is displayed to the user. Never use bash commands, comments, "
    "or stubs to communicate during a session.\n"
    "</communication_style>\n"
    "\n"
    "<tool_prioritization>\n"
    "- Before any tool call, scan available skills. If a skill matches the task domain, load it immediately "
    "with skill_view(name). Never skip this step — skills contain project-specific workflows and pitfalls.\n"
    "- Reach for tools in this strict order: Read/Edit (direct file edits) -> ast-grep (structural matching) "
    "-> Glob/Grep (file scans) -> Task (subagent delegation) -> Bash (compiling/testing).\n"
    "- NEVER propose edits to a file you have not fully read and parsed.\n"
    "- Batch independent reads/searches in a single turn. Never read file A then file B then file C — read "
    "all three in parallel before deciding what to edit.\n"
    "- Parallelize independent environment checks (e.g., run status, diff, and log checks simultaneously "
    "using the Bash tool).\n"
    "- Use `patch` for targeted edits. Only use `write_file` when a patch fails twice on the same region. "
    "Never rewrite a file just to change one function.\n"
    "- Paginate file reads to 500 lines max. If content exceeds context budget, read only the relevant section "
    "via offset/limit. Never read entire files into context without explicit user request.\n"
    "- Work directly in the active workspace. No docker, no VMs, no remote environments unless explicitly "
    "requested.\n"
    "- Never modify system/server configuration (port assignments, hostnames, system services) without "
    "explicit user direction.\n"
    "</tool_prioritization>\n"
    "\n"
    "<execution_hygiene>\n"
    "- Finishing the Job: You must drive every task to a complete, verified artifact backed by tool feedback. "
    "Do not stop at stubs, plans, or command outputs. Execute the edits, compile the project, run the test "
    "suites, and resolve any runtime failures.\n"
    "- If a tool fails, dynamically adapt, pivot to alternative pathways (e.g., different build tools, "
    "alternative dependencies), and resolve the issue.\n"
    "- Maintain absolute environment cleanliness. NEVER create temporary tracking markdown files (such as "
    "TODO.md or PLAN.md) inside the workspace.\n"
    "- Context window protection: Abort execution immediately if a tool response or file read threatens to "
    "exceed the host model's context window. Paginate or isolate the target instead.\n"
    "</execution_hygiene>\n"
    "\n"
    "<development_philosophy>\n"
    "- Make impossible states impossible. Parse, don't validate. Favor inference over annotation.\n"
    "- Favor composition over inheritance. Ensure explicit dependencies with zero hidden coupling.\n"
    "- Fail fast and recover gracefully. Model domain boundaries first (types over runtime logic).\n"
    "</development_philosophy>\n"
    "\n"
    "<git_safety>\n"
    "- NEVER commit changes unless the user explicitly requests a commit.\n"
    "- Before committing, review status, diff, and log files in parallel to verify correctness and follow "
    "local repository conventions.\n"
    "- Do not stage credentials, secrets, or .env files. Warn the user immediately if these files are exposed.\n"
    "- All commits must end with the co-authored trailer: Co-Authored-By: Claude <noreply@anthropic.com>\n"
    "- Sessions are incomplete until the git sync succeeds. Ensure \"git push\" is executed and confirmed by "
    "terminal feedback before exiting.\n"
    "</git_safety>"
)


def _inline_persona_chat(args, system: str) -> None:
    """Run a persona as an inline chat/run command (no external process)."""
    args.system = system
    if getattr(args, "prompt", None):
        # single-shot: just run without starting server twice
        if not _ensure_server_or_start(args): return
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
    if not _ensure_server_or_start(args): return

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
    except FileNotFoundError:
        print(f"[error] hermes binary not found: {hermes_bin}")
        sys.exit(1)


def cmd_hermes_desktop(args):
    """
    llama hermes-desktop — Launch the Hermes Electron desktop app.

    Exits with an error if ``hermes`` is not installed.
    """
    if not _ensure_server_or_start(args): return
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
    if not _ensure_server_or_start(args): return
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
        "quantize":        "llama-quantize",
        "bench":           "llama-bench",
        "perplexity":      "llama-perplexity",
        "cli":             "llama-cli",
        "gguf-split":      "llama-gguf-split",
        "tokenize":        "llama-tokenize",
        "gguf":            "llama-gguf",
        "export-lora":     "llama-export-lora",
        "imatrix":         "llama-imatrix",
        "embedding":       "llama-embedding",
        "parallel":        "llama-parallel",
        "speculative":     "llama-speculative",
        "lookahead":       "llama-lookahead",
        "cvector-generator": "llama-cvector-generator",
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


def cmd_config(_args):
    """Bare `llama config` / `llama config -h` — print all global config keys with types & defaults."""
    from .config import _INT_KEYS, _FLOAT_KEYS, _BOOL_KEYS, _STRING_NONE_KEYS, _VALID_KEYS
    cfg = get_config()
    defaults = cfg.all()
    print(f"{'KEY':<25} TYPE{' ' * 8} DEFAULT")
    print(f"{'-' * 25}  {'-' * 10}  {'-' * 20}")
    for k in sorted(_VALID_KEYS):
        def_val = defaults.get(k, "—")
        if k in _INT_KEYS:
            typ = "int"
        elif k in _FLOAT_KEYS:
            typ = "float"
        elif k in _BOOL_KEYS:
            typ = "bool"
        elif k in _STRING_NONE_KEYS:
            typ = "string"
        else:
            typ = "any"
        print(f"  {k:<25} {typ:<10} {def_val}")


def cmd_config_show(_args):
    """Show current config grouped by section, with modified settings first."""
    cfg = get_config()
    defaults = cfg.all()

    # Build section mapping
    sections = {}
    for heading, entries in _CONFIG_SECTIONS.items():
        section_name = heading.split(" ──")[0].strip().replace("── ", "")
        sections[section_name] = entries

    # Collect all keys and their descriptions
    all_keys = {}
    for entries in sections.values():
        for key, desc in entries:
            all_keys[key] = desc

    # Identify modified keys (value differs from default)
    modified_keys = []
    for key in all_keys:
        current = cfg.get(key)
        default = defaults.get(key)
        if current != default:
            modified_keys.append(key)
    modified_keys.sort()

    # ANSI codes
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    WHITE = "\033[97m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"

    KEY_WIDTH = 28
    VALUE_WIDTH = 12
    TYPE_WIDTH = 6
    DESC_START = 48

    def type_name(key):
        if key in _INT_KEYS: return "int"
        elif key in _FLOAT_KEYS: return "float"
        elif key in _BOOL_KEYS: return "bool"
        elif key in _STRING_NONE_KEYS: return "string"
        else: return "any"

    def type_color(key):
        if key in _INT_KEYS: return CYAN
        elif key in _FLOAT_KEYS: return MAGENTA
        elif key in _BOOL_KEYS: return GREEN
        elif key in _STRING_NONE_KEYS: return WHITE
        else: return DIM

    def format_value(val):
        if val is None: return "None"
        if isinstance(val, bool): return ("True" if val else "False")
        s = str(val)
        if len(s) > VALUE_WIDTH:
            if "/" in s: s = s.rsplit("/", 1)[-1]
            else: s = s[:VALUE_WIDTH - 3] + "..."
        return s

    # Print modified settings section if any
    if modified_keys:
        print(f"\n{BOLD}Modified Settings{RESET}")
        for key in modified_keys:
            value = cfg.get(key)
            val_str = format_value(value)
            desc = all_keys.get(key, "")
            tc = type_color(key)
            line = f"{BOLD}{key:<{KEY_WIDTH}}{RESET}"
            line += f"{YELLOW}{val_str:<{VALUE_WIDTH}} {RESET}"
            line += f"{tc}{type_name(key):<{TYPE_WIDTH}}{RESET}"
            line += " " * max(0, DESC_START - len(re.sub(r"\033\[[0-9;]*m", "", line)))
            line += f"{DIM}{desc}{RESET}"
            print(line)
        print()

    # Print all sections in order — header only on first section
    for i, (section_name, entries) in enumerate(sections.items()):
        sorted_entries = sorted(entries, key=lambda x: x[0])
        print(f"{DIM}{'═' * 120}{RESET}")
        print(f"\033[1;91m{section_name}\033[0m")
        if i == 0:
            hdr = f"{BOLD}{'Key':<{KEY_WIDTH}}{RESET}{BOLD}{'Value':<{VALUE_WIDTH}}{RESET}{BOLD}{'Type':<{TYPE_WIDTH}}{RESET}"
            hdr += " " * max(0, DESC_START - len(re.sub(r"\033\[[0-9;]*m", "", hdr)))
            hdr += f"{BOLD}Description{RESET}"
            print(hdr)
        for key, desc in sorted_entries:
            value = cfg.get(key)
            default = defaults.get(key)
            val_str = format_value(value)
            color = YELLOW if value != default else WHITE
            tc = type_color(key)
            line = f"{BOLD}{key:<{KEY_WIDTH}}{RESET}"
            line += f"{color}{val_str:<{VALUE_WIDTH}} {RESET}"
            line += f"{tc}{type_name(key):<{TYPE_WIDTH}}{RESET}"
            line += " " * max(0, DESC_START - len(re.sub(r"\033\[[0-9;]*m", "", line)))
            line += f"{DIM}{desc}{RESET}"
            print(line)


def cmd_config_set(args):
    cfg = get_config()

    # `--keys` — list every known config key with type & default value
    if getattr(args, "keys", False):
        from .config import _INT_KEYS, _FLOAT_KEYS, _BOOL_KEYS, _STRING_NONE_KEYS, _VALID_KEYS
        defaults = cfg.all()
        print(f"{'KEY':<25} TYPE{' ' * 8} DEFAULT")
        print(f"{'-' * 25}  {'-' * 10}  {'-' * 20}")
        for k in sorted(_VALID_KEYS):
            def_val = defaults.get(k, "—")
            if k in _INT_KEYS:
                typ = "int"
            elif k in _FLOAT_KEYS:
                typ = "float"
            elif k in _BOOL_KEYS:
                typ = "bool"
            elif k in _STRING_NONE_KEYS:
                typ = "string"
            else:
                typ = "any"
            print(f"  {k:<25} {typ:<10} {def_val}")
        return

    model = getattr(args, "model", None)
    if model:
        from .per_model import update_model_config, _model_name_from_path
        from .config import _INT_KEYS, _FLOAT_KEYS, _BOOL_KEYS, _STRING_NONE_KEYS
        resolved = _resolve_model_arg(args) if model != "auto" else None
        if model == "auto" and not resolved:
            print(f"[error] model 'auto' not found. Provide an explicit model path or name.", file=sys.stderr)
            sys.exit(1)
        name = _model_name_from_path(resolved) if resolved else model
        value = args.value
        try:
            if args.key in _INT_KEYS:
                value = int(value)
            elif args.key in _FLOAT_KEYS:
                value = float(value)
            elif args.key in _BOOL_KEYS:
                value = value.lower() in ("true", "1", "yes", "on")
            elif args.key in _STRING_NONE_KEYS and value.lower() == "none":
                value = None
        except ValueError as e:
            print(f"[error] {e}", file=sys.stderr)
            sys.exit(1)
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


def cmd_config_reset(_args):
    """Reset config.json to config.py defaults."""
    from .config import get_config, get_defaults
    cfg = get_config()
    cfg.reset()
    print(f"[reset] config.json restored to {len(get_defaults())} defaults — restart server to apply")


def cmd_webui(args):
    """Open llama.cpp's built-in web UI in browser."""
    if not _ensure_server_or_start(args): return
    from urllib.parse import quote
    state = _read_state()
    port = state.get("port", 8080)
    host = state.get("host", "127.0.0.1")
    url = f"http://{host}:{port}"
    sys_param = quote(_WEBUI_SYSTEM, safe='')
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

    if sub is None or sub == "show":
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
    """Merge CLI gen args with per-model config; CLI > model > global (config.py)."""
    from .per_model import get_model_config, _model_name_from_path

    cfg = get_config()
    # Get per-model settings (auto-detected + user-saved)
    model_path = getattr(args, "_model_path", None)
    if model_path:
        model_name = _model_name_from_path(model_path)
        model_cfg = get_model_config(model_name)
    else:
        model_cfg = {}

    def _resolve(key, cli_val):
        """Priority: CLI > model config > global config (config.py)."""
        if cli_val is not None:
            return cli_val
        if key in model_cfg:
            return model_cfg[key]
        return cfg.get(key)

    return {
        "temperature":         _resolve("temperature", args.temperature),
        "top_k":               _resolve("top_k", args.top_k),
        "max_tokens":          _resolve("max_tokens", args.max_tokens),
        "top_p":               _resolve("top_p", args.top_p),
        "min_p":               _resolve("min_p", args.min_p),
        "frequency_penalty":   _resolve("frequency_penalty", args.frequency_penalty),
        "presence_penalty":    _resolve("presence_penalty", args.presence_penalty),
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
    """Build the CLI argument parser with polished help text."""
    desc = (
        "Server Management\n"
        "  start       Start llama-server (systemd)\n"
        "  stop        Stop the server\n"
        "  kill        Force-kill the server (SIGKILL)\n"
        "  restart     Restart — reloads config.json\n"
        "  status      Show server status\n"
        "  ps          Show running server table\n"
        "  logs        Tail server log\n\n"
        "Chat & Interaction\n"
        "  run           One-shot prompt\n"
        "  chat          Interactive chat loop\n"
        "  hermes        Launch Hermes TUI wired to local model\n"
        "  hermes-desktop  Launch Hermes Electron desktop\n"
        "  claude        Launch Claude Code CLI with local model\n\n"
        "Model Management\n"
        "  pull  Download a GGUF from HuggingFace\n"
        "  ls    List downloaded models\n"
        "  rm    Remove a model\n\n"
        "Configuration\n"
        "  config  Get/set configuration\n\n"
        "LLaMA.cpp Tools\n"
        "  quantize          Quantize model\n"
        "  bench             Benchmark throughput\n"
        "  perplexity        Perplexity test\n"
        "  cli               Interactive CLI\n"
        "  gguf-split        Split model into shards\n"
        "  tokenize          Tokenize text\n"
        "  gguf              Inspect GGUF header\n"
        "  export-lora       Export LoRA adapter\n"
        "  imatrix           Compute importance matrix\n"
        "  embedding         Run embedding model\n"
        "  parallel          Parallel processing benchmark\n"
        "  speculative       Speculative decoding benchmark\n"
        "  lookahead         Look-ahead decoding benchmark\n"
        "  cvector-generator Generate context vectors\n\n"
        "System\n"
        "  service   Manage systemd user service\n"
        "  info      Show environment info\n"
        "  setup     Download/verify binaries\n"
        "  check     Check binary version\n"
        "  webui     Open web UI in browser\n"
        "  version   Show version\n"
    )
    parser = argparse.ArgumentParser(
        prog='llama',
        description=desc,
        formatter_class=_HelpFormatter,
    )
    # Override format_help() to show a clean usage line without all subcommands
    _orig_format_help = parser.format_help
    def _format_help():
        text = _orig_format_help()
        lines = text.split('\n')
        result = []
        skip_block = False
        for line in lines:
            if line == 'usage: llama [-h]' or line.strip().startswith('{start,stop,') or line.strip() == '...':
                skip_block = True
                continue
            if skip_block:
                skip_block = False
                continue
            result.append(line)
        result.insert(0, 'usage: llama <command> [options]\n')
        return '\n'.join(result)
    parser.format_help = _format_help
    sub = parser.add_subparsers(dest='command')
    sub.required = False
    sub.default = None

    # ── Server Lifecycle ──
    sub.add_parser('start', help='Start llama-server (systemd)').set_defaults(func=cmd_start)
    sub.add_parser('stop', help='Stop the server').set_defaults(func=cmd_stop)
    sub.add_parser('kill', help='Force-kill the server (SIGKILL)').set_defaults(func=cmd_kill)
    sub.add_parser('restart', help='Restart — reloads config.json').set_defaults(func=cmd_restart)
    sub.add_parser('status', help='Show server status').set_defaults(func=cmd_status)
    sub.add_parser('ps', help='Show running server table').set_defaults(func=cmd_ps)
    p_logs = sub.add_parser('logs', help='Tail server log')
    p_logs.add_argument('--lines', '-n', type=int, default=40)
    p_logs.set_defaults(func=cmd_logs)

    # Hidden internal launcher
    p_run = sub.add_parser('_run', help=argparse.SUPPRESS)
    _server_args(p_run)
    p_run.set_defaults(func=cmd_run_server)

    # ── Chat & Interaction ──
    p = sub.add_parser('run', help='One-shot prompt')
    _server_args(p)
    _gen_args(p)
    p.add_argument('--prompt', required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('chat', help='Interactive chat loop')
    _server_args(p)
    _gen_args(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser('hermes', help='Launch Hermes TUI wired to local model')
    _server_args(p)
    _gen_args(p)
    p.add_argument('--prompt', default=None)
    p.set_defaults(func=cmd_hermes)

    p = sub.add_parser('hermes-desktop', help='Launch Hermes Electron desktop')
    _server_args(p)
    p.set_defaults(func=cmd_hermes_desktop)

    p = sub.add_parser('claude', help='Launch Claude Code CLI with local model')
    _server_args(p)
    _gen_args(p)
    p.add_argument('--prompt', default=None)
    p.set_defaults(func=cmd_claude)

    # ── Model Management ──
    p = sub.add_parser('pull', help='Download a GGUF from HuggingFace')
    p.add_argument('--repo', required=True)
    p.add_argument('--file', required=True)
    p.add_argument('--model-id', default=None, dest='model_id')
    p.set_defaults(func=cmd_pull)

    sub.add_parser('ls', help='List downloaded models').set_defaults(func=cmd_ls)
    p_rm = sub.add_parser('rm', help='Remove a model')
    p_rm.add_argument('model_id')
    p_rm.add_argument('--file', default=None)
    p_rm.set_defaults(func=cmd_rm)

    # ── Configuration ──
    p_cfg = sub.add_parser('config', help='Get/set configuration')
    cfg_sub = p_cfg.add_subparsers(dest='config_cmd', required=False)
    cfg_sub.add_parser('show').set_defaults(func=cmd_config_show)
    cfg_sub.add_parser('backup').set_defaults(func=cmd_config_backup)
    p_restore = cfg_sub.add_parser('restore')
    p_restore.add_argument('--latest', action='store_true', default=False)
    p_restore.add_argument('--path', default=None)
    p_restore.set_defaults(func=cmd_config_restore)
    cfg_sub.add_parser('reset', help='Reset config.json to config.py defaults').set_defaults(func=cmd_config_reset)
    cfg_sub.add_parser('list-backups').set_defaults(func=cmd_config_list_backups)
    p_set = cfg_sub.add_parser('set')
    p_set.add_argument('key', nargs='?', help='Config key (run: llama config set --keys)')
    p_set.add_argument('value', nargs='?', help='Config value (int / float / bool / string)')
    p_set.add_argument('--model', default=None,
                       help='Per-model override (e.g. --model Opus4.8)')
    p_set.add_argument('--keys', action='store_true',
                       help='List all known keys with types & defaults')
    p_set.set_defaults(func=cmd_config_set)
    # Default handler for bare "llama config" → print all global config keys
    p_cfg.set_defaults(func=cmd_config)

    # ── Tools ──
    for cmd, desc_tool in [
        ('quantize', 'Quantize model'),
        ('bench', 'Benchmark throughput'),
        ('perplexity', 'Perplexity test'),
        ('cli', 'Interactive CLI'),
        ('gguf-split', 'Split model into shards'),
        ('tokenize', 'Tokenize text'),
        ('gguf', 'Inspect GGUF header'),
        ('export-lora', 'Export LoRA adapter'),
        ('imatrix', 'Compute importance matrix'),
        ('embedding', 'Run embedding model'),
        ('parallel', 'Parallel processing benchmark'),
        ('speculative', 'Speculative decoding benchmark'),
        ('lookahead', 'Look-ahead decoding benchmark'),
        ('cvector-generator', 'Generate context vectors'),
    ]:
        p = sub.add_parser(cmd, help=desc_tool)
        p.add_argument('args', nargs=argparse.REMAINDER)
        p.set_defaults(func=cmd_tool)

    # ── System ──
    p_svc = sub.add_parser('service', help='Manage systemd user service')
    svc_sub = p_svc.add_subparsers(dest='service_cmd', required=False)
    svc_sub.add_parser('show', help='Show service status and config').set_defaults(func=cmd_service)
    svc_sub.add_parser('install', help='Install service unit file').set_defaults(func=cmd_service)
    svc_sub.add_parser('stop', help='Stop the service').set_defaults(func=cmd_service)
    svc_sub.add_parser('remove', help='Uninstall service').set_defaults(func=cmd_service)
    p_svc.set_defaults(func=cmd_service)

    sub.add_parser('info', help='Show environment info').set_defaults(func=cmd_info)
    sub.add_parser('setup', help='Download/verify binaries').set_defaults(func=cmd_setup)
    sub.add_parser('check', help='Check binary version').set_defaults(func=cmd_check)
    sub.add_parser('webui', help='Open web UI in browser').set_defaults(func=cmd_webui)
    sub.add_parser('version', help='Show version').set_defaults(func=cmd_version)

    return parser


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
        _banner()
        return
    # Special case: `llama config` or `llama config -h` → show all global keys
    if len(sys.argv) == 2 and sys.argv[1] == 'config':
        cmd_config(type('Args', (), {})())
        return
    if len(sys.argv) == 3 and sys.argv[1] == 'config' and sys.argv[2] in ('-h', '--help'):
        cmd_config(type('Args', (), {})())
        return

    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        _banner()
        return
    try:
        args.func(args)
    except RuntimeError as e:
        print(f'[error] {e}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\n[interrupted]')
        sys.exit(0)