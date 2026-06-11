# llama-Light Pro Finish — Implementation Plan

## Problem
The previous session completed `_backup.py`, `config.py`, and `per_model.py` rewrites.
The next session was interrupted while attempting to rewrite `server.py`, `_cli.py`,
`__init__.py`, `pyproject.toml`, and `README.md`. The `Write` tool was failing with
missing `file_path`/`content` parameters.

Additionally, a critical bug was discovered: `config.py` was missing the `LLAMA_SERVER_BIN`
constant that `_bincheck.py`, `_cli.py`, and `server.py` all import. This was fixed
by adding it back to config.py line 66.

---

## Task 1: Rewrite `server.py` (lines 1-652 → ~400 lines)

### Remove
- **`chat_completion` function** (lines 457-514) — dead code, never called from CLI
- **`n_parallel` parameter** from `start()` (line 75) — unused, passed as `--parallel` from config
- **Redundant imports in `ps()`** (lines 347-349): `import os as _os`, `import subprocess as _sub`, `import time` — these are already imported at top of file
- **Dead `status()` function** — check if used; if not, remove

### Fix
- **Binary resolution**: Replace hardcoded `LLAMA_SERVER_BIN` checks with `_bincheck.locate_main_bin()`
  - In `start()` (line 81-85): use `bin_path = _bincheck.locate_main_bin()` instead of checking `LLAMA_SERVER_BIN`
  - In `start()` (line 119): use `bin_path` variable from locate_main_bin()
  - Better error message if binary not found
- **Service install error handling** (lines 638-640, 646-651): wrap `subprocess.run(check=True)` calls in try/except — `CalledProcessError` is unhandled when systemctl fails
- **Graceful shutdown**: Add signal handler in `start()` for SIGTERM/SIGINT to cleanly stop the server

### Keep
- `chat_messages` function (lines 519-587) — actively used by CLI
- `ps()`, `status()`, `logs()`, `restart()` — all actively used
- `_backup.py` import — check if needed in server.py

### New imports at top:
```python
from ._bincheck import locate_main_bin
```

### New code in `start()`:
```python
# Before existing binary check:
bin_path = locate_main_bin()
if not bin_path:
    raise RuntimeError(
        "llama-server binary not found.\n"
        "Compile llama.cpp: cd ~/llama.cpp && mkdir -p build && cd build && cmake .. && cmake --build . --config Release\n"
        "Or set LLAMA_SERVER_BIN environment variable."
    )
# Then replace all LLAMA_SERVER_BIN references with bin_path
```

---

## Task 2: Rewrite `_cli.py` (lines 1-628 → ~550 lines)

### Remove
- `chat_completion` from imports (line 9) — dead, not used in CLI
- Unused `Dict` type import if not needed
- `shutil` import if not used (check: it IS used in cmd_hermes_desktop line 253, keep it)

### Add: config backup subcommands
```python
# In imports:
from ._backup import backup, restore, list_backups

# Handlers:
def cmd_config_backup(args):
    path = backup()
    if path:
        print(f"[backup] saved → {path}")
    else:
        print("[backup] no config file to backup")

def cmd_config_restore(args):
    if args.latest:
        restore(None)
    else:
        restore(args.path)
    print("[restore] complete — restart server to apply")

def cmd_config_list_backups(args):
    backups = list_backups()
    if not backups:
        print("No backups found.")
        return
    print(f"{'PATH':<60} {'SIZE':>8} {'MODIFIED':>14}")
    print("-" * 90)
    for path, size, mtime in backups:
        from datetime import datetime
        dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{path:<60} {size:>7.1f}KB {dt:>14}")
```

### Add: webui subcommand
```python
# In imports:
import webbrowser

def cmd_webui(args):
    """Open llama.cpp's built-in web UI in browser."""
    _ensure_server_or_start(args)
    state = _read_state()
    port = state.get("port", 8080)
    host = state.get("host", "127.0.0.1")
    url = f"http://{host}:{port}/chat.html"
    print(f"[webui] opening {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[webui] could not open browser: {e}")
        print(f"       Open manually: {url}")
```

### Add: --model flag to config set parser
```python
# In build_parser(), modify config set parser:
p_set.add_argument("key")
p_set.add_argument("value")
p_set.add_argument("--model", default=None,
    help="Per-model config (creates/updates per-model settings)")
```

### Update: cmd_config_set to support --model
```python
def cmd_config_set(args):
    cfg = get_config()
    model = getattr(args, "model", None)
    if model:
        from .per_model import update_model_config, _model_name_from_path
        resolved = _resolve_model_arg(args) if model != "auto" else None
        name = _model_name_from_path(resolved) if resolved else model
        update_model_config(name, {args.key: args.value})
        print(f"[config] [{name}] {args.key} = {args.value}")
    else:
        cfg.set(args.key, args.value)
        print(f"[config] {args.key} = {cfg.get(args.key)}")
```

---

## Task 3: Update `pyproject.toml`

### Remove dead dependencies
- `psutil>=5.8.0` — never imported anywhere
- `requests>=2.25.0` — never imported anywhere

### Result:
```toml
[project]
name = "llama-light"
version = "0.2.0"
description = "Lightweight llama.cpp wrapper — Ollama-style CLI"
requires-python = ">=3.8"
dependencies = [
    "tqdm>=4.66.0",
    "huggingface_hub>=0.19.0",
]
```

---

## Task 4: Update `__init__.py`

### Remove `chat_completion` from exports
```python
# Before:
from .server import (
    start, stop, kill, restart,
    ps, status, logs,
    chat_completion, chat_messages,  # ← remove chat_completion
    install_service, uninstall_service,
)

# After:
from .server import (
    start, stop, kill, restart,
    ps, status, logs,
    chat_messages,
    install_service, uninstall_service,
)

__all__ = [
    "start", "stop", "kill", "restart",
    "ps", "status", "logs",
    "chat_messages",
    "install_service", "uninstall_service",
    "pull", "ls", "rm",
]
# Remove "chat_completion" from __all__ if present
```

---

## Task 5: Update `README.md`

### Document new features:
1. **Config Backup/Restore**: `llama config backup`, `llama config restore`, `llama config list-backups`
2. **Per-model config**: `llama config set key value --model <model>` or `--model auto`
3. **Web UI**: `llama webui` — opens llama.cpp chat.html in browser
4. **Smart VRAM tiers**: Auto-detects 4/6/8/12/16/24/48GB with context sizing
5. **Model-size awareness**: Large models (70B+) get smaller context, 405B even smaller
6. **Dynamic batch sizing**: Adjusted based on VRAM headroom
7. **Robust binary resolution**: Multi-strategy llama-server detection
8. **Config key validation**: Clear errors with valid key suggestions
9. **Per-model auto-detection**: opus/claude/codellama/qwen detection with optimized defaults
10. **Graceful shutdown**: Clean server stop on SIGTERM

---

## Execution Order
1. server.py — core changes, binary resolution, dead code removal
2. _cli.py — new subcommands, import fixes
3. __init__.py — remove dead export
4. pyproject.toml — remove dead deps
5. README.md — document everything

## Critical Files to Check After Changes
- Run: `python -m llama_light info` — should work without errors
- Run: `llama config show` — should display config
- Run: `llama version` — should display version
