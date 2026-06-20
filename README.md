# llama‑light

**Lightning‑fast local LLM inference** — direct `llama.cpp` server with zero API overhead, full hardware auto‑detection, and smart per‑model configuration.

## Features

- **Cross-platform**: Linux (systemd service) + Windows (prebuilt binaries, Windows service)
- **Zero‑overhead server**: Spawns `llama-server` directly — no Go daemon, no API middleware, no translation layer.
- **All `llama-server` flags**: ctx, ngl, flash_attn, cache types, RoPE, YaRN, MoE, reasoning, reasoning_format, reasoning_budget
- **Hardware auto‑detection**: GPU, CPU → `cache_type`, `ngl`, `threads`, `flash_attn`
- **Model registry**: scans Hugging Face cache and local models with fuzzy matching
- **Systemd service**: `install.sh` auto-creates user service; `llama start/stop/restart` delegates to systemd when active
- **Personas**: `claude`, `hermes`, `hermes-desktop` with no sandbox, no login prompt
- **Smart per‑model config**: auto‑detected defaults for reasoning models, user overrides persist per-model
- **Token‑efficient defaults**: low temperature, tight top-p, repetition penalties for Opus/Anthropic-like behaviour
- **Config backup/restore**: `llama config backup`, `restore`, `list-backups` — safety net with 10-file rotation
- **Web UI**: `llama webui` — opens llama.cpp chat UI in your browser
- **Robust binary resolution**: Multi-strategy llama-server detection via `_bincheck`
- **Graceful shutdown**: Clean server stop on SIGTERM/SIGINT

## Installation

```bash
pip install llama-light  # latest from PyPI
# or: pip install .  # from source directory
```

Prerequisite: NVIDIA driver with `nvidia-smi` available. The installer auto-detects CUDA version and GPU architecture.

### Windows

```powershell
# Run from an elevated PowerShell (right-click → Run as Administrator)
cd C:\path\to\llama-Light
.\install.ps1
```

The Windows installer (`install.ps1`):
- Detects NVIDIA GPU via `nvidia-smi` and downloads prebuilt CUDA binary from llama.cpp GitHub releases
- Falls back to CPU-only binary if no NVIDIA GPU is present
- Downloads CUDA runtime library (`cudart`) automatically
- Creates a Python virtual environment and installs `llama-light`
- Registers a Windows Service (`llama-light`) via `sc.exe`
- No CMake, no Visual Studio, no source build required

Post-install (non-elevated PowerShell):
```powershell
llama config set default_model C:\models\Opus4.8.gguf
llama start

# Or manage via Windows Service:
Start-Service llama-light
Stop-Service llama-light
Get-Service llama-light
```

## Quick Start

```bash
llama config set default_model ~/models/Opus4.8.gguf
llama start
llama run --prompt "Explain quantum computing"
llama chat
llama ps
llama service        # show systemd unit status
llama stop
```

## Commands

| Command | Description |
|---------|-------------|
| `llama start` | Start server (reads config, resolves model) |
| `llama stop` | Stop server (systemd or direct) |
| `llama kill` | Force-kill server (systemd or direct) |
| `llama restart` | Restart server (systemd or direct) |
| `llama run --prompt "..."` | Single prompt (auto-starts server) |
| `llama chat` | Interactive multi-turn chat |
| `llama hermes` | Chat with Hermes persona |
| `llama hermes-desktop` | Launch Hermes Electron desktop app |
| `llama claude` | Chat with Claude persona |
| `llama pull --repo <repo> --file <file>` | Download GGUF from HuggingFace |
| `llama ls` | List cached GGUF models (scans HF cache) |
| `llama rm <name>` | Remove model from registry and disk |
| `llama ps` | Server table (PID, ctx, ngl, batch, GPU VRAM, uptime) |
| `llama status` | Verbose status (PID, health, model, log) |
| `llama logs [-n N]` | Tail server log |
| `llama service` | Show systemd unit path, content, status, config defaults |
| `llama service install` | Install systemd user service |
| `llama service stop` | Stop the systemd service |
| `llama service remove` | Uninstall the systemd service |
| `llama version` | Show version |
| `llama info` | System info (Python, platform, binary, cache) |
| `llama webui` | Open llama.cpp chat UI in browser |
| `llama config show` | Show all settings |
| `llama config set <key> <value>` | Set a global config key |
| `llama config set --model <name> <key> <value>` | Save per-model setting |
| `llama config backup` | Snapshot current config |
| `llama config restore [--path <file>]` | Restore config |
| `llama config list-backups` | List available backups |
| `llama quantize [args]` | Quantize model |
| `llama bench [args]` | Benchmark throughput |
| `llama perplexity [args]` | Perplexity test |
| `llama cli [args]` | Interactive CLI |
| `llama gguf-split [args]` | Split model into shards |
| `llama tokenize [args]` | Tokenize text |
| `llama gguf [args]` | Inspect GGUF header |
| `llama export-lora [args]` | Export LoRA adapter |
| `llama imatrix [args]` | Compute importance matrix |
| `llama embedding [args]` | Run embedding model |

## Dual-Mode Start / Stop / Restart

`start`, `stop`, `kill`, and `restart` are **dual-mode**:

- If the **systemd unit exists** → delegate to `systemctl --user`
- If **systemd not active** → use direct signal (SIGTERM/SIGKILL)

This means:
- Running `llama start` works whether or not systemd is installed
- Running `llama service install` creates the unit file
- After that, `llama start` delegates to systemd automatically

## Smart Per‑Model Configuration

Every model you load gets auto‑detected defaults based on its family:

| Model family | temperature | top_k | top_p | freq_penalty | presence_penalty | max_tokens |
|---|---|---|---|---|---|---|
| **opus** (opus‑\*, claude‑\*) | 0.1 | 1 | 0.1 | 0.2 | 0.1 | 8192 |
| **claude** (claude‑\*) | 0.1 | 1 | 0.1 | 0.3 | 0.2 | 8192 |
| **codellama** (codellama‑\*) | 0.1 | 1 | 0.1 | 0.3 | 0.2 | 4096 |
| **qwen** (qwen‑\*, qwen2\*- \*) | 0.1 | 1 | 0.1 | 0.2 | 0.1 | 8192 |
| **default** (everything else) | 0.7 | 40 | 0.95 | 0.0 | 0.0 | 2048 |

These settings produce **Opus/Anthropic‑like** behaviour: clear, concise, deterministic answers with minimal wasted tokens.

**Configuration priority** (highest to lowest):

1. **CLI args** — `--temperature 0.3`, `--top-p 0.2`, etc.
2. **Per‑model config** — saved to `~/.config/llama_light/models/<model_name>.json`
3. **Global config** — `~/.config/llama_light/config.json`
4. **Auto‑detected family defaults**

**First time you load a new model** → auto‑detected defaults applied immediately (no file needed).

**Customize a model** → `llama config set --model <name> temperature 0.3` → saved to its own JSON file, persists across model swaps.

## Configuration

All settings live in `~/.config/llama_light/`:

| Path | Purpose |
|------|---------|
| `config.json` | Global settings (server, host, port, hardware defaults) |
| `models/<model>.json` | Per‑model settings (auto-created on first customisation) |
| `registry.json` | Model registry (HF cache + local) |

## systemd Service

The systemd service is auto-created by `install.sh`. It:
- Runs `llama _run` as the ExecStart command
- Uses `Type=simple` + `PIDFile=` for process tracking
- Has `KillMode=control-group` for clean shutdown
- Has `Restart=no` (no auto-restart)
- Has `TimeoutStartSec=300` for model loading
- Has `KillSignal=SIGKILL`

To change settings: `llama config set <key> <value>` then `llama restart`.

## Performance

| Metric | llama‑light | Traditional wrapper |
|--------|-------------|-------------------|
| Architecture | Direct `llama-server` subprocess | Go wrapper + API layer |
| Added latency | ~1‑2 ms | 20‑50 ms |
| Memory overhead | 0 MB (when stopped) | 200‑500 MB daemon |
| GPU utilisation | 100% via GPU layers | Often falls back to CPU |
| Time to first token | Same as llama.cpp | ~40% longer |

## Developer Blueprint

Core files:

- `_cli.py` — all CLI commands (argparse)
- `config.py` — hardware detection + persistent JSON config
- `server.py` — spawns/kills llama‑server, health checks, streaming chat, systemd helpers
- `model_manager.py` + `registry.py` — model registry, HF cache scanning
- `per_model.py` — per-model config auto-detection and merge
- `_bincheck.py` — robust binary detection

To extend:

- Add CLI command: handler in `_cli.py` + parser in `build_parser()`.
- Add config key: edit `get_defaults()` in `config.py`.
- Add `llama-server` flag: pass in `start()` using `cfg.get("key")`.
- Add auto-tuning: modify `get_defaults()` in `config.py`.
- Add persona: copy `cmd_claude` pattern, change system prompt.

State stored in:

- `~/.cache/llama_light/` — logs, runners, state
- `~/.config/llama_light/` — config, per-model settings, registry

## License

MIT — free to use, modify, and distribute.
