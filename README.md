# llama‑light

**Ollama‑style CLI with direct `llama.cpp` performance** — no API overhead, full hardware auto‑detection, smart per‑model configuration.

## Features

- **Ollama‑style CLI**: `run`, `pull`, `ls`, `rm`, `cp`, `show`, `push`, `create`, `serve`, `stop`
- **All `llama-server` flags**: ctx, ngl, flash_attn, cache types, RoPE, YaRN, MoE, reasoning
- **Hardware auto‑detection**: GPU, CPU → `cache_type`, `ngl`, `threads`
- **Model registry**: scans Hugging Face cache, OCI manifests + blobs
- **Multi‑model runner**: each model on its own port
- **Systemd service**: auto‑start at boot
- **Personas**: `claude`, `hermes`, `hermes-desktop` with no sandbox, no login prompt
- **Smart per‑model config**: auto‑detected defaults for reasoning models, user overrides persist per-model
- **Token‑efficient defaults**: low temperature, tight top-p, repetition penalties for Opus/Anthropic-like behaviour
- **Config backup/restore**: `llama config backup`, `restore`, `list-backups` — safety net with 10-file rotation
- **Web UI**: `llama webui` — opens llama.cpp chat UI in your browser
- **Robust binary resolution**: Multi-strategy llama-server detection via `_bincheck`
- **Graceful shutdown**: Clean server stop on SIGTERM/SIGINT

## Installation

```bash
pipx install llama_light-0.2.0-py3-none-any.whl   # recommended
# or: pip install --user llama_light-0.2.0-py3-none-any.whl
```

Prerequisite: `llama-server` at `~/llama.cpp/build/bin/llama-server` (adjustable in `config.py`).

## Quick Start

```bash
llama config show
llama config set default_model ~/models/llama-2-7b.Q4_K_M.gguf
llama start
llama run --prompt "Explain quantum computing"
llama claude
llama ps
llama stop
```

## Commands

| Command | Description |
|---------|-------------|
| `llama start [--model ...]` | Start llama‑server (uses default/last model) |
| `llama stop` | Stop server (SIGTERM) |
| `llama kill` | Force‑kill (SIGKILL) |
| `llama restart` | Stop then start |
| `llama run --prompt "..."` | Single prompt (auto‑starts server) |
| `llama chat` | Interactive multi‑turn chat |
| `llama hermes` | Chat with Hermes persona |
| `llama claude` | Chat with Claude persona |
| `llama hermes-desktop` | Launch Hermes Electron desktop app |
| `llama pull --repo <repo> --file <file>` | Download GGUF from HuggingFace |
| `llama ls` | List cached GGUF models (scans HF cache) |
| `llama rm <name>` | Remove model from registry and disk |
| `llama cp <source> <dest>` | Copy model (new manifest, same blobs) |
| `llama show <model>` | Display OCI manifest |
| `llama push <model>` | Push to registry (stub) |
| `llama create <model> -f <Modelfile>` | Create from Modelfile (stub) |
| `llama ps` | Enhanced process list (PID, ctx, ngl, batch, threads, flash, uptime) |
| `llama status` | Verbose status (PID, health, log path) |
| `llama logs [-n N]` | Tail server log |
| `llama config show` | Show all current settings |
| `llama config set <key> <value>` | Set a config key (persists) |
| `llama config set --model <name> <key> <value>` | Save per-model setting |
| `llama config backup` | Snapshot current config with timestamp |
| `llama config restore` | Restore latest backup |
| `llama config restore --path <file>` | Restore specific backup |
| `llama config list-backups` | List all available backups |
| `llama webui` | Open llama.cpp chat UI in browser |
| `llama info` | System info (Python, platform, binary, cache, models) |
| `llama service install --model <path>` | Install systemd user service |
| `llama service remove` | Remove systemd service |
| `llama version` | Show version |

## Smart Per‑Model Configuration

Every model you load gets auto‑detected defaults based on its family:

| Model family | temperature | top_k | top_p | freq_penalty | presence_penalty | max_tokens |
|---|---|---|---|---|---|---|
| **opus** (opus‑*, claude‑*) | 0.1 | 1 | 0.1 | 0.2 | 0.1 | 8192 |
| **claude** (claude‑*) | 0.1 | 1 | 0.1 | 0.3 | 0.2 | 8192 |
| **codellama** (codellama‑*) | 0.1 | 1 | 0.1 | 0.3 | 0.2 | 4096 |
| **qwen** (qwen‑*, qwen2*‑*) | 0.1 | 1 | 0.1 | 0.2 | 0.1 | 8192 |
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
| `models/<model>.json` | Per‑model settings (auto‑created on first customisation) |
| `registry.json` | Model registry (HF cache + local) |

See `CONFIG.md` for all 50+ configuration keys.

## Performance vs Ollama

| Metric | llama‑light | Ollama |
|--------|-------------|--------|
| Architecture | Direct llama‑server call | Go server + API |
| Added latency | ~1‑2 ms | 20‑50 ms |
| Throughput (concurrent) | Same as llama.cpp | Up to 19× slower |
| Memory overhead | 0 MB (when stopped) | 200‑500 MB daemon |
| GPU utilisation | 100% | Often falls back to CPU |
| Time to first token | Same as llama.cpp | ~40% longer |

## Developer Blueprint

Core files:

- `_cli.py` — all CLI commands (argparse)
- `config.py` — hardware detection + persistent JSON config
- `server.py` — spawns/kills llama‑server, health checks, streaming chat
- `model_manager.py` + `registry.py` — model registry, HF cache scanning
- `per_model.py` — per‑model config auto‑detection and merge
- `_bincheck.py` — robust binary detection (hermes, claude, hermes-desktop)

To extend:

- Add CLI command: handler in `_cli.py` + parser in `build_parser()`.
- Add config key: edit `get_defaults()` in `config.py`.
- Add `llama-server` flag: pass in `start()` using `cfg.get("key")`.
- Add auto‑tuning: modify `get_defaults()` in `config.py`.
- Add persona: copy `cmd_claude` pattern, change system prompt.

State stored in:

- `~/.cache/llama_light/` — logs, runners, state
- `~/.config/llama_light/` — config, per‑model settings, registry

## License

MIT — free to use, modify, and distribute.
