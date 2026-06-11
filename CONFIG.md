# Full Configuration Reference

All settings are stored in **`~/.config/llama_light/`**. Defaults are auto‑tuned to your hardware.

## Global Config — `~/.config/llama_light/config.json`

| Key | Auto‑default | Description | `llama-server` flag |
|-----|--------------|-------------|----------------------|
| `ctx` | 8192 – 65536 (VRAM‑based) | Context length | `-c` |
| `ngl` | 99 (GPU) / 0 (CPU) | Number of GPU layers | `-ngl` |
| `flash_attn` | `auto` | Flash attention | `--flash-attn` |
| `batch_size` | 2048 | Batch size | `-b` |
| `ubatch_size` | 512 | Micro‑batch size | `--ubatch-size` |
| `parallel` | 1 | Parallel sequences | `--parallel` |
| `threads` | min(CPU cores, 8) | CPU threads | `--threads` |
| `threads_batch` | min(CPU cores, 8) | Batch threads | `--threads-batch` |
| `cache_type_k` | `f16`/`q4_0`/`q8_0` (VRAM‑based) | KV cache type (K) | `--cache-type-k` |
| `cache_type_v` | `f16`/`q4_0`/`q8_0` (VRAM‑based) | KV cache type (V) | `--cache-type-v` |
| `temperature` | 0.7 | Sampling temperature | (passed to `/v1/chat/completions`) |
| `top_k` | 40 | Top‑k sampling | (passed to `/v1/chat/completions`) |
| `max_tokens` | 2048 | Max tokens to generate | (passed to `/v1/chat/completions`) |
| `mlock` | false | Lock model in memory | `--mlock` |
| `mmap` | true | Memory‑map model | `--mmap` / `--no-mmap` |
| `rope_scaling` | `none` | RoPE scaling method | `--rope-scaling` |
| `rope_freq_base` | none | Base frequency | `--rope-freq-base` |
| `rope_scale` | none | Scale factor | `--rope-scale` |
| `rope_freq_scale` | none | Frequency scale | `--rope-freq-scale` |
| `numa` | none | NUMA policy | `--numa` |
| `split_mode` | `layer` | Multi‑GPU split mode | `--split-mode` |
| `kv_offload` | true | Offload KV cache to GPU | `--kv-offload` |
| `repack` | true | Repack KV cache | `--repack` |
| `direct_io` | false | Direct I/O for file loading | `--direct-io` |
| `no_host` | false | Disable host memory mapping | `--no-host` |
| `device` | none | Specific device (e.g., `cuda:0`) | `--device` |
| `override_tensor` | none | Override tensor placement | `--override-tensor` |
| `cpu_moe` | false | Run MoE layers on CPU | `--cpu-moe` |
| `n_cpu_moe` | none | CPU threads for MoE | `--n-cpu-moe` |
| `yarn_orig_ctx` | 0 | YaRN original context | `--yarn-orig-ctx` |
| `yarn_ext_factor` | -1.0 | YaRN extrapolation factor | `--yarn-ext-factor` |
| `yarn_attn_factor` | -1.0 | YaRN attention factor | `--yarn-attn-factor` |
| `yarn_beta_slow` | -1.0 | YaRN beta slow | `--yarn-beta-slow` |
| `yarn_beta_fast` | -1.0 | YaRN beta fast | `--yarn-beta-fast` |
| `swa_full` | false | Full SWA (sliding window) | `--swa-full` |
| `perf` | false | Performance metrics | `--perf` |
| `escape` | true | Escape special characters | `--escape` / `--no-escape` |
| `ui_mcp_proxy` | `on` | UI MCP proxy | `--ui-mcp-proxy` |
| `tools` | `all` | Enable tool calls | `--tools` |
| `reasoning` | false | Enable reasoning/thinking | `--reasoning-format` + `--chat-template-kwargs` |
| `reasoning_format` | `none` | Format (`none`, `auto`, `qwen3`, etc.) | `--reasoning-format` |
| `reasoning_budget` | 0 | Max reasoning tokens | `--reasoning-budget` |
| `active_profile` | `default` | Profile name (not yet wired) | – |

## Per‑Model Config — `~/.config/llama_light/models/<model_name>.json`

One JSON file per model. Only store values that differ from family defaults. Auto‑created when you customise a model.

**Token‑efficiency keys** (passed to `/v1/chat/completions`):

| Key | Description | Auto‑default by family |
|-----|-------------|----------------------|
| `temperature` | Sampling temperature | 0.1 (opus/claude/qwen), 0.7 (default) |
| `top_k` | Top‑k sampling | 1 (reasoning), 40 (default) |
| `top_p` | Nucleus sampling threshold | 0.1 (reasoning), 0.95 (default) |
| `min_p` | Min‑probability pruning | 0.05 |
| `frequency_penalty` | Penalise repeated tokens | 0.2–0.3 (reasoning), 0.0 (default) |
| `presence_penalty` | Encourage new topics | 0.1–0.2 (reasoning), 0.0 (default) |
| `max_tokens` | Max output tokens | 8192 (reasoning), 2048 (default) |
| `stream_options` | `{include_usage: true}` | tracks token usage |

**Server‑side keys** (merged into `llama-server` start args):

| Key | Description | `llama-server` flag |
|-----|-------------|----------------------|
| `ctx` | Context length | `-c` |
| `ngl` | GPU layers | `-ngl` |
| `threads` | CPU threads | `--threads` |
| `flash_attn` | Flash attention | `--flash-attn` |
| `keep` | Keep model in VRAM | `--keep` |
| `predict` | Predictions per request | `-np` |
| `reasoning` | Enable reasoning | `--reasoning-format` |
| `reasoning_budget` | Max reasoning tokens | `--reasoning-budget` |

## Example

```bash
# Global settings
llama config set ctx 32768
llama config set ngl 40
llama config set flash_attn on
llama config set cache_type_k q4_0

# Per-model: make opus-4-7 slightly warmer
llama config set --model opus-4-7 --temperature 0.3

# Per-model: force 16k context for a large model
llama config set --model large-model.gguf --ctx 16384

# Override for a single chat
llama run --prompt "..." --temperature 0.5 --top-p 0.2
```

## Auto‑Tuning

On first run, llama‑light auto‑detects:

| Resource | Setting | Values |
|----------|---------|--------|
| GPU type | — | CUDA / Metal / ROCm / CPU |
| Total VRAM | — | NVIDIA only |
| CPU cores | — | All available |
| `<8 GB VRAM` | `ctx` | 8192 |
| 8–16 GB VRAM | `ctx` | 32768 |
| `>16 GB VRAM` | `ctx` | 65536 |
| `<8 GB VRAM` | `cache_type_k/v` | `f16` |
| 8–16 GB VRAM | `cache_type_k/v` | `q4_0` |
| `>16 GB VRAM` | `cache_type_k/v` | `q8_0` |
| GPU present | `ngl` | 99 |
| No GPU | `ngl` | 0 |
| Any | `threads` | min(cores, 8) |

Override any value with `llama config set`.
