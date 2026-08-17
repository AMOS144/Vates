<div align="center">

# vates

**English** | [简体中文](README.zh-CN.md)

**Run Qwen3-Next-80B-A3B on Apple Silicon with a reproducible, disk-streamed K=3 inference path**

An out-of-core streaming MoE inference engine for MLX, with Qwen3-Next MTP self-speculative decoding.

[![GitHub Stars](https://img.shields.io/github/stars/AMOS144/Vates?style=flat&logo=github&label=Stars)](https://github.com/AMOS144/Vates/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/AMOS144/Vates?style=flat&logo=github&label=Forks)](https://github.com/AMOS144/Vates/forks)
[![GitHub Issues](https://img.shields.io/github/issues/AMOS144/Vates?style=flat&logo=github&label=Issues)](https://github.com/AMOS144/Vates/issues)
[![Last Commit](https://img.shields.io/github/last-commit/AMOS144/Vates?style=flat&logo=git&label=Last%20Commit)](https://github.com/AMOS144/Vates/commits/main)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Required-000000?style=flat&logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-report-mchlp1176/mac)
[![MLX](https://img.shields.io/badge/MLX-0.31%2B-8A2BE2?style=flat)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Demo](#demo) · [Quick Start](#quick-start) · [Usage](#usage) · [FAQ](#faq) · [Contributing](#contributing)

</div>

---

## Overview

The complete weights of a large Mixture-of-Experts (MoE) model often exceed the unified memory available on a Mac. vates keeps most expert weights on disk, loading them on demand and predictively prefetching the experts needed at runtime. This substantially reduces the model's resident memory requirement.

vates targets **Qwen3-Next-80B-A3B 4-bit MLX**, which has 48 MoE transformer layers. Of those layers, 12 use full attention and the remaining 36 use linear attention. Each layer selects 10 routed experts from 512 and also uses a resident shared expert. vates uses the model's built-in MTP head for self-speculative decoding to improve generation throughput.

> [!NOTE]
> The 4-bit main-model weights occupy approximately 41 GB on disk. The profile merged in [PR #1](https://github.com/AMOS144/Vates/pull/1) measured approximately 10.96 GiB of active MLX memory and an approximately 11.51 GiB MLX peak. These figures are not process RSS or total system memory, and they do not mean that a machine with only that much unified memory is sufficient. macOS, mapped files, native allocations, the filesystem cache, and other applications require additional headroom. Actual memory use and speed depend on hardware, prompt length, model files, and cache warmth. The repository does not include the main model, expert blobs, MTP weights, or streamed MTP expert files.

## Demo

Click the image below to play the demo:

[![vates demo](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo-poster.png)](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo.mp4)

## Features

- **Low-memory inference:** streams expert weights from disk on demand while retaining bounded main-model and MTP expert pools.
- **Bounded Expert-major prefill:** groups prompt tokens by routed expert instead of materializing the old token-major activation path.
- **MTP self-speculative decoding:** drafts tokens with Qwen3-Next's built-in MTP head, then batch-verifies them with the main model and either accepts or falls back.
- **Measured K=3 decode profile:** fixes the main-model pool, streamed 4-bit MTP pool, physical read budget, reranking policy, and prefetch targets as one reproducible configuration.
- **Native benchmark-validated path:** a C++ extension handles parallel `pread`, pool-state management, predictive prefetching, and fused MoE computation.
- **Long-context optimization:** IsoQuant K4/V3 with SO(4) block rotations reduces KV storage for a 128k context from approximately 3.0 GiB to approximately 0.68 GiB.
- **Correctness checks:** includes targeted regression tests, capacity-invariance checks, byte-level pool oracles, and a 32K prefill validator.
- **Nonblocking interactive terminal:** detokenization and Textual rendering run outside the inference hot path; the TUI reports prefill and decode separately.

## Architecture and data flow

Each turn has two separately timed phases with different I/O policies:

```text
prompt → PREFILL → first-token boundary → DECODE → token stream
           │                              │
           ├─ Expert-major MoE             ├─ K=3 MTP draft + batch verify
           ├─ synchronous expert reads     ├─ async demand + native prefetch
           └─ build KV/recurrent state     └─ nonblocking detokenizer + TUI
```

Both phases use the same disk-backed native runtime stack:

```text
┌──────────────────────────────────────────────────────────────┐
│  vates TUI / CLI                                             │
├──────────────────────────────────────────────────────────────┤
│  MTP self-speculative decoding                               │
│  drafter → main-model batch verification → accept/fallback   │
├──────────────────────────────────────────────────────────────┤
│  Streaming MoE expert pool                                   │
│  capacity region ∪ LFU side region → one GPU gather          │
│  miss → C++ on-demand pread → pool; cross-layer prefetch     │
├──────────────────────────────────────────────────────────────┤
│  Native C++ extension                                        │
│  unified pool state · parallel pread · fused MoE · KV quant  │
├──────────────────────────────────────────────────────────────┤
│  Disk: per-expert blobs, one contiguous byte range each      │
└──────────────────────────────────────────────────────────────┘
```

Core mechanisms:

1. **Bounded expert pools:** the main model uses a checked-in per-layer capacity profile and MTP uses a separate 256-slot 4-bit pool. Missing experts are loaded from SSD into those pools.
2. **Unified C++ pool state:** the native extension owns slot state, eviction, demand reads, and prefetching, reducing synchronization work on the Python main thread.
3. **Predictive cross-layer prefetching:** routing results from the current layer predict experts needed by later layers, allowing reads to overlap computation.
4. **Adaptive-depth MTP:** draft confidence controls verification depth, avoiding unnecessary expert loads during low-confidence steps.
5. **KV quantization:** only the 12 full-attention layers have their KV caches compressed; recurrent state in the linear-attention layers remains unchanged.

### Prefill: ingest the prompt

Prefill processes the entire uncached prompt and builds the KV cache and linear-attention recurrent state needed by generation. Its route unions are much wider than decode, so the production path deliberately keeps expert demand **synchronous** during this phase.

The Expert-major implementation groups tokens by routed expert and reuses one bounded transient bank. This avoids the old token-major large-activation path and makes memory use depend on the fixed superblock rather than growing unchecked with prompt length. The implementation preserves canonical route-rank accumulation order for deterministic reduction. A dedicated 32K-boundary validator compares logits, hidden state, argmax, cache offsets, and memory across runs.

Prefill latency is time to first token, not decode throughput. The CLI and TUI report prefill tokens, seconds, and tokens/second separately. In a multi-turn chat, a valid prefix cache means only the newly appended suffix is prefetched; if the prefix is not authoritative, the cache is discarded and the prompt is rebuilt.

### Decode: generate new tokens

At the exact prefill/decode boundary, the fixed profile enables asynchronous expert demand and predictive cross-layer prefetch. Decode uses Qwen3-Next's streamed 4-bit MTP experts to draft up to K=3 tokens, then batch-verifies them with the main model. Only target-verified tokens are committed; low-confidence steps can stop at a shallower adaptive depth.

The main-model expert pool uses a checked-in per-layer capacity profile, native C++ pool state, batched `preadv` demand reads, reranked prefetch candidates, and a physical SSD-read budget. K4/V3 rotated KV quantization applies only to the 12 full-attention layers; the 36 linear-attention recurrent states remain unquantized.

Decode throughput uses the engine's decode-only `wall_s`, which starts after prefill. Token detokenization runs in a separate thread and publishes the latest text snapshot to the TUI without backpressure; final text is reconciled from the complete authoritative token sequence.

## Tech stack

| Category | Technology | Role |
| --- | --- | --- |
| Runtime | Python 3.11+ | CLI, model assembly, inference flow, and tooling |
| Inference | MLX 0.31+, mlx-lm 0.31+ | Unified-memory inference on Apple Silicon |
| Numerical computing | NumPy 2.0+ | Data preparation and numerical processing |
| Terminal UI | Textual 0.80+ | Full-screen interactive TUI |
| Native extension | C++, nanobind, MLX Primitive | Expert pool, I/O, prefetching, and fused computation |
| Build tools | uv, CMake, Make | Dependency management and native builds |
| Testing | pytest, pytest-asyncio | Unit, integration, and correctness testing |

## Quick Start

### Requirements

- An Apple Silicon Mac
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- CMake, Make, and a working C++ build environment
- Model files and sufficient local disk space for real inference
- Sufficient unified-memory headroom beyond the MLX allocation high-water mark

### Installation and native build

Clone the repository:

```bash
git clone https://github.com/AMOS144/Vates.git
cd Vates
```

Create the virtual environment and install the dependencies pinned by `uv.lock`:

```bash
uv sync
source .venv/bin/activate
```

Build the native extension with the virtual environment's Python. Pass an absolute path because `make -C` changes the working directory before evaluating the Makefile:

```bash
VATES_PYTHON="$PWD/.venv/bin/python"
make -C native/ext PYTHON="$VATES_PYTHON" native_moe_ext
```

> [!TIP]
> Use `uv sync` to reproduce the repository's locked dependency set and avoid incompatibilities introduced by transitive dependency upgrades.

### Minimal examples

Preview the interface without preparing a model:

```bash
vates --demo
```

After preparing all four model assets below, launch the measured profile from the repository root:

```bash
.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 --chat --stats
```

This launcher installs the fixed profile before importing the model runtime. Running bare `vates` uses the configurable generic CLI and is not the same benchmark configuration.

## Data preparation

The example below uses separate directories for intermediate per-expert files and the final runtime blobs:

```text
models/
├── qwen3_next_80b_4bit/                  # 4-bit MLX main model
├── qwen3_next_expert_files_4bit_g64/     # Intermediate per-expert files
├── qwen3_next_experts_4bit_g64/          # CLI default expert directory
│   └── blobs/                            # Final runtime blobs
├── qn_mtp_weights.safetensors            # Source MTP weights
└── qn_mtp_experts_4bit_g64/               # Streamed 4-bit MTP experts
```

Expert weights must be converted into blobs where each expert occupies one contiguous byte range, allowing a runtime expert read to use a single `pread`.

Split the stacked `switch_mlp` weights from the main model into the intermediate per-expert directory:

```bash
.venv/bin/python -m mlx_streaming.prep.split_experts \
  models/qwen3_next_80b_4bit \
  models/qwen3_next_expert_files_4bit_g64
```

Pack those files into one contiguous blob per layer under the CLI's default expert directory, and generate `blob_index.json`. With the default `--expert-dir models/qwen3_next_experts_4bit_g64`, `model_builder.py` resolves runtime blobs from its `blobs/` subdirectory:

```bash
EXPERT_DIR=models/qwen3_next_expert_files_4bit_g64 \
BLOB_DIR=models/qwen3_next_experts_4bit_g64/blobs \
BITS=4 GROUP=64 LAYERS=all \
  .venv/bin/python -m mlx_streaming.prep.pack_blob_from_experts
```

Extract and organize MTP weights. This command downloads the original `model-00041-of-00041.safetensors` shard, approximately 3.30 GB (3.07 GiB), and writes the default `models/qn_mtp_weights.safetensors` output:

```bash
.venv/bin/python -m mlx_streaming.prep.extract_mtp
```

Split and quantize the MTP experts for the bounded 4-bit MTP pool used by the optimal profile:

```bash
.venv/bin/python -m mlx_streaming.tools.split_mtp_experts \
  --bits 4 \
  --group-size 64 \
  --out models/qn_mtp_experts_4bit_g64
```

`mlx_streaming/prep/blob_layout.py` is the single definition of the byte layout and stays aligned with the runtime blob loader.

> [!WARNING]
> Data preparation needs substantially more disk space than the approximately 41 GB main-model directory alone. The split files and final blobs coexist during packing, and the MTP source shard adds approximately 3.30 GB (3.07 GiB). After validating `models/qwen3_next_experts_4bit_g64/blobs` and its `blob_index.json`, you may delete the intermediate `models/qwen3_next_expert_files_4bit_g64` directory. The 41 GB figure describes only the main-model weights, not peak preparation storage.

## Usage

### Interactive chat

Run all commands from the repository root.

Start the full-screen TUI with the fixed K=3 profile:

```bash
.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 --chat
```

Set the generation length and print separate prefill/decode statistics:

```bash
.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 \
  --chat -n 800 --stats
```

Set a system prompt:

```bash
.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 \
  --chat --system "You are a concise assistant."
```

Preview the interface without loading a model:

```bash
vates --demo
```

Use the plain-text REPL when the terminal is incompatible with the TUI:

```bash
.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 \
  --chat --plain --stats
```

Without activating the virtual environment, run either:

```bash
.venv/bin/vates --demo
.venv/bin/python -m mlx_streaming.cli --demo
```

TUI controls:

| Input | Action |
| --- | --- |
| `Enter` | Send the message |
| `Esc` | Interrupt the current generation |
| `Ctrl+C` | Exit |
| `/help` | Show help |
| `/reset` | Clear conversation history |
| `/clear` | Clear the screen |
| `/exit` | Exit |

### CLI options

| Option | Description | Default |
| --- | --- | --- |
| `--model` | Path to the 4-bit MLX main model | `models/qwen3_next_80b_4bit` |
| `--expert-dir` | Expert root directory; blobs are read from its `blobs/` subdirectory by default. If specifying the blob directory directly, set `BLOB_DIR` | `models/qwen3_next_experts_4bit_g64` |
| `--mtp-out` | MTP weights file | `models/qn_mtp_weights.safetensors` |
| `--qn-config` | Qwen3-Next configuration file | `models/qwen3_next_80b_4bit/config.json` |
| `-k`, `--k` | MTP speculative width; keep at the validated value | `3` |
| `-n`, `--max-tokens` | Maximum new tokens per turn | `4096` |
| `--expert-slots` | Main-model expert-pool capacity; fixed by the launcher | `152` |
| `--spec-slots` | Legacy side-region rows; fixed by the launcher | `0` |
| `--system` | System prompt | None |
| `--stats` | Print token count, throughput, and accepted length | Off |
| `--plain` | Use the plain-text REPL | Off |
| `--demo` | Preview the TUI with a mock backend | Off |

View the complete option set for the installed version:

```bash
.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 --chat --help
```

## Configuration

`mlx_streaming.runtime.run_qwen_k3_sub10` is the authority for the measured profile. It deliberately overwrites performance-related environment variables so a stale shell cannot silently alter the result. Its key contract is:

- **Prefill:** Expert-major prompt ingestion, synchronous demand, and a fixed bounded superblock;
- **Decode:** K=3 batch MTP verification, adaptive depth capped at 3, and asynchronous demand after the phase boundary;
- streamed 4-bit MTP experts with 256 MTP slots;
- a 152-slot main pool plus checked-in per-layer capacity overrides;
- K4/V3 rotated KV quantization; and
- native predictive prefetch with fixed reranking, target-layer, and physical-read-budget policies.

Model paths, the system prompt, and the maximum generated-token count remain CLI options. Use bare `vates` only when intentionally experimenting with other pool sizes or environment switches. Results from that generic path must not be presented as measurements of the fixed profile.

## Repository layout

```text
.
├── mlx_streaming/
│   ├── cli.py                 # vates command entry point
│   ├── config.py              # Environment variables and defaults
│   ├── model_builder.py       # Streaming model assembly
│   ├── core/
│   │   ├── cache/             # Expert pools, blob loader, and KV quantization
│   │   ├── moe/               # Streaming MoE, gate, and fused computation
│   │   ├── prefetch/          # Cross-layer prediction and background prefetch
│   │   └── linear_attn/       # Qwen3-Next linear attention
│   ├── mtp/                   # MTP drafting, verification, and KV reuse
│   ├── prep/                  # Expert splitting, blob packing, and weight extraction
│   ├── runtime/               # Benchmarks and runtime entry points
│   ├── tools/                 # Analysis and diagnostic tools
│   ├── tui/                   # Full-screen Textual interface
│   └── tests/                 # Python tests
├── native/
│   ├── ext/                   # C++/nanobind extension used by the benchmarked path
│   └── bench/                 # Native microbenchmarks
├── benchmarks/
│   └── reports/               # Ablation studies and performance reports
├── docs/
│   └── superpowers/           # Design specifications and implementation plans
├── pyproject.toml             # Project metadata and dependencies
└── uv.lock                    # Locked dependency versions
```

## Benchmarks, correctness, and rejected experiments

The current production numbers come from the fixed profile merged in [PR #1](https://github.com/AMOS144/Vates/pull/1). Older reports under `benchmarks/reports/` are independent ablations with different pool sizes, prompts, warmup lengths, and clocks; their gains **must not be added together**.

| Item | Result |
| --- | --- |
| Storage and memory | Main-model weights: approximately 41 GB on disk. Fixed profile: approximately 10.96 GiB active MLX memory and approximately 11.51 GiB MLX peak; neither is process RSS or total system memory. |
| Prefill | Reported independently as prompt tokens, seconds, and tokens/second. It is synchronous and excluded from the decode figure. Time to first token varies with uncached prompt length and SSD state. |
| Steady decode | Approximately 31–37 tok/s in existing fixed 128-token steady-state runs; cache warmth and prompts materially affect the result. |
| TUI overhead A/B | 31.65 tok/s with asynchronous TUI streaming versus 31.76 tok/s with streaming disabled in the same warm-pool comparison, a difference of approximately 0.35%. |
| Decode clock | Uses the engine's decode-only `wall_s`, which starts after prefill; detokenization and UI rendering do not run on the inference hot path. |
| KV storage | Approximately 3.0 GiB to approximately 0.68 GiB at 128k context |
| Long-prefill validation | Includes a 32K-boundary tool that compares logits, hidden state, argmax, cache offsets, and memory between runs. |

Correctness validation includes:

- byte-for-byte identical output for the deterministic greedy prompts and configurations exercised by the capacity-invariance reports;
- `0 BAD` as the acceptance condition for the byte-level pool-oracle checks `DUAL_VERIFY` and `STG_VERIFY` in the reports that used them;
- deterministic Expert-major route reduction that preserves canonical route-rank accumulation order;
- main-model batch verification that commits only target-verified tokens;
- final TUI text reconciled from the complete authoritative token sequence; and
- test coverage for both Python and native paths.

These oracles cover the documented deterministic greedy prompts, pool states, and benchmark configurations. They are regression evidence for those cases, not a mathematical guarantee of byte-identical output for every prompt, decoding mode, hardware state, or configuration.

Complete experimental records are available in [`benchmarks/reports/`](benchmarks/reports/).

Evaluated approaches that were rejected from the benchmark-validated path include full tree verification, event-gated asynchronous demand loading, sliding-window expert pools, and per-layer capacity reclamation. They were excluded because of throughput regressions, insufficient I/O budget, or no benefit under the current configuration. See the benchmark reports for details.

## Tests

After installing development dependencies, run the complete Python test suite:

```bash
.venv/bin/python -m pytest
```

The suite covers Expert-major prefill, expert pools, blob layout, MTP, KV quantization, predictive prefetch, native I/O, the fixed launcher, the TUI, and streaming detokenization. The benchmarked path is also validated with native tests, capacity-invariance checks, byte-level pool-oracle checks, and the 32K prefill validator.

## FAQ

<details>
<summary><strong>Why is only Apple Silicon supported?</strong></summary>

The project is built on MLX and depends on Apple Silicon's unified-memory architecture and native MLX capabilities. There is currently no CUDA, ROCm, or CPU-only backend.

</details>

<details>
<summary><strong>Why do I get <code>vates: command not found</code>?</strong></summary>

`vates` is installed in `.venv/bin/`. Run `source .venv/bin/activate` first, or invoke `.venv/bin/vates` directly. The measured profile can always be launched with `.venv/bin/python -m mlx_streaming.runtime.run_qwen_k3_sub10 --chat`. If the virtual environment was moved or renamed after creation, rebuild it with `uv venv --clear && uv sync`.

</details>

<details>
<summary><strong>Can vates run without compiling the native extension?</strong></summary>

It can fall back to a slower path, but native features such as predictive prefetching and the unified pool will be disabled. For real inference, pass the virtual environment's Python executable to Make:

```bash
VATES_PYTHON="$PWD/.venv/bin/python"
make -C native/ext PYTHON="$VATES_PYTHON" native_moe_ext
```

</details>

<details>
<summary><strong>Why is <code>uv sync</code> recommended?</strong></summary>

`uv sync` installs the dependency set validated in `uv.lock`, preventing transitive packages such as `transformers` from drifting to incompatible versions.

</details>

<details>
<summary><strong>Where should model files be placed?</strong></summary>

The defaults are under `models/` at the repository root. The fixed profile also requires `models/qn_mtp_experts_4bit_g64`. Use `--model`, `--expert-dir`, `--mtp-out`, and `--qn-config` to select other main-model locations.

</details>

<details>
<summary><strong>How can I inspect the interface without a model?</strong></summary>

Run `vates --demo`. This mode uses a mock backend and reads no model files, so it can be used to inspect the TUI, streaming display, and status bar.

</details>

## Contributing

Issues and pull requests are welcome. Read the [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

1. For substantial changes, open an Issue first to describe the problem, goal, and proposed approach.
2. Create a dedicated branch from `main` and keep unrelated changes out of the pull request.
3. Keep each change focused, and add or update tests for behavioral changes.
4. Run `.venv/bin/python -m pytest` before submitting.
5. In the pull request description, explain the context, implementation, validation results, and potential impact.
6. Performance changes should include reproducible benchmark commands, comparison data, and correctness results.
7. Bug reports should include the device model, macOS version, Python version, reproduction command, and complete error output.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 AMOS144

## Contact

- Author: [AMOS144](https://github.com/AMOS144)
- Repository: [github.com/AMOS144/Vates](https://github.com/AMOS144/Vates)
- Issues: [github.com/AMOS144/Vates/issues](https://github.com/AMOS144/Vates/issues)
- Email: [3108424075@qq.com](mailto:3108424075@qq.com)

---

Technical feedback and contributions are welcome through [Issues](https://github.com/AMOS144/Vates/issues).
