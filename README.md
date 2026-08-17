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
> The prepared runtime bundle is estimated at approximately 42.7 GiB (shown as roughly 43 GB by `du`, below the 44 GB target); the downloaded 4-bit main-model source alone is approximately 41.8 GiB. The profile merged in [PR #1](https://github.com/AMOS144/Vates/pull/1) measured approximately 10.96 GiB of active MLX memory and an approximately 11.51 GiB MLX peak. These figures are not process RSS or total system memory, and they do not mean that a machine with only that much unified memory is sufficient. macOS, mapped files, native allocations, the filesystem cache, and other applications require additional headroom. Actual memory use and speed depend on hardware, prompt length, model files, and cache warmth. The repository does not include model weights.

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

After preparing the runtime bundle below, launch the measured profile from the repository root:

```bash
vates --stats
```

The public `vates` command installs the fixed profile before importing the model runtime. Internal experiment and benchmark runners remain available to developers, but they are not the supported user entry point.

## Data preparation

The supported path is one command. It downloads the official 4-bit MLX main model and the original Qwen MTP shard, converts them, performs a full output hash pass, verifies every recorded SHA-256 digest, and then deletes the original weight shards:

```bash
vates prepare --download
```

The command writes the runtime-ready `models/vates-runtime` directory:

```text
models/vates-runtime/
├── model/                 # Compact main-model core without routed experts
├── experts/blobs/         # 48 direct-from-shard main expert blobs
├── mtp/core.safetensors   # Compact MTP core
├── mtp/experts/           # One 4-bit MTP expert blob
└── vates_manifest.json    # Sizes and SHA-256 integrity records
```

`vates prepare` never creates the old 24,576 per-expert intermediate files. It copies non-expert tensors into compact main-model shards while writing each stacked `switch_mlp` tensor directly to its final blob offset. MTP is split directly into a compact core and a 4-bit/group-64 expert blob. The current source headers give an estimated result of approximately 42.7 GiB (roughly 43 GB as shown by `du`), depending slightly on model metadata.

If the files were downloaded separately, use the Hugging Face CLI and then prepare them locally:

```bash
hf download mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
  --local-dir models/.vates-source/main

hf download Qwen/Qwen3-Next-80B-A3B-Instruct \
  model-00041-of-00041.safetensors \
  --local-dir models/.vates-source/mtp

vates prepare
```

Existing source locations can also be passed explicitly:

```bash
vates prepare \
  --main-source /path/to/Qwen3-Next-80B-A3B-Instruct-4bit \
  --mtp-source /path/to/model-00041-of-00041.safetensors
```

> [!CAUTION]
> Source cleanup is the default and occurs only after structural checks and output hash verification succeed. Use `--keep-source` if the original shards must remain. Existing output directories are never overwritten.

> [!NOTE]
> When downloading and converting in one run, allow roughly 90 GiB of temporary free space because source and output weights coexist before verification. After successful cleanup, only the approximately 42.7 GiB runtime bundle remains (plus negligible download metadata). This flow removes the former per-expert-file duplication, which pushed the old preparation layout to roughly 128 GB.

## Usage

### Interactive chat

Run all commands from the repository root.

Start the full-screen TUI with the fixed K=3 profile:

```bash
vates
```

Set the generation length and print separate prefill/decode statistics:

```bash
vates -n 800 --stats
```

Set a system prompt:

```bash
vates --system "You are a concise assistant."
```

Preview the interface without loading a model:

```bash
vates --demo
```

Use the plain-text REPL when the terminal is incompatible with the TUI:

```bash
vates --plain --stats
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
| `--model` | Path to the compact 4-bit MLX main-model core | `models/vates-runtime/model` |
| `--expert-dir` | Expert root directory; blobs are read from its `blobs/` subdirectory by default. If specifying the blob directory directly, set `BLOB_DIR` | `models/vates-runtime/experts` |
| `--mtp-out` | Compact MTP core file | `models/vates-runtime/mtp/core.safetensors` |
| `--qn-config` | Qwen3-Next configuration file | `models/vates-runtime/model/config.json` |
| `-k`, `--k` | MTP speculative width; fixed by the public CLI | `3` |
| `-n`, `--max-tokens` | Maximum new tokens per turn | `4096` |
| `--expert-slots` | Main-model expert-pool capacity; fixed by the public CLI | `152` |
| `--spec-slots` | Legacy side-region rows; fixed by the public CLI | `0` |
| `--system` | System prompt | None |
| `--stats` | Print token count, throughput, and accepted length | Off |
| `--plain` | Use the plain-text REPL | Off |
| `--demo` | Preview the TUI with a mock backend | Off |

View the complete option set for the installed version:

```bash
vates --help
```

## Configuration

The public `vates` entry point is the authority for the measured profile. It installs the fixed configuration before importing the inference runtime and deliberately overwrites performance-related environment variables so a stale shell cannot silently alter the result. Its key contract is:

- **Prefill:** Expert-major prompt ingestion, synchronous demand, and a fixed bounded superblock;
- **Decode:** K=3 batch MTP verification, adaptive depth capped at 3, and asynchronous demand after the phase boundary;
- streamed 4-bit MTP experts with 256 MTP slots;
- a 152-slot main pool plus checked-in per-layer capacity overrides;
- K4/V3 rotated KV quantization; and
- native predictive prefetch with fixed reranking, target-layer, and physical-read-budget policies.

Model paths, the system prompt, and the maximum generated-token count remain CLI options. Pool sizes and profile-critical switches are fixed on the public command; developers testing variants should use the internal benchmark runners and must not present those results as measurements of the fixed profile.

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
| Storage and memory | Prepared runtime bundle: approximately 42.7 GiB (roughly 43 GB as shown by `du`) on disk. Fixed profile: approximately 10.96 GiB active MLX memory and approximately 11.51 GiB MLX peak; neither is process RSS or total system memory. |
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

The suite covers Expert-major prefill, expert pools, blob layout, MTP, KV quantization, predictive prefetch, native I/O, the public entry point, the TUI, and streaming detokenization. The benchmarked path is also validated with native tests, capacity-invariance checks, byte-level pool-oracle checks, and the 32K prefill validator.

## FAQ

<details>
<summary><strong>Why is only Apple Silicon supported?</strong></summary>

The project is built on MLX and depends on Apple Silicon's unified-memory architecture and native MLX capabilities. There is currently no CUDA, ROCm, or CPU-only backend.

</details>

<details>
<summary><strong>Why do I get <code>vates: command not found</code>?</strong></summary>

`vates` is installed in `.venv/bin/`. Run `source .venv/bin/activate` first, or invoke `.venv/bin/vates` directly. If the virtual environment was moved or renamed after creation, rebuild it with `uv venv --clear && uv sync`.

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

Run `vates prepare --download`; the default bundle is `models/vates-runtime`. Set `VATES_RUNTIME_DIR` to relocate the whole bundle, or use `--model`, `--expert-dir`, `--mtp-out`, and `--qn-config` for individual paths.

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
