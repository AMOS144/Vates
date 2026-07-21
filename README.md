<div align="center">

# vates

**English** | [简体中文](README.zh-CN.md)

**Run an 80B MoE model on Apple Silicon with about an 8.3 GiB MLX in-flight tensor-allocation high-water mark**

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
> The 4-bit main-model weights occupy approximately 41 GB on disk. The approximately 8.23–8.27 GiB figure comes from MLX `get_peak_memory` results in the repository's benchmark reports and measures the in-flight tensor-allocation high-water mark for the reported configuration. It is not process RSS, total system memory use, or evidence that a machine with only that much unified memory has sufficient capacity. The end-to-end configuration was tested on a MacBook Pro with an Apple M5 (10-core CPU), 32 GB of physical unified memory, and a 1 TB internal Apple SSD. The system requires additional memory headroom for macOS and non-MLX allocations. Actual memory use and speed depend on hardware, context length, model files, and configuration. The repository does not include the main model, expert data, or MTP weights, and there is no single download location for all three.

## Demo

Click the image below to play the demo:

[![vates demo](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo-poster.png)](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo.mp4)

## Features

- **Low-memory inference:** streams expert weights from disk on demand while retaining only a small resident pool and an LFU side-region cache.
- **MTP self-speculative decoding:** drafts tokens with Qwen3-Next's built-in MTP head, then batch-verifies them with the main model and either accepts or falls back.
- **Zero-copy dual-source expert pool:** the capacity and side regions share one expert pool, with no pool-to-pool copy when selecting between those regions. Disk reads still copy expert bytes into the pool.
- **Native benchmark-validated path:** a C++ extension handles parallel `pread`, pool-state management, predictive prefetching, and fused MoE computation.
- **Long-context optimization:** IsoQuant K4/V3 with SO(4) block rotations reduces KV storage for a 128k context from approximately 3.0 GiB to approximately 0.68 GiB.
- **Correctness checks:** runs capacity-invariance tests, byte-level pool-oracle checks, and the full test suite for the configurations covered by the repository's reports.
- **Interactive terminal:** includes a full-screen Textual TUI, a plain-text REPL, streaming output, and throughput and memory status.

## Architecture and data flow

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

1. **Zero-copy dual-source pool:** each layer maintains a capacity region and a persistent LFU side region inside one unified expert pool. Selecting experts from either region requires no pool-to-pool copy; loading a miss from SSD into the pool is still a data transfer.
2. **Unified C++ pool state:** the native extension owns slot state, eviction, demand reads, and prefetching, reducing synchronization work on the Python main thread.
3. **Predictive cross-layer prefetching:** routing results from the current layer predict experts needed by later layers, allowing reads to overlap computation.
4. **Adaptive-depth MTP:** draft confidence controls verification depth, avoiding unnecessary expert loads during low-confidence steps.
5. **KV quantization:** only the 12 full-attention layers have their KV caches compressed; recurrent state in the linear-attention layers remains unchanged.

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

Build the native extension used by the benchmark-validated path. The Makefile's default `PY_SITE` is version-specific, so derive the active virtual environment's `purelib` path and override it:

```bash
PY_SITE="$("./.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
make -C native/ext PY_SITE="$PY_SITE" native_moe_ext
```

> [!TIP]
> Use `uv sync` to reproduce the repository's locked dependency set and avoid incompatibilities introduced by transitive dependency upgrades.

### Minimal examples

Preview the interface without preparing a model:

```bash
vates --demo
```

After preparing the model and expert data, run this from the repository root:

```bash
vates
```

## Data preparation

The example below uses separate directories for intermediate per-expert files and the final runtime blobs:

```text
models/
├── qwen3_next_80b_4bit/                  # 4-bit MLX main model
├── qwen3_next_expert_files_4bit_g64/     # Intermediate per-expert files
├── qwen3_next_experts_4bit_g64/          # CLI default expert directory
│   └── blobs/                            # Final runtime blobs
└── qn_mtp_weights.safetensors            # MTP weights
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

`mlx_streaming/prep/blob_layout.py` is the single definition of the byte layout and stays aligned with the runtime blob loader.

> [!WARNING]
> Data preparation needs substantially more disk space than the approximately 41 GB main-model directory alone. The split files and final blobs coexist during packing, and the MTP source shard adds approximately 3.30 GB (3.07 GiB). After validating `models/qwen3_next_experts_4bit_g64/blobs` and its `blob_index.json`, you may delete the intermediate `models/qwen3_next_expert_files_4bit_g64` directory. The 41 GB figure describes only the main-model weights, not peak preparation storage.

## Usage

### Interactive chat

Run all commands from the repository root.

Start the full-screen TUI:

```bash
vates
```

Set the MTP speculative width and generation length, and print statistics:

```bash
vates -k 4 -n 800 --stats
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
vates chat --plain
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
| `-k`, `--k` | MTP speculative width | `3` |
| `-n`, `--max-tokens` | Maximum new tokens per turn | `4096` |
| `--expert-slots` | Resident expert-pool capacity | `32` |
| `--spec-slots` | Number of side-region rows | Same as `--expert-slots` |
| `--system` | System prompt | None |
| `--stats` | Print token count, throughput, and accepted length | Off |
| `--plain` | Use the plain-text REPL | Off |
| `--demo` | Preview the TUI with a mock backend | Off |

View the complete option set for the installed version:

```bash
vates chat --help
```

## Configuration

The CLI uses `setdefault` to select the benchmark-validated configuration. Explicitly defined environment variables therefore take precedence.

| Environment variable | Production default | Purpose |
| --- | --- | --- |
| `STREAM_BLOB_LOADER` | `1` | Handle expert-pool misses with direct blob reads |
| `ZEROCOPY_DUAL_SOURCE` | `1` | Enable the zero-copy dual-source expert pool |
| `NATIVE_FUSED_PREFETCH` | `1` | Enable native predictive cross-layer prefetching |
| `SIDEREGION_LFU` | `1` | Enable the persistent LFU side-region cache |
| `KV_QUANT` | `1` | Enable K4/V3 KV quantization with SO(4) rotations |
| `MTP_ADAPTIVE_DEPTH` | `1` | Enable confidence-gated adaptive depth |
| `MTP_CONF_TAU` | `0.3` | Adaptive-depth confidence threshold |
| `MTP_DEPTH_MAX` | `3` | Maximum adaptive depth |

For example, explicitly override the resident-pool and side-region capacities:

```bash
EXPERT_SLOTS=32 POOL_SPEC_SLOTS=16 vates --stats
```

See `mlx_streaming/config.py` for additional experimental switches and their defaults.

> [!WARNING]
> `EXPERT_SLOTS` affects memory use, speed, and correctness. For K=3 and top-k=10, the default configuration uses `32` as the validated capacity floor. Rerun capacity-invariance and byte-level pool-oracle checks after changing it. The cap=48 unified-pool ablation below is a separate experiment, not the CLI default.

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

The figures below come from the ablation reports in this repository. Each ablation was measured in an independent experiment; the results were not obtained in one common hardware run and **must not be added together**. The observed 13–15 tok/s range comes from several end-to-end experiments with different configurations, prompts, warmup lengths, and output lengths. It is not a single standardized benchmark. Actual performance depends on the device, model files, context, and configuration.

**End-to-end production configuration test device:** MacBook Pro, Apple M5 (10-core CPU), 32 GB of physical unified memory, and a 1 TB internal Apple SSD.

| Item | Result |
| --- | --- |
| Storage and memory | 4-bit main-model weights: approximately 41 GB on disk; reported configuration: approximately 8.23–8.27 GiB from MLX `get_peak_memory`, measuring the in-flight tensor-allocation high-water mark—not process RSS or total system memory. Additional system headroom is required. See the [peak-memory report](benchmarks/reports/peak-shrink-2026-07-03.md). |
| Observed generation range | Approximately 13–15 tok/s across several end-to-end experiments; configurations, prompts, warmup lengths, and output lengths differ, so this is not a standardized benchmark |
| KV storage | Approximately 3.0 GiB to approximately 0.68 GiB at 128k context |
| Persistent LFU | Hit rate from 0.76 to approximately 0.81; measured throughput gain of approximately 8%–12% |
| Unified C++ pool ablation | With cap=48, K=3, MAXTOK=48, WARMUP=48, and REPEAT=2: 13.70 to 14.80 tok/s; the side region changed from double buffering to single buffering. This cap differs from the default cap=32. See the [unified-pool report](benchmarks/reports/cpp-unified-pool-final-2026-07-04.md). |
| MTP top-2 rescue | Approximately 10.8% throughput gain with zero output mismatches on the deterministic greedy prompts and configurations covered by the [top-2 rescue report](benchmarks/reports/tree-top2-rescue-2026-07-05.md) |
| Adaptive-depth MTP | Approximately 5%–6% throughput gain with zero output mismatches on the deterministic greedy prompts and configurations covered by the [adaptive-depth report](benchmarks/reports/adaptive-depth-2026-07-05.md) |
| Peak optimization | Avoiding unnecessary KV snapshots reduced the MLX allocation high-water mark by approximately 0.18–0.22 GiB |

Correctness validation includes:

- byte-for-byte identical output for the deterministic greedy prompts and configurations exercised by the capacity-invariance reports;
- `0 BAD` as the acceptance condition for the byte-level pool-oracle checks `DUAL_VERIFY` and `STG_VERIFY` in the reports that used them;
- a measured maximum per-layer routed-expert union of 30 in a single forward pass, which is why the default configuration uses 32 slots; and
- test coverage for both Python and native paths.

These oracles cover the documented deterministic greedy prompts, pool states, and benchmark configurations. They are regression evidence for those cases, not a mathematical guarantee of byte-identical output for every prompt, decoding mode, hardware state, or configuration.

Complete experimental records are available in [`benchmarks/reports/`](benchmarks/reports/).

Evaluated approaches that were rejected from the benchmark-validated path include full tree verification, event-gated asynchronous demand loading, sliding-window expert pools, and per-layer capacity reclamation. They were excluded because of throughput regressions, insufficient I/O budget, or no benefit under the current configuration. See the benchmark reports for details.

## Tests

After installing development dependencies, run the complete Python test suite:

```bash
.venv/bin/python -m pytest
```

The repository currently contains 60 test files covering expert pools, dual-source caching, blob layout, MTP, KV quantization, the TUI, configuration, and related modules. The benchmarked path is also validated with native tests, capacity-invariance checks, and byte-level pool-oracle checks.

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

It can fall back to a slower path, but native features such as predictive prefetching and the unified pool will be disabled. For real inference, derive the virtual environment's Python library path and pass it to Make:

```bash
PY_SITE="$("./.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
make -C native/ext PY_SITE="$PY_SITE" native_moe_ext
```

</details>

<details>
<summary><strong>Why is <code>uv sync</code> recommended?</strong></summary>

`uv sync` installs the dependency set validated in `uv.lock`, preventing transitive packages such as `transformers` from drifting to incompatible versions.

</details>

<details>
<summary><strong>Where should model files be placed?</strong></summary>

The defaults are under `models/` at the repository root. Use `--model`, `--expert-dir`, `--mtp-out`, and `--qn-config` to select other locations.

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
