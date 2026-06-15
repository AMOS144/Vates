# 自定义 Pack Loader de-risk 结论

日期: 2026-06-10

目标: 验证 `layerXX.pack + layerXX.index` 是否能减少 per-expert 小文件 `mx.load` / safetensors 解析开销。

## 做法

新增离线 pack writer:

- `mlx_streaming/prep/pack_expert_ranges.py`
- 直接解析 safetensors header，复制 tensor payload bytes。
- 生成 `layerXX.pack`、`layerXX.index.json`、`layerXX.idx`。

新增 C++ range loader:

- `native/pack_loader/main.cpp`
- 读取 `.idx` TSV。
- 按 expert id 收集 tensor byte ranges。
- 合并相邻 range 后用 `pread` 读取。
- 输出 `ranges / bytes / elapsed_ms / checksum`。

新增 microbench:

- `mlx_streaming/cli/probe_pack_loader.py`
- 对比同一层 N 个专家:
  - `N × mx.load(layerXX_expertYYY.safetensors) + mx.eval`
  - C++ `pack_loader bench --pack ... --index ... --experts ...`

## 结果

测试目录:

```text
models/qwen3_next_experts_bnd12_l43_l47_6_g128
layer=43
repeat=32
```

v1 tensor-range + checksum 结果:

```text
n=1   small_files=0.6184ms  pack_read=2.2410ms  speedup=0.276
n=2   small_files=0.1854ms  pack_read=4.4368ms  speedup=0.042
n=4   small_files=0.3901ms  pack_read=8.8574ms  speedup=0.044
n=8   small_files=0.8023ms  pack_read=17.8703ms speedup=0.045
n=16  small_files=1.6139ms  pack_read=35.7450ms speedup=0.045
```

随后修正两个问题：

- 改为 expert blob range（每个 expert 一个连续 blob）。
- 增加 `--no-checksum`，避免把逐字节 checksum CPU 开销算入 I/O 上界。

v2 expert-blob range read 结果:

```text
n=1   small_files=0.6686ms  pack_read=0.1257ms  speedup=5.321
n=2   small_files=0.3000ms  pack_read=0.1573ms  speedup=1.907
n=4   small_files=0.4781ms  pack_read=0.2464ms  speedup=1.940
n=8   small_files=0.9086ms  pack_read=0.5487ms  speedup=1.656
n=16  small_files=1.9873ms  pack_read=0.9689ms  speedup=2.051
```

这说明“range read 上界”在 n=16 时达到继续条件。

进一步用 MLX C++ raw pointer 构造 array 并 eval:

```text
n=8:
  C++ pack read + MLX array eval = 47.8689ms / 32 = 1.496ms
  Python small files mx.load      = 1.5326ms

n=16:
  C++ pack read + MLX array eval = 69.0527ms / 32 = 2.158ms
  Python small files mx.load      = 2.5323ms
```

## 结论

自定义 pack range read 的 I/O 上界在 v2 后达到门槛，但 MLX array 构造/eval 把大部分收益吃掉。

计划中的继续条件是:

```text
pack range read 对 8 或 16 个专家至少比小文件 mx.load 快 2 倍
```

v2 纯 range read 满足 n=16 的进入条件，但更关键的 MLX tensor 构造 de-risk 显示:

```text
pack pread + mx::array construction + eval 只比 Python mx.load 小幅快，未达到端到端接入价值。
```

因此不继续做:

- `FileExpertStore` 的 `EXPERT_PACK=1` 正式运行时接入

## 判断

这个结果说明 pack 格式确实能降低底层 range read 成本，但当前收益不足以覆盖 MLX array 构造成本、Python/C++ 边界和运行时接入复杂度。真正瓶颈已经从“小文件读”转移到“tensor construction / MLX array materialization”。

## 后续建议

停止 custom pack loader 路线。

当前可用主配置仍保持:

```bash
EXPERT_DIR=models/qwen3_next_experts_bnd12_l43_l47_6_g128
K=3
EXPERT_SLOTS=208
PIN_HOT=16
PIN_CAL_TOK=32
EVICT_POLICY=lru
CROSS_LAYER_PREFETCH=0
EXPERT_BUNDLE=0
```
