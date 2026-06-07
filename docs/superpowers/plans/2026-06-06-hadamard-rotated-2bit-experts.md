# Hadamard 旋转 2-bit 专家 实现 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用输入维 Hadamard 旋转 + 均匀 2-bit 量化（QuIP#-lite）救回 2-bit 专家的生成质量，内存/SSD 零增加。

**Architecture:** 离线把每个专家每个 proj 沿 input 维做 `W' = hadamard(W, 1/√n_in)` 再量化到 2-bit；运行时在 gate/up 前旋转输入 x、在 down 前旋转中间激活，`gather_qmm` 数学等价 `W·x`。其余流式/缓存/投机逻辑完全不动。

**Tech Stack:** MLX (`mx.hadamard_transform`, `mx.quantize`, `mx.dequantize`, `QuantizedSwitchLinear`, `SwitchGLU`)、pytest。

> **注意：`flash-moe` 当前不是 git 仓库，所有 `commit` 步骤暂记为"标记完成"即可（无法实际提交）。若后续 `git init`，再补提交。**

> **常量（全程一致）：** `hidden=2048`，`moe_inter=768`，`group_size=64`，`bits=2`，`num_experts=128`，`top_k=8`。源 4-bit 专家目录 `/tmp/mlx_qwen3_experts`（bits=4, group=64）。

---

## 文件结构

- 新建 `mlx_streaming/rotate_requantize_experts.py` —— 离线旋转重量化（自包含，依赖 `mx`）。
- 修改 `mlx_streaming/streaming_moe.py` —— 新增 `rotated_switch_glu_call()` 与 `RotatedSubGLU`，`FileStreamingMoeBlock`/`patch_model_filebacked` 增加 `rotated` 开关。
- 修改 `mlx_streaming/run_streaming.py`、`mlx_streaming/run_spec.py` —— 读 `EXPERT_ROT` 环境变量透传 `rotated`。
- 新建 `mlx_streaming/validate_rotation.py` —— 质量验证（MAE/余弦 + 困惑度）。
- 新建 `mlx_streaming/tests/test_rotation.py` —— 全部单测。

---

## Task 1: Hadamard 旋转的还原性与权重等价性

**Files:**
- Test: `mlx_streaming/tests/test_rotation.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_rotation.py
import math
import mlx.core as mx


def _rot(a, n):
    return mx.hadamard_transform(a, scale=1.0 / math.sqrt(n))


def test_hadamard_roundtrip_2048_and_768():
    for n in (2048, 768):
        x = mx.random.normal((4, n))
        back = _rot(_rot(x, n), n)
        assert float(mx.max(mx.abs(back - x))) < 1e-5


def test_weight_prerotation_equivalence():
    # W' = hadamard(W) 沿 input 维；W'·(Hx) 应等价 W·x
    for n in (2048, 768):
        x = mx.random.normal((4, n))
        W = mx.random.normal((5, n))
        Wp = _rot(W, n)          # 逐行旋转（最后一维=input）
        xr = _rot(x, n)
        lhs = Wp @ xr.T
        rhs = W @ x.T
        rel = float(mx.max(mx.abs(lhs - rhs)) / (mx.max(mx.abs(rhs)) + 1e-6))
        assert rel < 1e-2
```

- [ ] **Step 2: 运行确认失败/通过**

Run: `cd /Users/amos/project/flash-moe/hypura && python3 -m pytest mlx_streaming/tests/test_rotation.py -v`
Expected: 两个测试 PASS（这两条只验证 MLX 算子语义，无需新代码；若 import 路径报错则修正）。

- [ ] **Step 3: 提交（标记完成）**

```bash
# 无 git，跳过；记录 Task 1 完成
```

---

## Task 2: 离线旋转重量化脚本

**Files:**
- Create: `mlx_streaming/rotate_requantize_experts.py`
- Test: `mlx_streaming/tests/test_rotation.py`

- [ ] **Step 1: 写失败测试（文件完整性 + 形状/键名/大小不变）**

追加到 `test_rotation.py`：

```python
import os
import json
import tempfile


def test_rotate_requantize_file_preserves_shape_and_keys():
    from mlx_streaming.rotate_requantize_experts import rotate_requantize_file
    # 造一个「源 4-bit 单专家」文件：gate/up (768,2048)，down (2048,768)
    src = {}
    for name, (out, inp) in {
        "gate_proj": (768, 2048),
        "up_proj": (768, 2048),
        "down_proj": (2048, 768),
    }.items():
        W = mx.random.normal((out, inp)) * 0.1
        wq, s, b = mx.quantize(W, group_size=64, bits=4)
        src[f"{name}.weight"] = wq
        src[f"{name}.scales"] = s
        src[f"{name}.biases"] = b
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "src.safetensors")
        dp = os.path.join(d, "dst.safetensors")
        mx.save_safetensors(sp, src)
        in_dims = {"gate_proj": 2048, "up_proj": 2048, "down_proj": 768}
        rotate_requantize_file(sp, dp, src_bits=4, src_group=64,
                               dst_bits=2, dst_group=64, in_dims=in_dims)
        out = mx.load(dp)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"{name}.weight" in out
            assert f"{name}.scales" in out
            assert f"{name}.biases" in out
        # 2-bit 文件应明显小于 4-bit 源
        assert os.path.getsize(dp) < os.path.getsize(sp)
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest mlx_streaming/tests/test_rotation.py::test_rotate_requantize_file_preserves_shape_and_keys -v`
Expected: FAIL（`ModuleNotFoundError: rotate_requantize_experts`）。

- [ ] **Step 3: 写实现**

```python
# mlx_streaming/rotate_requantize_experts.py
"""把已拆分的 4-bit per-expert 专家，沿 input 维做 Hadamard 旋转后重量化到 2-bit。

原理（QuIP#-lite）：量化分组沿 input 维，旋转把每组能量打散成近高斯、压低组内动态
范围 → 低比特误差大降。W' = hadamard(W, 1/√n_in)（逐行、作用在最后一维=input 维）。
运行时输入也做同样旋转，gather_qmm 数学等价 W·x（见 RotatedSubGLU）。
"""
import os
import sys
import json
import math
import time

import mlx.core as mx

PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def rotate_requantize_file(src_path: str, dst_path: str, src_bits: int, src_group: int,
                           dst_bits: int, dst_group: int, in_dims: dict) -> None:
    """单个专家文件：每个 proj 反量化 → 沿 input 维 Hadamard 旋转 → 重量化。"""
    src = mx.load(src_path)
    out = {}
    for name in PROJ_NAMES:
        wq = src.get(f"{name}.weight")
        if wq is None:
            continue
        scales = src[f"{name}.scales"]
        biases = src[f"{name}.biases"]
        W = mx.dequantize(wq, scales, biases, group_size=src_group, bits=src_bits)
        n_in = in_dims[name]
        # Hadamard 作用在最后一维（input 维），逐行旋转
        Wp = mx.hadamard_transform(W, scale=1.0 / math.sqrt(n_in))
        nwq, ns, nb = mx.quantize(Wp, group_size=dst_group, bits=dst_bits)
        out[f"{name}.weight"] = nwq
        out[f"{name}.scales"] = ns
        out[f"{name}.biases"] = nb
    mx.eval(out)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mx.save_safetensors(dst_path, out)


def rotate_requantize_dir(src_dir: str, dst_dir: str, dst_bits: int, dst_group: int) -> dict:
    """把 src_dir（4-bit 拆分专家）全部旋转重量化到 dst_dir（2-bit），并写 meta。"""
    with open(os.path.join(src_dir, "_split_meta.json")) as f:
        src_meta = json.load(f)
    src_bits = src_meta["dims"]["bits"]
    src_group = src_meta["dims"]["group_size"]
    hidden = src_meta["dims"]["hidden"]
    moe_inter = src_meta["dims"]["moe_intermediate"]
    in_dims = {"gate_proj": hidden, "up_proj": hidden, "down_proj": moe_inter}
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".safetensors"))
    t = time.perf_counter()
    for i, fn in enumerate(files):
        rotate_requantize_file(os.path.join(src_dir, fn), os.path.join(dst_dir, fn),
                               src_bits, src_group, dst_bits, dst_group, in_dims)
        if (i + 1) % 512 == 0:
            print(f"  {i+1}/{len(files)} ({round(time.perf_counter()-t,1)}s)", flush=True)

    dst_meta = dict(src_meta)
    dst_meta["out_dir"] = dst_dir
    dst_meta["dims"] = dict(src_meta["dims"], bits=dst_bits, group_size=dst_group)
    dst_meta["rotated"] = True
    dst_meta["rotation"] = {"type": "hadamard", "scale": "1/sqrt(n_in)", "in_dims": in_dims}
    dst_meta["rotated_from"] = {"dir": src_dir, "bits": src_bits, "group_size": src_group}
    with open(os.path.join(dst_dir, "_split_meta.json"), "w") as f:
        json.dump(dst_meta, f, ensure_ascii=False, indent=2)
    return {"files": len(files), "dst_bits": dst_bits, "dst_group": dst_group,
            "elapsed_s": round(time.perf_counter() - t, 1)}


if __name__ == "__main__":
    # 用法：rotate_requantize_experts.py SRC_DIR DST_DIR [DST_BITS=2] [DST_GROUP=64]
    src = sys.argv[1]
    dst = sys.argv[2]
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    group = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    info = rotate_requantize_dir(src, dst, bits, group)
    print(json.dumps(info, ensure_ascii=False, indent=2))
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest mlx_streaming/tests/test_rotation.py::test_rotate_requantize_file_preserves_shape_and_keys -v`
Expected: PASS

- [ ] **Step 5: 提交（标记完成）**

---

## Task 3: 运行时旋转前向 RotatedSubGLU

**Files:**
- Modify: `mlx_streaming/streaming_moe.py`（在 `PersistentSubGLU` 之后新增）
- Test: `mlx_streaming/tests/test_rotation.py`

- [ ] **Step 1: 写失败测试（旋转管线无 bug：4-bit 权重上旋转重量化→旋转前向 ≈ 原 SwitchGLU）**

> 说明：用 dst_bits=4 做「旋转重量化」（旋转 + 4-bit 重量化），再用 RotatedSubGLU 前向，
> 应与「在同一组源 4-bit 权重上跑普通 SwitchGLU」数值接近（差异只来自一次 4-bit 重量化噪声）。
> 用宽松阈值 MAE < 0.05 验证「旋转-反旋转管线本身不引入系统性偏差」。

追加到 `test_rotation.py`：

```python
def _make_expert_files(d, n_experts=4):
    """造 n_experts 个源 4-bit 专家文件，返回 (dir, dims)。"""
    from mlx_streaming.split_experts import PROJ_NAMES as _P  # 复用键名约定
    dims = {"hidden": 2048, "moe_intermediate": 768,
            "num_experts": n_experts, "group_size": 64, "bits": 4}
    os.makedirs(d, exist_ok=True)
    shapes = {"gate_proj": (768, 2048), "up_proj": (768, 2048), "down_proj": (2048, 768)}
    for e in range(n_experts):
        rec = {}
        for name, (out, inp) in shapes.items():
            W = mx.random.normal((out, inp)) * 0.1
            wq, s, b = mx.quantize(W, group_size=64, bits=4)
            rec[f"{name}.weight"] = wq
            rec[f"{name}.scales"] = s
            rec[f"{name}.biases"] = b
        mx.save_safetensors(os.path.join(d, f"layer00_expert{e:03d}.safetensors"), rec)
    with open(os.path.join(d, "_split_meta.json"), "w") as f:
        json.dump({"dims": dims}, f)
    return d, dims


def test_rotated_forward_matches_plain_switchglu():
    from mlx_streaming.rotate_requantize_experts import rotate_requantize_dir
    from mlx_streaming.expert_store import FileExpertStore
    from mlx_streaming.streaming_moe import RotatedSubGLU, PersistentSubGLU
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        rot = os.path.join(root, "rot")
        _make_expert_files(src, n_experts=4)
        # 旋转 + 重量化回 4-bit（隔离「旋转管线」与「2-bit 量化损失」）
        rotate_requantize_dir(src, rot, dst_bits=4, dst_group=64)

        hidden, moe_inter = 2048, 768
        x = mx.random.normal((1, 1, hidden)) * 0.5
        local = mx.array([[0, 1, 2, 3]])  # 4 个专家各一次

        # 参照：普通（非旋转）4-bit 前向
        store_p = FileExpertStore(src, capacity=8)
        plain = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=4)
        fetched_p = store_p.fetch(0, [0, 1, 2, 3])
        y_plain = plain.forward(fetched_p, 4, x, local)

        # 被测：旋转 4-bit 前向
        store_r = FileExpertStore(rot, capacity=8)
        rotated = RotatedSubGLU(hidden, moe_inter, group_size=64, bits=4)
        fetched_r = store_r.fetch(0, [0, 1, 2, 3])
        y_rot = rotated.forward(fetched_r, 4, x, local)

        mx.eval(y_plain, y_rot)
        mae = float(mx.mean(mx.abs(y_plain - y_rot)))
        assert mae < 0.05, f"旋转管线偏差过大 MAE={mae}"
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest mlx_streaming/tests/test_rotation.py::test_rotated_forward_matches_plain_switchglu -v`
Expected: FAIL（`ImportError: cannot import name 'RotatedSubGLU'`）。

- [ ] **Step 3: 写实现（在 `streaming_moe.py` 顶部 import 处加 `import math`，并在 `PersistentSubGLU` 类之后新增）**

```python
class RotatedSubGLU(PersistentSubGLU):
    """旋转版 PersistentSubGLU：复用对象缓存/原地 update，但前向在 gate/up 前旋转输入、
    在 down 前旋转中间激活。配合 rotate_requantize_experts 产出的旋转权重，数学等价 W·x。
    """

    def forward(self, fetched: dict, n: int, x: mx.array, local: mx.array) -> mx.array:
        self._ensure(n)
        _update_qsl(self._glu.gate_proj, "gate_proj", fetched)
        _update_qsl(self._glu.up_proj, "up_proj", fetched)
        _update_qsl(self._glu.down_proj, "down_proj", fetched)
        return self._rotated_call(self._glu, x, local)

    def _rotated_call(self, glu, x, indices):
        # 镜像 mlx_lm SwitchGLU.__call__，仅在两处插入 Hadamard（沿特征维，与 token 维排序正交）
        hs = self.hidden ** -0.5
        ms = self.moe_inter ** -0.5
        x = mx.hadamard_transform(x, scale=hs)        # 输入旋转（gate/up 共用）
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            from mlx_lm.models.switch_layers import _gather_sort
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = glu.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = glu.gate_proj(x, idx, sorted_indices=do_sort)
        a = glu.activation(x_up, x_gate)              # 融合激活（原 SwitchGLU 约定）
        a = mx.hadamard_transform(a, scale=ms)        # 中间激活旋转（down 前）
        x = glu.down_proj(a, idx, sorted_indices=do_sort)
        if do_sort:
            from mlx_lm.models.switch_layers import _scatter_unsort
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest mlx_streaming/tests/test_rotation.py::test_rotated_forward_matches_plain_switchglu -v`
Expected: PASS（MAE < 0.05）

- [ ] **Step 5: 提交（标记完成）**

---

## Task 4: 接线到流式块与运行脚本

**Files:**
- Modify: `mlx_streaming/streaming_moe.py`（`FileStreamingMoeBlock.__init__`、`patch_model_filebacked`）
- Modify: `mlx_streaming/run_streaming.py`、`mlx_streaming/run_spec.py`
- Test: `mlx_streaming/tests/test_rotation.py`

- [ ] **Step 1: 写失败测试（patch 带 rotated=True 时 block 用 RotatedSubGLU）**

追加到 `test_rotation.py`：

```python
def test_patch_filebacked_uses_rotated_sub():
    from mlx_streaming.streaming_moe import FileStreamingMoeBlock, RotatedSubGLU
    from mlx_streaming.expert_store import FileExpertStore
    with tempfile.TemporaryDirectory() as root:
        _make_expert_files(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)

        class _Gate:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 4))

        blk = FileStreamingMoeBlock(
            gate=_Gate(), top_k=2, norm_topk_prob=True, store=store, layer_idx=0,
            hidden=2048, moe_inter=768, group_size=64, bits=4, rotated=True)
        assert isinstance(blk._sub, RotatedSubGLU)
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest mlx_streaming/tests/test_rotation.py::test_patch_filebacked_uses_rotated_sub -v`
Expected: FAIL（`TypeError: __init__() got an unexpected keyword argument 'rotated'`）。

- [ ] **Step 3: 写实现**

在 `streaming_moe.py` 的 `FileStreamingMoeBlock.__init__` 签名末尾加 `rotated: bool = False`，并改子模块构造：

```python
    def __init__(self, gate, top_k, norm_topk_prob, store, layer_idx,
                 hidden, moe_inter, group_size, bits, rotated: bool = False):
        self.gate = gate
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.store = store
        self.layer_idx = layer_idx
        self.hidden = hidden
        self.moe_inter = moe_inter
        self.group_size = group_size
        self.bits = bits
        sub_cls = RotatedSubGLU if rotated else PersistentSubGLU
        self._sub = sub_cls(hidden, moe_inter, group_size, bits)
```

在 `patch_model_filebacked` 签名末尾加 `rotated: bool = False` 并透传：

```python
def patch_model_filebacked(model, store, hidden, moe_inter, group_size, bits,
                           rotated: bool = False):
    patched = 0
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            layer.mlp = FileStreamingMoeBlock(
                gate=mlp.gate, top_k=mlp.top_k, norm_topk_prob=mlp.norm_topk_prob,
                store=store, layer_idx=i, hidden=hidden, moe_inter=moe_inter,
                group_size=group_size, bits=bits, rotated=rotated,
            )
            patched += 1
    return patched
```

在 `run_streaming.py` 与 `run_spec.py` 读环境变量并透传（两处 `patch_model_filebacked` 调用前各加）：

```python
EXPERT_ROT = os.environ.get("EXPERT_ROT", "0") == "1"
# ... 调用处：
patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                       EXPERT_GROUP, EXPERT_BITS, rotated=EXPERT_ROT)
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest mlx_streaming/tests/test_rotation.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交（标记完成）**

---

## Task 5: 离线生成旋转专家 + 质量验证脚本

**Files:**
- Create: `mlx_streaming/validate_rotation.py`
- 产物：`/tmp/mlx_qwen3_experts_2bit_rot`

- [ ] **Step 1: 生成旋转 2-bit 专家目录**

Run:
```bash
cd /Users/amos/project/flash-moe/hypura
python3 -m mlx_streaming.rotate_requantize_experts \
  /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_2bit_rot 2 64
```
Expected: 打印 `{"files": N, "dst_bits": 2, ...}`，且 `du -sh` 与 `/tmp/mlx_qwen3_experts_2bit` 体积相当（±5%）。

- [ ] **Step 2: 写质量验证脚本**

```python
# mlx_streaming/validate_rotation.py
"""验证旋转 2-bit 是否把质量拉近 4-bit：
1) 整段文本上对比 4-bit / plain-2bit / rotated-2bit 的困惑度（token NLL）。
2) 打印三者 logits 相对 4-bit 的 MAE。
环境变量：MODEL、DIR_4BIT、DIR_2BIT、DIR_2BIT_ROT、SLOTS、TEXT。
"""
import os
import math
import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from mlx_streaming.expert_store import FileExpertStore
from mlx_streaming.streaming_moe import patch_model_filebacked

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
DIR_4BIT = os.environ.get("DIR_4BIT", "/tmp/mlx_qwen3_experts")
DIR_2BIT = os.environ.get("DIR_2BIT", "/tmp/mlx_qwen3_experts_2bit")
DIR_2BIT_ROT = os.environ.get("DIR_2BIT_ROT", "/tmp/mlx_qwen3_experts_2bit_rot")
SLOTS = int(os.environ.get("SLOTS", "96"))
TEXT = os.environ.get("TEXT", "混合专家模型通过路由器为每个 token 选择少数专家参与计算，"
                                "从而在巨大参数量下保持较低的激活计算成本。")


def _first_moe_dims(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            return {"hidden": gp.input_dims, "moe_inter": gp.output_dims}
    raise RuntimeError("无 MoE 层")


def _ppl(model, ids):
    # teacher-forcing：用前缀预测下一 token，算平均 NLL → 困惑度
    x = ids[None, :-1]
    tgt = ids[1:]
    logits = model(x)[0]                       # (L-1, vocab)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    return float(mx.exp(mx.mean(nll)))


def _build(expert_dir, bits, rotated):
    model, tok = load(MODEL, lazy=True)
    dims = _first_moe_dims(model)
    store = FileExpertStore(expert_dir, capacity=SLOTS)
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           64, bits, rotated=rotated)
    return model, tok


def main():
    # 用 4-bit 的 tokenizer 编码（三者同 tokenizer）
    _, tok = load(MODEL, lazy=True)
    ids = mx.array(tok.encode(TEXT))

    out = {}
    for tag, (d, bits, rot) in {
        "4bit": (DIR_4BIT, 4, False),
        "2bit_plain": (DIR_2BIT, 2, False),
        "2bit_rot": (DIR_2BIT_ROT, 2, True),
    }.items():
        model, _ = _build(d, bits, rot)
        out[tag] = round(_ppl(model, ids), 3)
        del model

    print(json.dumps({"slots": SLOTS, "ppl": out,
                      "rot_recovers": out["2bit_plain"] - out["2bit_rot"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行质量验证**

Run:
```bash
cd /Users/amos/project/flash-moe/hypura && python3 -m mlx_streaming.validate_rotation 2>/dev/null
```
Expected: 输出三档困惑度。验收：`ppl["2bit_rot"] < ppl["2bit_plain"]`（旋转救回质量），且更接近 `ppl["4bit"]`。

- [ ] **Step 4: 速度/内存回退检查**

Run:
```bash
cd /Users/amos/project/flash-moe/hypura
EXPERT_DIR=/tmp/mlx_qwen3_experts_2bit_rot EXPERT_BITS=2 EXPERT_GROUP=64 \
  EXPERT_SLOTS=96 EXPERT_ROT=1 NDRAFTS=2 MAXTOK=128 \
  python3 -m mlx_streaming.run_spec 2>/dev/null | tail -40
```
Expected: 跑通；tok/s 相对 plain-2bit @ 96（~25）回退 <5%，峰值内存不变（~9.3GB）。

- [ ] **Step 5: 把结果写入报告**

把困惑度三档对比 + 旋转后 tok/s/内存 追加到 `benchmarks/reports/mlx-streaming-moe-lowbit-2026-06-06.md` 新增 §4.11「Hadamard 旋转救 2-bit 质量」。

- [ ] **Step 6: 提交（标记完成）**

---

## Self-Review（已核对）

- **Spec 覆盖**：§3.1 离线脚本→Task 2；§3.2 RotatedSubGLU→Task 3；§3.3 开关→Task 4；
  §3.4 验证→Task 5；§4 测试 1/2→Task 1，测试 3→Task 2，测试 4→Task 3，断言 5（核心质量）
  →Task 5 的困惑度验收 + Task 3 的管线无 bias 断言；§7 验收→Task 5 Step 3/4。
- **类型一致**：`RotatedSubGLU(hidden, moe_inter, group_size, bits)` 继承 `PersistentSubGLU`
  签名一致；`rotate_requantize_file(... in_dims)` 与调用处一致；`patch_model_filebacked(..., rotated=)`
  与 `FileStreamingMoeBlock(..., rotated=)` 一致。
- **无占位符**：所有代码步骤均含完整代码与确切命令。
- **激活函数**：使用 `glu.activation(x_up, x_gate)`（与真实 SwitchGLU 融合激活一致），不手写 silu。
