"""下载 Qwen3-Next 原版末分片并抽取/整理 MTP 权重为单文件。

MTP 全部权重在 model-00041-of-00041.safetensors(~3.9GB)。整理规则:
  1) 把 mlp.experts.{e}.{proj}.weight(512 个)stack 成 mlp.switch_mlp.{proj}.weight。
  2) 对所有 RMSNorm 权重 +1.0(Qwen3-Next 用 zero-centered RMSNorm,
     与 mlx-lm sanitize 对主模型的处理一致;MTP 被 sanitize 过滤,故需自行补)。
"""
import json
import os
import subprocess

import mlx.core as mx

# 与 mlx-lm qwen3_next.sanitize 一致的 norm 后缀(去掉 model.norm,因 MTP 用 mtp.norm)
_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _is_mtp_norm(key: str) -> bool:
    if key.endswith(".pre_fc_norm_hidden.weight") or key.endswith(
        ".pre_fc_norm_embedding.weight"
    ):
        return True
    if key == "mtp.norm.weight":
        return True
    return any(key.endswith(sfx) for sfx in _NORM_SUFFIXES)


def bump_mtp_norms(weights: dict) -> dict:
    out = {}
    for k, v in weights.items():
        if _is_mtp_norm(k) and v.ndim == 1:
            out[k] = v + 1.0
        else:
            out[k] = v
    return out


def stack_mtp_experts(weights: dict, num_experts: int) -> dict:
    out = dict(weights)
    prefix = "mtp.layers.0.mlp"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        keys = [f"{prefix}.experts.{e}.{proj}.weight" for e in range(num_experts)]
        if keys[0] not in out:
            continue
        stacked = mx.stack([out.pop(k) for k in keys])
        out[f"{prefix}.switch_mlp.{proj}.weight"] = stacked
    return out


SHARD = "model-00041-of-00041.safetensors"
REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"
SHARD_DIR = os.environ.get("MTP_SHARD_DIR", "/tmp/qn_mtp_shard")
OUT_PATH = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")

_DL_URL = (
    f"https://www.modelscope.cn/api/v1/models/{REPO}/repo"
    f"?Revision=master&FilePath={SHARD}"
)


def _expected_size() -> int:
    """从 ModelScope 文件列表 API 取 SHARD 的真实字节数。

    不能用 HEAD/content-length:API 重定向只回小 JSON,content-length 不可靠。
    """
    url = (
        f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files"
        f"?Revision=master&Recursive=true"
    )
    out = subprocess.run(
        ["curl", "-sL", "--max-time", "60", url],
        capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    files = data.get("Data", {}).get("Files") or data.get("Data", {}).get("files") or []
    for f in files:
        name = f.get("Name") or f.get("Path") or f.get("name")
        if name and SHARD in str(name):
            return int(f.get("Size") or f.get("size") or 0)
    return 0


def download_shard() -> str:
    os.makedirs(SHARD_DIR, exist_ok=True)
    out = os.path.join(SHARD_DIR, SHARD)
    expect = _expected_size()
    for attempt in range(1, 9):
        cur = os.path.getsize(out) if os.path.exists(out) else 0
        if expect and cur == expect:
            print(f"SKIP shard already complete ({cur}B)")
            return out
        print(f"download attempt {attempt}: have {cur}B / {expect}B")
        subprocess.run(
            ["curl", "-sL", "--max-time", "3600",
             "--speed-limit", "51200", "--speed-time", "30",
             _DL_URL, "-o", out],
        )
    cur = os.path.getsize(out) if os.path.exists(out) else 0
    if expect and cur != expect:
        raise RuntimeError(f"download failed: {cur}B != {expect}B")
    return out


def extract(shard_path: str, config_path: str, out_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    num_experts = cfg["num_experts"]
    all_w = mx.load(shard_path)
    mtp = {k: v for k, v in all_w.items() if k.startswith("mtp.")}
    print(f"原始 mtp 张量数: {len(mtp)}")
    mtp = stack_mtp_experts(mtp, num_experts)
    mtp = bump_mtp_norms(mtp)
    mx.eval(list(mtp.values()))
    mx.save_safetensors(out_path, mtp)
    print(f"已写出 {len(mtp)} 个张量 -> {out_path}")


def main():
    shard = download_shard()
    extract(shard, CONFIG, OUT_PATH)


if __name__ == "__main__":
    main()
