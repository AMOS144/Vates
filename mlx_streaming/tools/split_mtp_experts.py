"""Create per-expert quantized files for the bounded MTP expert cache."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mtp.qwen3_next_mtp import load_mtp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/qwen3_next_80b_4bit/config.json")
    parser.add_argument("--weights", default="models/qn_mtp_weights.safetensors")
    parser.add_argument("--out", default="models/qn_mtp_experts_2bit_g64")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    with open(args.config) as file:
        model_args = ModelArgs.from_dict(json.load(file))
    model = load_mtp(
        model_args, args.weights, quantize=True,
        bits=args.bits, group_size=args.group_size,
    )
    switch = model.layer.mlp.switch_mlp
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    projections = ("gate_proj", "up_proj", "down_proj")
    blob = open(output / "layer100.blob", "wb")
    for expert in range(int(model_args.num_experts)):
        arrays = {}
        for projection in projections:
            params = getattr(switch, projection).parameters()
            for name in ("weight", "scales", "biases"):
                arrays[f"{projection}.{name}"] = mx.contiguous(params[name][expert])
        mx.eval(arrays)
        mx.save_safetensors(str(output / f"expert{expert:03d}.safetensors"), arrays)
        for projection in projections:
            for name in ("weight", "scales", "biases"):
                value = arrays[f"{projection}.{name}"]
                if value.dtype == mx.bfloat16:
                    value = value.view(mx.uint16)
                blob.write(np.asarray(value).tobytes(order="C"))
    blob.close()
    (output / "meta.json").write_text(json.dumps({
        "bits": args.bits,
        "group_size": args.group_size,
        "num_experts": int(model_args.num_experts),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
