"""专家 blob 字节布局的单一真相源（v1 affine / v2 mxfp4）。

Segment 五元组 (proj, tensor, np_dtype_name, shape, nbytes)，与现有
blob_loader._layout 的元组结构一致 → blob_loader 可直接消费、pack 取子集。
"""
from typing import List, Tuple

BLOB_V1_AFFINE = "expert_blob_v1"
BLOB_V2_MXFP4 = "expert_blob_v2_mxfp4"

Segment = Tuple[str, str, str, Tuple[int, int], int]  # proj, tensor, dtype, shape, nbytes


def _projs(hidden: int, inter: int) -> Tuple[Tuple[str, int, int], ...]:
    return (("gate_proj", inter, hidden), ("up_proj", inter, hidden), ("down_proj", hidden, inter))


def layout_for(fmt: str, hidden: int, inter: int, bits: int, group: int) -> Tuple[List[Segment], int]:
    segs: List[Segment] = []
    for proj, out_d, in_d in _projs(hidden, inter):
        words = in_d * bits // 32
        groups = in_d // group
        segs.append((proj, "weight", "uint32", (out_d, words), out_d * words * 4))
        if fmt == BLOB_V1_AFFINE:
            segs.append((proj, "scales", "uint16", (out_d, groups), out_d * groups * 2))
            segs.append((proj, "biases", "uint16", (out_d, groups), out_d * groups * 2))
        elif fmt == BLOB_V2_MXFP4:
            segs.append((proj, "scales", "uint8", (out_d, groups), out_d * groups * 1))
        else:
            raise ValueError(f"未知 blob format: {fmt}")
    stride = sum(s[-1] for s in segs)
    return segs, stride
