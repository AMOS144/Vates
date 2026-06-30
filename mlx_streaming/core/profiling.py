"""热路径计时与诊断埋点（从 streaming_moe 抽出，集中管理）。

- PROF：细粒度分段计时（STREAM_PROF=1），probe 读取。
- WINDOW_PROF：同层 submit→promote 的 attention/GDN 窗口时间（WINDOW_PROF=1）。
- PREDICT_RECALL_PROF：运行时预测对实际路由的覆盖率（分诊预取覆盖问题，PREDICT_RECALL_PROF=1）。
这些 dict 为共享可变对象：各模块 import 后原地累加，外部（run_mtp_spec）读同一对象。
"""
import time

import mlx.core as mx

from mlx_streaming import config

_PROF_ON = config.stream_prof()
PROF = {"route": 0.0, "pyremap": 0.0, "fetch": 0.0, "matmul": 0.0, "combine": 0.0,
        "n_calls": 0}
WINDOW_PROF = {"sum_s": 0.0, "n": 0}
PREDICT_RECALL_PROF = {"hit": 0, "routed": 0, "n": 0}
# 主动预取路径 host 墙钟探针（PREFETCH_TPROF=1）：量主线程"不可与 GPU 重叠"的 CPU 时间。
# 注意：MLX 惰性——gate matmul / pool scatter 的 GPU 计算落在前向末尾统一 eval，本探针看不到，
# 那部分用消融吞吐口径(NATIVE_NO_SUBMIT/NATIVE_NO_PROMOTE 差值)测。本探针专测 host 段：
# - predict_s：block._native_fused_prefetch 里建预测图(gate(x)+argpartition[+predicted_set tolist])
# - submit_s ：stg.submit(prefetch_into_staging 的 host 调用，注册 GPU 完成回调)
# - promote_s：promote 总时长，再细分：
#   · take_s ：prefetch_staging_take（C++ 锁读已就绪记录）
#   · route_s：route_used_subset 的成员测试 .tolist()（GPU→host 同步，drain≤budget）
#   · place_s：_place_expert 循环（建惰性切片 + 池 scatter 入图，不含其 GPU 执行）
# *_n 为各段调用次数（按"层·前向"计）。关时各调用点直接跳过，零开销。
PREFETCH_TPROF = {
    "predict_s": 0.0, "predict_n": 0,
    "submit_s": 0.0, "submit_n": 0,
    "promote_s": 0.0, "promote_n": 0,
    "take_s": 0.0, "route_s": 0.0, "place_s": 0.0, "place_experts": 0,
}
TPROF_ON = config.prefetch_tprof()


def note_tprof(seg: str, dt: float, *, count_key: "str | None" = None, count: int = 1) -> None:
    """累加预取某段 host 墙钟。dt 为秒;可选 count_key 同步累加调用/专家计数。"""
    PREFETCH_TPROF[seg] += dt
    if count_key is not None:
        PREFETCH_TPROF[count_key] += count


def tprof_reset() -> None:
    for k in PREFETCH_TPROF:
        PREFETCH_TPROF[k] = 0 if isinstance(PREFETCH_TPROF[k], int) else 0.0
# miss 归因（MISS_ATTRIB=1，promote 之后 / acquire 之前统计本层真实路由的命中构成）：
# - resident_hit：acquire 前已驻留（LRU 历史 + 本次 promote 命中）
# - miss_A_predicted：miss 且"在预测集里"——预测对了但没进池（budget 丢/时序没到/被驱逐）
# - miss_B_unpredicted：miss 且"不在预测集里"——预测器召回缺口（根本没预测到）
# dec_* 为 decode/verify 热路径专用桶（seq 短，与 prefill 长 seq 分离）：原 miss_attrib
# 仅埋在 host 分支、被 prefill 样本主导，无法反映 decode 热路径的真实 A/B 构成。
MISS_ATTRIB = {"routed": 0, "resident_hit": 0, "miss_A_predicted": 0,
               "miss_B_unpredicted": 0, "n": 0,
               "dec_routed": 0, "dec_resident_hit": 0, "dec_miss_A": 0,
               "dec_miss_B": 0, "dec_n": 0,
               # miss_A 细分（decode 桶）：时序=就绪集里没有它（pread 没完成）；
               # 驱逐=就绪过（pread 完成）但 acquire 前已不在池（被驱逐/没写进）。
               "dec_miss_A_timing": 0, "dec_miss_A_evicted": 0}


def note_miss_attrib(uniq, predicted, resident, is_decode: bool, ready=None) -> None:
    """统计本层真实路由 uniq 的命中构成，分 prefill / decode 两桶累加。

    uniq：本层真实路由专家集合；predicted：本层预测集；resident：acquire 前驻留集；
    ready：本层最近 promote take 到的"已就绪(pread 完成)"专家集（用于细分 miss_A）。
    供 host 路径与 GPU remap 路径共用，确保两条路径口径一致。
    """
    pset = predicted or set()
    rset = ready or set()
    for e in uniq:
        MISS_ATTRIB["routed"] += 1
        if is_decode:
            MISS_ATTRIB["dec_routed"] += 1
        if e in resident:
            MISS_ATTRIB["resident_hit"] += 1
            if is_decode:
                MISS_ATTRIB["dec_resident_hit"] += 1
        elif e in pset:
            MISS_ATTRIB["miss_A_predicted"] += 1
            if is_decode:
                MISS_ATTRIB["dec_miss_A"] += 1
                # 就绪集里有它=pread 完成过却仍 miss → 驱逐；否则 pread 没完成 → 时序。
                if e in rset:
                    MISS_ATTRIB["dec_miss_A_evicted"] += 1
                else:
                    MISS_ATTRIB["dec_miss_A_timing"] += 1
        else:
            MISS_ATTRIB["miss_B_unpredicted"] += 1
            if is_decode:
                MISS_ATTRIB["dec_miss_B"] += 1
    MISS_ATTRIB["n"] += 1
    if is_decode:
        MISS_ATTRIB["dec_n"] += 1


# 并集专家数探针(UNION_PROF=1):按本次前向的 seq 长度分桶,记每层路由专家"去重并集"大小。
# seq=K(MTP verify,如 K=3)的桶即"K 个 token 的专家并集",决定池 cap 下限;seq=1 为 decode、
# seq=chunk 为分块 prefill。值为 {seq: [sum_union, n_layer_calls]}。默认关、零开销。
UNION_PROF: "dict[int, list]" = {}
UNION_ON = config.union_prof()


def note_union(seq: int, union_count: int) -> None:
    e = UNION_PROF.setdefault(int(seq), [0, 0])
    e[0] += int(union_count)
    e[1] += 1


def union_reset() -> None:
    UNION_PROF.clear()


def prof_reset():
    for k in PROF:
        PROF[k] = 0.0


def _tick(seg, t0):
    mx.eval  # noqa  (占位,避免误用)
    PROF[seg] += time.perf_counter() - t0
