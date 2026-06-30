"""Task1 GATE spike：验证图内物化方案的两条地基机制。

机制1（依赖/排序）：自定义 primitive 在 eval_gpu 里写输出 buffer，下游 mx.take 从输出 gather
  → MLX 必须把 gather 排在 fill 之后（依赖边成立），200 次确定性读到填好的字节。
机制2（零拷贝别名 + 保留）：out 经 copy_shared_buffer 别名 in[0]，原地只写「偶数行」
  → 偶数行=fill、奇数行保留原值、out 与 src 同 buffer（零拷贝）。

注意：正确顺序是「先 copy_shared_buffer 别名、再原地写」；反过来（先写 in[0] 再别名）不生效。
两条全成立 = 图内物化侧区方案地基稳，可进 Task2。
"""
import mlx.core as mx
import mlx_streaming.native_moe_ext as N


def test_ingraph_alias_gather_deterministic_and_preserves_rows():
    N_ROWS, FILL = 64, 777
    idx = mx.array([0, 5, 17, 63], dtype=mx.uint32)   # 含偶数行(填)与奇数行(保留)
    bad = 0
    for _ in range(200):
        src = mx.arange(N_ROWS, dtype=mx.uint32) + 1000   # 哨兵：每行=1000+row
        mx.eval(src)                                        # 模拟池来自上一前向已落地
        out = N.materialize_spike(src, FILL)                # eval_gpu 别名+原地写偶数行
        g = mx.take(out, idx)
        mx.eval(g)
        # 偶数行应为 FILL，奇数行应保留 1000+row
        expect = [FILL if int(i) % 2 == 0 else 1000 + int(i) for i in idx.tolist()]
        if g.tolist() != expect:
            bad += 1
    assert bad == 0, f"别名/排序机制 {bad}/200 次不符 → 机制不成立，停"
