"""Route 3 底座回归：验证 C++ 拥有的池 buffer（pool_owned_zeros）的两条关键性质。

1. 地址稳定性：把 owned 池数组反复喂给 MLX 图（matmul/take/加法）并 eval 后，
   底层 buffer 指针跨前向恒定——即 MLX 不会 donation/迁移这块 C++-owned buffer。
2. 直写可见性：C++ 通过 pool_write_stacked 直写某物理行后，MLX 读该行能看到新值——
   即「C++ 直写 + MLX 读」在 C++-owned 池 buffer 上成立且不错槽。

这两条是 Route 3 底座（C++ 拥有池 buffer、侧区/demand 异步直写、删除 MLX scatter）的地基，
用生产接口 pool_owned_zeros / pool_write_stacked 覆盖，作为长期不变量守护。
"""
import mlx.core as mx

from mlx_streaming import native_moe_ext as N


def test_owned_buffer_address_stable_across_mlx_graph():
    """C++-owned 池数组反复参与 MLX 图运算后底层指针不变。"""
    pool = N.pool_owned_zeros([32, 32], "uint32")   # (32,32) uint32，全 0
    p0 = N.array_data_ptr(pool)
    assert p0 != 0

    # 模拟稳态解码：把 pool 当只读输入反复搭图 + eval（matmul 触发 donation 的典型场景）。
    for _ in range(50):
        w = pool.astype(mx.float32)
        y = (w @ w.T).sum()
        mx.eval(y)
        z = mx.take(pool.reshape(-1), mx.array([0, 5, 17, 1000], dtype=mx.uint32))
        mx.eval(z)

    p1 = N.array_data_ptr(pool)
    assert p1 == p0, f"buffer 迁移了: {hex(p0)} -> {hex(p1)}"


def test_cpp_direct_write_visible_to_mlx_read():
    """C++ 经 pool_write_stacked 直写某行后，MLX 读该行得到新值。"""
    pool = N.pool_owned_zeros([64, 4], "uint32")
    # 先确认目标行初始为 0。
    assert mx.take(pool, mx.array([10], dtype=mx.uint32), axis=0).reshape(-1).tolist() == [0, 0, 0, 0]

    # C++ 直写第 10 行（模拟侧区/demand 预取落该行）。
    src = mx.array([[0xABCD1234, 1, 2, 3]], dtype=mx.uint32)   # stacked (m=1, 4)
    N.pool_write_stacked([pool], [src], [10])

    got = mx.take(pool, mx.array([10], dtype=mx.uint32), axis=0).reshape(-1)
    mx.eval(got)
    assert got.tolist() == [0xABCD1234, 1, 2, 3], got.tolist()


def test_direct_write_survives_graph_reuse():
    """先跑一堆图（可能触发迁移）→ 断言地址不变 → 再 C++ 直写 → MLX 读得到新值。

    这一条把"地址稳定"和"直写可见"串起来，最贴近底座的实际用法：
    池 buffer 长期存活、反复被读，侧区随时异步直写新行。
    """
    pool = N.pool_owned_zeros([256, 4], "uint32")
    p0 = N.array_data_ptr(pool)
    for i in range(30):
        y = (pool.astype(mx.float32) * float(i)).sum()
        mx.eval(y)
    assert N.array_data_ptr(pool) == p0

    src = mx.array([[777, 0, 0, 0]], dtype=mx.uint32)
    N.pool_write_stacked([pool], [src], [200])
    got = mx.take(pool, mx.array([200], dtype=mx.uint32), axis=0).reshape(-1)
    mx.eval(got)
    assert got.tolist() == [777, 0, 0, 0]
