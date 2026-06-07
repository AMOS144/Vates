"""Phase 0 环境探针：核对 MLX 版本、内存 API、use_mmap/lazy 是否可用。"""
import inspect
import mlx.core as mx


def main():
    print("== 内存/缓存/常驻相关符号 ==")
    print([n for n in dir(mx) if any(k in n for k in ("mem", "cache", "wired", "peak"))])

    print("== metal 可用性与设备信息 ==")
    print("metal.is_available:", mx.metal.is_available())
    try:
        print("device_info:", mx.metal.device_info())
    except Exception as e:  # 不同版本可能在 mx.device_info
        print("mx.metal.device_info 失败:", e)
        try:
            print("mx.device_info:", mx.device_info())
        except Exception as e2:
            print("mx.device_info 也失败:", e2)

    print("== mlx_lm.load 签名（确认 lazy / use_mmap）==")
    from mlx_lm import load
    print(inspect.signature(load))
    try:
        from mlx_lm.utils import load_model
        print("load_model 签名:", inspect.signature(load_model))
    except Exception as e:
        print("load_model 取签名失败:", e)

    print("== mx.load 是否支持 use_mmap ==")
    print(inspect.signature(mx.load))


if __name__ == "__main__":
    main()
