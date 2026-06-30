"""预取时间分配报表生成器：读 4 档消融 JSON + host 探针，产出 markdown。

口径(为何要两套):
- 消融吞吐:MLX 惰性，gate matmul / pool scatter 的 GPU 执行落在前向末尾统一 eval，
  无法用墙钟单测;故用 4 档差值把"每段对端到端 tok/s 的真实影响"差出来。
    OFF   = 纯按需(无 staging)
    NOSUB = staging 已分配但完全不预取(地板:测 staging 分配本身的代价)
    NOPRO = predict + submit 开,promote 关(测预测 matmul + 后台 pread 洪水)
    ON    = 全开(再叠加 promote scatter)
  → staging 分配代价 = NOSUB - OFF
  → predict+submit  = NOPRO - NOSUB
  → promote          = ON   - NOPRO
  → 净预取代价        = ON   - OFF
- host 探针(PREFETCH_TPROF=1):量主线程"不可与 GPU 重叠"的 CPU 时间(锁/同步/建图/Python)。
  与吞吐口径互补:吞吐差里"墙钟探针看不到"的部分 ≈ GPU/IO(gate matmul 执行 + pread 抢带宽)。

用法:python benchmarks/make_prefetch_tprof_report.py /tmp/ab > 报表.md
其中目录需含 tprof_off.json / tprof_nosub.json / tprof_nopro.json / tprof_on.json。
"""
import json
import sys
from pathlib import Path

ORDER = [("OFF", "tprof_off.json"), ("NOSUB", "tprof_nosub.json"),
         ("NOPRO", "tprof_nopro.json"), ("ON", "tprof_on.json")]


def _load(d: Path):
    rows = {}
    for name, fn in ORDER:
        p = d / fn
        if p.exists():
            rows[name] = json.loads(p.read_text())
    return rows


def _g(d, *path, default=None):
    for k in path:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab")
    r = _load(base)
    if "ON" not in r or "OFF" not in r:
        sys.exit("缺少 ON/OFF JSON")

    def spec(n): return _g(r.get(n), "spec_tok_per_s", default=float("nan"))
    def tv(n): return _g(r.get(n), "t_verify_s", default=float("nan"))

    out = []
    out.append("# 主动预取路径 · 时间分配报表\n")
    out.append("> 64 槽 / K=3 / MAXTOK=96 / REPEAT=2(中位) / blob+双 nocache / EVICT=lfu / budget=16 / predict_width=32\n")

    # 1. 吞吐总表
    out.append("## 1. 四档消融总表\n")
    out.append("| 档位 | 含义 | spec tok/s | baseline | t_verify(s) | active(GB) | 读盘 | decode 命中 |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    desc = {"OFF": "纯按需(无 staging)", "NOSUB": "staging 分配但不预取",
            "NOPRO": "predict+submit,无 promote", "ON": "全开"}
    for n, _ in ORDER:
        d = r.get(n)
        if not d:
            continue
        out.append(f"| {n} | {desc[n]} | {d['spec_tok_per_s']:.2f} | "
                   f"{d['baseline_tok_per_s']:.2f} | {d['t_verify_s']:.3f} | "
                   f"{d['mlx_active_gb']:.2f} | {d['spec_disk_loads']} | "
                   f"{_g(d, 'miss_attrib', 'decode', 'hit_rate')} |")
    out.append("")

    # 2. 分段归因(吞吐差)
    out.append("## 2. 时间分配(按吞吐差归因)\n")
    out.append("| 阶段 | 口径 | Δspec tok/s | Δ% | Δt_verify(s) | 结论 |")
    out.append("|---|---|---:|---:|---:|---|")

    def seg(label, formula, a, b, verdict):
        if a not in r or b not in r:
            return
        ds = spec(a) - spec(b)
        dpct = ds / spec(b) * 100
        dtv = tv(a) - tv(b)
        out.append(f"| {label} | {formula} | {ds:+.2f} | {dpct:+.1f}% | {dtv:+.3f} | {verdict} |")

    seg("staging 分配", "NOSUB−OFF", "NOSUB", "OFF", "≈免费")
    seg("predict + submit", "NOPRO−NOSUB", "NOPRO", "NOSUB", "**主凶:后台 pread 抢带宽**")
    seg("promote(scatter)", "ON−NOPRO", "ON", "NOPRO", "**净正收益**(省读盘)")
    seg("净预取(总)", "ON−OFF", "ON", "OFF", "净亏损")
    out.append("")

    # 3. host 探针(ON)
    t = _g(r["ON"], "prefetch_tprof")
    if t:
        out.append("## 3. host 墙钟探针(ON,最终测量轮)\n")
        out.append("> 主线程不可重叠的 CPU 时间。占比小 → 说明预取税的大头在 GPU/IO(被 §2 吞吐口径捕获),不在 Python。\n")
        out.append("| 段 | total(s) | 占 wall | 单次(ms) |")
        out.append("|---|---:|---:|---:|")
        for key, lab in [("predict", "predict 建图"), ("submit", "submit 调用"),
                         ("promote", "promote 总"), ("promote_take", "·take 锁读"),
                         ("promote_route", "·route 同步"), ("promote_place", "·place 切片+scatter入图")]:
            s = t.get(key, {})
            out.append(f"| {lab} | {s.get('total_s', 0)} | "
                       f"{s.get('pct_wall', 0)}% | {s.get('avg_ms', '-')} |")
        out.append(f"| **host 合计** | **{t.get('host_total_s')}** | "
                   f"**{t.get('host_total_pct_wall')}%** | — |")
        c = t.get("counts", {})
        out.append(f"\n计数:predict_n={c.get('predict_n')} submit_n={c.get('submit_n')} "
                   f"promote_n={c.get('promote_n')} place_experts={c.get('place_experts')}\n")

    # 4. 结论
    out.append("## 4. 结论与修复方向\n")
    pp = spec("NOPRO") - spec("NOSUB")
    pm = spec("ON") - spec("NOPRO")
    out.append(f"1. **predict+submit 是瓶颈({pp:+.2f} tok/s)**,且几乎全是 GPU/IO:host 探针仅占 "
               f"~{_g(r['ON'], 'prefetch_tprof', 'host_total_pct_wall')}% wall。"
               f"submit 每层把 ≤budget 个专家 pread 进 staging,在统一内存带宽受限的 decode 上与计算抢带宽。")
    out.append(f"2. **promote 反而是净收益({pm:+.2f} tok/s)**:读盘 "
               f"{_g(r['NOPRO'], 'spec_disk_loads')}→{_g(r['ON'], 'spec_disk_loads')},"
               f"命中 {_g(r['NOPRO'], 'miss_attrib', 'decode', 'hit_rate')}→"
               f"{_g(r['ON'], 'miss_attrib', 'decode', 'hit_rate')}。之前\"promote 是瓶颈\"的结论被推翻。")
    out.append("3. **修复方向**:别动 promote;砍 submit 的 pread 体量——只 pread \"真·缺口\"(预测∩非常驻∩高分),"
               "或带宽空闲时才发,或进一步降 budget。本硬件(快 NVMe + 带宽受限 decode)上,任何后台 pread 都与计算争带宽,是预取转正的根本矛盾。")
    print("\n".join(out))


if __name__ == "__main__":
    main()
