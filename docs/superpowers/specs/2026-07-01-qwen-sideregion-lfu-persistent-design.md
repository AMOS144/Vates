# 侧区持久 LFU 二级缓存 设计(Qwen dual-source)

> 状态:设计稿,待评审。日期:2026-07-01。模型:Qwen3-Next-80B-A3B(4bit,MoE+SSD 流式)。

## 1. 背景

流式 MoE 把专家权重放在磁盘,常驻池(resident pool)按 LFU 缓存热专家,miss 时按需从 SSD 读。为提高命中,有一条 `ZEROCOPY_DUAL_SOURCE` 的"双源"路径:预取的专家**直接落进常驻池的侧区(side region)物理行**,`acquire` 时把"真实区 slot 表 ∪ 侧区 e2r"合成一次 GPU gather 读出——不经 `promote` 拷贝,理论上一个专家字节只存一份。

**目标场景**:cap=32 单池只花 6.64GB 但命中 0.763;cap=64 单池命中 0.869 却要 9.38GB。我们想用**接近 cap=32 的内存拿到接近 cap=64 的命中**。

## 2. 现状问题(已实测)

端到端 A/B(80B,cap=32,64 token,`benchmarks/bench_dual_source.py`):

| 配置 | 命中 | 读盘 | active | tok/s | 正确性 |
|---|---|---|---|---|---|
| cap=32 单池(基线) | 0.763 | 8019 | 6.64GB | 5.13 | 确定(off-vs-off 0 失配) |
| cap=64 单池(目标) | 0.869 | 4450 | 9.38GB | 4.75 | 确定 |
| dual `POOL_SPEC_SLOTS=3` | 0.709 | 9857 | 4.59GB | 6.77 | 确定、对基线仅 1(本底) |
| dual `POOL_SPEC_SLOTS=32` | 0.768 | 7863 | 9.51GB | 4.70 | **不确定(自身 32 失配)** |

两个症状:**(A) 命中上不去**(spec=32 仅 0.768,远够不到 0.87);**(B) spec 放大后输出不确定**。

## 3. 根因(系统调试结论)

- **侧区字节永远正确**:快路径与 fallback 路径都加 `DUAL_VERIFY` 校验,spec=32 全程 **0 BAD**。所以漂移不是"侧区装错字节"。
- **`SIDEREGION_SYNC=1`(同步 fill、无后台线程)仍不确定** → 与后台线程无关,是"Metal 完成回调线程改 e2r vs 主线程读"这一层。
- **spec=3 确定、spec=32 不确定**,唯一变量是每层并发 pread 数(= `budget` = `spec_slots`)。spec=32 时每层 32 个 pread 洪水,放大了 demand 路径本就存在的异步 I/O 浮点噪声(streaming 前向本身有 ~5-logit 抖动)→ 边界 token 翻转。
- **共同的根**:C++ `reserve()`(`native/ext/native_prefetch.cpp` 约 470-477 行)**每次 fill 都把"不在当前预测集合 P"的侧区行全部淘汰**,给缺口重新 pread。于是侧区**永远只是"这一步的预测批"**:
  - 既**不跨步累积工作集** → 命中上不去(症状 A);
  - 又**每步重读洪水** → 放大噪声(症状 B)。

> 一句话:侧区当前是"一次性预取批",不是"持久缓存"。把它改成**持久 LFU** 同时解决 A 和 B。

## 4. 目标 / 非目标

**目标**
1. 侧区变成**跨步持久的 LFU 二级缓存**:热专家跨 token 保留,每步只 pread 真正新增的专家。
2. 命中随时间累积到接近 cap=64(目标 ≥0.85 @ ~7GB),内存显著低于 cap=64 单池(9.38GB)。
3. 正确性:warm 后 `exact_match` 对齐基线(或退化到与基线同级的良性本底,≤1~2 个晚位 token)。

**非目标**
- 不改 DeepSeek 路径(gen 默认 0、单缓冲行为保持)。
- 不追求 prefill 阶段走双源(prefill 仍走 host 路径,现状不变)。
- 不动 MTP 逻辑。

## 5. 设计概览

保持"零拷贝、字节只存一份"的核心;把侧区从"每步全清重填"改成"**单代(single-gen)持久 LFU**",并把 e2r 的**发布与淘汰收归主线程**做到 race-free。

### 5.1 单代(single-gen),不用双缓冲

双缓冲(spec_gens=2)会让两个 gen 各存一份工作集 → 内存翻倍,和"省内存"冲突。持久 LFU 用**单代**:一份工作集,`spec_gens=1`,物理行 = `cap + spec_slots`。

race-free 不再依赖"读写不同 gen",而依赖下面的**行不变式**。

### 5.2 行不变式(正确性核心)

**任一物理侧区行,在它被本前向 gather 引用期间,字节不被改写。**

用"三态行 + 主线程掌控 e2r"保证:
- **已发布(e2r)**:expert→row 已进 e2r,可被 gather 读;字节只读。
- **在飞(inflight)**:主线程已分配该行给某新专家,回调/后台线程正在 pread 写它;**不在 e2r**,故本前向 gather 绝不读它 → 写它安全。
- **空闲(free)**:未占用。

规则:
1. **只有主线程改 e2r 与 free**(发布 inflight→e2r、LFU 淘汰 e2r→free、分配 free→inflight)。时机在每层"预取提交之前"的固定点(block 已有 `begin_forward` 钩子附近)。→ 前向内 e2r 快照稳定,回调线程不会在 gather 中途改它。
2. **回调/后台线程只**:按主线程给定的 `(expert,row)` 列表 pread+memcpy 进 inflight 行,写完置 `done`(不进 e2r)。
3. **LFU 淘汰只淘 e2r 已发布行**,淘汰后进 free;绝不淘 inflight 行。被淘汰行在本前向之后才可能被复用 pread,复用写入要等**下一前向**主线程发布才进 e2r → 不影响任何在读前向。

### 5.3 每步流程(每层)

主线程(每层,预取提交前,称 `plan_commit`):
1. **发布**:把上一步 inflight 中 `done` 的行移入 e2r。
2. **算新增**:`P = 预测 ∖ 常驻(真实区) ∖ e2r ∖ inflight`(真正尚未缓存的)。
3. **分配行**:为 P 取行——优先 free;free 空则按 **LFU(freq 最小,LRU tie-break)淘汰 e2r 行**腾出;仍不足则本步少读几个(下步再补)。分配的行置 inflight。
4. 触发回调/后台对这些 inflight 行 pread(复用现有 `prefetch_pool_sideregion` 的读发布机制,但"发布"改为只置 done、不写 e2r)。

主线程 `acquire`(每层):
5. 读 e2r 快照 → 合成 `真实区表 ∪ 侧区 e2r` → 单次 gather(现有 `acquire_gpu_dual`)。

**LFU 频次(零 host 同步)**:不在 `acquire` 快路径回传(那会强制 `.tolist` host 同步、毁掉快路径)。改为**在 `reserve` 内计**:对"本步预测 P 中、且已在 e2r"的专家 `freq += 1`。语义 =「越常被预测越热」,完全在 C++、无 host 往返。新专家发布时 `freq = 1`。淘汰选 `freq` 最小且 ∉ 当前 P 者(tie-break:最小 expert id,确定性)。

### 5.4 与现有代码的差异(最小化)

- 现 `reserve()` 已"跳过已在 e2r 的专家"(只读新增),这部分**保留**。
- 主要改动:把 470-477 的"∉P 全淘汰"换成"**只在 free 空且需要行时按 LFU 淘汰**";新增 freq 计数;把发布/淘汰时机从回调线程挪到主线程 `plan_commit`(新增 C++ 入口)。

### 5.5 实现分期(降风险、可测)

上面 §5.2/5.3 是**目标架构**(主线程掌控 e2r、完全 race-free)。为降风险,分两期落地,以实测门控:

- **第一期(最小改动,先测)**:只把 `reserve()` 的淘汰从"∉P 全弃"改成"**LFU 持久:仅当 free 空且需要行时淘汰 freq 最小者**",并加 freq 计数(命中回传)。发布仍在原处。预期同时:命中累积↑ + 每步只读新增使 I/O 洪水消失 → 大概率恢复确定。**若第一期端到端 `exact_match` 达标且命中达标,即完成,不做第二期。**
- **第二期(仅在第一期确定性不达标时)**:把 e2r 的发布与淘汰收归主线程 `plan_commit`(§5.3),行分配随之上移;回调/后台只 pread 进 inflight 行、置 done。彻底 race-free。

两期共用同一套测试(§8),第一期后跑端到端 5/6/7 做 go/no-go。

## 6. 组件与接口

### 6.1 C++ `native/ext/native_prefetch.cpp`(第一期核心)
- `struct SideLayer` 增字段:`std::map<int,uint32_t> freq`(预测频次,LFU 分数)。
- `reserve()` 淘汰逻辑改 LFU 持久(见 5.3、5.5 第一期);freq 在 reserve 内维护(再预测 +1、新专家发布 =1)。
- 门控:`std::getenv("SIDEREGION_LFU")`(每次 reserve 读,便于测试切换;默认 0 → 走原"∉P 全弃"路径,DeepSeek/回退零影响)。
- `sideregion_contents(layer, gen)` 语义不变(读 e2r)。
- (第二期,仅需要时)新增 `sideregion_commit(layer, gen)`:主线程发布 done→e2r。

### 6.2 `mlx_streaming/core/cache/virtual_pool.py`
- 持久 LFU 模式用**单代**:`read_gen == fill_gen == 0`(不再 `&1` 双缓冲)。
- 保留调度器 `target_for/ahead_for` 不变。

### 6.3 `mlx_streaming/model_builder.py`
- LFU 模式:`spec_gens=1`(物理行 `cap+spec_slots`);不变量断言相应保持 `budget==spec_slots`。

### 6.4 `mlx_streaming/config.py`
- `sideregion_lfu() -> bool`(默认 0)。

## 7. 数据流

第一期数据流(reserve 内 LFU、零 host 同步):

```
预测(下层 gate) ──► reserve(回调线程,持锁)
                     ├─ P = 预测 ∖ 常驻
                     ├─ 对 P∩e2r 的专家 freq+1(越常预测越热)
                     ├─ 对 P∖e2r 的新专家:free 有则取;free 空则 LFU 淘汰(freq 最小且 ∉P)腾行
                     └─ (不再 ∉P 全弃 → 跨步累积)
                          │
                          ▼
                     pread+memcpy → 写行 → 发布 e2r(freq=1)
acquire(主线程,每层) ──► 读 e2r 快照 ──► 真实区 ∪ 侧区 → 单次 gather(零 .tolist)
```

第二期(仅需要时):把发布/淘汰上移主线程 `plan_commit`,回调只写 inflight 行、置 done。

## 8. 测试策略(TDD)

**C++/单元(不需大模型,经 `native_moe_ext` 直调,`SIDEREGION_LFU=1`)**
1. `test_sideregion_lfu_persist`:fill1 P={5,6}(spec=4,留空行);fill2 P={7,8}。断言 fill2 后 e2r ⊇ {5,6}(旧热专家**未被清**,与旧路径"∉P 全弃"相反),={5,6,7,8}。
2. `test_sideregion_lfu_evicts_min_freq`:spec=2,fill1 P={5,6} 填满;多次 fill P⊇{5} 使 freq[5]>freq[6];再 fill P={7}(free 空)→ 断言淘汰 6(freq 最小)、保留 5,e2r={5,7}。
3. `test_sideregion_lfu_off_is_legacy`:`SIDEREGION_LFU` 未设时,fill2 P 不含旧专家 → 旧专家被清(保持旧行为回退)。
4. (第二期,仅需要时)`test_sideregion_commit_publishes`:pread 完置 done,`commit` 后才进 e2r。

**端到端(80B,`bench_dual_source.py`)**
5. **确定性**:`SIDEREGION_LFU=1` dual-on 跑两次,on-vs-on `exact_match=true`(或 ≤1 本底)。
6. **正确性**:dual-on vs 基线 off,warm 后 `exact_match`(或 ≤1~2 晚位)。
7. **命中/内存**:目标 spec≈32 持久 LFU,命中 ≥0.85、active ≤ ~7.5GB、读盘显著低于基线;对照 cap=64 单池。

每个 C++/Python 改动配失败测试先行;每步验 `exact_match` 门控,破则停。

## 9. 风险与回退

- **风险1:确定性未完全恢复**。若 warm 后仍漂,先确认是否 demand 路径既有的良性本底(与基线同级则接受);否则 `plan_commit` 主线程发布是主要防线,已在设计内。
- **风险2:命中不达 0.85**。若持久 LFU 累积仍不足,回退为"仅证明 race-free + 省内存",或调 `spec_slots`/预测宽度。
- **风险3:C++ 并发回归**。所有 e2r/free 变更收归主线程,回调只写 inflight 行 → 显著缩小并发面。
- **总回退**:`SIDEREGION_LFU=0` 完全回到现状(甚至 `ZEROCOPY_DUAL_SOURCE=0` 回到单池),零影响。

## 10. 预期结果

- 命中:0.763(cap=32)→ 目标 ≥0.85,逼近 cap=64 的 0.869。
- 内存:目标 ~7GB,远低于 cap=64 的 9.38GB(省 ~2.4GB)。
- 正确性:warm 后 `exact_match` 对齐基线。
- 门控:任一端到端指标不达标即在对应 Task 停下复盘,不硬推。
