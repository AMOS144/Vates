# Qwen3-Next-80B MTP 草稿接受率实测(Spike)

日期：2026-06-07
模型：`Qwen/Qwen3-Next-80B-A3B-Instruct`(主模型用 mlx-community 4-bit 流式 + 抽取的 BF16 MTP 头)
机器：Apple Silicon,32GB 统一内存

## 结论(go/no-go)

**GO —— 值得做完整 MTP 自投机。** 主决策指标 `mtp_vs_greedy_acc` ≈ **0.73–0.75**,远超 0.60 阈值。

## 实测数据

| hidden 变体 | mtp_vs_greedy_acc(主依据) | mtp_vs_text_acc | 位置数 |
|---|---|---|---|
| pre_final_norm | 0.7282 | 0.3889 | 103 |
| **post_final_norm** ✅ | **0.7476** | 0.4722 | 103 |

- **决策依据**是 `mtp_vs_greedy_acc`(MTP 对主模型贪心输出 t+2 的命中率,即自投机的真实接受率代理)。
- `post_final_norm` 略优(0.748 vs 0.728)→ MTP 消费的主模型 hidden 应取 **final-norm 之后**(与 vLLM/sglang 把 lm_head 前的 hidden 传给 MTP、MTP 再叠自己的 `pre_fc_norm_hidden` 一致)。
- `mtp_vs_text_acc`(对自然语料)较低属正常:语料 token 不是模型贪心选择,本就不该高;自投机对齐的是模型自身输出,看 greedy 指标。

## 速度收益预估

单 MTP 头预测 1 token ahead,接受率 0.748:
- K=1 草稿:每次主验证平均产出 1 + 0.748 ≈ **1.75 token** → ~1.75x。
- K=2–3 迭代草稿(接受率按位复合 0.748² ≈ 0.56、0.748³ ≈ 0.42):每次验证平均产出 ≈ 2.0–2.7 token。流式下一次验证 forward 的专家加载在多 token 间摊薄 → 预计 **~2–2.7x**,把当前 ~7.5 tok/s 顶到 **~15–20 tok/s**。仍不到 30,但显著优于 7.5。

## 关键实现坑(本 spike 踩到并已修)

1. **zero-centered RMSNorm**:Qwen3-Next 所有 RMSNorm 权重需 +1.0(mlx-lm `sanitize` 对主模型做,但过滤掉了 MTP)。抽取时对 MTP 的全部 norm 权重补 +1.0。
2. **MTP 共享主模型 embedding**:MTP 权重里**没有** `embed_tokens`。最初 MTP 用自建随机 embedding,接受率虚低到 ~12%;改为共享 `model.model.embed_tokens` 后跳到 0.73。
3. **下载大小**:ModelScope HEAD 的 content-length 不可靠(回 18B),改用文件列表 API 取真实大小(shard 41 = 3301131296B)。
4. **内存**:32GB 装不下 41GB 非流式主模型,测量走现成文件后端流式 patch(主模型 ~10GB 驻留)。

## 复现

```bash
cd hypura && source mlx_streaming/.venv/bin/activate
# 1) 下载末分片 + 抽取 MTP 权重(~3.3GB)
python -m mlx_streaming.extract_mtp
# 2) 测接受率(主模型走流式,专家目录 /tmp/qwen3_next_experts 需已拆分)
MAXTOK=96 HIDDEN_VARIANT=post_final_norm python -m mlx_streaming.validate_mtp
```

## 下一步(若推进完整自投机)

1. MTP 专家量化到 4-bit(当前 BF16 ~3GB 常驻),或并入流式专家池。
2. 实现自回归生成循环:MTP 迭代抽 K 个草稿 → 主模型一次性验证 K 个 → 接受最长匹配前缀。
3. **递归 cache 回滚**:验证拒绝时,gated-delta-net 的 `ArraysCache`(conv_state/ssm_state)不可裁剪,需在验证前 snapshot `cache.state`(小数组),按接受数 restore + 重放接受的 token;全注意力层的 `KVCache` 可裁剪,直接 trim。
4. 与流式 LRU 专家缓存集成,实测端到端 tok/s 与内存。
