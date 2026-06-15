// fused MoE 计算：合成/真实 mmap staging、staged(预切片权重)、slot(local slot 索引)三条工厂。
// 实现见 native_fused.cpp；这里只暴露给 bindings 的自由函数声明（默认参数放在声明处）。
#pragma once
#include "native_common.h"

mx::array fused_moe(
    const mx::array& x, const mx::array& expert_ids, const mx::array& scores,
    const std::string& compute_dir, int layer, int hidden, int inter, int group,
    int bits, int num_experts, bool synthetic, mx::StreamOrDevice s);

mx::array fused_moe_staged(
    const mx::array& x, const mx::array& scores,
    const mx::array& gate_w, const mx::array& gate_s, const mx::array& gate_b,
    const mx::array& up_w, const mx::array& up_s, const mx::array& up_b,
    const mx::array& down_w, const mx::array& down_s, const mx::array& down_b,
    int hidden, int inter, int group, int bits, mx::StreamOrDevice s);

mx::array fused_moe_slots(
    const mx::array& x, const mx::array& local_slots, const mx::array& scores,
    const mx::array& gate_w, const mx::array& gate_s, const mx::array& gate_b,
    const mx::array& up_w, const mx::array& up_s, const mx::array& up_b,
    const mx::array& down_w, const mx::array& down_s, const mx::array& down_b,
    int hidden, int inter, int group, int bits, mx::StreamOrDevice s);
