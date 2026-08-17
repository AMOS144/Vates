#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

mx::array k4v3_fused_causal_attention(
    const mx::array& q,
    const mx::array& k_weight,
    const mx::array& k_scales,
    const mx::array& k_biases,
    const mx::array& v_weight,
    const mx::array& v_scales,
    const mx::array& v_biases,
    float scale,
    int q_block = 32,
    int k_block = 8,
    mx::StreamOrDevice s = {});

mx::array dense_fused_causal_attention(
    const mx::array& q, const mx::array& k, const mx::array& v,
    float scale, mx::StreamOrDevice s = {});
