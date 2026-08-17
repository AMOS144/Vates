"""Generate a packed K4/V3 specialization from MLX's Steel attention.

The attention math stays exactly upstream Steel (MMA + online softmax); only
the K/V block loaders and ABI are specialized for affine group-64 caches.
"""

from __future__ import annotations

import pathlib
import sys


LOADER = r'''
template <typename T, int BROWS, int BD, int BITS, int DST_LD,
          bool TRANSPOSE, int TGP_SIZE>
struct AffineQuantBlockLoader {
  const device uint* w;
  const device T* scales;
  const device T* biases;
  threadgroup T* dst;
  ushort thread_idx;

  METAL_FUNC AffineQuantBlockLoader(
      const device uint* w_, const device T* scales_,
      const device T* biases_, threadgroup T* dst_,
      ushort simd_group_id, ushort simd_lane_id)
      : w(w_), scales(scales_), biases(biases_), dst(dst_),
        thread_idx(simd_group_id * 32 + simd_lane_id) {}

  METAL_FUNC void load_safe(short2 dims) const {
    constexpr int WORDS = BD * BITS / 32;
    constexpr uint MASK = (1u << BITS) - 1u;
    for (int linear = thread_idx; linear < BROWS * BD; linear += TGP_SIZE) {
      const int row = linear / BD;
      const int col = linear - row * BD;
      T value = T(0);
      if (row < dims.y && col < dims.x) {
        const int bit = col * BITS;
        const int word = bit >> 5;
        const int shift = bit & 31;
        uint raw = w[row * WORDS + word] >> shift;
        if (shift + BITS > 32) {
          raw |= w[row * WORDS + word + 1] << (32 - shift);
        }
        raw &= MASK;
        const int group = col >> 6;
        value = T(raw) * scales[row * (BD / 64) + group]
              + biases[row * (BD / 64) + group];
      }
      if constexpr (TRANSPOSE) {
        dst[col * DST_LD + row] = value;
      } else {
        dst[row * DST_LD + col] = value;
      }
    }
  }

  METAL_FUNC void load_unsafe() const { load_safe(short2(BD, BROWS)); }
  METAL_FUNC void next() {
    w += BROWS * (BD * BITS / 32);
    scales += BROWS * (BD / 64);
    biases += BROWS * (BD / 64);
  }
};
'''


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one match, got {source.count(old)}: {old[:80]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    source_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])
    source = source_path.read_text()
    source = source.replace(
        'using namespace mlx::steel;\n',
        'using namespace mlx::steel;\n' + LOADER + '\n',
        1,
    )
    source = replace_once(source, ']] void attention(\n', ']] void quant_attention(\n')
    source = replace_once(
        source,
        '    const device T* K [[buffer(1)]],\n'
        '    const device T* V [[buffer(2)]],\n'
        '    device T* O [[buffer(3)]],\n'
        '    const constant AttnParams* params [[buffer(4)]],',
        '    const device uint* K [[buffer(1)]],\n'
        '    const device T* K_scales [[buffer(2)]],\n'
        '    const device T* K_biases [[buffer(3)]],\n'
        '    const device uint* V [[buffer(4)]],\n'
        '    const device T* V_scales [[buffer(5)]],\n'
        '    const device T* V_biases [[buffer(6)]],\n'
        '    device T* O [[buffer(7)]],\n'
        '    const constant AttnParams* params [[buffer(8)]],'
    )
    # Optional inputs disappear under function constants in our specialization;
    # sharing their declared IDs with live quant buffers is legal in Metal.
    source = replace_once(
        source,
        '  K += tidl.z * params->K_strides[0] + // Batch\n'
        '      kv_head_idx * params->K_strides[1]; // Head\n\n'
        '  V += tidl.z * params->V_strides[0] + // Batch\n'
        '      kv_head_idx * params->V_strides[1]; // Head',
        '  const ulong k_words = BD * 4 / 32;\n'
        '  const ulong kv_token_base = tidl.z * (params->K_strides[0] / k_words)\n'
        '      + kv_head_idx * (params->K_strides[1] / k_words);\n'
        '  K += tidl.z * params->K_strides[0]\n'
        '      + kv_head_idx * params->K_strides[1];\n'
        '  K_scales += kv_token_base * (BD / 64);\n'
        '  K_biases += kv_token_base * (BD / 64);\n'
        '  V += kv_token_base * (BD * 3 / 32);\n'
        '  V_scales += kv_token_base * (BD / 64);\n'
        '  V_biases += kv_token_base * (BD / 64);'
    )
    k_old = '''  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;'''
    k_new = '''  using KBlockLoader = AffineQuantBlockLoader<
      T, BK, BD, 4, LDK_tgp, true, WM * WN * 32>;
  using VBlockLoader = AffineQuantBlockLoader<
      T, BK, BD, 3, LDV_tgp, false, WM * WN * 32>;'''
    source = replace_once(source, k_old, k_new)
    source = replace_once(
        source,
        '''  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);''',
        '''  KBlockLoader loader_k(
      K, K_scales, K_biases, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, V_scales, V_biases, Vs, simd_group_id, simd_lane_id);'''
    )
    output_path.write_text(source)


if __name__ == '__main__':
    main()
