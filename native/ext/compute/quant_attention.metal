#include "mlx/backend/metal/kernels/utils.h"
#include "quant_attention_generated.h"

#define instantiate_quant_attn(tname, dtype, bq, bk, bd, wm, wn)             \
  instantiate_kernel(                                                        \
      "vates_k4v3_fa256_" #tname "_bq" #bq "_bk" #bk "_bd" #bd          \
      "_wm" #wm "_wn" #wn "_mask" #tname,                                \
      quant_attention, dtype, bq, bk, bd, wm, wn, dtype, float)

instantiate_quant_attn(float16, half, 16, 8, 256, 2, 1);
instantiate_quant_attn(float16, half, 32, 8, 256, 4, 1);
instantiate_quant_attn(float16, half, 16, 16, 256, 2, 1);
instantiate_quant_attn(float16, half, 32, 16, 256, 4, 1);
instantiate_quant_attn(bfloat16, bfloat16_t, 16, 8, 256, 2, 1);
instantiate_quant_attn(bfloat16, bfloat16_t, 32, 8, 256, 4, 1);
instantiate_quant_attn(bfloat16, bfloat16_t, 16, 16, 256, 2, 1);
instantiate_quant_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 1);
