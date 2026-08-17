#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_dense_attn(tname, dtype, bq, bk, bd, wm, wn)             \
  instantiate_kernel(                                                        \
      "vates_dense_fa256_" #tname "_bq" #bq "_bk" #bk "_bd" #bd          \
      "_wm" #wm "_wn" #wn "_mask" #tname,                                \
      attention, dtype, bq, bk, bd, wm, wn, dtype, float)

instantiate_dense_attn(float16, half, 32, 8, 256, 4, 1);
instantiate_dense_attn(bfloat16, bfloat16_t, 32, 8, 256, 4, 1);
