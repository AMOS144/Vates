# 一次性 de-risk 基准程序（非生产；各自独立 main）。生产扩展在 ../ext。
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
PY_SITE := $(abspath ../../.venv/lib/python3.12/site-packages)
MLX_INC := $(PY_SITE)/mlx/include
MLX_LIB := $(PY_SITE)/mlx/lib

MLX_FLAGS := -I$(MLX_INC) -L$(MLX_LIB) -lmlx -Wl,-rpath,$(MLX_LIB)
METAL_FLAGS := -O3 -std=c++17 -fobjc-arc -framework Foundation -framework Metal

all: pack_loader array_bench qlinear_bench compute_buffer_bench \
     metal_staging_bench metal_fused_moe_bench native_moe_runner

pack_loader: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

array_bench: array_bench.cpp
	$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $@ $<

qlinear_bench: qlinear_bench.cpp
	$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $@ $<

compute_buffer_bench: compute_buffer_bench.cpp
	$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $@ $<

metal_staging_bench: metal_staging_bench.mm
	$(CXX) $(METAL_FLAGS) -o $@ $<

metal_fused_moe_bench: metal_fused_moe_bench.mm
	$(CXX) $(METAL_FLAGS) -o $@ $<

native_moe_runner: native_moe_runner.mm
	$(CXX) $(METAL_FLAGS) -o $@ $<

clean:
	rm -f pack_loader array_bench qlinear_bench compute_buffer_bench \
	      metal_staging_bench metal_fused_moe_bench native_moe_runner
