# 一次性 de-risk 基准程序（非生产；各自独立 main）。生产扩展在 ../ext。
# 按类别分级：io/（blob/array 读写）、compute/（量化线性、compute buffer）、metal/（Metal 内核）。
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
PYTHON ?= python3
PY_SITE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
MLX_INC := $(PY_SITE)/mlx/include
MLX_LIB := $(PY_SITE)/mlx/lib

MLX_FLAGS := -I$(MLX_INC) -L$(MLX_LIB) -lmlx -Wl,-rpath,$(MLX_LIB)
METAL_FLAGS := -O3 -std=c++17 -fobjc-arc -framework Foundation -framework Metal

all: io/pack_loader io/array_bench compute/qlinear_bench compute/compute_buffer_bench \
     metal/metal_staging_bench metal/metal_fused_moe_bench metal/native_moe_runner

io/pack_loader: io/pack_loader.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

io/array_bench: io/array_bench.cpp
	$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $@ $<

compute/qlinear_bench: compute/qlinear_bench.cpp
	$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $@ $<

compute/compute_buffer_bench: compute/compute_buffer_bench.cpp
	$(CXX) $(CXXFLAGS) $(MLX_FLAGS) -o $@ $<

metal/metal_staging_bench: metal/metal_staging_bench.mm
	$(CXX) $(METAL_FLAGS) -o $@ $<

metal/metal_fused_moe_bench: metal/metal_fused_moe_bench.mm
	$(CXX) $(METAL_FLAGS) -o $@ $<

metal/native_moe_runner: metal/native_moe_runner.mm
	$(CXX) $(METAL_FLAGS) -o $@ $<

clean:
	rm -f io/pack_loader io/array_bench compute/qlinear_bench compute/compute_buffer_bench \
	      metal/metal_staging_bench metal/metal_fused_moe_bench metal/native_moe_runner
