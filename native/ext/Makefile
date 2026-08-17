# 生产 MLX 扩展构建：cmake 把 native_moe_mlx_ext.cpp 编成
# mlx_streaming/native_moe_ext$(EXT_SUFFIX).so（nanobind + MLX Primitive）。
PYTHON ?= python3
PY_SITE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NANOBIND_DIR := $(shell $(PYTHON) -m nanobind --cmake_dir)
MLX_CMAKE_DIR := $(PY_SITE)/mlx/share/cmake/MLX

native_moe_ext:
	cmake -S . -B build -DPython_EXECUTABLE=$(abspath $(PYTHON)) -Dnanobind_DIR=$(NANOBIND_DIR) -DMLX_DIR=$(MLX_CMAKE_DIR) -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(abspath ../../mlx_streaming)
	cmake --build build --target native_moe_ext -j

clean:
	rm -rf build
	rm -f ../../mlx_streaming/native_moe_ext*.so
