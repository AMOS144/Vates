cmake_minimum_required(VERSION 3.22)
project(native_moe_mlx_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(MLX CONFIG REQUIRED)

nanobind_add_module(
  native_moe_ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN mlx
  compute/fused_moe.cpp
  io/blob_load.cpp
  io/bg_reader.cpp
  prefetch/prefetch.cpp
  pool/owned_pool.cpp
  pool/side_region.cpp
  pool/demand.cpp
  bindings.cpp
)

target_link_libraries(native_moe_ext PRIVATE mlx)

if(BUILD_SHARED_LIBS)
  target_link_options(native_moe_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
