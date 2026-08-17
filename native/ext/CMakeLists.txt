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
  compute/quant_attention.cpp
  io/blob_load.cpp
  io/bg_reader.cpp
  prefetch/prefetch.cpp
  pool/owned_pool.cpp
  pool/side_region.cpp
  pool/demand.cpp
  bindings.cpp
)

target_link_libraries(native_moe_ext PRIVATE mlx)

execute_process(
  COMMAND xcrun -sdk macosx metal --version
  RESULT_VARIABLE VATES_METAL_COMPILER_RESULT
  OUTPUT_QUIET ERROR_QUIET)
if(MLX_BUILD_METAL AND VATES_METAL_COMPILER_RESULT EQUAL 0)
  list(GET MLX_INCLUDE_DIRS 0 VATES_MLX_INCLUDE_DIR)
  set(VATES_QUANT_GENERATED
      ${CMAKE_CURRENT_BINARY_DIR}/quant_attention_generated.h)
  set(VATES_STEEL_ATTENTION
      ${VATES_MLX_INCLUDE_DIR}/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h)
  add_custom_command(
    OUTPUT ${VATES_QUANT_GENERATED}
    COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_LIST_DIR}/compute/gen_quant_attention_header.py
            ${VATES_STEEL_ATTENTION} ${VATES_QUANT_GENERATED}
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/compute/gen_quant_attention_header.py
            ${VATES_STEEL_ATTENTION}
    VERBATIM)
  set(VATES_QUANT_AIRS)
  foreach(VATES_METAL_SOURCE quant_attention dense_attention)
    set(VATES_AIR ${CMAKE_CURRENT_BINARY_DIR}/${VATES_METAL_SOURCE}.air)
    add_custom_command(
      OUTPUT ${VATES_AIR}
      COMMAND xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math
              -Wno-c++17-extensions -Wno-c++20-extensions
              -I${CMAKE_CURRENT_BINARY_DIR} -I${VATES_MLX_INCLUDE_DIR}
              -c ${CMAKE_CURRENT_LIST_DIR}/compute/${VATES_METAL_SOURCE}.metal
              -o ${VATES_AIR}
      DEPENDS ${CMAKE_CURRENT_LIST_DIR}/compute/${VATES_METAL_SOURCE}.metal
              ${VATES_QUANT_GENERATED}
      VERBATIM)
    list(APPEND VATES_QUANT_AIRS ${VATES_AIR})
  endforeach()
  set(VATES_QUANT_METALLIB
      ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/vates_quant_attention.metallib)
  add_custom_command(
    OUTPUT ${VATES_QUANT_METALLIB}
    COMMAND xcrun -sdk macosx metal ${VATES_QUANT_AIRS}
            -o ${VATES_QUANT_METALLIB}
    DEPENDS ${VATES_QUANT_AIRS}
    VERBATIM)
  add_custom_target(vates_quant_attention_metallib
                    DEPENDS ${VATES_QUANT_METALLIB})
  add_dependencies(native_moe_ext vates_quant_attention_metallib)
else()
  message(WARNING
    "Offline Metal toolchain unavailable: packed attention primitive builds, "
    "but its metallib cannot be generated on this host")
endif()

if(BUILD_SHARED_LIBS)
  target_link_options(native_moe_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
