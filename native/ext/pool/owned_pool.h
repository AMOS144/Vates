// Route 3 Phase 1 底座：C++ 拥有的池 buffer + 直写（替代 mx.zeros 建池 + 消费侧 MLX scatter）。
#pragma once
#include "../common.h"

// C++ 拥有的池 buffer 包成 mx.array（no-op deleter，进程内持有、永不迁移）。
// 供侧区/demand 后台 pread 安全直写；替代原 mx.zeros 建池 + 消费侧 MLX scatter。
mx::array pool_owned_zeros(const std::vector<int>& shape, const std::string& dtype);
// demand 真实区落池：把已加载专家段直接 memcpy 进 owned 池行（无 MLX scatter，保 buffer 稳定）。
void pool_write_rows(const std::vector<mx::array>& pool_list,
                     const std::vector<mx::array>& srcs_flat, const std::vector<int>& slots);
void pool_write_stacked(const std::vector<mx::array>& pool_list,
                        const std::vector<mx::array>& stacked_list, const std::vector<int>& slots);
// 诊断：mx.array 底层 buffer 原始指针（uintptr），用于对拍侧区写入 buffer 与 consume 读到 buffer 是否同一块。
uintptr_t array_data_ptr(const mx::array& a);
