// 自由后台读线程（de-risk）：脱离 GPU 完成回调、零 GIL，pread 进调用方 MLX buffer。
// 线程全程只碰 dst 原始指针 / 路径 / 整数，绝不接触 Python 对象 → 无需 GIL。
#pragma once
#include "../common.h"
#include <functional>

void bg_reader_start(int workers, int low_cap = 0);
long bg_reader_submit(const mx::array& dst, const std::vector<int>& experts,
                      const std::vector<int>& rows, const std::string& path,
                      int stride, long ticket, int prio = 0);
bool bg_reader_ready(long ticket);
void bg_reader_wait(long ticket);
void bg_reader_stop();
long bg_pread_into_pool(
    const std::vector<mx::array>& dst,
    const std::vector<long>& seg_off,
    const std::vector<long>& seg_nb,
    long slot, long expert,
    const std::string& path, long stride, long ticket, int prio = 0, bool nocache = true);

// 通用后台任务入口。普通 early 预取进低优队列；progressive T-1 最终补位可用
// prio>0 进入高优队列，避免已经失去大部分计算窗口的少量关键字节排在后续层 early
// 批次之后。内部接口，不绑 Python。
void bg_submit_task(std::function<void()> fn, int prio = 0);
