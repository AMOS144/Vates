// 共用底层 IO 助手：打开 blob 文件。多个模块（侧区/后台读线程等）的 pread 都用它。
#pragma once
#include <fcntl.h>
#include <unistd.h>

#ifndef F_NOCACHE
#define F_NOCACHE 48          // macOS：提示内核读过的页不留 page cache
#endif

// 打开 blob 并设 F_NOCACHE：与 demand 侧 blob_loader 对齐，避免预取读把 page cache 灌满 →
// 在内存受限机上累积压力触发"双稳态"慢挡翻转（实测 zerocopy 每轮翻慢挡的根因）。
static inline int open_blob_nocache(const char* path) {
  int fd = ::open(path, O_RDONLY);
  if (fd >= 0) ::fcntl(fd, F_NOCACHE, 1);
  return fd;
}
