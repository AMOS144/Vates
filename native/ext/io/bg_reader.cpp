// [7] 自由后台读线程（de-risk）：双队列 + 低优并发上限的后台 pread 线程池。
#include "bg_reader.h"
#include "blob_io.h"

#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

namespace {
struct ReadOp { uint8_t* dst; size_t nbytes; off_t file_off; };
struct BgJob {
  std::vector<ReadOp> ops;
  std::string path;
  long ticket;
  std::vector<mx::array> keep;     // 持 buffer 引用，保证 dst 指针在读期间存活
  std::function<void()> task;      // 若设置，worker 直接执行它（侧区异步读用），不走 ops/ticket
  int prio = 0;                    // >0=高优(route 读)，=0=低优(投机兜底)
  bool nocache = true;             // demand（route）读设 false 走 page cache：段偏移非页对齐，
                                   // F_NOCACHE 下多 MB 非对齐 pread 会短读 → 池槽脏字节（实测）。
  bool vectored = false;            // contiguous file segments -> one preadv
};

// 双队列 + 低优并发上限：高优(route)读永远优先取、且独占大多数 worker；
// 低优(投机)读最多占 low_cap_ 个 worker → 给 route 读留出 worker 和 SSD 带宽，
// 实现"高分抢带宽先就绪、低分延后不阻塞关键路径"的真 IO 优先级。
class BgReader {
 public:
  ~BgReader() { stop(); }            // 进程退出兜底：避免 joinable 线程析构触发 std::terminate
  void start(int workers, int low_cap) {
    { std::lock_guard<std::mutex> lk(m_); low_cap_ = low_cap; }
    ensure_started(workers);
  }
  // 懒启动：侧区预取的 GPU 回调可能在 Python 没调用 bg_reader_start 时就要派任务，
  // 故 submit/submit_task 自动确保线程已起（默认 4 worker），调用方无需显式 start。
  void ensure_started(int workers) {
    std::lock_guard<std::mutex> lk(m_);
    if (running_) return;
    running_ = true;
    worker_count_ = workers;
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { loop(); });
  }
  void submit(BgJob job) {
    ensure_started(4);
    { std::lock_guard<std::mutex> lk(m_);
      if (job.prio > 0) high_q_.push(std::move(job)); else low_q_.push(std::move(job)); }
    cv_.notify_all();
  }
  void submit_task(std::function<void()> fn, int prio) {
    ensure_started(4);
    BgJob job;
    job.task = std::move(fn);
    job.prio = prio;
    { std::lock_guard<std::mutex> lk(m_);
      if (job.prio > 0) high_q_.push(std::move(job)); else low_q_.push(std::move(job)); }
    cv_.notify_all();
  }
  bool ready(long ticket) {
    std::lock_guard<std::mutex> lk(dm_);
    return done_.count(ticket) > 0;
  }
  void wait(long ticket) {
    std::unique_lock<std::mutex> lk(dm_);
    dcv_.wait(lk, [&] { return done_.count(ticket) > 0; });
    done_.erase(ticket);
  }
  void wait_all(const std::vector<long>& tickets) {
    if (tickets.empty()) return;
    std::unique_lock<std::mutex> lk(dm_);
    dcv_.wait(lk, [&] {
      for (long ticket : tickets)
        if (done_.count(ticket) == 0) return false;
      return true;
    });
    for (long ticket : tickets) done_.erase(ticket);
  }
  void wait_high_idle() {
    std::unique_lock<std::mutex> lk(m_);
    if (low_cap_ <= 0 || low_cap_ >= worker_count_) return;
    cv_.wait(lk, [&] {
      return (high_q_.empty() && active_high_ == 0) || !running_;
    });
  }
  void stop() {
    { std::lock_guard<std::mutex> lk(m_); running_ = false; }
    cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
    threads_.clear();
    { std::lock_guard<std::mutex> lk(dm_); done_.clear(); }
    { std::lock_guard<std::mutex> lk(m_); active_low_ = 0; }
  }
 private:
  int low_budget() const { return low_cap_ <= 0 ? 1 << 30 : low_cap_; }  // <=0 视为不限流
  bool can_take_low() const { return !low_q_.empty() && active_low_ < low_budget(); }
  void loop() {
    std::unordered_map<std::string, int> fds;       // 每线程各自缓存 fd
    while (true) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this] { return !high_q_.empty() || can_take_low() || !running_; });
      if (!running_ && high_q_.empty() && low_q_.empty()) break;
      if (!running_ && high_q_.empty() && !can_take_low()) break;  // 退出时低优可超额排空
      bool is_low = false;
      BgJob job;
      if (!high_q_.empty()) {                          // 高优(route)读永远先取
        job = std::move(high_q_.front()); high_q_.pop(); ++active_high_;
      } else if (can_take_low()) {                     // 低优限流：仅在额度内取
        job = std::move(low_q_.front()); low_q_.pop(); ++active_low_; is_low = true;
      } else {
        continue;                                      // 假醒（低优已满额）→ 回去等
      }
      lk.unlock();
      if (job.task) { job.task(); }                    // 通用任务（侧区异步读）：直接执行，无 ticket
      else {
        int fd;
        std::string key = (job.nocache ? "N:" : "C:") + job.path;   // 按 nocache 分别缓存 fd
        auto it = fds.find(key);
        if (it == fds.end()) {
          fd = job.nocache ? open_blob_nocache(job.path.c_str()) : ::open(job.path.c_str(), O_RDONLY);
          fds[key] = fd;
        } else fd = it->second;
        if (fd >= 0 && job.vectored && !job.ops.empty()) {
          std::vector<struct iovec> iov(job.ops.size());
          size_t expected = 0;
          for (size_t i = 0; i < job.ops.size(); ++i) {
            iov[i].iov_base = job.ops[i].dst;
            iov[i].iov_len = job.ops[i].nbytes;
            expected += job.ops[i].nbytes;
          }
          ssize_t got = ::preadv(
              fd, iov.data(), static_cast<int>(iov.size()),
              job.ops.front().file_off);
          if (got != static_cast<ssize_t>(expected))
            fprintf(stderr, "[bg preadv SHORT] got=%zd want=%zu off=%lld\n",
                    got, expected,
                    static_cast<long long>(job.ops.front().file_off));
        } else if (fd >= 0) {
          for (auto& op : job.ops) {
            ssize_t got = ::pread(fd, op.dst, op.nbytes, op.file_off);
            if (got != static_cast<ssize_t>(op.nbytes))
              fprintf(stderr, "[bg pread SHORT] got=%zd want=%zu off=%lld\n",
                      got, op.nbytes, static_cast<long long>(op.file_off));
          }
        }
        { std::lock_guard<std::mutex> lk2(dm_); done_.insert(job.ticket); }
        dcv_.notify_all();
      }
      if (is_low) {                                    // 释放低优额度 → 唤醒别的 worker 再取
        std::lock_guard<std::mutex> lk2(m_); --active_low_; cv_.notify_all();
      } else if (job.prio > 0) {
        std::lock_guard<std::mutex> lk2(m_); --active_high_; cv_.notify_all();
      }
    }
    for (auto& kv : fds) if (kv.second >= 0) ::close(kv.second);
  }
  std::mutex m_, dm_;
  std::condition_variable cv_, dcv_;
  std::queue<BgJob> high_q_, low_q_;
  int active_low_ = 0;             // 当前正在执行的低优读数
  int active_high_ = 0;
  int low_cap_ = 0;               // 低优并发上限（<=0 不限流，保持旧行为）
  int worker_count_ = 0;
  std::unordered_set<long> done_;
  std::vector<std::thread> threads_;
  bool running_ = false;
};
BgReader g_bg;
}  // namespace

void bg_submit_task(std::function<void()> fn, int prio) {
  g_bg.submit_task(std::move(fn), prio);
}

void bg_reader_start(int workers, int low_cap) { g_bg.start(workers, low_cap); }

long bg_reader_submit(const mx::array& dst, const std::vector<int>& experts,
                      const std::vector<int>& rows, const std::string& path,
                      int stride, long ticket, int prio) {
  mx::array d = dst;
  d.eval();
  uint8_t* base = d.data<uint8_t>();
  size_t st = static_cast<size_t>(stride);
  BgJob job;
  job.path = path;
  job.ticket = ticket;
  job.prio = prio;
  job.keep.push_back(d);
  for (size_t i = 0; i < experts.size(); ++i)
    job.ops.push_back(ReadOp{base + static_cast<size_t>(rows[i]) * st, st,
                             static_cast<off_t>(static_cast<size_t>(experts[i]) * st)});
  g_bg.submit(std::move(job));
  return ticket;
}

// 专家各段直写进多个池段张量的 slot 行（消费侧零 MLX 算子）。
long bg_pread_into_pool(
    const std::vector<mx::array>& dst,
    const std::vector<long>& seg_off,
    const std::vector<long>& seg_nb,
    long slot, long expert,
    const std::string& path, long stride, long ticket, int prio, bool nocache) {
  BgJob job;
  job.path = path;
  job.ticket = ticket;
  job.prio = prio;
  job.nocache = nocache;
  for (size_t i = 0; i < dst.size(); ++i) {
    mx::array d = dst[i];
    d.eval();
    uint8_t* base = d.data<uint8_t>();
    job.keep.push_back(d);
    job.ops.push_back(ReadOp{
        base + static_cast<size_t>(slot) * static_cast<size_t>(seg_nb[i]),
        static_cast<size_t>(seg_nb[i]),
        static_cast<off_t>(static_cast<size_t>(expert) * static_cast<size_t>(stride)
                           + static_cast<size_t>(seg_off[i]))});
  }
  g_bg.submit(std::move(job));
  return ticket;
}

long bg_preadv_into_pool(
    const std::vector<mx::array>& dst,
    const std::vector<long>& seg_off,
    const std::vector<long>& seg_nb,
    long slot, long expert,
    const std::string& path, long stride, long ticket, int prio, bool nocache) {
  if (dst.size() != seg_off.size() || dst.size() != seg_nb.size())
    throw std::invalid_argument("bg_preadv_into_pool segment size mismatch");
  BgJob job;
  job.path = path;
  job.ticket = ticket;
  job.prio = prio;
  job.nocache = nocache;
  job.vectored = true;
  for (size_t i = 0; i < dst.size(); ++i) {
    mx::array d = dst[i];
    d.eval();
    uint8_t* base = d.data<uint8_t>();
    job.keep.push_back(d);
    job.ops.push_back(ReadOp{
        base + static_cast<size_t>(slot) * static_cast<size_t>(seg_nb[i]),
        static_cast<size_t>(seg_nb[i]),
        static_cast<off_t>(static_cast<size_t>(expert) * static_cast<size_t>(stride)
                           + static_cast<size_t>(seg_off[i]))});
  }
  g_bg.submit(std::move(job));
  return ticket;
}

bool bg_reader_ready(long ticket) { return g_bg.ready(ticket); }
void bg_reader_wait(long ticket) { g_bg.wait(ticket); }
void bg_reader_wait_all(const std::vector<long>& tickets) { g_bg.wait_all(tickets); }
void bg_reader_wait_high_idle() { g_bg.wait_high_idle(); }
void bg_reader_stop() { g_bg.stop(); }
