#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include <vector>

struct Range {
    uint64_t offset;
    uint64_t nbytes;
};

struct Row {
    int expert;
    Range range;
    bool is_expert;
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) out.push_back(item);
    return out;
}

static std::unordered_set<int> parse_experts(const std::string& s) {
    std::unordered_set<int> out;
    for (auto& part : split(s, ',')) {
        if (!part.empty()) out.insert(std::stoi(part));
    }
    return out;
}

static std::vector<Row> read_index(const std::string& idx_path, const std::unordered_set<int>& experts, bool tensor_ranges) {
    FILE* fp = std::fopen(idx_path.c_str(), "r");
    if (!fp) {
        std::perror("open index");
        std::exit(2);
    }
    std::vector<Row> rows;
    char* line = nullptr;
    size_t cap = 0;
    ssize_t n = getline(&line, &cap, fp); // header
    (void)n;
    while ((n = getline(&line, &cap, fp)) != -1) {
        std::string s(line, static_cast<size_t>(n));
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        auto cols = split(s, '\t');
        if (cols.size() < 8) continue;
        bool is_expert = cols[0] == "EXPERT";
        if (tensor_ranges && is_expert) continue;
        if (!tensor_ranges && !is_expert) continue;
        int expert = std::stoi(cols[2]);
        if (!experts.empty() && experts.find(expert) == experts.end()) continue;
        uint64_t offset = std::stoull(cols[6]);
        uint64_t nbytes = std::stoull(cols[7]);
        rows.push_back(Row{expert, Range{offset, nbytes}, is_expert});
    }
    if (line) free(line);
    std::fclose(fp);
    return rows;
}

static std::vector<Range> merge_ranges(std::vector<Row>& rows) {
    std::vector<Range> ranges;
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.range.offset < b.range.offset;
    });
    for (auto& row : rows) {
        Range cur = row.range;
        if (ranges.empty()) {
            ranges.push_back(cur);
            continue;
        }
        Range& last = ranges.back();
        uint64_t last_end = last.offset + last.nbytes;
        uint64_t cur_end = cur.offset + cur.nbytes;
        if (cur.offset <= last_end) {
            if (cur_end > last_end) last.nbytes = cur_end - last.offset;
        } else {
            ranges.push_back(cur);
        }
    }
    return ranges;
}

static uint64_t read_ranges(const std::string& pack_path, const std::vector<Range>& ranges,
                            uint64_t& checksum, bool do_checksum) {
    int fd = open(pack_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::perror("open pack");
        std::exit(3);
    }
    std::vector<char> buf;
    uint64_t total = 0;
    checksum = 1469598103934665603ull; // FNV-ish
    for (auto& r : ranges) {
        buf.resize(static_cast<size_t>(r.nbytes));
        uint64_t done = 0;
        while (done < r.nbytes) {
            ssize_t n = pread(fd, buf.data() + done, static_cast<size_t>(r.nbytes - done),
                              static_cast<off_t>(r.offset + done));
            if (n <= 0) {
                std::perror("pread");
                close(fd);
                std::exit(4);
            }
            done += static_cast<uint64_t>(n);
        }
        total += r.nbytes;
        if (do_checksum) {
            for (unsigned char c : buf) {
                checksum ^= c;
                checksum *= 1099511628211ull;
            }
        }
    }
    close(fd);
    return total;
}

int main(int argc, char** argv) {
    std::string pack, index, experts_s;
    int repeat = 1;
    uint64_t merge_gap = 0;
    bool tensor_ranges = false;
    bool checksum_enabled = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "bench") continue;
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--index" && i + 1 < argc) index = argv[++i];
        else if (a == "--experts" && i + 1 < argc) experts_s = argv[++i];
        else if (a == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
        else if (a == "--merge-gap" && i + 1 < argc) merge_gap = std::stoull(argv[++i]);
        else if (a == "--tensor-ranges") tensor_ranges = true;
        else if (a == "--no-checksum") checksum_enabled = false;
    }
    if (pack.empty() || index.empty()) {
        std::cerr << "usage: pack_loader bench --pack P --index I --experts 1,2 --repeat N\n";
        return 1;
    }
    auto experts = parse_experts(experts_s);
    auto rows = read_index(index, experts, tensor_ranges);
    auto ranges = merge_ranges(rows);
    if (merge_gap > 0) {
        std::vector<Row> expanded;
        for (auto& r : ranges) expanded.push_back(Row{0, r, true});
        std::sort(expanded.begin(), expanded.end(), [](const Row& a, const Row& b) {
            return a.range.offset < b.range.offset;
        });
        std::vector<Range> merged;
        for (auto& row : expanded) {
            if (merged.empty()) {
                merged.push_back(row.range);
                continue;
            }
            auto& last = merged.back();
            uint64_t last_end = last.offset + last.nbytes;
            uint64_t cur_end = row.range.offset + row.range.nbytes;
            if (row.range.offset <= last_end + merge_gap) {
                if (cur_end > last_end) last.nbytes = cur_end - last.offset;
            } else {
                merged.push_back(row.range);
            }
        }
        ranges = merged;
    }
    uint64_t bytes = 0, checksum = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        bytes = read_ranges(pack, ranges, checksum, checksum_enabled);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "{"
              << "\"experts\":" << experts.size()
              << ",\"tensors\":" << rows.size()
              << ",\"ranges\":" << ranges.size()
              << ",\"tensor_ranges\":" << (tensor_ranges ? "true" : "false")
              << ",\"merge_gap\":" << merge_gap
              << ",\"bytes\":" << bytes
              << ",\"repeat\":" << repeat
              << ",\"elapsed_ms\":" << ms
              << ",\"checksum\":" << checksum
              << ",\"checksum_enabled\":" << (checksum_enabled ? "true" : "false")
              << "}\n";
    return 0;
}
