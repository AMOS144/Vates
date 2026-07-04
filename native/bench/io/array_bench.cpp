#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include "mlx/array.h"

using mlx::core::array;
using mlx::core::bfloat16;
using mlx::core::float16;
using mlx::core::float32;
using mlx::core::int32;
using mlx::core::uint32;
using mlx::core::Shape;

struct TensorRow {
    int expert;
    std::string key;
    std::string dtype;
    std::vector<int32_t> shape;
    uint64_t offset;
    uint64_t nbytes;
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
    for (auto& part : split(s, ',')) if (!part.empty()) out.insert(std::stoi(part));
    return out;
}

static std::vector<int32_t> parse_shape(const std::string& s) {
    std::vector<int32_t> out;
    for (auto& part : split(s, ',')) if (!part.empty()) out.push_back(std::stoi(part));
    return out;
}

static mlx::core::Dtype dtype_of(const std::string& s) {
    if (s == "U32") return uint32;
    if (s == "I32") return int32;
    if (s == "F32") return float32;
    if (s == "F16") return float16;
    if (s == "BF16") return bfloat16;
    std::cerr << "unsupported dtype: " << s << "\n";
    std::exit(5);
}

static std::vector<TensorRow> read_tensor_rows(const std::string& idx, const std::unordered_set<int>& experts) {
    FILE* fp = std::fopen(idx.c_str(), "r");
    if (!fp) { std::perror("open index"); std::exit(2); }
    std::vector<TensorRow> rows;
    char* line = nullptr;
    size_t cap = 0;
    ssize_t n = getline(&line, &cap, fp);
    (void)n;
    while ((n = getline(&line, &cap, fp)) != -1) {
        std::string s(line, static_cast<size_t>(n));
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        auto cols = split(s, '\t');
        if (cols.size() < 8 || cols[0] != "TENSOR") continue;
        int expert = std::stoi(cols[2]);
        if (!experts.empty() && experts.find(expert) == experts.end()) continue;
        rows.push_back(TensorRow{
            expert, cols[3], cols[4], parse_shape(cols[5]),
            static_cast<uint64_t>(std::stoull(cols[6])),
            static_cast<uint64_t>(std::stoull(cols[7]))
        });
    }
    if (line) free(line);
    std::fclose(fp);
    return rows;
}

int main(int argc, char** argv) {
    std::string pack, idx, experts_s;
    int repeat = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--index" && i + 1 < argc) idx = argv[++i];
        else if (a == "--experts" && i + 1 < argc) experts_s = argv[++i];
        else if (a == "--repeat" && i + 1 < argc) repeat = std::stoi(argv[++i]);
    }
    if (pack.empty() || idx.empty()) {
        std::cerr << "usage: array_bench --pack P --index I --experts 1,2 --repeat N\n";
        return 1;
    }
    auto experts = parse_experts(experts_s);
    auto rows = read_tensor_rows(idx, experts);
    int fd = open(pack.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("open pack"); return 2; }
    size_t arrays = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) {
        for (auto& row : rows) {
            void* data = std::malloc(row.nbytes);
            if (!data) return 3;
            uint64_t done = 0;
            while (done < row.nbytes) {
                ssize_t n = pread(fd, static_cast<char*>(data) + done, row.nbytes - done, row.offset + done);
                if (n <= 0) { std::perror("pread"); return 4; }
                done += static_cast<uint64_t>(n);
            }
            Shape shape;
            for (auto dim : row.shape) shape.push_back(dim);
            array a(data, shape, dtype_of(row.dtype), [](void* p) { std::free(p); });
            a.eval();
            arrays++;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    close(fd);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "{"
              << "\"experts\":" << experts.size()
              << ",\"tensors\":" << rows.size()
              << ",\"arrays\":" << arrays
              << ",\"repeat\":" << repeat
              << ",\"elapsed_ms\":" << ms
              << "}\n";
    return 0;
}
