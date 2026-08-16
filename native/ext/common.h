// 共用前导：系统/Metal/MLX/nanobind 头 + 命名空间别名。所有 TU 都包含。
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <Metal/Metal.hpp>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <random>
#include <utility>
#include <tuple>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
namespace mx = mlx::core;
