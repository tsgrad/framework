#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define TSGRAD_HD __host__ __device__
#else
#define TSGRAD_HD
#endif

namespace tsgrad {

// One value per index. The same seed and index produce the same float on CPU and CUDA.
TSGRAD_HD inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// [0, 1), matching randomFloat's use of Math.random.
TSGRAD_HD inline float uniform01(uint64_t seed, uint64_t index) {
    uint32_t h = mix32(static_cast<uint32_t>(index) ^ mix32(static_cast<uint32_t>(seed)));
    h ^= mix32(static_cast<uint32_t>(index >> 32) ^ mix32(static_cast<uint32_t>(seed >> 32) + 0x9e3779b9u));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

} // namespace tsgrad

#undef TSGRAD_HD
