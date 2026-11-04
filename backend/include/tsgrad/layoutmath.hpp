#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define TSGRAD_HD __host__ __device__
#else
#define TSGRAD_HD
#endif

namespace tsgrad {

// Axes of `out` that a broadcast reduction sums. Passed to a kernel by value.
struct SumAxes {
    int32_t axes[16];
    int32_t sizes[16];
    int32_t count;
    long long volume;
};

TSGRAD_HD inline long long layout_position(const int32_t* index, const int32_t* stride, int32_t size) {
    long long res = 0;
    for (int32_t i = 0; i < size; i++)
        res += static_cast<long long>(index[i]) * static_cast<long long>(stride[i]);
    return res;
}

TSGRAD_HD inline void layout_index(long long ordinal, const int32_t* shape, int32_t* out_index, int32_t size) {
    for (int32_t i = size - 1; i >= 0; i--) {
        out_index[i] = static_cast<int32_t>(ordinal % shape[i]);
        ordinal /= shape[i];
    }
}

// small_index follows broadcastIndex: a size-1 axis reads 0. big is the larger rank.
TSGRAD_HD inline void layout_broadcast(const int32_t* big_index, const int32_t* big_shape, int32_t big_size, const int32_t* small_shape, int32_t* small_index, int32_t small_size) {
    int32_t diff = big_size - small_size;
    for (int32_t i = 0; i < small_size; i++) {
        if (small_shape[i] == 1)
            small_index[i] = 0;
        else
            small_index[i] = big_index[i + diff];
    }
}

// index[dim] is left at 0. Element j along dim is base + j * stride[dim].
TSGRAD_HD inline void layout_row_index(long long row, const int32_t* shape, int32_t rank, int32_t dim, long long inner, int32_t* index) {
    long long outer_i = row / inner;
    long long inner_i = row - outer_i * inner;
    for (int32_t i = dim - 1; i >= 0; i--) {
        index[i] = static_cast<int32_t>(outer_i % shape[i]);
        outer_i /= shape[i];
    }
    index[dim] = 0;
    for (int32_t i = rank - 1; i > dim; i--) {
        index[i] = static_cast<int32_t>(inner_i % shape[i]);
        inner_i /= shape[i];
    }
}

} // namespace tsgrad

#undef TSGRAD_HD
