#pragma once
#include "tsgrad/ops.h"
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <climits>

#define THRESHOLD 2048

namespace tsgrad{
inline long long index_to_position(const int32_t* index, const int32_t* stride, int32_t size){
    long long res = 0;
    for (int32_t i = 0; i < size; i++)
        res += static_cast<long long>(index[i]) * static_cast<long long>(stride[i]);
    
    return res;
}

inline void position_to_index(long long ordinal, const int32_t* shape, int32_t* out_index, int32_t size){
    for (int32_t i = size - 1; i >= 0; i--){
        out_index[i] = ordinal % shape[i];
        ordinal /= shape[i];
    }
}

// writes to small_index, reads from big_index
inline void broadcast_index(const int32_t* big_index, const int32_t*  big_shape, int32_t big_size, const int32_t*  small_shape, int32_t* small_index, int32_t small_size){
    if (big_size < small_size) throw std::runtime_error("big_size cannot be less than small_size");

    int32_t diff = big_size - small_size;

    for (int32_t i = 0; i < small_size; i++){
        if (small_shape[i] == 1)
            small_index[i] = 0;
        else
            small_index[i] = big_index[i + diff];
    }
}

inline long long prod(const int32_t* shape, int32_t size){
    long long res = 1;
    for (int32_t i = 0; i < size; i++){
        if (shape[i] < 0) throw std::runtime_error("Cannot take product, negative dimension in shape");
        if (shape[i] == 0) return 0;
        if (res > LLONG_MAX  / shape[i]) throw std::runtime_error("Overflow, shape must be less than int 64 max");
        res *= static_cast<long long>(shape[i]);
    }

    return res;
}

inline bool is_contiguous(const View& a){
    long long expected_stride = 1;

    for (int32_t i = a.rank - 1; i >= 0; i--) {
        if (a.shape[i] > 1) {
            if (static_cast<long long>(a.stride[i]) != expected_stride) {
                return false;
            }
        }

        expected_stride *= static_cast<long long>(a.shape[i]);
    }

    return true;
}

inline bool shapesAreEqual(const int32_t* a, const int32_t* b, int32_t a_rank, int32_t b_rank){
    if (a_rank != b_rank) return false;
    for (int32_t i = 0; i < a_rank; i++)
        if (a[i] != b[i]) return false;

    return true;
}

// expects small < big
inline bool is_broadcastable(const int32_t* big, const int32_t* small, int32_t big_rank, int32_t small_rank){
    if (big_rank < small_rank) throw std::runtime_error("big cannot be less than small");
    
    int32_t i = big_rank - 1, j = small_rank - 1;
    for (; j >= 0; i--, j--)
        if (small[j] != 1 && big[i] != small[j]) return false;

    return true;
}

} // namespace