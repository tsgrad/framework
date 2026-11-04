#pragma once
#include "tsgrad/ops.h"
#include "tsgrad/layoutmath.hpp"

#include <climits>
#include <format>
#include <stdexcept>
#include <string>

namespace tsgrad {

struct RowSplit {
    long long rows;
    long long inner;
    int32_t dim_size;
};

inline long long checked_mul(long long a, long long b) {
    if (b < 0) throw std::runtime_error("Cannot take product, negative dimension in shape");
    if (a == 0 || b == 0) return 0;
    if (a > 0) {
        if (a > LLONG_MAX / b) throw std::runtime_error("Overflow, shape must be less than int 64 max");
    } else if (a < LLONG_MIN / b) {
        throw std::runtime_error("Overflow, shape must be less than int 64 max");
    }
    return a * b;
}

inline void check_shape(const View& view, const char* name) {
    if (view.rank > 16)
        throw std::runtime_error(std::format("{} received exceptionally large shape: {}", name, view.rank));
    for (int32_t i = 0; i < view.rank; i++) {
        if (view.shape[i] < 0)
            throw std::runtime_error("Cannot take product, negative dimension in shape");
    }
}

inline long long shape_numel(const View& view) {
    long long n = 1;
    for (int32_t i = 0; i < view.rank; i++)
        n = checked_mul(n, view.shape[i]);
    return n;
}

inline bool same_shape(const View& a, const View& b) {
    if (a.rank != b.rank) return false;
    for (int32_t i = 0; i < a.rank; i++) {
        if (a.shape[i] != b.shape[i]) return false;
    }
    return true;
}

inline void require_same(const View& a, const View& b, const char* name) {
    if (!same_shape(a, b))
        throw std::runtime_error(std::string(name) + " output shape must match the input");
}

inline bool can_broadcast(const View& big, const View& small) {
    if (big.rank < small.rank) return false;
    int32_t i = big.rank - 1;
    int32_t j = small.rank - 1;
    for (; j >= 0; i--, j--) {
        if (small.shape[j] != 1 && big.shape[i] != small.shape[j]) return false;
    }
    return true;
}

// rows is the product of every axis except dim. inner is the product after dim.
// inner is 1 when dim is the last axis, so a row index can divide by it.
inline RowSplit split_rows(const View& in, int32_t dim, const char* name) {
    check_shape(in, name);
    if (dim < 0 || dim >= in.rank)
        throw std::runtime_error(std::string(name) + " received dim outside the boundaries of the shape");

    long long inner = 1;
    for (int32_t i = dim + 1; i < in.rank; i++)
        inner = checked_mul(inner, in.shape[i]);

    long long rows = 1;
    for (int32_t i = 0; i < in.rank; i++) {
        if (i == dim) continue;
        rows = checked_mul(rows, in.shape[i]);
    }
    if (inner == 0) inner = 1;
    return {rows, inner, in.shape[dim]};
}

inline int32_t broadcast_shape(const View& a, const View& b, int32_t* out) {
    check_shape(a, "bias_act");
    check_shape(b, "bias_act");
    int32_t rank = a.rank > b.rank ? a.rank : b.rank;
    int32_t i = a.rank - 1;
    int32_t j = b.rank - 1;
    int32_t k = rank - 1;
    for (; i >= 0 && j >= 0; i--, j--, k--) {
        if (a.shape[i] == 1)
            out[k] = b.shape[j];
        else if (b.shape[j] != 1 && a.shape[i] != b.shape[j])
            throw std::runtime_error("Incompatible shapes for bias_act");
        else
            out[k] = a.shape[i];
    }
    for (; i >= 0; i--, k--) out[k] = a.shape[i];
    for (; j >= 0; j--, k--) out[k] = b.shape[j];
    return rank;
}

struct PoolGeom {
    int32_t kh;
    int32_t kw;
};

inline PoolGeom prepare_pool(const View& input, const View& out, int32_t kh, int32_t kw, const char* name) {
    check_shape(input, name);
    check_shape(out, name);
    if (input.rank != 4)
        throw std::runtime_error(std::string(name) + " expected rank 4");
    if (kh <= 0 || kw <= 0)
        throw std::runtime_error(std::string(name) + " kernel must be positive");
    if (input.shape[2] % kh != 0 || input.shape[3] % kw != 0)
        throw std::runtime_error(std::string(name) + " expected height and width to be divisible by the kernel");

    int32_t expect_h = input.shape[2] / kh;
    int32_t expect_w = input.shape[3] / kw;
    if (out.rank != 4 || out.shape[0] != input.shape[0] || out.shape[1] != input.shape[1] || out.shape[2] != expect_h || out.shape[3] != expect_w)
        throw std::runtime_error(std::string(name) + " output shape must be the pooled size");
    return {kh, kw};
}

inline SumAxes sum_axes(const View& out, const View& small) {
    SumAxes axes{};
    axes.volume = 1;
    int32_t diff = out.rank - small.rank;
    for (int32_t i = 0; i < out.rank; i++) {
        bool fixed = i >= diff && small.shape[i - diff] != 1;
        if (fixed) continue;
        axes.axes[axes.count] = i;
        axes.sizes[axes.count] = out.shape[i];
        axes.volume = checked_mul(axes.volume, out.shape[i]);
        axes.count++;
    }
    return axes;
}

} // namespace tsgrad
