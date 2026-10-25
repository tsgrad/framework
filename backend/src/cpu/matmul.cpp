#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <format>
#include <span>
#include <ranges>

// Body is intentionally empty.
// tensor_functions.matmul only calls this for rank-2 inputs:
//   out [M, N] = a [M, K] @ b [K, N]
// Strides are not necessarily contiguous, because permute shares storage.
//
// CUDA handoff: tsgrad::cuda::matmul(out, a, b), defined in src/cuda/matmul.cu.

namespace tsgrad {

bool shapes_can_be_mul(const int32_t* a_shape, const int32_t* b_shape, int32_t a_rank, int32_t b_rank){
    if (a_rank != b_rank || a_rank < 2) return false;
    for (int32_t i = 0; i < a_rank - 2; i++)
        if (a_shape[i] != b_shape[i]) return false;

    return a_shape[a_rank - 1] == b_shape[b_rank - 2];
}

void matmul(const View& out, const View& a, const View& b) {
    if constexpr (TSGRAD_HAS_CUDA){
        tsgrad::cuda::matmul(out, a, b);
        return;
    }
    else{
        if (!shapes_can_be_mul(a.shape, b.shape, a.rank, b.rank)){
            std::span<const int32_t> aview{a.shape, static_cast<size_t>(a.rank)};
            std::span<const int32_t> bview{b.shape, static_cast<size_t>(b.rank)};

            throw std::runtime_error(std::format("Incompatible shapes for matrix mul, a: {:n}, b: {:n}", aview,  bview ));
        }

        if (a.rank != 2)
            throw std::runtime_error("Shape length must be 2 for now");

        const int32_t m = a.shape[0];
        const int32_t k = a.shape[1];
        const int32_t n = b.shape[1];

        const long long size = static_cast<long long>(m) * static_cast<long long>(n) * static_cast<long long>(k);
        const long long out_size = static_cast<long long>(m) * static_cast<long long>(n);

        #pragma omp parallel for if (size >= THRESHOLD)
        for (long long ordinal = 0; ordinal < out_size; ordinal++){
            // should always fit in int32_t
            const int32_t row = ordinal / n;
            const int32_t column = ordinal % n;

            int32_t a_pos = row * a.stride[0];
            int32_t b_pos = column * b.stride[1];
            float cur = 0;

            for (long long j = 0; j < k; j++){
                cur += a.data[a_pos] * b.data[b_pos];
                a_pos += a.stride[1];
                b_pos += b.stride[0];
            }

            out.data[row * out.stride[0] + column * out.stride[1]] = cur;
        }
    }
}

} // namespace tsgrad
