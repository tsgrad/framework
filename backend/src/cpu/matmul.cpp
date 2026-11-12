#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <format>
#include <span>
#include <ranges>
#include <vector>

// out[..., M, N] = a[..., M, K] @ b[..., K, N]
// Leading dimensions are a batch and must match. Both inputs have the same rank, at least 2.
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

        const int32_t rank = a.rank;
        const int32_t m = a.shape[rank - 2];
        const int32_t k = a.shape[rank - 1];
        const int32_t n = b.shape[rank - 1];

        if (rank == 2){
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
            return;
        }

        const int32_t batch_rank = rank - 2;
        const long long batch = prod(a.shape, batch_rank);
        const long long matrix = static_cast<long long>(m) * static_cast<long long>(n);
        if (batch > 0 && matrix > 0 && batch > LLONG_MAX / matrix)
            throw std::runtime_error("Overflow, shape must be less than int 64 max");
        const long long out_size = batch * matrix;
        if (out_size <= 0) return;
        if (k > 0 && out_size > LLONG_MAX / static_cast<long long>(k))
            throw std::runtime_error("Overflow, shape must be less than int 64 max");
        const long long size = out_size * static_cast<long long>(k);

        #pragma omp parallel if (size >= THRESHOLD)
        {
            std::vector<int32_t> batch_index(static_cast<size_t>(batch_rank), 0);

            #pragma omp for
            for (long long ordinal = 0; ordinal < out_size; ordinal++){
                const long long matrix_ordinal = ordinal % matrix;
                const long long batch_ordinal = ordinal / matrix;
                // should always fit in int32_t
                const int32_t row = static_cast<int32_t>(matrix_ordinal / n);
                const int32_t column = static_cast<int32_t>(matrix_ordinal % n);

                position_to_index(batch_ordinal, a.shape, batch_index.data(), batch_rank);
                const long long a_base = index_to_position(batch_index.data(), a.stride, batch_rank);
                const long long b_base = index_to_position(batch_index.data(), b.stride, batch_rank);
                const long long out_base = index_to_position(batch_index.data(), out.stride, batch_rank);

                long long a_pos = a_base + static_cast<long long>(row) * a.stride[rank - 2];
                long long b_pos = b_base + static_cast<long long>(column) * b.stride[rank - 1];
                float cur = 0;

                for (long long j = 0; j < k; j++){
                    cur += a.data[a_pos] * b.data[b_pos];
                    a_pos += a.stride[rank - 1];
                    b_pos += b.stride[rank - 2];
                }

                out.data[out_base + static_cast<long long>(row) * out.stride[rank - 2] + static_cast<long long>(column) * out.stride[rank - 1]] = cur;
            }
        }
    }
}

} // namespace tsgrad
