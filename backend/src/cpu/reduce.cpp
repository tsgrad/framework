#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <climits>
#include <limits>


// CUDA handoff: tsgrad::cuda::reduce_sum(out, in, dim), defined in
// src/cuda/elementwise.cu. Same borrowed host Views as the CPU function.

namespace tsgrad {

template <typename Func>
void reduce_ops(Func&& operation, const View& out, const View& in, int32_t dim, float base){
    auto size = prod(out.shape, out.rank);

    if (dim < 0 || dim >= in.rank) throw std::runtime_error("reduce_ops received dim outside the boundaries of in.shape");

    #pragma omp parallel
    {
        int32_t n = in.rank;
        int32_t dimSize = in.shape[dim];

        std::vector<int32_t> out_index(n, 0);

        #pragma omp for
        for (long long i = 0; i < size; i++){
            position_to_index(i, out.shape, out_index.data(), n);
            int32_t temp = out_index[dim];
            float cur = base;

            for (int32_t j = 0; j < dimSize; j++){
                out_index[dim] = j;
                cur = operation(cur, in.data[index_to_position(out_index.data(), in.stride, n)]);
            }

            out_index[dim] = temp;
            out.data[index_to_position(out_index.data(), out.stride, n)] = cur;
        }
    }
}

void reduce_max(const View& out, const View& in, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA){
        cuda::reduce_max(out, in, dim);
    }
    else{
        reduce_ops([](float x, float y) -> float { return x > y ? x : y; }, out, in, dim, -std::numeric_limits<float>::infinity());
    }
}

void reduce_sum(const View& out, const View& in, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA){
        cuda::reduce_sum(out, in, dim);
    }
    else{
        reduce_ops([](float x, float y) -> float { return x + y; }, out, in, dim, 0.0F);
    }
}

void reduce_prod(const View& out, const View& in, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA){
        cuda::reduce_prod(out, in, dim);
    }
    else{
        reduce_ops([](float x, float y) -> float { return x * y; }, out, in, dim, 1.0F);
    }
}

} // namespace tsgrad
