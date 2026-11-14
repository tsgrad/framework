#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>
#include <limits>

namespace tsgrad::cuda {
    __global__ void gather_kernel(float* out, Layout out_layout, const float* indicies, Layout indicies_layout, const float* values, Layout values_layout, int32_t dim, long long n){
        long long i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= n) return;

        int32_t out_index[16];

        // i represents an index in out.data
        // Get the out/indicies positions for this i
        position_to_index(i, out_layout.shape, out_index, out_layout.rank);

        const long long out_pos = index_to_position(out_index, out_layout.stride, out_layout.rank);
        out_index[dim] = indicies[index_to_position(out_index, indicies_layout.stride, indicies_layout.rank)];
        out[out_pos] = values[index_to_position(out_index, values_layout.stride, values_layout.rank)];
    }

    void gather(const View& out, const View& indicies, const View& values, const int32_t dim){
        if (out.rank > 16 || indicies.rank > 16 || values.rank > 16)
            throw std::runtime_error(std::format("CUDA gather received exceptionally large shape, received out.rank: {}, indicies.rank: {}, values.rank: {}", out.rank, indicies.rank, values.rank));
    
        Context& ctx = Context::get();
        long long outsize = prod(out.shape, out.rank);
        if (outsize == 0) return;

        int32_t threads = 256;
        int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

        gather_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), indicies.data, layout_from(indicies), values.data, layout_from(values), dim, outsize);
        ctx.checkLastError();
    }
}