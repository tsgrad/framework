#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>
#include <limits>

namespace tsgrad::cuda {
    __global__ void scatter_kernel(float* out, Layout out_layout, const float* indicies, Layout indicies_layout, const float* grad, Layout grad_layout, int32_t dim, long long n){
        long long i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= n) return;

        int32_t grad_index[16];

        position_to_index(i, grad_layout.shape, grad_index, grad_layout.rank);
        const long long grad_pos = index_to_position(grad_index, grad_layout.stride, grad_layout.rank);
        grad_index[dim] = indicies[index_to_position(grad_index, indicies_layout.stride, indicies_layout.rank)];
        
        atomicAdd(&out[index_to_position(grad_index, out_layout.stride, out_layout.rank)], grad[grad_pos]);
    }

    void scatter(const View& out, const View& indicies, const View& grad, const int32_t dim){
        if (out.rank > 16 || indicies.rank > 16 || grad.rank > 16)
            throw std::runtime_error(std::format("CUDA scatter received exceptionally large shape, received out.rank: {}, indicies.rank: {}, grad.rank: {}", out.rank, indicies.rank, grad.rank));
    
        Context& ctx = Context::get();
        long long grad_size = prod(grad.shape, grad.rank);
        if (grad_size == 0) return;

        int32_t threads = 256;
        int32_t blocks = static_cast<int32_t>((grad_size + threads - 1) / threads);

        scatter_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), indicies.data, layout_from(indicies), grad.data, layout_from(grad), dim, grad_size);
        ctx.checkLastError();
    }
}