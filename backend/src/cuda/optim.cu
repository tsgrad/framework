#include "tsgrad/ops.h"
#include "cudahelpers.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace tsgrad::cuda {

__global__ void sgd_kernel(float* param, const float* grad, long long n, float lr) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] -= lr * grad[i];
}

__global__ void sgd_momentum_kernel(float* param, const float* grad, float* velocity, long long n, float lr, float beta) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float updated = beta * velocity[i] + (1.0F - beta) * grad[i];
    velocity[i] = updated;
    param[i] -= lr * updated;
}

void sgd(const View& param, const View& grad, float lr) {
    if (!shapesAreEqual(param.shape, grad.shape, param.rank, grad.rank))
        throw std::runtime_error("sgd received mismatched shapes");
    if (!is_contiguous(param) || !is_contiguous(grad))
        throw std::runtime_error("sgd requires contiguous storage");

    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    sgd_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(param.data, grad.data, n, lr);
    ctx.synchronize();
}

void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta) {
    if (!shapesAreEqual(param.shape, grad.shape, param.rank, grad.rank) || !shapesAreEqual(param.shape, velocity.shape, param.rank, velocity.rank))
        throw std::runtime_error("sgd_momentum received mismatched shapes");
    if (!is_contiguous(param) || !is_contiguous(grad) || !is_contiguous(velocity))
        throw std::runtime_error("sgd_momentum requires contiguous storage");

    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    sgd_momentum_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(param.data, grad.data, velocity.data, n, lr, beta);
    ctx.synchronize();
}

} // namespace tsgrad::cuda
