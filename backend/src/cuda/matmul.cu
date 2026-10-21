#include "tsgrad/ops.h"

// tsgrad::cuda::matmul is the device side of tsgrad::matmul in src/cpu/matmul.cpp.
// addon.cc does not call this directly.
//
// Rank 2: out[M, N] = a[M, K] @ b[K, N]. Strides still apply, because a
// permute() result shares its Float32Array and only changes stride.
//
// Sequence, same as elementwise.cu:
//   Context::get()
//   copy a, b, and the output buffer to the device
//   matmul_kernel<<<grid, block, 0, stream>>>
//   copy the output buffer back
//   Context::synchronize()

namespace tsgrad::cuda {

__global__ void matmul_kernel(
    float* out, int32_t out_stride_m, int32_t out_stride_n,
    const float* a, int32_t a_stride_m, int32_t a_stride_k,
    const float* b, int32_t b_stride_k, int32_t b_stride_n,
    int32_t M, int32_t N, int32_t K) {
    // int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    // int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row >= M || col >= N) return;
    // accumulate a[row, k] * b[k, col] using the strides above
    (void)out; (void)out_stride_m; (void)out_stride_n;
    (void)a; (void)a_stride_m; (void)a_stride_k;
    (void)b; (void)b_stride_k; (void)b_stride_n;
    (void)M; (void)N; (void)K;
}

void matmul(const View& out, const View& a, const View& b) {
    // Context& ctx = Context::get();
    // dim3 block(16, 16);
    // dim3 grid((N + 15) / 16, (M + 15) / 16);
    // matmul_kernel<<<grid, block, 0, stream>>>(d_out, ..., M, N, K);
    // ctx.synchronize();
    (void)out;
    (void)a;
    (void)b;
}

} // namespace tsgrad::cuda
