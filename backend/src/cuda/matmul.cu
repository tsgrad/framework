#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>
#include <span>

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

__global__ void matmul_kernel(float* out, int32_t out_stride_m, int32_t out_stride_n,
    const float* a, int32_t a_stride_m, int32_t a_stride_k,
    const float* b, int32_t b_stride_k, int32_t b_stride_n,
    int32_t M, int32_t N, int32_t K) {
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || column >= N) return;

    int32_t a_pos = row * a_stride_m;
    int32_t b_pos = column * b_stride_n;
    float cur = 0;

    for (long long j = 0; j < K; j++){
        cur += a[a_pos] * b[b_pos];
        a_pos += a_stride_k;
        b_pos += b_stride_k;
    }

    out[row * out_stride_m + column * out_stride_n] = cur;
}

void matmul(const View& out, const View& a, const View& b) {
    if (out.rank > 16 || a.rank > 16 || b.rank > 16)
        throw std::runtime_error(std::format("CUDA matmul received exceptionally large shape: {}", std::max(std::max(out.rank, a.rank), b.rank)));
    
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

    const long long outsize = static_cast<long long>(m) * static_cast<long long>(n);
    if (outsize == 0) return;
    
    long long asize = prod(a.shape, a.rank);
    long long bsize = prod(b.shape, b.rank);

    Context& ctx = Context::get();

    float *outdata, *adata, *bdata;
    size_t outfbytes = outsize * sizeof(float), afbytes = asize * sizeof(float), bfbytes = bsize * sizeof(float); 
    
    dim3 threads(32, 8);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    cudaMalloc(&outdata, outfbytes);
    cudaMalloc(&adata, afbytes);
    cudaMalloc(&bdata, bfbytes);
    
    cudaMemcpy(outdata, out.data, outfbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(adata, a.data, afbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdata, b.data, bfbytes, cudaMemcpyHostToDevice);

    matmul_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        outdata, out.stride[0], out.stride[1],
        adata, a.stride[0], a.stride[1],
        bdata, b.stride[0], b.stride[1],
        m, n, k
    );

    ctx.synchronize();
    cudaMemcpy(out.data, outdata, outfbytes, cudaMemcpyDeviceToHost);
    cudaFree(outdata); cudaFree(adata); cudaFree(bdata);
}

} // namespace tsgrad::cuda
