#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include "cudaops.hpp"
#include <format>
#include <cuda_runtime.h>
#include <limits>

// Host functions here are tsgrad::cuda::*. The CPU functions in
// src/cpu/elementwise.cpp and src/cpu/reduce.cpp are what addon.cc calls.
// Those CPU functions pass the same host Views here.
//
// Each host function below is the same sequence:
//   1. Context::get()            current device and stream
//   2. kernel<<<grid, block, 0, stream>>> on View::data, already device memory
//   3. Context::synchronize()    the next copyToHost sees the result
//
// Shape and stride arrays are tiny and still copied per call. Do not free View::data.

namespace tsgrad::cuda {
    
template <float (*Func)(float)>
__global__ void map_kernel(float* out, const int32_t* out_shape, const int32_t* out_stride, int32_t out_rank,
    const float* in, const int32_t* in_shape, const int32_t* in_stride, int32_t in_rank, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t out_index[16];
    int32_t in_index[16];

    position_to_index(i, out_shape, out_index, out_rank);
    long long outPos = index_to_position(out_index, out_stride, out_rank);

    broadcast_index(out_index, out_shape, out_rank, in_shape, in_index, in_rank);
    long long inPos = index_to_position(in_index, in_stride, in_rank);

    out[outPos] = Func(in[inPos]);
}

template <float (*Func)(float)>
__global__ void map_kernel_contiguous(float* out, const float* in, long long n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(in[i]);
}

template <float (*Func)(float, float)>
__global__ void zip_kernel(float* out, const int32_t* out_shape, const int32_t* out_stride, int32_t out_rank,
    const float* a, const int32_t* a_shape, const int32_t* a_stride, int32_t a_rank,
    const float* b, const int32_t* b_shape, const int32_t* b_stride, int32_t b_rank, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t out_index[16];
    int32_t a_index[16];
    int32_t b_index[16];

    position_to_index(i, out_shape, out_index, out_rank);
    long long outPos = index_to_position(out_index, out_stride, out_rank);

    broadcast_index(out_index, out_shape, out_rank, a_shape, a_index, a_rank);
    long long aPos = index_to_position(a_index, a_stride, a_rank);

    broadcast_index(out_index, out_shape, out_rank, b_shape, b_index, b_rank);
    long long bPos = index_to_position(b_index, b_stride, b_rank);

    out[outPos] = Func(a[aPos], b[bPos]);
}

template <float (*Func)(float, float)>
__global__ void zip_kernel_contiguous(float* out, const float* a, const float* b, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(a[i], b[i]);
}

template <float (*Func)(float, float)>
__global__ void reduce_kernel(float* out, const int32_t* out_shape, const int32_t* out_stride,
    const float* in, const int32_t* in_shape, const int32_t* in_stride, int32_t rank, int32_t dim, float base, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t dimSize = in_shape[dim];
    int32_t out_index[16];

    position_to_index(i, out_shape, out_index, rank);
    int32_t temp = out_index[dim];
    float cur = base;

    for (int32_t j = 0; j < dimSize; j++){
        out_index[dim] = j;
        cur = Func(cur, in[index_to_position(out_index, in_stride, rank)]);
    }

    out_index[dim] = temp;
    out[index_to_position(out_index, out_stride, rank)] = cur;
}

int32_t* upload_i32(const int32_t* src, size_t bytes) {
    if (bytes == 0) return nullptr;
    int32_t* dst = nullptr;
    cudaMalloc(&dst, bytes);
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    return dst;
}

template<float (*Func)(float)>
void map_op(const View& out, const View& in){
    if (out.rank > 16 || in.rank > 16)
        throw std::runtime_error(std::format("CUDA map_op received exceptionally large shape: {}", std::max(out.rank, in.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    if (is_contiguous(out) && is_contiguous(in) && shapesAreEqual(out.shape, in.shape, out.rank, in.rank)){
        map_kernel_contiguous<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, in.data, outsize);
        ctx.synchronize();
        return;
    }

    const size_t outotherbytes = static_cast<size_t>(out.rank) * sizeof(int32_t);
    const size_t inotherbytes = static_cast<size_t>(in.rank) * sizeof(int32_t);
    int32_t* outshape = upload_i32(out.shape, outotherbytes);
    int32_t* outstride = upload_i32(out.stride, outotherbytes);
    int32_t* inshape = upload_i32(in.shape, inotherbytes);
    int32_t* instride = upload_i32(in.stride, inotherbytes);

    map_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, outshape, outstride, out.rank, in.data, inshape, instride, in.rank, outsize);

    ctx.synchronize();
    cudaFree(outshape); cudaFree(outstride); cudaFree(inshape); cudaFree(instride);
}

void map_neg(const View& out, const View& in){ map_op<neg>(out, in); }

void map_inv(const View& out, const View& in) { map_op<inv>(out, in); }
void map_sigmoid(const View& out, const View& in) { map_op<sigmoid>(out, in); }
void map_relu(const View& out, const View& in) { map_op<relu>(out, in); }
void map_log(const View& out, const View& in) { map_op<log>(out, in); }
void map_exp(const View& out, const View& in) { map_op<exp>(out, in); }
void map_id(const View& out, const View& in) { map_op<id>(out, in); }

template<float (*Func)(float, float)>
void zip_op(const View& out, const View& a, const View& b){
    if (out.rank > 16 || a.rank > 16 || b.rank > 16)
        throw std::runtime_error(std::format("CUDA zip_op received exceptionally large shape: {}", std::max(std::max(out.rank, a.rank), b.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    if (is_contiguous(out) && is_contiguous(a) && is_contiguous(b) && shapesAreEqual(out.shape, a.shape, out.rank, a.rank) && shapesAreEqual(out.shape, b.shape, out.rank, b.rank)){
        zip_kernel_contiguous<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, a.data, b.data, outsize);
        ctx.synchronize();
        return;
    }

    if (!is_broadcastable(out.shape, a.shape, out.rank, a.rank) || !is_broadcastable(out.shape, b.shape, out.rank, b.rank))
        throw std::runtime_error("zip_op received unbroadcastable shapes");

    const size_t outotherbytes = static_cast<size_t>(out.rank) * sizeof(int32_t);
    const size_t aotherbytes = static_cast<size_t>(a.rank) * sizeof(int32_t);
    const size_t botherbytes = static_cast<size_t>(b.rank) * sizeof(int32_t);
    int32_t* outshape = upload_i32(out.shape, outotherbytes);
    int32_t* outstride = upload_i32(out.stride, outotherbytes);
    int32_t* ashape = upload_i32(a.shape, aotherbytes);
    int32_t* astride = upload_i32(a.stride, aotherbytes);
    int32_t* bshape = upload_i32(b.shape, botherbytes);
    int32_t* bstride = upload_i32(b.stride, botherbytes);

    zip_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, outshape, outstride, out.rank,
        a.data, ashape, astride, a.rank,
        b.data, bshape, bstride, b.rank, outsize
    );

    ctx.synchronize();
    cudaFree(outshape); cudaFree(outstride); cudaFree(ashape); cudaFree(astride); cudaFree(bshape); cudaFree(bstride);
}

void zip_add(const View& out, const View& a, const View& b) { zip_op<add>(out, a, b); }

void zip_mul(const View& out, const View& a, const View& b) { zip_op<mul>(out, a, b); }
void zip_lt(const View& out, const View& a, const View& b) { zip_op<lt>(out, a, b); }
void zip_eq(const View& out, const View& a, const View& b) { zip_op<eq>(out, a, b); }
void zip_is_close(const View& out, const View& a, const View& b) { zip_op<is_close>(out, a, b); }

template<float (*Func)(float, float)>
void reduce_op(const View& out, const View& in, int32_t dim, float base){
    if (dim < 0 || dim >= in.rank)
        throw std::runtime_error("reduce_op received dim outside the boundaries of in.shape");

    if (out.rank > 16 || in.rank > 16)
        throw std::runtime_error(std::format("CUDA reduce_op received exceptionally large shape: {}", std::max(out.rank, in.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    const size_t outotherbytes = static_cast<size_t>(out.rank) * sizeof(int32_t);
    const size_t inotherbytes = static_cast<size_t>(in.rank) * sizeof(int32_t);
    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    int32_t* outshape = upload_i32(out.shape, outotherbytes);
    int32_t* outstride = upload_i32(out.stride, outotherbytes);
    int32_t* inshape = upload_i32(in.shape, inotherbytes);
    int32_t* instride = upload_i32(in.stride, inotherbytes);

    reduce_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, outshape, outstride, in.data, inshape, instride, in.rank, dim, base, outsize);

    ctx.synchronize();
    cudaFree(outshape); cudaFree(outstride); cudaFree(inshape); cudaFree(instride);
}

void reduce_max(const View& out, const View& in, int32_t dim) { reduce_op<max>(out, in, dim, -std::numeric_limits<float>::infinity()); }
void reduce_sum(const View& out, const View& in, int32_t dim) { reduce_op<add>(out, in, dim, 0); }
void reduce_prod(const View& out, const View& in, int32_t dim) { reduce_op<mul>(out, in, dim, 1); }

} // namespace tsgrad::cuda
