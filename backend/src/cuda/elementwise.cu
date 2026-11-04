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
//   1. Context::get()              current device and stream
//   2. kernel<<<grid, block, 0, stream>>> on View::data, already device memory
//   3. Context::checkLastError()   a bad launch is reported here; the stream is not waited on
//
// Shape and stride ride along as a Layout argument. Do not free View::data.

namespace tsgrad::cuda {
    
template <float (*Func)(float)>
__global__ void map_kernel(float* out, Layout out_layout, const float* in, Layout in_layout, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t out_index[16];
    int32_t in_index[16];

    position_to_index(i, out_layout.shape, out_index, out_layout.rank);
    long long outPos = index_to_position(out_index, out_layout.stride, out_layout.rank);

    broadcast_index(out_index, out_layout.shape, out_layout.rank, in_layout.shape, in_index, in_layout.rank);
    long long inPos = index_to_position(in_index, in_layout.stride, in_layout.rank);

    out[outPos] = Func(in[inPos]);
}

template <float (*Func)(float)>
__global__ void map_kernel_contiguous(float* out, const float* in, long long n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(in[i]);
}

template <float (*Func)(float, float)>
__global__ void zip_kernel(float* out, Layout out_layout, const float* a, Layout a_layout, const float* b, Layout b_layout, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t out_index[16];
    int32_t a_index[16];
    int32_t b_index[16];

    position_to_index(i, out_layout.shape, out_index, out_layout.rank);
    long long outPos = index_to_position(out_index, out_layout.stride, out_layout.rank);

    broadcast_index(out_index, out_layout.shape, out_layout.rank, a_layout.shape, a_index, a_layout.rank);
    long long aPos = index_to_position(a_index, a_layout.stride, a_layout.rank);

    broadcast_index(out_index, out_layout.shape, out_layout.rank, b_layout.shape, b_index, b_layout.rank);
    long long bPos = index_to_position(b_index, b_layout.stride, b_layout.rank);

    out[outPos] = Func(a[aPos], b[bPos]);
}

template <float (*Func)(float, float)>
__global__ void zip_kernel_contiguous(float* out, const float* a, const float* b, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(a[i], b[i]);
}

template <float (*Func)(float, float)>
__global__ void reduce_kernel(float* out, Layout out_layout, const float* in, Layout in_layout, int32_t dim, float base, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    const int32_t rank = in_layout.rank;
    int32_t dimSize = in_layout.shape[dim];
    int32_t out_index[16];

    position_to_index(i, out_layout.shape, out_index, rank);
    int32_t temp = out_index[dim];
    float cur = base;

    for (int32_t j = 0; j < dimSize; j++){
        out_index[dim] = j;
        cur = Func(cur, in[index_to_position(out_index, in_layout.stride, rank)]);
    }

    out_index[dim] = temp;
    out[index_to_position(out_index, out_layout.stride, rank)] = cur;
}

__global__ void fill_kernel(float* data, long long n, float value) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = value;
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
        ctx.checkLastError();
        return;
    }

    map_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), in.data, layout_from(in), outsize);
    ctx.checkLastError();
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
        ctx.checkLastError();
        return;
    }

    if (!is_broadcastable(out.shape, a.shape, out.rank, a.rank) || !is_broadcastable(out.shape, b.shape, out.rank, b.rank))
        throw std::runtime_error("zip_op received unbroadcastable shapes");

    zip_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out),
        a.data, layout_from(a),
        b.data, layout_from(b),
        outsize
    );
    ctx.checkLastError();
}

void zip_add(const View& out, const View& a, const View& b) { zip_op<add>(out, a, b); }
void zip_sub(const View& out, const View& a, const View& b) { zip_op<sub>(out, a, b); }

void zip_mul(const View& out, const View& a, const View& b) { zip_op<mul>(out, a, b); }
void zip_div(const View& out, const View& a, const View& b) { zip_op<div>(out, a, b); }
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

    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    reduce_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), in.data, layout_from(in), dim, base, outsize);
    ctx.checkLastError();
}

void fill(float* data, size_t numel, float value) {
    if (numel == 0 || data == nullptr) return;

    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(data, static_cast<long long>(numel), value);
    ctx.checkLastError();
}

void reduce_max(const View& out, const View& in, int32_t dim) { reduce_op<max>(out, in, dim, -std::numeric_limits<float>::infinity()); }
void reduce_sum(const View& out, const View& in, int32_t dim) { reduce_op<add>(out, in, dim, 0); }
void reduce_prod(const View& out, const View& in, int32_t dim) { reduce_op<mul>(out, in, dim, 1); }

} // namespace tsgrad::cuda
