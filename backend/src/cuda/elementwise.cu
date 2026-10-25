#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include "cudaops.hpp"
#include <format>
#include <cuda_runtime.h>

// Host functions here are tsgrad::cuda::*. The CPU functions in
// src/cpu/elementwise.cpp and src/cpu/reduce.cpp are what addon.cc calls.
// Those CPU functions pass the same host Views here.
//
// Each host function below is the same sequence:
//   1. Context::get()            current device and stream
//   2. cudaMalloc + cudaMemcpy   host Float32Array -> device
//   3. kernel<<<grid, block, 0, stream>>>
//   4. cudaMemcpy                device -> the output Float32Array
//   5. Context::synchronize()    JS reads `out` as soon as this returns
//
// Shape and stride arrays are tiny. Copy them to the device with the floats,
// or pass them in constant memory once you care about that.

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
__global__ void map_kernel_contiguous(float* out, const int32_t* out_shape, const int32_t* out_stride, int32_t out_rank,
    const float* in, const int32_t* in_shape, const int32_t* in_stride, int32_t in_rank, long long n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(in[i]);
}

__global__ void zip_add_kernel(
    float* out, const int32_t* out_shape, const int32_t* out_stride, int32_t out_rank,
    const float* a, const int32_t* a_shape, const int32_t* a_stride,
    const float* b, const int32_t* b_shape, const int32_t* b_stride) {
    // one thread per output element, then broadcastIndex like tensorZip
    (void)out; (void)out_shape; (void)out_stride; (void)out_rank;
    (void)a; (void)a_shape; (void)a_stride;
    (void)b; (void)b_shape; (void)b_stride;
}

__global__ void reduce_sum_kernel(
    float* out, const int32_t* out_shape, const int32_t* out_stride,
    const float* in, const int32_t* in_shape, const int32_t* in_stride,
    int32_t rank, int32_t dim) {
    // one thread per output element, fold `dim` the way tensorReduce does
    (void)out; (void)out_shape; (void)out_stride;
    (void)in; (void)in_shape; (void)in_stride;
    (void)rank; (void)dim;
}

template<typename Func>
void map_op(const View& out, const View& in){
    if (out.rank > 16 || in.rank > 16)
    throw std::runtime_error(std::format("CUDA map_neg received exceptionally large shape: {}", std::max(out.rank, in.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    long long insize = prod(in.shape, in.rank);

    float *outdata, *indata;
    int32_t *outshape, *outstride, *inshape, *instride;
    size_t outfbytes = outsize * sizeof(float), infbytes = insize * sizeof(float); 
    
    size_t outotherbytes = out.rank * sizeof(int32_t);
    size_t inotherbytes = in.rank * sizeof(int32_t);

    cudaMalloc(&outdata, outfbytes);
    cudaMalloc(&indata, infbytes);

    cudaMalloc(&outshape, outotherbytes);
    cudaMalloc(&outstride, outotherbytes);
    cudaMalloc(&inshape, inotherbytes);
    cudaMalloc(&instride, inotherbytes);

    cudaMemcpy(outdata, out.data, outfbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(indata, in.data, infbytes, cudaMemcpyHostToDevice);

    cudaMemcpy(outshape, out.shape, outotherbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(outstride, out.stride, outotherbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(inshape, in.shape, inotherbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(instride, in.stride, inotherbytes, cudaMemcpyHostToDevice);

    int32_t threads = 256;
    int32_t blocks = (outsize + threads - 1) / threads;

    if (is_contiguous(out) && is_contiguous(in) && shapesAreEqual(out.shape, in.shape, out.rank, in.rank)){
        map_kernel_contiguous<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(outdata, outshape, outstride, out.rank, indata, inshape, instride, in.rank, outsize);
    }
    else{
        map_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(outdata, outshape, outstride, out.rank, indata, inshape, instride, in.rank, outsize);
    }
    ctx.synchronize();

    cudaMemcpy(out.data, outdata, outfbytes, cudaMemcpyDeviceToHost);
    cudaFree(outdata); cudaFree(outshape); cudaFree(outstride); cudaFree(indata); cudaFree(inshape); cudaFree(instride);
}

void map_neg(const View& out, const View& in){ map_op<neg>(out, in); }

void map_inv(const View& out, const View& in) { (void)out; (void)in; }
void map_sigmoid(const View& out, const View& in) { (void)out; (void)in; }
void map_relu(const View& out, const View& in) { (void)out; (void)in; }
void map_log(const View& out, const View& in) { (void)out; (void)in; }
void map_exp(const View& out, const View& in) { (void)out; (void)in; }
void map_id(const View& out, const View& in) { (void)out; (void)in; }

void zip_add(const View& out, const View& a, const View& b) {
    // Context& ctx = Context::get();
    // zip_add_kernel<<<grid, block, 0, stream>>>(d_out, ..., d_a, ..., d_b, ...);
    // ctx.synchronize();
    (void)out;
    (void)a;
    (void)b;
}

void zip_mul(const View& out, const View& a, const View& b) { (void)out; (void)a; (void)b; }
void zip_lt(const View& out, const View& a, const View& b) { (void)out; (void)a; (void)b; }
void zip_eq(const View& out, const View& a, const View& b) { (void)out; (void)a; (void)b; }
void zip_is_close(const View& out, const View& a, const View& b) { (void)out; (void)a; (void)b; }

void reduce_max(const View& out, const View& in, int32_t dim) { (void)out; (void)in; (void)dim; }
void reduce_sum(const View& out, const View& in, int32_t dim) {
    // reduce_sum_kernel<<<grid, block, 0, stream>>>(...);
    // Context::get().synchronize();
    (void)out;
    (void)in;
    (void)dim;
}
void reduce_prod(const View& out, const View& in, int32_t dim) { (void)out; (void)in; (void)dim; }

} // namespace tsgrad::cuda
