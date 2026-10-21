#include "tsgrad/ops.h"

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

__global__ void map_neg_kernel(
    float* out, const int32_t* out_shape, const int32_t* out_stride, int32_t out_rank,
    const float* in, const int32_t* in_shape, const int32_t* in_stride, int32_t in_rank) {
    // int32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    // one thread per output element, then the same index walk as tensorMap
    (void)out; (void)out_shape; (void)out_stride; (void)out_rank;
    (void)in; (void)in_shape; (void)in_stride; (void)in_rank;
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

void map_neg(const View& out, const View& in) {
    // Context& ctx = Context::get();
    // map_neg_kernel<<<grid, block, 0, stream>>>(d_out, ..., d_in, ...);
    // ctx.synchronize();
    (void)out;
    (void)in;
}

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
