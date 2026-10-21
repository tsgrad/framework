#pragma once

#include <cstdint>

// Host-side view of one tensor argument.
//
// data, shape, and stride are borrowed. addon.cc owns the shape and stride
// buffers for the duration of the call, and data points into a JavaScript
// Float32Array. Do not keep these pointers after the function returns.
//
// out is written. Every other View is read. Shape and stride follow the same
// rules as indexToPosition / broadcastIndex in src/tensor_data.ts.
namespace tsgrad {

struct View {
    float* data;
    const int32_t* shape;
    const int32_t* stride;
    int32_t rank;
};

// One output element per input element. Same indexing as tensorMap.
void map_neg(const View& out, const View& in);
void map_inv(const View& out, const View& in);
void map_sigmoid(const View& out, const View& in); // stable split at 0, same as operators.sigmoid
void map_relu(const View& out, const View& in);
void map_log(const View& out, const View& in);
void map_exp(const View& out, const View& in);
void map_id(const View& out, const View& in);

// One output element from a broadcast pair. Same indexing as tensorZip.
// zip_is_close writes 1 when abs(a - b) < 1e-2, else 0. That tolerance is operators.isClose.
void zip_add(const View& out, const View& a, const View& b);
void zip_mul(const View& out, const View& a, const View& b);
void zip_lt(const View& out, const View& a, const View& b);
void zip_eq(const View& out, const View& a, const View& b);
void zip_is_close(const View& out, const View& a, const View& b);

// out's reduced dimension has length 1. base is the identity:
// max uses -infinity, sum uses 0, prod uses 1. Same walk as tensorReduce.
void reduce_max(const View& out, const View& in, int32_t dim);
void reduce_sum(const View& out, const View& in, int32_t dim);
void reduce_prod(const View& out, const View& in, int32_t dim);

// Rank-2 matmul. out is [M, N], a is [M, K], b is [K, N], strides included.
void matmul(const View& out, const View& a, const View& b);

// Same signatures as above. Defined in src/cuda/*.cu when TSGRAD_HAS_CUDA is set.
// The pointers in these Views are still host pointers. The .cu function copies
// to the device, launches, and copies back before it returns.
namespace cuda {

class Context {
public:
    static Context& get();
    void synchronize();
    int device() const;

private:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    int device_;
    void* stream_; // cudaStream_t, stored without including cuda_runtime.h here
};

void map_neg(const View& out, const View& in);
void map_inv(const View& out, const View& in);
void map_sigmoid(const View& out, const View& in);
void map_relu(const View& out, const View& in);
void map_log(const View& out, const View& in);
void map_exp(const View& out, const View& in);
void map_id(const View& out, const View& in);

void zip_add(const View& out, const View& a, const View& b);
void zip_mul(const View& out, const View& a, const View& b);
void zip_lt(const View& out, const View& a, const View& b);
void zip_eq(const View& out, const View& a, const View& b);
void zip_is_close(const View& out, const View& a, const View& b);

void reduce_max(const View& out, const View& in, int32_t dim);
void reduce_sum(const View& out, const View& in, int32_t dim);
void reduce_prod(const View& out, const View& in, int32_t dim);

void matmul(const View& out, const View& a, const View& b);

} // namespace cuda
} // namespace tsgrad
