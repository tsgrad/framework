#pragma once

#ifndef TSGRAD_HAS_CUDA
    #define TSGRAD_HAS_CUDA 0
#endif

#include <cstddef>
#include <cstdint>

// Host-side view of one tensor argument.
//
// data is Storage::data(): a host vector on a CPU build, a device pointer on
// a CUDA build. Do not free it. shape and stride are borrowed for this call.
// Do not keep these pointers after the function returns.
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
void zip_sub(const View& out, const View& a, const View& b);
void zip_mul(const View& out, const View& a, const View& b);
void zip_div(const View& out, const View& a, const View& b);
void zip_lt(const View& out, const View& a, const View& b);
void zip_eq(const View& out, const View& a, const View& b);
void zip_is_close(const View& out, const View& a, const View& b);

// param and grad are the same shape. Both are updated through param's data.
void sgd(const View& param, const View& grad, float lr);
void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta);

// out's reduced dimension has length 1. base is the identity:
// max uses -infinity, sum uses 0, prod uses 1. Same walk as tensorReduce.
void reduce_max(const View& out, const View& in, int32_t dim);
void reduce_sum(const View& out, const View& in, int32_t dim);
void reduce_prod(const View& out, const View& in, int32_t dim);

// Rank-2 matmul. out is [M, N], a is [M, K], b is [K, N], strides included.
void matmul(const View& out, const View& a, const View& b);

// bias_act activation. relu is 0, sigmoid is 1.
constexpr int32_t bias_relu = 0;
constexpr int32_t bias_sigmoid = 1;

// One row along dim. softmax writes exp(x - max) / sum. log_softmax writes x - max - log(sum).
void softmax(const View& out, const View& in, int32_t dim);
void softmax_backward(const View& out, const View& softmax, const View& grad, int32_t dim);
void log_softmax(const View& out, const View& in, int32_t dim);
void log_softmax_backward(const View& out, const View& log_softmax, const View& grad, int32_t dim);

// loss is one element: the mean over rows of -sum(target * log_softmax).
// softmax is the same shape as logits and is kept for the backward.
void softmax_cross_entropy(const View& loss, const View& softmax, const View& logits, const View& target, int32_t dim);
void softmax_cross_entropy_backward(const View& grad_logits, const View& softmax, const View& target, int32_t dim);

// relu: out = input > 0 ? grad : 0. sigmoid: out = grad * y * (1 - y).
void relu_backward(const View& out, const View& grad, const View& input);
void sigmoid_backward(const View& out, const View& grad, const View& output);

// Copy in into out. A size-1 axis of in is read at index 0. out is the larger shape.
void expand(const View& out, const View& in);

// out = act(input + bias). bias broadcasts from the right, same rule as zip add.
void bias_act(const View& out, const View& input, const View& bias, int32_t act);
// grad_input is act'(pre) * grad at the broadcast shape. grad_bias sums that over the broadcast axes.
// saved is the input for relu and the forward output for sigmoid.
void bias_act_backward(const View& grad_input, const View& grad_bias, const View& grad, const View& saved, const View& bias, int32_t act);

// Normalize the last dimension. weight and bias have that length.
// mean and rstd are rank-1, one value per row.
void layer_norm(const View& out, const View& mean, const View& rstd, const View& input, const View& weight, const View& bias, float eps);
void layer_norm_backward(const View& grad_input, const View& grad_weight, const View& grad_bias, const View& grad, const View& input, const View& mean, const View& rstd, const View& weight);

// NCHW. The kernel divides height and width, and the stride equals the kernel.
void max_pool2d(const View& out, const View& input, int32_t kh, int32_t kw);
void max_pool2d_backward(const View& grad_input, const View& grad, const View& input, const View& output, int32_t kh, int32_t kw);
void avg_pool2d(const View& out, const View& input, int32_t kh, int32_t kw);
void avg_pool2d_backward(const View& grad_input, const View& grad, int32_t kh, int32_t kw);

// Write low + u * (high - low), u in [0, 1), one value per element. Same formula on CPU and CUDA.
void uniform(float* data, size_t numel, float low, float high);

// conv1d: out [batch, outChannels, outWidth], input [batch, inChannels, width],
// weight [outChannels, inChannels, kWidth].
// conv2d: out [batch, outChannels, outHeight, outWidth],
// input [batch, inChannels, height, width], weight [outChannels, inChannels, kHeight, kWidth].
// reverse walks the kernel backward. Samples outside the input contribute 0.
// Same indexing as fastConv1d / fastConv2d. Strides included.
void conv1d(const View& out, const View& input, const View& weight, bool reverse);
void conv2d(const View& out, const View& input, const View& weight, bool reverse);

// Same signatures as above. Defined in src/cuda/*.cu when TSGRAD_HAS_CUDA is set.
// On a CUDA build View::data is already a device pointer. The .cu function
// launches on the process-wide stream and returns without waiting. It does not
// free data. copyToHost and synchronize wait.
namespace cuda {

class Context {
public:
    static Context& get();
    void* getStream();
    void* getCublas();
    void synchronize();
    void checkLastError();
    int device() const;

private:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    int device_;
    void* stream_; // cudaStream_t, stored without including cuda_runtime.h here
    void* cublas_; // cublasHandle_t, stored without including cublas_v2.h here
};

void map_neg(const View& out, const View& in);
void map_inv(const View& out, const View& in);
void map_sigmoid(const View& out, const View& in);
void map_relu(const View& out, const View& in);
void map_log(const View& out, const View& in);
void map_exp(const View& out, const View& in);
void map_id(const View& out, const View& in);

void zip_add(const View& out, const View& a, const View& b);
void zip_sub(const View& out, const View& a, const View& b);
void zip_mul(const View& out, const View& a, const View& b);
void zip_div(const View& out, const View& a, const View& b);
void zip_lt(const View& out, const View& a, const View& b);
void zip_eq(const View& out, const View& a, const View& b);
void zip_is_close(const View& out, const View& a, const View& b);

void sgd(const View& param, const View& grad, float lr);
void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta);

void reduce_max(const View& out, const View& in, int32_t dim);
void reduce_sum(const View& out, const View& in, int32_t dim);
void reduce_prod(const View& out, const View& in, int32_t dim);

// Leading dimensions are a batch. A uniform stride goes to cuBLAS.
// The stride kernel covers every other layout. CPU matmul is still rank 2.
void matmul(const View& out, const View& a, const View& b);

void conv1d(const View& out, const View& input, const View& weight, bool reverse);
void conv2d(const View& out, const View& input, const View& weight, bool reverse);

// Non-zero storage fill. Queued on the process-wide stream. Does not wait.
void fill(float* data, size_t numel, float value);

// seed selects the sequence. The host uniform() advances it.
void uniform(float* data, size_t numel, float low, float high, uint64_t seed);

void softmax(const View& out, const View& in, int32_t dim);
void softmax_backward(const View& out, const View& softmax, const View& grad, int32_t dim);
void log_softmax(const View& out, const View& in, int32_t dim);
void log_softmax_backward(const View& out, const View& log_softmax, const View& grad, int32_t dim);
void softmax_cross_entropy(const View& loss, const View& softmax, const View& logits, const View& target, int32_t dim);
void softmax_cross_entropy_backward(const View& grad_logits, const View& softmax, const View& target, int32_t dim);
void relu_backward(const View& out, const View& grad, const View& input);
void sigmoid_backward(const View& out, const View& grad, const View& output);
void expand(const View& out, const View& in);
void bias_act(const View& out, const View& input, const View& bias, int32_t act);
void bias_act_backward(const View& grad_input, const View& grad_bias, const View& grad, const View& saved, const View& bias, int32_t act);
void layer_norm(const View& out, const View& mean, const View& rstd, const View& input, const View& weight, const View& bias, float eps);
void layer_norm_backward(const View& grad_input, const View& grad_weight, const View& grad_bias, const View& grad, const View& input, const View& mean, const View& rstd, const View& weight);
void max_pool2d(const View& out, const View& input, int32_t kh, int32_t kw);
void max_pool2d_backward(const View& grad_input, const View& grad, const View& input, const View& output, int32_t kh, int32_t kw);
void avg_pool2d(const View& out, const View& input, int32_t kh, int32_t kw);
void avg_pool2d_backward(const View& grad_input, const View& grad, int32_t kh, int32_t kw);

} // namespace cuda
} // namespace tsgrad
