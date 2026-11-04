#include "tsgrad/ops.h"
#include "tsgrad/rows.hpp"
#include "cudahelpers.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <limits>
#include <string>

// tsgrad::cuda fused kernels. src/cpu/fused.cpp forwards here on a CUDA build.
//
// Sequence, same as elementwise.cu:
//   Context::get()
//   kernel on View::data
//   Context::checkLastError()
//
// A row kernel uses one thread per row. LayerNorm uses one block per row.
// Shape and stride ride along as a Layout. Do not free View::data.

namespace tsgrad::cuda {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

cudaStream_t stream() {
    return static_cast<cudaStream_t>(Context::get().getStream());
}

int32_t grid_for(long long n) {
    if (n <= 0) return 0;
    const int32_t threads = 256;
    long long blocks = (n + threads - 1) / threads;
    if (blocks > INT32_MAX)
        throw std::runtime_error("Overflow, row count must fit in int");
    return static_cast<int32_t>(blocks);
}

void require_blocks(long long n) {
    if (n > INT32_MAX)
        throw std::runtime_error("Overflow, row count must fit in int");
}

void require_act(int32_t act) {
    if (act != bias_relu && act != bias_sigmoid)
        throw std::runtime_error("bias_act received an unknown activation");
}

void require_loss(const View& loss) {
    if (shape_numel(loss) != 1)
        throw std::runtime_error("softmax_cross_entropy loss must be one element");
}

void require_rows(const View& stats, long long rows, const char* name) {
    if (rows > INT32_MAX || stats.rank != 1 || static_cast<long long>(stats.shape[0]) != rows)
        throw std::runtime_error(std::string(name) + " must have one value per row");
}

void check_bias_out(const View& out, const View& input, const View& bias) {
    int32_t expect[16] = {};
    int32_t rank = broadcast_shape(input, bias, expect);
    if (out.rank != rank) throw std::runtime_error("bias_act output shape must be the broadcast shape");
    for (int32_t i = 0; i < rank; i++) {
        if (out.shape[i] != expect[i])
            throw std::runtime_error("bias_act output shape must be the broadcast shape");
    }
    if (!can_broadcast(out, input) || !can_broadcast(out, bias))
        throw std::runtime_error("Incompatible shapes for bias_act");
}

__device__ float block_sum(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    float total = shared[0];
    __syncthreads();
    return total;
}

template <bool LogSpace>
__global__ void normalize_kernel(float* out, Layout out_l, const float* in, Layout in_l, int32_t dim, long long rows, long long inner) {
    long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int32_t index[16];
    layout_row_index(row, in_l.shape, in_l.rank, dim, inner, index);
    long long in_base = layout_position(index, in_l.stride, in_l.rank);
    long long out_base = layout_position(index, out_l.stride, out_l.rank);
    int32_t n = in_l.shape[dim];
    int32_t in_step = in_l.stride[dim];
    int32_t out_step = out_l.stride[dim];

    float m = -INFINITY;
    for (int32_t j = 0; j < n; j++)
        m = fmaxf(m, in[in_base + static_cast<long long>(j) * in_step]);

    float sum = 0.0f;
    for (int32_t j = 0; j < n; j++)
        sum += expf(in[in_base + static_cast<long long>(j) * in_step] - m);

    float log_sum = logf(sum);
    for (int32_t j = 0; j < n; j++) {
        float x = in[in_base + static_cast<long long>(j) * in_step];
        float y = LogSpace ? (x - m - log_sum) : expf(x - m) / sum;
        out[out_base + static_cast<long long>(j) * out_step] = y;
    }
}

__global__ void softmax_backward_kernel(float* out, Layout out_l, const float* softmax, Layout s_l, const float* grad, Layout g_l, int32_t dim, long long rows, long long inner) {
    long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int32_t index[16];
    layout_row_index(row, s_l.shape, s_l.rank, dim, inner, index);
    long long base_s = layout_position(index, s_l.stride, s_l.rank);
    long long base_g = layout_position(index, g_l.stride, g_l.rank);
    long long base_o = layout_position(index, out_l.stride, out_l.rank);
    int32_t n = s_l.shape[dim];
    int32_t step_s = s_l.stride[dim];
    int32_t step_g = g_l.stride[dim];
    int32_t step_o = out_l.stride[dim];

    float dot = 0.0f;
    for (int32_t j = 0; j < n; j++)
        dot += grad[base_g + static_cast<long long>(j) * step_g] * softmax[base_s + static_cast<long long>(j) * step_s];
    for (int32_t j = 0; j < n; j++) {
        float s = softmax[base_s + static_cast<long long>(j) * step_s];
        float g = grad[base_g + static_cast<long long>(j) * step_g];
        out[base_o + static_cast<long long>(j) * step_o] = s * (g - dot);
    }
}

__global__ void log_softmax_backward_kernel(float* out, Layout out_l, const float* log_softmax, Layout l_l, const float* grad, Layout g_l, int32_t dim, long long rows, long long inner) {
    long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int32_t index[16];
    layout_row_index(row, l_l.shape, l_l.rank, dim, inner, index);
    long long base_l = layout_position(index, l_l.stride, l_l.rank);
    long long base_g = layout_position(index, g_l.stride, g_l.rank);
    long long base_o = layout_position(index, out_l.stride, out_l.rank);
    int32_t n = l_l.shape[dim];
    int32_t step_l = l_l.stride[dim];
    int32_t step_g = g_l.stride[dim];
    int32_t step_o = out_l.stride[dim];

    float sum_g = 0.0f;
    for (int32_t j = 0; j < n; j++)
        sum_g += grad[base_g + static_cast<long long>(j) * step_g];
    for (int32_t j = 0; j < n; j++) {
        float g = grad[base_g + static_cast<long long>(j) * step_g];
        float p = expf(log_softmax[base_l + static_cast<long long>(j) * step_l]);
        out[base_o + static_cast<long long>(j) * step_o] = g - p * sum_g;
    }
}

__global__ void cross_entropy_kernel(float* loss, float* softmax, Layout s_l, const float* logits, Layout x_l, const float* target, Layout t_l, int32_t dim, long long rows, long long inner) {
    long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int32_t index[16];
    layout_row_index(row, x_l.shape, x_l.rank, dim, inner, index);
    long long base_x = layout_position(index, x_l.stride, x_l.rank);
    long long base_t = layout_position(index, t_l.stride, t_l.rank);
    long long base_p = layout_position(index, s_l.stride, s_l.rank);
    int32_t n = x_l.shape[dim];
    int32_t step_x = x_l.stride[dim];
    int32_t step_t = t_l.stride[dim];
    int32_t step_p = s_l.stride[dim];

    float m = -INFINITY;
    for (int32_t j = 0; j < n; j++)
        m = fmaxf(m, logits[base_x + static_cast<long long>(j) * step_x]);

    float sum = 0.0f;
    for (int32_t j = 0; j < n; j++)
        sum += expf(logits[base_x + static_cast<long long>(j) * step_x] - m);

    float log_z = m + logf(sum);
    float nll = 0.0f;
    for (int32_t j = 0; j < n; j++) {
        float x = logits[base_x + static_cast<long long>(j) * step_x];
        softmax[base_p + static_cast<long long>(j) * step_p] = expf(x - m) / sum;
        nll -= target[base_t + static_cast<long long>(j) * step_t] * (x - log_z);
    }
    atomicAdd(loss, nll);
}

__global__ void scale_one_kernel(float* value, float scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) value[0] *= scale;
}

__global__ void cross_entropy_backward_kernel(float* out, Layout out_l, const float* softmax, Layout s_l, const float* target, Layout t_l, int32_t dim, float inv_rows, long long rows, long long inner) {
    long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    int32_t index[16];
    layout_row_index(row, s_l.shape, s_l.rank, dim, inner, index);
    long long base_p = layout_position(index, s_l.stride, s_l.rank);
    long long base_t = layout_position(index, t_l.stride, t_l.rank);
    long long base_o = layout_position(index, out_l.stride, out_l.rank);
    int32_t n = s_l.shape[dim];
    int32_t step_p = s_l.stride[dim];
    int32_t step_t = t_l.stride[dim];
    int32_t step_o = out_l.stride[dim];
    for (int32_t j = 0; j < n; j++) {
        float p = softmax[base_p + static_cast<long long>(j) * step_p];
        float y = target[base_t + static_cast<long long>(j) * step_t];
        out[base_o + static_cast<long long>(j) * step_o] = (p - y) * inv_rows;
    }
}

__global__ void relu_contig_kernel(float* out, const float* grad, const float* input, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = input[i] > 0.0f ? grad[i] : 0.0f;
}

__global__ void relu_strided_kernel(float* out, Layout out_l, const float* grad, Layout g_l, const float* input, Layout in_l, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t index[16];
    layout_index(i, out_l.shape, index, out_l.rank);
    float value = input[layout_position(index, in_l.stride, in_l.rank)];
    float g = grad[layout_position(index, g_l.stride, g_l.rank)];
    out[layout_position(index, out_l.stride, out_l.rank)] = value > 0.0f ? g : 0.0f;
}

__global__ void sigmoid_contig_kernel(float* out, const float* grad, const float* y, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = y[i];
    out[i] = grad[i] * s * (1.0f - s);
}

__global__ void sigmoid_strided_kernel(float* out, Layout out_l, const float* grad, Layout g_l, const float* y, Layout y_l, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t index[16];
    layout_index(i, out_l.shape, index, out_l.rank);
    float s = y[layout_position(index, y_l.stride, y_l.rank)];
    float g = grad[layout_position(index, g_l.stride, g_l.rank)];
    out[layout_position(index, out_l.stride, out_l.rank)] = g * s * (1.0f - s);
}

__global__ void expand_kernel(float* out, Layout out_l, const float* in, Layout in_l, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t out_index[16];
    int32_t in_index[16];
    layout_index(i, out_l.shape, out_index, out_l.rank);
    layout_broadcast(out_index, out_l.shape, out_l.rank, in_l.shape, in_index, in_l.rank);
    out[layout_position(out_index, out_l.stride, out_l.rank)] = in[layout_position(in_index, in_l.stride, in_l.rank)];
}

__device__ float bias_relu_value(float a) {
    return a > 0.0f ? a : 0.0f;
}

__device__ float bias_sigmoid_value(float a) {
    if (a >= 0.0f) return 1.0f / (1.0f + expf(-a));
    return expf(a) / (1.0f + expf(a));
}

__global__ void bias_act_kernel(float* out, Layout out_l, const float* input, Layout in_l, const float* bias, Layout bias_l, int32_t act, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t out_index[16];
    int32_t in_index[16];
    int32_t bias_index[16];
    layout_index(i, out_l.shape, out_index, out_l.rank);
    layout_broadcast(out_index, out_l.shape, out_l.rank, in_l.shape, in_index, in_l.rank);
    layout_broadcast(out_index, out_l.shape, out_l.rank, bias_l.shape, bias_index, bias_l.rank);
    float pre = input[layout_position(in_index, in_l.stride, in_l.rank)] + bias[layout_position(bias_index, bias_l.stride, bias_l.rank)];
    float y = act == 0 ? bias_relu_value(pre) : bias_sigmoid_value(pre);
    out[layout_position(out_index, out_l.stride, out_l.rank)] = y;
}

__global__ void bias_act_backward_kernel(float* grad_input, Layout out_l, const float* grad, Layout g_l, const float* saved, Layout saved_l, const float* bias, Layout bias_l, int32_t act, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t out_index[16];
    layout_index(i, out_l.shape, out_index, out_l.rank);
    float g = grad[layout_position(out_index, g_l.stride, g_l.rank)];
    if (act == 0) {
        int32_t in_index[16];
        int32_t bias_index[16];
        layout_broadcast(out_index, out_l.shape, out_l.rank, saved_l.shape, in_index, saved_l.rank);
        layout_broadcast(out_index, out_l.shape, out_l.rank, bias_l.shape, bias_index, bias_l.rank);
        float pre = saved[layout_position(in_index, saved_l.stride, saved_l.rank)] + bias[layout_position(bias_index, bias_l.stride, bias_l.rank)];
        g = pre > 0.0f ? g : 0.0f;
    } else {
        float y = saved[layout_position(out_index, saved_l.stride, saved_l.rank)];
        g = g * y * (1.0f - y);
    }
    grad_input[layout_position(out_index, out_l.stride, out_l.rank)] = g;
}

__global__ void broadcast_sum_kernel(float* grad_bias, Layout bias_l, const float* grad_input, Layout out_l, SumAxes axes, long long bias_n) {
    long long b = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= bias_n) return;

    int32_t bias_index[16];
    int32_t out_index[16];
    for (int32_t i = 0; i < 16; i++) out_index[i] = 0;
    layout_index(b, bias_l.shape, bias_index, bias_l.rank);
    int32_t diff = out_l.rank - bias_l.rank;
    for (int32_t i = 0; i < bias_l.rank; i++) {
        if (bias_l.shape[i] != 1)
            out_index[i + diff] = bias_index[i];
    }

    float acc = 0.0f;
    for (long long v = 0; v < axes.volume; v++) {
        long long t = v;
        for (int32_t k = axes.count - 1; k >= 0; k--) {
            out_index[axes.axes[k]] = static_cast<int32_t>(t % axes.sizes[k]);
            t /= axes.sizes[k];
        }
        acc += grad_input[layout_position(out_index, out_l.stride, out_l.rank)];
    }
    grad_bias[layout_position(bias_index, bias_l.stride, bias_l.rank)] = acc;
}

__global__ void layer_norm_kernel(float* out, Layout out_l, float* mean, int32_t mean_stride, float* rstd, int32_t rstd_stride, const float* input, Layout in_l, const float* weight, int32_t weight_stride, const float* bias, int32_t bias_stride, float eps, long long rows, long long inner) {
    long long row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float shared[256];
    int32_t dim = in_l.rank - 1;
    int32_t n = in_l.shape[dim];
    int32_t index[16];
    layout_row_index(row, in_l.shape, in_l.rank, dim, inner, index);
    long long base = layout_position(index, in_l.stride, in_l.rank);
    long long out_base = layout_position(index, out_l.stride, out_l.rank);
    int32_t step = in_l.stride[dim];
    int32_t out_step = out_l.stride[dim];

    float sum = 0.0f;
    for (int32_t j = threadIdx.x; j < n; j += blockDim.x)
        sum += input[base + static_cast<long long>(j) * step];
    sum = block_sum(sum, shared);
    float m = sum / static_cast<float>(n);

    float var = 0.0f;
    for (int32_t j = threadIdx.x; j < n; j += blockDim.x) {
        float d = input[base + static_cast<long long>(j) * step] - m;
        var += d * d;
    }
    var = block_sum(var, shared) / static_cast<float>(n);
    float rs = rsqrtf(var + eps);
    if (threadIdx.x == 0) {
        mean[row * mean_stride] = m;
        rstd[row * rstd_stride] = rs;
    }
    __syncthreads();

    for (int32_t j = threadIdx.x; j < n; j += blockDim.x) {
        float xhat = (input[base + static_cast<long long>(j) * step] - m) * rs;
        out[out_base + static_cast<long long>(j) * out_step] = xhat * weight[static_cast<long long>(j) * weight_stride] + bias[static_cast<long long>(j) * bias_stride];
    }
}

__global__ void layer_norm_backward_kernel(float* grad_input, Layout out_l, const float* grad, Layout g_l, const float* input, Layout in_l, const float* mean, int32_t mean_stride, const float* rstd, int32_t rstd_stride, const float* weight, int32_t weight_stride, long long rows, long long inner) {
    long long row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float shared[256];
    int32_t dim = in_l.rank - 1;
    int32_t n = in_l.shape[dim];
    int32_t index[16];
    layout_row_index(row, in_l.shape, in_l.rank, dim, inner, index);
    long long base = layout_position(index, in_l.stride, in_l.rank);
    long long g_base = layout_position(index, g_l.stride, g_l.rank);
    long long o_base = layout_position(index, out_l.stride, out_l.rank);
    int32_t step = in_l.stride[dim];
    int32_t g_step = g_l.stride[dim];
    int32_t o_step = out_l.stride[dim];
    float m = mean[row * mean_stride];
    float rs = rstd[row * rstd_stride];

    float sum_g = 0.0f;
    float sum_gx = 0.0f;
    for (int32_t j = threadIdx.x; j < n; j += blockDim.x) {
        float xhat = (input[base + static_cast<long long>(j) * step] - m) * rs;
        float g = grad[g_base + static_cast<long long>(j) * g_step] * weight[static_cast<long long>(j) * weight_stride];
        sum_g += g;
        sum_gx += g * xhat;
    }
    float reduced_g = block_sum(sum_g, shared);
    float reduced_gx = block_sum(sum_gx, shared);
    float inv_n = 1.0f / static_cast<float>(n);
    float mean_g = reduced_g * inv_n;
    float mean_gx = reduced_gx * inv_n;

    for (int32_t j = threadIdx.x; j < n; j += blockDim.x) {
        long long pos = base + static_cast<long long>(j) * step;
        long long gpos = g_base + static_cast<long long>(j) * g_step;
        float xhat = (input[pos] - m) * rs;
        float g = grad[gpos] * weight[static_cast<long long>(j) * weight_stride];
        grad_input[o_base + static_cast<long long>(j) * o_step] = rs * (g - mean_g - xhat * mean_gx);
    }
}

__global__ void layer_norm_affine_kernel(float* grad_weight, int32_t grad_w_stride, float* grad_bias, int32_t grad_b_stride, const float* grad, Layout g_l, const float* input, Layout in_l, const float* mean, int32_t mean_stride, const float* rstd, int32_t rstd_stride, int32_t n, long long rows, long long inner) {
    long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int32_t dim = in_l.rank - 1;
    float gw = 0.0f;
    float gb = 0.0f;
    for (long long row = 0; row < rows; row++) {
        int32_t index[16];
        layout_row_index(row, in_l.shape, in_l.rank, dim, inner, index);
        index[dim] = static_cast<int32_t>(j);
        long long pos = layout_position(index, in_l.stride, in_l.rank);
        long long gpos = layout_position(index, g_l.stride, g_l.rank);
        float xhat = (input[pos] - mean[row * mean_stride]) * rstd[row * rstd_stride];
        float dy = grad[gpos];
        gw += dy * xhat;
        gb += dy;
    }
    grad_weight[static_cast<long long>(j) * grad_w_stride] = gw;
    grad_bias[static_cast<long long>(j) * grad_b_stride] = gb;
}

__global__ void max_pool_kernel(float* out, Layout out_l, const float* input, Layout in_l, int32_t kh, int32_t kw, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t index[4];
    layout_index(i, out_l.shape, index, 4);
    float m = -INFINITY;
    for (int32_t ih = 0; ih < kh; ih++) {
        for (int32_t iw = 0; iw < kw; iw++) {
            int32_t in_index[4] = {index[0], index[1], index[2] * kh + ih, index[3] * kw + iw};
            m = fmaxf(m, input[layout_position(in_index, in_l.stride, 4)]);
        }
    }
    out[layout_position(index, out_l.stride, 4)] = m;
}

__global__ void max_pool_backward_kernel(float* grad_input, Layout in_l, const float* grad, Layout g_l, const float* input, const float* output, Layout out_l, int32_t kh, int32_t kw, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t in_index[4];
    layout_index(i, in_l.shape, in_index, 4);
    int32_t out_index[4] = {in_index[0], in_index[1], in_index[2] / kh, in_index[3] / kw};
    float value = input[layout_position(in_index, in_l.stride, 4)];
    float pooled = output[layout_position(out_index, out_l.stride, 4)];
    float g = grad[layout_position(out_index, g_l.stride, 4)];
    grad_input[layout_position(in_index, in_l.stride, 4)] = value == pooled ? g : 0.0f;
}

__global__ void avg_pool_kernel(float* out, Layout out_l, const float* input, Layout in_l, int32_t kh, int32_t kw, float inv, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t index[4];
    layout_index(i, out_l.shape, index, 4);
    float sum = 0.0f;
    for (int32_t ih = 0; ih < kh; ih++) {
        for (int32_t iw = 0; iw < kw; iw++) {
            int32_t in_index[4] = {index[0], index[1], index[2] * kh + ih, index[3] * kw + iw};
            sum += input[layout_position(in_index, in_l.stride, 4)];
        }
    }
    out[layout_position(index, out_l.stride, 4)] = sum * inv;
}

__global__ void avg_pool_backward_kernel(float* grad_input, Layout in_l, const float* grad, Layout g_l, int32_t kh, int32_t kw, float inv, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t in_index[4];
    layout_index(i, in_l.shape, in_index, 4);
    int32_t out_index[4] = {in_index[0], in_index[1], in_index[2] / kh, in_index[3] / kw};
    grad_input[layout_position(in_index, in_l.stride, 4)] = grad[layout_position(out_index, g_l.stride, 4)] * inv;
}

void softmax(const View& out, const View& in, int32_t dim) {
    require_same(out, in, "softmax");
    RowSplit split = split_rows(in, dim, "softmax");
    if (shape_numel(in) == 0) return;
    int blocks = grid_for(split.rows);
    normalize_kernel<false><<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), in.data, layout_from(in), dim, split.rows, split.inner);
    Context::get().checkLastError();
}

void softmax_backward(const View& out, const View& softmax, const View& grad, int32_t dim) {
    require_same(out, softmax, "softmax_backward");
    require_same(out, grad, "softmax_backward");
    RowSplit split = split_rows(softmax, dim, "softmax_backward");
    if (shape_numel(softmax) == 0) return;
    int blocks = grid_for(split.rows);
    softmax_backward_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), softmax.data, layout_from(softmax), grad.data, layout_from(grad), dim, split.rows, split.inner);
    Context::get().checkLastError();
}

void log_softmax(const View& out, const View& in, int32_t dim) {
    require_same(out, in, "log_softmax");
    RowSplit split = split_rows(in, dim, "log_softmax");
    if (shape_numel(in) == 0) return;
    int blocks = grid_for(split.rows);
    normalize_kernel<true><<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), in.data, layout_from(in), dim, split.rows, split.inner);
    Context::get().checkLastError();
}

void log_softmax_backward(const View& out, const View& log_softmax, const View& grad, int32_t dim) {
    require_same(out, log_softmax, "log_softmax_backward");
    require_same(out, grad, "log_softmax_backward");
    RowSplit split = split_rows(log_softmax, dim, "log_softmax_backward");
    if (shape_numel(log_softmax) == 0) return;
    int blocks = grid_for(split.rows);
    log_softmax_backward_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), log_softmax.data, layout_from(log_softmax), grad.data, layout_from(grad), dim, split.rows, split.inner);
    Context::get().checkLastError();
}

void softmax_cross_entropy(const View& loss, const View& softmax, const View& logits, const View& target, int32_t dim) {
    require_loss(loss);
    require_same(softmax, logits, "softmax_cross_entropy");
    require_same(target, logits, "softmax_cross_entropy");
    RowSplit split = split_rows(logits, dim, "softmax_cross_entropy");
    check_cuda(cudaMemsetAsync(loss.data, 0, sizeof(float), stream()), "cudaMemsetAsync");
    if (shape_numel(logits) == 0) {
        Context::get().checkLastError();
        return;
    }
    int blocks = grid_for(split.rows);
    cross_entropy_kernel<<<blocks, 256, 0, stream()>>>(loss.data, softmax.data, layout_from(softmax), logits.data, layout_from(logits), target.data, layout_from(target), dim, split.rows, split.inner);
    scale_one_kernel<<<1, 1, 0, stream()>>>(loss.data, 1.0f / static_cast<float>(split.rows));
    Context::get().checkLastError();
}

void softmax_cross_entropy_backward(const View& grad_logits, const View& softmax, const View& target, int32_t dim) {
    require_same(grad_logits, softmax, "softmax_cross_entropy_backward");
    require_same(target, softmax, "softmax_cross_entropy_backward");
    RowSplit split = split_rows(softmax, dim, "softmax_cross_entropy_backward");
    if (shape_numel(softmax) == 0) return;
    int blocks = grid_for(split.rows);
    float inv_rows = 1.0f / static_cast<float>(split.rows);
    cross_entropy_backward_kernel<<<blocks, 256, 0, stream()>>>(grad_logits.data, layout_from(grad_logits), softmax.data, layout_from(softmax), target.data, layout_from(target), dim, inv_rows, split.rows, split.inner);
    Context::get().checkLastError();
}

void relu_backward(const View& out, const View& grad, const View& input) {
    check_shape(out, "relu_backward");
    require_same(out, grad, "relu_backward");
    require_same(out, input, "relu_backward");
    long long n = shape_numel(out);
    if (n == 0) return;
    int blocks = grid_for(n);
    if (is_contiguous(out) && is_contiguous(grad) && is_contiguous(input))
        relu_contig_kernel<<<blocks, 256, 0, stream()>>>(out.data, grad.data, input.data, n);
    else
        relu_strided_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), grad.data, layout_from(grad), input.data, layout_from(input), n);
    Context::get().checkLastError();
}

void sigmoid_backward(const View& out, const View& grad, const View& output) {
    check_shape(out, "sigmoid_backward");
    require_same(out, grad, "sigmoid_backward");
    require_same(out, output, "sigmoid_backward");
    long long n = shape_numel(out);
    if (n == 0) return;
    int blocks = grid_for(n);
    if (is_contiguous(out) && is_contiguous(grad) && is_contiguous(output))
        sigmoid_contig_kernel<<<blocks, 256, 0, stream()>>>(out.data, grad.data, output.data, n);
    else
        sigmoid_strided_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), grad.data, layout_from(grad), output.data, layout_from(output), n);
    Context::get().checkLastError();
}

void expand(const View& out, const View& in) {
    check_shape(out, "expand");
    check_shape(in, "expand");
    if (!can_broadcast(out, in))
        throw std::runtime_error("expand received a shape that does not broadcast");
    long long n = shape_numel(out);
    if (n == 0) return;
    if (same_shape(out, in) && is_contiguous(out) && is_contiguous(in)) {
        check_cuda(cudaMemcpyAsync(out.data, in.data, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice, stream()), "cudaMemcpyAsync");
        Context::get().checkLastError();
        return;
    }
    int blocks = grid_for(n);
    expand_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), in.data, layout_from(in), n);
    Context::get().checkLastError();
}

void bias_act(const View& out, const View& input, const View& bias, int32_t act) {
    require_act(act);
    check_bias_out(out, input, bias);
    long long n = shape_numel(out);
    if (n == 0) return;
    int blocks = grid_for(n);
    bias_act_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), input.data, layout_from(input), bias.data, layout_from(bias), act, n);
    Context::get().checkLastError();
}

void bias_act_backward(const View& grad_input, const View& grad_bias, const View& grad, const View& saved, const View& bias, int32_t act) {
    require_act(act);
    require_same(grad_input, grad, "bias_act_backward");
    check_shape(grad_bias, "bias_act_backward");
    check_shape(bias, "bias_act_backward");
    if (!same_shape(grad_bias, bias))
        throw std::runtime_error("bias_act_backward bias gradient shape must match the bias");
    if (!can_broadcast(grad, bias))
        throw std::runtime_error("Incompatible shapes for bias_act");
    if (act == bias_relu) {
        if (!can_broadcast(grad, saved))
            throw std::runtime_error("Incompatible shapes for bias_act");
    } else if (!same_shape(saved, grad)) {
        throw std::runtime_error("bias_act_backward saved output shape must match the gradient");
    }

    long long n = shape_numel(grad);
    long long bias_n = shape_numel(grad_bias);
    if (n == 0) {
        if (bias_n == 0) return;
        int blocks = grid_for(bias_n);
        // A zero-length input still writes zeros into the bias gradient. Reuse the sum kernel with an empty volume.
        SumAxes axes{};
        axes.volume = 0;
        broadcast_sum_kernel<<<blocks, 256, 0, stream()>>>(grad_bias.data, layout_from(grad_bias), grad_input.data, layout_from(grad_input), axes, bias_n);
        Context::get().checkLastError();
        return;
    }

    int blocks = grid_for(n);
    bias_act_backward_kernel<<<blocks, 256, 0, stream()>>>(grad_input.data, layout_from(grad_input), grad.data, layout_from(grad), saved.data, layout_from(saved), bias.data, layout_from(bias), act, n);
    if (bias_n > 0) {
        SumAxes axes = sum_axes(grad_input, grad_bias);
        int bias_blocks = grid_for(bias_n);
        broadcast_sum_kernel<<<bias_blocks, 256, 0, stream()>>>(grad_bias.data, layout_from(grad_bias), grad_input.data, layout_from(grad_input), axes, bias_n);
    }
    Context::get().checkLastError();
}

void layer_norm(const View& out, const View& mean, const View& rstd, const View& input, const View& weight, const View& bias, float eps) {
    check_shape(input, "layer_norm");
    if (input.rank < 1) throw std::runtime_error("layer_norm expected a last dimension");
    require_same(out, input, "layer_norm");
    int32_t dim = input.rank - 1;
    int32_t n = input.shape[dim];
    if (n == 0) throw std::runtime_error("layer_norm received an empty last dimension");
    RowSplit split = split_rows(input, dim, "layer_norm");
    require_rows(mean, split.rows, "layer_norm mean");
    require_rows(rstd, split.rows, "layer_norm rstd");
    if (weight.rank != 1 || weight.shape[0] != n || bias.rank != 1 || bias.shape[0] != n)
        throw std::runtime_error("layer_norm weight must have the last dimension's length");
    if (shape_numel(input) == 0) return;
    require_blocks(split.rows);
    layer_norm_kernel<<<static_cast<int>(split.rows), 256, 0, stream()>>>(out.data, layout_from(out), mean.data, mean.stride[0], rstd.data, rstd.stride[0], input.data, layout_from(input), weight.data, weight.stride[0], bias.data, bias.stride[0], eps, split.rows, split.inner);
    Context::get().checkLastError();
}

void layer_norm_backward(const View& grad_input, const View& grad_weight, const View& grad_bias, const View& grad, const View& input, const View& mean, const View& rstd, const View& weight) {
    check_shape(input, "layer_norm_backward");
    if (input.rank < 1) throw std::runtime_error("layer_norm expected a last dimension");
    require_same(grad_input, input, "layer_norm_backward");
    require_same(grad, input, "layer_norm_backward");
    int32_t dim = input.rank - 1;
    int32_t n = input.shape[dim];
    if (n == 0) throw std::runtime_error("layer_norm received an empty last dimension");
    RowSplit split = split_rows(input, dim, "layer_norm_backward");
    require_rows(mean, split.rows, "layer_norm mean");
    require_rows(rstd, split.rows, "layer_norm rstd");
    if (weight.rank != 1 || weight.shape[0] != n)
        throw std::runtime_error("layer_norm weight must have the last dimension's length");
    if (!same_shape(grad_weight, weight) || !same_shape(grad_bias, weight))
        throw std::runtime_error("layer_norm weight gradient must have the last dimension's length");
    if (shape_numel(input) == 0) return;
    require_blocks(split.rows);
    layer_norm_backward_kernel<<<static_cast<int>(split.rows), 256, 0, stream()>>>(grad_input.data, layout_from(grad_input), grad.data, layout_from(grad), input.data, layout_from(input), mean.data, mean.stride[0], rstd.data, rstd.stride[0], weight.data, weight.stride[0], split.rows, split.inner);
    int affine_blocks = grid_for(n);
    layer_norm_affine_kernel<<<affine_blocks, 256, 0, stream()>>>(grad_weight.data, grad_weight.stride[0], grad_bias.data, grad_bias.stride[0], grad.data, layout_from(grad), input.data, layout_from(input), mean.data, mean.stride[0], rstd.data, rstd.stride[0], n, split.rows, split.inner);
    Context::get().checkLastError();
}

void max_pool2d(const View& out, const View& input, int32_t kh, int32_t kw) {
    prepare_pool(input, out, kh, kw, "max_pool2d");
    long long n = shape_numel(out);
    if (n == 0) return;
    int blocks = grid_for(n);
    max_pool_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), input.data, layout_from(input), kh, kw, n);
    Context::get().checkLastError();
}

void max_pool2d_backward(const View& grad_input, const View& grad, const View& input, const View& output, int32_t kh, int32_t kw) {
    prepare_pool(input, output, kh, kw, "max_pool2d_backward");
    require_same(grad, output, "max_pool2d_backward");
    if (!same_shape(grad_input, input))
        throw std::runtime_error("max_pool2d_backward input gradient shape must match the input");
    long long n = shape_numel(input);
    if (n == 0) return;
    int blocks = grid_for(n);
    max_pool_backward_kernel<<<blocks, 256, 0, stream()>>>(grad_input.data, layout_from(grad_input), grad.data, layout_from(grad), input.data, output.data, layout_from(output), kh, kw, n);
    Context::get().checkLastError();
}

void avg_pool2d(const View& out, const View& input, int32_t kh, int32_t kw) {
    prepare_pool(input, out, kh, kw, "avg_pool2d");
    long long n = shape_numel(out);
    if (n == 0) return;
    float inv = 1.0f / static_cast<float>(static_cast<long long>(kh) * static_cast<long long>(kw));
    int blocks = grid_for(n);
    avg_pool_kernel<<<blocks, 256, 0, stream()>>>(out.data, layout_from(out), input.data, layout_from(input), kh, kw, inv, n);
    Context::get().checkLastError();
}

void avg_pool2d_backward(const View& grad_input, const View& grad, int32_t kh, int32_t kw) {
    prepare_pool(grad_input, grad, kh, kw, "avg_pool2d_backward");
    long long n = shape_numel(grad_input);
    if (n == 0) return;
    float inv = 1.0f / static_cast<float>(static_cast<long long>(kh) * static_cast<long long>(kw));
    int blocks = grid_for(n);
    avg_pool_backward_kernel<<<blocks, 256, 0, stream()>>>(grad_input.data, layout_from(grad_input), grad.data, layout_from(grad), kh, kw, inv, n);
    Context::get().checkLastError();
}

} // namespace tsgrad::cuda
