#include "tsgrad/ops.h"
#include "tsgrad/rows.hpp"
#include "helpers.hpp"

#include <cmath>
#include <limits>

// Host entry points for the fused ops. addon.cc calls these.
// A CUDA build forwards to tsgrad::cuda in src/cuda/fused.cu.
// The CPU loops are the same math, so a build without nvcc still runs.

namespace tsgrad {

namespace {

float stable_sigmoid(float a) {
    if (a >= 0.0f) return 1.0f / (1.0f + std::exp(-a));
    return std::exp(a) / (1.0f + std::exp(a));
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

void normalize_rows(const View& out, const View& in, int32_t dim, bool log_space) {
    require_same(out, in, log_space ? "log_softmax" : "softmax");
    RowSplit split = split_rows(in, dim, log_space ? "log_softmax" : "softmax");
    if (shape_numel(in) == 0) return;

    const int32_t n = split.dim_size;
    const int32_t in_step = in.stride[dim];
    const int32_t out_step = out.stride[dim];

    #pragma omp parallel for if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, in.shape, in.rank, dim, split.inner, index);
        long long in_base = layout_position(index, in.stride, in.rank);
        long long out_base = layout_position(index, out.stride, out.rank);

        float m = -std::numeric_limits<float>::infinity();
        for (int32_t j = 0; j < n; j++)
            m = std::max(m, in.data[in_base + static_cast<long long>(j) * in_step]);

        float sum = 0.0f;
        for (int32_t j = 0; j < n; j++)
            sum += std::exp(in.data[in_base + static_cast<long long>(j) * in_step] - m);

        float log_sum = std::log(sum);
        for (int32_t j = 0; j < n; j++) {
            float x = in.data[in_base + static_cast<long long>(j) * in_step];
            float y = log_space ? (x - m - log_sum) : std::exp(x - m) / sum;
            out.data[out_base + static_cast<long long>(j) * out_step] = y;
        }
    }
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

} // namespace

void softmax(const View& out, const View& in, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::softmax(out, in, dim);
        return;
    }
    normalize_rows(out, in, dim, false);
}

void softmax_backward(const View& out, const View& softmax, const View& grad, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::softmax_backward(out, softmax, grad, dim);
        return;
    }
    require_same(out, softmax, "softmax_backward");
    require_same(out, grad, "softmax_backward");
    RowSplit split = split_rows(softmax, dim, "softmax_backward");
    if (shape_numel(softmax) == 0) return;

    const int32_t n = split.dim_size;
    const int32_t step_s = softmax.stride[dim];
    const int32_t step_g = grad.stride[dim];
    const int32_t step_o = out.stride[dim];

    #pragma omp parallel for if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, softmax.shape, softmax.rank, dim, split.inner, index);
        long long base_s = layout_position(index, softmax.stride, softmax.rank);
        long long base_g = layout_position(index, grad.stride, grad.rank);
        long long base_o = layout_position(index, out.stride, out.rank);
        float dot = 0.0f;
        for (int32_t j = 0; j < n; j++)
            dot += grad.data[base_g + static_cast<long long>(j) * step_g] * softmax.data[base_s + static_cast<long long>(j) * step_s];
        for (int32_t j = 0; j < n; j++) {
            float s = softmax.data[base_s + static_cast<long long>(j) * step_s];
            float g = grad.data[base_g + static_cast<long long>(j) * step_g];
            out.data[base_o + static_cast<long long>(j) * step_o] = s * (g - dot);
        }
    }
}

void log_softmax(const View& out, const View& in, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::log_softmax(out, in, dim);
        return;
    }
    normalize_rows(out, in, dim, true);
}

void log_softmax_backward(const View& out, const View& log_softmax, const View& grad, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::log_softmax_backward(out, log_softmax, grad, dim);
        return;
    }
    require_same(out, log_softmax, "log_softmax_backward");
    require_same(out, grad, "log_softmax_backward");
    RowSplit split = split_rows(log_softmax, dim, "log_softmax_backward");
    if (shape_numel(log_softmax) == 0) return;

    const int32_t n = split.dim_size;
    const int32_t step_l = log_softmax.stride[dim];
    const int32_t step_g = grad.stride[dim];
    const int32_t step_o = out.stride[dim];

    #pragma omp parallel for if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, log_softmax.shape, log_softmax.rank, dim, split.inner, index);
        long long base_l = layout_position(index, log_softmax.stride, log_softmax.rank);
        long long base_g = layout_position(index, grad.stride, grad.rank);
        long long base_o = layout_position(index, out.stride, out.rank);
        float sum_g = 0.0f;
        for (int32_t j = 0; j < n; j++)
            sum_g += grad.data[base_g + static_cast<long long>(j) * step_g];
        for (int32_t j = 0; j < n; j++) {
            float g = grad.data[base_g + static_cast<long long>(j) * step_g];
            float p = std::exp(log_softmax.data[base_l + static_cast<long long>(j) * step_l]);
            out.data[base_o + static_cast<long long>(j) * step_o] = g - p * sum_g;
        }
    }
}

void softmax_cross_entropy(const View& loss, const View& softmax, const View& logits, const View& target, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::softmax_cross_entropy(loss, softmax, logits, target, dim);
        return;
    }
    require_loss(loss);
    require_same(softmax, logits, "softmax_cross_entropy");
    require_same(target, logits, "softmax_cross_entropy");
    RowSplit split = split_rows(logits, dim, "softmax_cross_entropy");
    if (shape_numel(logits) == 0) {
        loss.data[0] = 0.0f;
        return;
    }

    const int32_t n = split.dim_size;
    const int32_t step_x = logits.stride[dim];
    const int32_t step_t = target.stride[dim];
    const int32_t step_p = softmax.stride[dim];
    float total = 0.0f;

    #pragma omp parallel for reduction(+:total) if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, logits.shape, logits.rank, dim, split.inner, index);
        long long base_x = layout_position(index, logits.stride, logits.rank);
        long long base_t = layout_position(index, target.stride, target.rank);
        long long base_p = layout_position(index, softmax.stride, softmax.rank);

        float m = -std::numeric_limits<float>::infinity();
        for (int32_t j = 0; j < n; j++)
            m = std::max(m, logits.data[base_x + static_cast<long long>(j) * step_x]);

        float sum = 0.0f;
        for (int32_t j = 0; j < n; j++)
            sum += std::exp(logits.data[base_x + static_cast<long long>(j) * step_x] - m);

        float log_z = m + std::log(sum);
        float nll = 0.0f;
        for (int32_t j = 0; j < n; j++) {
            float x = logits.data[base_x + static_cast<long long>(j) * step_x];
            softmax.data[base_p + static_cast<long long>(j) * step_p] = std::exp(x - m) / sum;
            nll -= target.data[base_t + static_cast<long long>(j) * step_t] * (x - log_z);
        }
        total += nll;
    }

    loss.data[0] = total / static_cast<float>(split.rows);
}

void softmax_cross_entropy_backward(const View& grad_logits, const View& softmax, const View& target, int32_t dim) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::softmax_cross_entropy_backward(grad_logits, softmax, target, dim);
        return;
    }
    require_same(grad_logits, softmax, "softmax_cross_entropy_backward");
    require_same(target, softmax, "softmax_cross_entropy_backward");
    RowSplit split = split_rows(softmax, dim, "softmax_cross_entropy_backward");
    if (shape_numel(softmax) == 0) return;

    const float inv_rows = 1.0f / static_cast<float>(split.rows);
    const int32_t n = split.dim_size;
    const int32_t step_p = softmax.stride[dim];
    const int32_t step_t = target.stride[dim];
    const int32_t step_o = grad_logits.stride[dim];

    #pragma omp parallel for if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, softmax.shape, softmax.rank, dim, split.inner, index);
        long long base_p = layout_position(index, softmax.stride, softmax.rank);
        long long base_t = layout_position(index, target.stride, target.rank);
        long long base_o = layout_position(index, grad_logits.stride, grad_logits.rank);
        for (int32_t j = 0; j < n; j++) {
            float p = softmax.data[base_p + static_cast<long long>(j) * step_p];
            float y = target.data[base_t + static_cast<long long>(j) * step_t];
            grad_logits.data[base_o + static_cast<long long>(j) * step_o] = (p - y) * inv_rows;
        }
    }
}

void relu_backward(const View& out, const View& grad, const View& input) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::relu_backward(out, grad, input);
        return;
    }
    check_shape(out, "relu_backward");
    require_same(out, grad, "relu_backward");
    require_same(out, input, "relu_backward");
    long long n = shape_numel(out);
    if (n == 0) return;

    if (is_contiguous(out) && is_contiguous(grad) && is_contiguous(input)) {
        #pragma omp parallel for if (n >= THRESHOLD)
        for (long long i = 0; i < n; i++)
            out.data[i] = input.data[i] > 0.0f ? grad.data[i] : 0.0f;
        return;
    }

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t index[16] = {};
        layout_index(i, out.shape, index, out.rank);
        long long pos_in = layout_position(index, input.stride, input.rank);
        long long pos_g = layout_position(index, grad.stride, grad.rank);
        long long pos_o = layout_position(index, out.stride, out.rank);
        out.data[pos_o] = input.data[pos_in] > 0.0f ? grad.data[pos_g] : 0.0f;
    }
}

void sigmoid_backward(const View& out, const View& grad, const View& output) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::sigmoid_backward(out, grad, output);
        return;
    }
    check_shape(out, "sigmoid_backward");
    require_same(out, grad, "sigmoid_backward");
    require_same(out, output, "sigmoid_backward");
    long long n = shape_numel(out);
    if (n == 0) return;

    if (is_contiguous(out) && is_contiguous(grad) && is_contiguous(output)) {
        #pragma omp parallel for if (n >= THRESHOLD)
        for (long long i = 0; i < n; i++) {
            float y = output.data[i];
            out.data[i] = grad.data[i] * y * (1.0f - y);
        }
        return;
    }

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t index[16] = {};
        layout_index(i, out.shape, index, out.rank);
        float y = output.data[layout_position(index, output.stride, output.rank)];
        float g = grad.data[layout_position(index, grad.stride, grad.rank)];
        out.data[layout_position(index, out.stride, out.rank)] = g * y * (1.0f - y);
    }
}

void expand(const View& out, const View& in) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::expand(out, in);
        return;
    }
    check_shape(out, "expand");
    check_shape(in, "expand");
    if (!can_broadcast(out, in))
        throw std::runtime_error("expand received a shape that does not broadcast");
    long long n = shape_numel(out);
    if (n == 0) return;

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t out_index[16] = {};
        int32_t in_index[16] = {};
        layout_index(i, out.shape, out_index, out.rank);
        layout_broadcast(out_index, out.shape, out.rank, in.shape, in_index, in.rank);
        out.data[layout_position(out_index, out.stride, out.rank)] = in.data[layout_position(in_index, in.stride, in.rank)];
    }
}

void bias_act(const View& out, const View& input, const View& bias, int32_t act) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::bias_act(out, input, bias, act);
        return;
    }
    require_act(act);
    check_bias_out(out, input, bias);
    long long n = shape_numel(out);
    if (n == 0) return;

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t out_index[16] = {};
        int32_t in_index[16] = {};
        int32_t bias_index[16] = {};
        layout_index(i, out.shape, out_index, out.rank);
        layout_broadcast(out_index, out.shape, out.rank, input.shape, in_index, input.rank);
        layout_broadcast(out_index, out.shape, out.rank, bias.shape, bias_index, bias.rank);
        float pre = input.data[layout_position(in_index, input.stride, input.rank)] + bias.data[layout_position(bias_index, bias.stride, bias.rank)];
        float y = act == bias_relu ? (pre > 0.0f ? pre : 0.0f) : stable_sigmoid(pre);
        out.data[layout_position(out_index, out.stride, out.rank)] = y;
    }
}

void bias_act_backward(const View& grad_input, const View& grad_bias, const View& grad, const View& saved, const View& bias, int32_t act) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::bias_act_backward(grad_input, grad_bias, grad, saved, bias, act);
        return;
    }
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
    if (n == 0) {
        long long bias_n = shape_numel(grad_bias);
        for (long long i = 0; i < bias_n; i++) {
            int32_t index[16] = {};
            layout_index(i, grad_bias.shape, index, grad_bias.rank);
            grad_bias.data[layout_position(index, grad_bias.stride, grad_bias.rank)] = 0.0f;
        }
        return;
    }

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t out_index[16] = {};
        layout_index(i, grad.shape, out_index, grad.rank);
        float g = grad.data[layout_position(out_index, grad.stride, grad.rank)];
        if (act == bias_relu) {
            int32_t in_index[16] = {};
            int32_t bias_index[16] = {};
            layout_broadcast(out_index, grad.shape, grad.rank, saved.shape, in_index, saved.rank);
            layout_broadcast(out_index, grad.shape, grad.rank, bias.shape, bias_index, bias.rank);
            float pre = saved.data[layout_position(in_index, saved.stride, saved.rank)] + bias.data[layout_position(bias_index, bias.stride, bias.rank)];
            g = pre > 0.0f ? g : 0.0f;
        } else {
            float y = saved.data[layout_position(out_index, saved.stride, saved.rank)];
            g = g * y * (1.0f - y);
        }
        grad_input.data[layout_position(out_index, grad_input.stride, grad_input.rank)] = g;
    }

    SumAxes axes = sum_axes(grad_input, grad_bias);
    long long bias_n = shape_numel(grad_bias);
    #pragma omp parallel for if (bias_n >= THRESHOLD)
    for (long long b = 0; b < bias_n; b++) {
        int32_t bias_index[16] = {};
        int32_t out_index[16] = {};
        layout_index(b, grad_bias.shape, bias_index, grad_bias.rank);
        int32_t diff = grad_input.rank - grad_bias.rank;
        for (int32_t i = 0; i < grad_bias.rank; i++) {
            if (grad_bias.shape[i] != 1)
                out_index[i + diff] = bias_index[i];
        }
        float acc = 0.0f;
        for (long long v = 0; v < axes.volume; v++) {
            long long t = v;
            for (int32_t k = axes.count - 1; k >= 0; k--) {
                out_index[axes.axes[k]] = static_cast<int32_t>(t % axes.sizes[k]);
                t /= axes.sizes[k];
            }
            acc += grad_input.data[layout_position(out_index, grad_input.stride, grad_input.rank)];
        }
        grad_bias.data[layout_position(bias_index, grad_bias.stride, grad_bias.rank)] = acc;
    }
}

void layer_norm(const View& out, const View& mean, const View& rstd, const View& input, const View& weight, const View& bias, float eps) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::layer_norm(out, mean, rstd, input, weight, bias, eps);
        return;
    }
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

    const int32_t step = input.stride[dim];
    const int32_t out_step = out.stride[dim];
    const int32_t w_step = weight.stride[0];
    const int32_t b_step = bias.stride[0];

    #pragma omp parallel for if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, input.shape, input.rank, dim, split.inner, index);
        long long base = layout_position(index, input.stride, input.rank);
        long long out_base = layout_position(index, out.stride, out.rank);
        float sum = 0.0f;
        for (int32_t j = 0; j < n; j++)
            sum += input.data[base + static_cast<long long>(j) * step];
        float m = sum / static_cast<float>(n);
        float var = 0.0f;
        for (int32_t j = 0; j < n; j++) {
            float d = input.data[base + static_cast<long long>(j) * step] - m;
            var += d * d;
        }
        var /= static_cast<float>(n);
        float rs = 1.0f / std::sqrt(var + eps);
        mean.data[row * mean.stride[0]] = m;
        rstd.data[row * rstd.stride[0]] = rs;
        for (int32_t j = 0; j < n; j++) {
            float xhat = (input.data[base + static_cast<long long>(j) * step] - m) * rs;
            out.data[out_base + static_cast<long long>(j) * out_step] = xhat * weight.data[static_cast<long long>(j) * w_step] + bias.data[static_cast<long long>(j) * b_step];
        }
    }
}

void layer_norm_backward(const View& grad_input, const View& grad_weight, const View& grad_bias, const View& grad, const View& input, const View& mean, const View& rstd, const View& weight) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::layer_norm_backward(grad_input, grad_weight, grad_bias, grad, input, mean, rstd, weight);
        return;
    }
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

    const int32_t step = input.stride[dim];
    const int32_t g_step = grad.stride[dim];
    const int32_t o_step = grad_input.stride[dim];
    const int32_t w_step = weight.stride[0];

    #pragma omp parallel for if (split.rows >= THRESHOLD)
    for (long long row = 0; row < split.rows; row++) {
        int32_t index[16] = {};
        layout_row_index(row, input.shape, input.rank, dim, split.inner, index);
        long long base = layout_position(index, input.stride, input.rank);
        long long g_base = layout_position(index, grad.stride, grad.rank);
        long long o_base = layout_position(index, grad_input.stride, grad_input.rank);
        float m = mean.data[row * mean.stride[0]];
        float rs = rstd.data[row * rstd.stride[0]];
        float sum_g = 0.0f;
        float sum_gx = 0.0f;
        for (int32_t j = 0; j < n; j++) {
            float xhat = (input.data[base + static_cast<long long>(j) * step] - m) * rs;
            float g = grad.data[g_base + static_cast<long long>(j) * g_step] * weight.data[static_cast<long long>(j) * w_step];
            sum_g += g;
            sum_gx += g * xhat;
        }
        float inv_n = 1.0f / static_cast<float>(n);
        float mean_g = sum_g * inv_n;
        float mean_gx = sum_gx * inv_n;
        for (int32_t j = 0; j < n; j++) {
            float xhat = (input.data[base + static_cast<long long>(j) * step] - m) * rs;
            float g = grad.data[g_base + static_cast<long long>(j) * g_step] * weight.data[static_cast<long long>(j) * w_step];
            grad_input.data[o_base + static_cast<long long>(j) * o_step] = rs * (g - mean_g - xhat * mean_gx);
        }
    }

    #pragma omp parallel for if (n >= THRESHOLD)
    for (int32_t j = 0; j < n; j++) {
        float gw = 0.0f;
        float gb = 0.0f;
        for (long long row = 0; row < split.rows; row++) {
            int32_t index[16] = {};
            layout_row_index(row, input.shape, input.rank, dim, split.inner, index);
            index[dim] = j;
            long long pos = layout_position(index, input.stride, input.rank);
            long long gpos = layout_position(index, grad.stride, grad.rank);
            float xhat = (input.data[pos] - mean.data[row * mean.stride[0]]) * rstd.data[row * rstd.stride[0]];
            float dy = grad.data[gpos];
            gw += dy * xhat;
            gb += dy;
        }
        grad_weight.data[static_cast<long long>(j) * grad_weight.stride[0]] = gw;
        grad_bias.data[static_cast<long long>(j) * grad_bias.stride[0]] = gb;
    }
}

void max_pool2d(const View& out, const View& input, int32_t kh, int32_t kw) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::max_pool2d(out, input, kh, kw);
        return;
    }
    prepare_pool(input, out, kh, kw, "max_pool2d");
    long long n = shape_numel(out);
    if (n == 0) return;

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t index[4] = {};
        layout_index(i, out.shape, index, 4);
        float m = -std::numeric_limits<float>::infinity();
        for (int32_t ih = 0; ih < kh; ih++) {
            for (int32_t iw = 0; iw < kw; iw++) {
                int32_t in_index[4] = {index[0], index[1], index[2] * kh + ih, index[3] * kw + iw};
                m = std::max(m, input.data[layout_position(in_index, input.stride, 4)]);
            }
        }
        out.data[layout_position(index, out.stride, 4)] = m;
    }
}

void max_pool2d_backward(const View& grad_input, const View& grad, const View& input, const View& output, int32_t kh, int32_t kw) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::max_pool2d_backward(grad_input, grad, input, output, kh, kw);
        return;
    }
    prepare_pool(input, output, kh, kw, "max_pool2d_backward");
    require_same(grad, output, "max_pool2d_backward");
    if (!same_shape(grad_input, input))
        throw std::runtime_error("max_pool2d_backward input gradient shape must match the input");
    long long n = shape_numel(input);
    if (n == 0) return;

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t in_index[4] = {};
        layout_index(i, input.shape, in_index, 4);
        int32_t out_index[4] = {in_index[0], in_index[1], in_index[2] / kh, in_index[3] / kw};
        float value = input.data[layout_position(in_index, input.stride, 4)];
        float pooled = output.data[layout_position(out_index, output.stride, 4)];
        float g = grad.data[layout_position(out_index, grad.stride, 4)];
        grad_input.data[layout_position(in_index, grad_input.stride, 4)] = value == pooled ? g : 0.0f;
    }
}

void avg_pool2d(const View& out, const View& input, int32_t kh, int32_t kw) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::avg_pool2d(out, input, kh, kw);
        return;
    }
    prepare_pool(input, out, kh, kw, "avg_pool2d");
    long long n = shape_numel(out);
    if (n == 0) return;
    float inv = 1.0f / static_cast<float>(static_cast<long long>(kh) * static_cast<long long>(kw));

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t index[4] = {};
        layout_index(i, out.shape, index, 4);
        float sum = 0.0f;
        for (int32_t ih = 0; ih < kh; ih++) {
            for (int32_t iw = 0; iw < kw; iw++) {
                int32_t in_index[4] = {index[0], index[1], index[2] * kh + ih, index[3] * kw + iw};
                sum += input.data[layout_position(in_index, input.stride, 4)];
            }
        }
        out.data[layout_position(index, out.stride, 4)] = sum * inv;
    }
}

void avg_pool2d_backward(const View& grad_input, const View& grad, int32_t kh, int32_t kw) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::avg_pool2d_backward(grad_input, grad, kh, kw);
        return;
    }
    prepare_pool(grad_input, grad, kh, kw, "avg_pool2d_backward");
    long long n = shape_numel(grad_input);
    if (n == 0) return;
    float inv = 1.0f / static_cast<float>(static_cast<long long>(kh) * static_cast<long long>(kw));

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        int32_t in_index[4] = {};
        layout_index(i, grad_input.shape, in_index, 4);
        int32_t out_index[4] = {in_index[0], in_index[1], in_index[2] / kh, in_index[3] / kw};
        grad_input.data[layout_position(in_index, grad_input.stride, 4)] = grad.data[layout_position(out_index, grad.stride, 4)] * inv;
    }
}

} // namespace tsgrad
