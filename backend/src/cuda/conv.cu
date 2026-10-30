#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>
#include <span>

// tsgrad::cuda::conv1d / conv2d are the device side of tsgrad::conv1d and
// tsgrad::conv2d in src/cpu/conv.cpp. addon.cc does not call these directly.
//
// conv1d: out [batch, outChannels, outWidth], input [batch, inChannels, width],
// weight [outChannels, inChannels, kWidth].
// conv2d adds a height axis on input, weight, and out.
// reverse walks the kernel backward. Samples outside the input are 0.
// Strides still apply, because a permute() result shares its Storage.
//
// Sequence, same as elementwise.cu:
//   Context::get()
//   conv*_kernel<<<grid, block, 0, stream>>> on View::data
//   Context::synchronize()

namespace tsgrad::cuda {

__global__ void conv1d_kernel(
    float* out, int32_t out_stride_b, int32_t out_stride_c, int32_t out_stride_w,
    const float* input, int32_t in_stride_b, int32_t in_stride_c, int32_t in_stride_w,
    const float* weight, int32_t weight_stride_oc, int32_t weight_stride_ic, int32_t weight_stride_k,
    int32_t batch, int32_t out_channels, int32_t out_width,
    int32_t in_channels, int32_t width, int32_t k_width,
    bool reverse) {
    const long long ordinal = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long plane = static_cast<long long>(out_channels) * static_cast<long long>(out_width);
    const long long out_size = static_cast<long long>(batch) * plane;
    if (ordinal >= out_size) return;

    const int32_t b = static_cast<int32_t>(ordinal / plane);
    const long long rem = ordinal % plane;
    const int32_t oc = static_cast<int32_t>(rem / out_width);
    const int32_t w = static_cast<int32_t>(rem % out_width);

    const int32_t in_b = b * in_stride_b;
    const int32_t weight_oc = oc * weight_stride_oc;
    float total = 0;

    for (int32_t ic = 0; ic < in_channels; ic++) {
        const int32_t in_ic = in_b + ic * in_stride_c;
        const int32_t weight_ic = weight_oc + ic * weight_stride_ic;
        for (int32_t k = 0; k < k_width; k++) {
            const int32_t x = reverse ? w - k : w + k;
            if (x >= 0 && x < width)
                total += input[in_ic + x * in_stride_w] * weight[weight_ic + k * weight_stride_k];
        }
    }

    out[b * out_stride_b + oc * out_stride_c + w * out_stride_w] = total;
}

__global__ void conv2d_kernel(
    float* out, int32_t out_stride_b, int32_t out_stride_c, int32_t out_stride_h, int32_t out_stride_w,
    const float* input, int32_t in_stride_b, int32_t in_stride_c, int32_t in_stride_h, int32_t in_stride_w,
    const float* weight, int32_t weight_stride_oc, int32_t weight_stride_ic, int32_t weight_stride_kh, int32_t weight_stride_kw,
    int32_t batch, int32_t out_channels, int32_t out_height, int32_t out_width,
    int32_t in_channels, int32_t height, int32_t width, int32_t k_height, int32_t k_width,
    bool reverse) {
    const long long ordinal = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long spatial = static_cast<long long>(out_height) * static_cast<long long>(out_width);
    const long long plane = static_cast<long long>(out_channels) * spatial;
    const long long out_size = static_cast<long long>(batch) * plane;
    if (ordinal >= out_size) return;

    const int32_t b = static_cast<int32_t>(ordinal / plane);
    const long long rem = ordinal % plane;
    const int32_t oc = static_cast<int32_t>(rem / spatial);
    const long long at = rem % spatial;
    const int32_t h = static_cast<int32_t>(at / out_width);
    const int32_t w = static_cast<int32_t>(at % out_width);

    const int32_t in_b = b * in_stride_b;
    const int32_t weight_oc = oc * weight_stride_oc;
    float total = 0;

    for (int32_t ic = 0; ic < in_channels; ic++) {
        const int32_t in_ic = in_b + ic * in_stride_c;
        const int32_t weight_ic = weight_oc + ic * weight_stride_ic;
        for (int32_t kh = 0; kh < k_height; kh++) {
            const int32_t xh = reverse ? h - kh : h + kh;
            if (xh < 0 || xh >= height) continue;
            const int32_t in_h = in_ic + xh * in_stride_h;
            const int32_t weight_h = weight_ic + kh * weight_stride_kh;
            for (int32_t kw = 0; kw < k_width; kw++) {
                const int32_t xw = reverse ? w - kw : w + kw;
                if (xw >= 0 && xw < width)
                    total += input[in_h + xw * in_stride_w] * weight[weight_h + kw * weight_stride_kw];
            }
        }
    }

    out[b * out_stride_b + oc * out_stride_c + h * out_stride_h + w * out_stride_w] = total;
}

void conv1d(const View& out, const View& input, const View& weight, bool reverse) {
    if (input.rank != 3 || weight.rank != 3 || out.rank != 3)
        throw std::runtime_error("Shape length must be 3 for conv1d");

    const int32_t batch = input.shape[0];
    const int32_t in_channels = input.shape[1];
    const int32_t width = input.shape[2];
    const int32_t out_channels = weight.shape[0];
    const int32_t k_width = weight.shape[2];
    const int32_t out_width = out.shape[2];

    if (in_channels != weight.shape[1] || out.shape[0] != batch || out.shape[1] != out_channels) {
        std::span<const int32_t> in_view{input.shape, static_cast<size_t>(input.rank)};
        std::span<const int32_t> weight_view{weight.shape, static_cast<size_t>(weight.rank)};
        std::span<const int32_t> out_view{out.shape, static_cast<size_t>(out.rank)};
        throw std::runtime_error(std::format("Incompatible shapes for conv1d, input: {:n}, weight: {:n}, out: {:n}", in_view, weight_view, out_view));
    }

    const long long out_size = static_cast<long long>(batch) * static_cast<long long>(out_channels) * static_cast<long long>(out_width);
    if (out_size == 0) return;

    Context& ctx = Context::get();

    const int32_t threads = 256;
    const int32_t blocks = static_cast<int32_t>((out_size + threads - 1) / threads);

    conv1d_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, out.stride[0], out.stride[1], out.stride[2],
        input.data, input.stride[0], input.stride[1], input.stride[2],
        weight.data, weight.stride[0], weight.stride[1], weight.stride[2],
        batch, out_channels, out_width,
        in_channels, width, k_width,
        reverse
    );

    ctx.synchronize();
}

void conv2d(const View& out, const View& input, const View& weight, bool reverse) {
    if (input.rank != 4 || weight.rank != 4 || out.rank != 4)
        throw std::runtime_error("Shape length must be 4 for conv2d");

    const int32_t batch = input.shape[0];
    const int32_t in_channels = input.shape[1];
    const int32_t height = input.shape[2];
    const int32_t width = input.shape[3];
    const int32_t out_channels = weight.shape[0];
    const int32_t k_height = weight.shape[2];
    const int32_t k_width = weight.shape[3];
    const int32_t out_height = out.shape[2];
    const int32_t out_width = out.shape[3];

    if (in_channels != weight.shape[1] || out.shape[0] != batch || out.shape[1] != out_channels) {
        std::span<const int32_t> in_view{input.shape, static_cast<size_t>(input.rank)};
        std::span<const int32_t> weight_view{weight.shape, static_cast<size_t>(weight.rank)};
        std::span<const int32_t> out_view{out.shape, static_cast<size_t>(out.rank)};
        throw std::runtime_error(std::format("Incompatible shapes for conv2d, input: {:n}, weight: {:n}, out: {:n}", in_view, weight_view, out_view));
    }

    const long long spatial = static_cast<long long>(out_height) * static_cast<long long>(out_width);
    const long long plane = static_cast<long long>(out_channels) * spatial;
    const long long out_size = static_cast<long long>(batch) * plane;
    if (out_size == 0) return;

    Context& ctx = Context::get();

    const int32_t threads = 256;
    const int32_t blocks = static_cast<int32_t>((out_size + threads - 1) / threads);

    conv2d_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, out.stride[0], out.stride[1], out.stride[2], out.stride[3],
        input.data, input.stride[0], input.stride[1], input.stride[2], input.stride[3],
        weight.data, weight.stride[0], weight.stride[1], weight.stride[2], weight.stride[3],
        batch, out_channels, out_height, out_width,
        in_channels, height, width, k_height, k_width,
        reverse
    );

    ctx.synchronize();
}

} // namespace tsgrad::cuda
