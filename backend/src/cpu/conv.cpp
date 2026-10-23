#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <format>
#include <span>
#include <ranges>

// tensor_functions.conv1d / conv2d call these.
// conv1d: out [batch, outChannels, outWidth] from
//   input [batch, inChannels, width] and weight [outChannels, inChannels, kWidth]
// conv2d: out [batch, outChannels, outHeight, outWidth] from
//   input [batch, inChannels, height, width] and
//   weight [outChannels, inChannels, kHeight, kWidth]
// reverse walks the kernel backward. Samples outside the input are 0.
// Strides are not necessarily contiguous, because permute shares storage.
//
// CUDA handoff: tsgrad::cuda::conv1d / conv2d, defined in src/cuda/conv.cu.

namespace tsgrad {

void conv1d(const View& out, const View& input, const View& weight, bool reverse) {
    if (input.rank != 3 || weight.rank != 3 || out.rank != 3)
        throw std::runtime_error("Shape length must be 3 for conv1d");

    const int32_t batch = input.shape[0];
    const int32_t in_channels = input.shape[1];
    const int32_t width = input.shape[2];
    const int32_t out_channels = weight.shape[0];
    const int32_t k_width = weight.shape[2];
    const int32_t out_width = out.shape[2];

    if (in_channels != weight.shape[1] || out.shape[0] != batch || out.shape[1] != out_channels){
        std::span<const int32_t> in_view{input.shape, static_cast<size_t>(input.rank)};
        std::span<const int32_t> weight_view{weight.shape, static_cast<size_t>(weight.rank)};
        std::span<const int32_t> out_view{out.shape, static_cast<size_t>(out.rank)};
        throw std::runtime_error(std::format("Incompatible shapes for conv1d, input: {:n}, weight: {:n}, out: {:n}", in_view, weight_view, out_view));
    }

    const long long out_size = static_cast<long long>(batch) * static_cast<long long>(out_channels) * static_cast<long long>(out_width);
    const long long work = out_size * static_cast<long long>(in_channels) * static_cast<long long>(k_width);
    const long long plane = static_cast<long long>(out_channels) * static_cast<long long>(out_width);

    #pragma omp parallel for if (work >= THRESHOLD)
    for (long long ordinal = 0; ordinal < out_size; ordinal++) {
        const int32_t b = static_cast<int32_t>(ordinal / plane);
        const long long rem = ordinal % plane;
        const int32_t oc = static_cast<int32_t>(rem / out_width);
        const int32_t w = static_cast<int32_t>(rem % out_width);

        const int32_t in_b = b * input.stride[0];
        const int32_t weight_oc = oc * weight.stride[0];
        float total = 0;

        for (int32_t ic = 0; ic < in_channels; ic++) {
            const int32_t in_ic = in_b + ic * input.stride[1];
            const int32_t weight_ic = weight_oc + ic * weight.stride[1];
            for (int32_t k = 0; k < k_width; k++) {
                const int32_t x = reverse ? w - k : w + k;
                if (x >= 0 && x < width)
                    total += input.data[in_ic + x * input.stride[2]] * weight.data[weight_ic + k * weight.stride[2]];
            }
        }

        out.data[b * out.stride[0] + oc * out.stride[1] + w * out.stride[2]] = total;
    }
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

    if (in_channels != weight.shape[1] || out.shape[0] != batch || out.shape[1] != out_channels){
        std::span<const int32_t> in_view{input.shape, static_cast<size_t>(input.rank)};
        std::span<const int32_t> weight_view{weight.shape, static_cast<size_t>(weight.rank)};
        std::span<const int32_t> out_view{out.shape, static_cast<size_t>(out.rank)};
        throw std::runtime_error(std::format("Incompatible shapes for conv2d, input: {:n}, weight: {:n}, out: {:n}", in_view, weight_view, out_view));
    }

    const long long spatial = static_cast<long long>(out_height) * static_cast<long long>(out_width);
    const long long plane = static_cast<long long>(out_channels) * spatial;
    const long long out_size = static_cast<long long>(batch) * plane;
    const long long work = out_size * static_cast<long long>(in_channels) * static_cast<long long>(k_height) * static_cast<long long>(k_width);

    #pragma omp parallel for if (work >= THRESHOLD)
    for (long long ordinal = 0; ordinal < out_size; ordinal++) {
        const int32_t b = static_cast<int32_t>(ordinal / plane);
        const long long rem = ordinal % plane;
        const int32_t oc = static_cast<int32_t>(rem / spatial);
        const long long at = rem % spatial;
        const int32_t h = static_cast<int32_t>(at / out_width);
        const int32_t w = static_cast<int32_t>(at % out_width);

        const int32_t in_b = b * input.stride[0];
        const int32_t weight_oc = oc * weight.stride[0];
        float total = 0;

        for (int32_t ic = 0; ic < in_channels; ic++) {
            const int32_t in_ic = in_b + ic * input.stride[1];
            const int32_t weight_ic = weight_oc + ic * weight.stride[1];
            for (int32_t kh = 0; kh < k_height; kh++) {
                const int32_t xh = reverse ? h - kh : h + kh;
                if (xh < 0 || xh >= height) continue;
                const int32_t in_h = in_ic + xh * input.stride[2];
                const int32_t weight_h = weight_ic + kh * weight.stride[2];
                for (int32_t kw = 0; kw < k_width; kw++) {
                    const int32_t xw = reverse ? w - kw : w + kw;
                    if (xw >= 0 && xw < width)
                        total += input.data[in_h + xw * input.stride[3]] * weight.data[weight_h + kw * weight.stride[3]];
                }
            }
        }

        out.data[b * out.stride[0] + oc * out.stride[1] + h * out.stride[2] + w * out.stride[3]] = total;
    }
}

} // namespace tsgrad
