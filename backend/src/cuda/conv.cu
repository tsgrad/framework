#include "tsgrad/ops.h"

// tsgrad::cuda::conv1d / conv2d are the device side of tsgrad::conv1d and
// tsgrad::conv2d in src/cpu/conv.cpp. addon.cc does not call these directly.
//
// conv1d: out [batch, outChannels, outWidth], input [batch, inChannels, width],
// weight [outChannels, inChannels, kWidth].
// conv2d adds a height axis on input, weight, and out.
// reverse walks the kernel backward. Samples outside the input are 0.
// Strides still apply, because a permute() result shares its Float32Array.
//
// Sequence, same as elementwise.cu:
//   Context::get()
//   copy input, weight, and the output buffer to the device
//   conv*_kernel<<<grid, block, 0, stream>>>
//   copy the output buffer back
//   Context::synchronize()

namespace tsgrad::cuda {

__global__ void conv1d_kernel(
    float* out, int32_t out_stride_b, int32_t out_stride_c, int32_t out_stride_w,
    const float* input, int32_t in_stride_b, int32_t in_stride_c, int32_t in_stride_w,
    const float* weight, int32_t weight_stride_oc, int32_t weight_stride_ic, int32_t weight_stride_k,
    int32_t batch, int32_t out_channels, int32_t out_width,
    int32_t in_channels, int32_t width, int32_t k_width,
    bool reverse) {
    // int32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    // one thread per output element, then the same walk as conv1d
    (void)out; (void)out_stride_b; (void)out_stride_c; (void)out_stride_w;
    (void)input; (void)in_stride_b; (void)in_stride_c; (void)in_stride_w;
    (void)weight; (void)weight_stride_oc; (void)weight_stride_ic; (void)weight_stride_k;
    (void)batch; (void)out_channels; (void)out_width;
    (void)in_channels; (void)width; (void)k_width;
    (void)reverse;
}

__global__ void conv2d_kernel(
    float* out, int32_t out_stride_b, int32_t out_stride_c, int32_t out_stride_h, int32_t out_stride_w,
    const float* input, int32_t in_stride_b, int32_t in_stride_c, int32_t in_stride_h, int32_t in_stride_w,
    const float* weight, int32_t weight_stride_oc, int32_t weight_stride_ic, int32_t weight_stride_kh, int32_t weight_stride_kw,
    int32_t batch, int32_t out_channels, int32_t out_height, int32_t out_width,
    int32_t in_channels, int32_t height, int32_t width, int32_t k_height, int32_t k_width,
    bool reverse) {
    // one thread per output element, then the same walk as conv2d
    (void)out; (void)out_stride_b; (void)out_stride_c; (void)out_stride_h; (void)out_stride_w;
    (void)input; (void)in_stride_b; (void)in_stride_c; (void)in_stride_h; (void)in_stride_w;
    (void)weight; (void)weight_stride_oc; (void)weight_stride_ic; (void)weight_stride_kh; (void)weight_stride_kw;
    (void)batch; (void)out_channels; (void)out_height; (void)out_width;
    (void)in_channels; (void)height; (void)width; (void)k_height; (void)k_width;
    (void)reverse;
}

void conv1d(const View& out, const View& input, const View& weight, bool reverse) {
    // Context& ctx = Context::get();
    // conv1d_kernel<<<grid, block, 0, stream>>>(d_out, ..., reverse);
    // ctx.synchronize();
    (void)out;
    (void)input;
    (void)weight;
    (void)reverse;
}

void conv2d(const View& out, const View& input, const View& weight, bool reverse) {
    // Context& ctx = Context::get();
    // conv2d_kernel<<<grid, block, 0, stream>>>(d_out, ..., reverse);
    // ctx.synchronize();
    (void)out;
    (void)input;
    (void)weight;
    (void)reverse;
}

} // namespace tsgrad::cuda
