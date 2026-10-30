#include "tsgrad/storage.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if TSGRAD_HAS_CUDA
#include "tsgrad/ops.h"
#include <cuda_runtime.h>
#endif

namespace tsgrad {
namespace {

#if TSGRAD_HAS_CUDA
void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}
#endif

} // namespace

Storage::Storage(size_t numel, float fill)
    : numel_(numel)
#if TSGRAD_HAS_CUDA
    , device_(nullptr)
#else
    , host_(numel, fill)
#endif
{
#if TSGRAD_HAS_CUDA
    if (numel_ == 0) return;

    const size_t bytes = numel_ * sizeof(float);
    check_cuda(cudaMalloc(&device_, bytes), "cudaMalloc");
    cudaError_t err;
    if (fill == 0.0F) {
        err = cudaMemset(device_, 0, bytes);
    } else {
        std::vector<float> tmp(numel_, fill);
        err = cudaMemcpy(device_, tmp.data(), bytes, cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        cudaFree(device_);
        device_ = nullptr;
        check_cuda(err, "storage fill");
    }
#endif
}

Storage::~Storage() {
#if TSGRAD_HAS_CUDA
    if (device_ != nullptr) cudaFree(device_);
#endif
}

float* Storage::data() {
#if TSGRAD_HAS_CUDA
    return device_;
#else
    return host_.data();
#endif
}

void Storage::copy_to_host(float* dst) const {
    if (numel_ == 0) return;
#if TSGRAD_HAS_CUDA
    cuda::Context::get().synchronize();
    check_cuda(cudaMemcpy(dst, device_, numel_ * sizeof(float), cudaMemcpyDeviceToHost), "copyToHost");
#else
    std::memcpy(dst, host_.data(), numel_ * sizeof(float));
#endif
}

void Storage::copy_from_host(const float* src) {
    if (numel_ == 0) return;
#if TSGRAD_HAS_CUDA
    check_cuda(cudaMemcpy(device_, src, numel_ * sizeof(float), cudaMemcpyHostToDevice), "copyFromHost");
#else
    std::memcpy(host_.data(), src, numel_ * sizeof(float));
#endif
}

} // namespace tsgrad
