#include "tsgrad/storage.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

struct BlockPool {
    bool alive = true;
#if TSGRAD_HAS_CUDA
    std::unordered_map<size_t, std::vector<float*>> device;
#else
    std::unordered_map<size_t, std::vector<std::vector<float>>> host;
#endif

    ~BlockPool() {
        alive = false;
#if TSGRAD_HAS_CUDA
        for (auto& entry : device) {
            for (float* ptr : entry.second) {
                if (ptr != nullptr) cudaFree(ptr);
            }
        }
        device.clear();
#endif
    }
};

BlockPool& pool() {
    static BlockPool instance;
    return instance;
}

#if TSGRAD_HAS_CUDA
float* take_device(size_t bytes) {
    BlockPool& blocks = pool();
    if (!blocks.alive) return nullptr;
    auto found = blocks.device.find(bytes);
    if (found == blocks.device.end() || found->second.empty()) return nullptr;
    float* ptr = found->second.back();
    found->second.pop_back();
    return ptr;
}

void give_device(float* ptr, size_t bytes) {
    if (ptr == nullptr) return;
    BlockPool& blocks = pool();
    if (!blocks.alive) {
        cudaFree(ptr);
        return;
    }
    blocks.device[bytes].push_back(ptr);
}

void write_device(float* device, size_t numel, float fill) {
    const size_t bytes = numel * sizeof(float);
    cuda::Context& ctx = cuda::Context::get();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.getStream());
    if (fill == 0.0F) {
        check_cuda(cudaMemsetAsync(device, 0, bytes, stream), "cudaMemsetAsync");
    } else {
        cuda::fill(device, numel, fill);
    }
}
#else
std::vector<float> take_host(size_t numel) {
    BlockPool& blocks = pool();
    if (!blocks.alive) return {};
    auto found = blocks.host.find(numel);
    if (found == blocks.host.end() || found->second.empty()) return {};
    std::vector<float> values = std::move(found->second.back());
    found->second.pop_back();
    return values;
}

void give_host(std::vector<float>&& values) {
    if (values.empty()) return;
    BlockPool& blocks = pool();
    if (!blocks.alive) return;
    const size_t numel = values.size();
    blocks.host[numel].push_back(std::move(values));
}
#endif

} // namespace

Storage::Storage(size_t numel, bool initialize, float fill)
    : numel_(numel)
#if TSGRAD_HAS_CUDA
    , device_(nullptr)
#endif
{
#if TSGRAD_HAS_CUDA
    if (numel_ == 0) return;

    const size_t bytes = numel_ * sizeof(float);
    device_ = take_device(bytes);
    if (device_ == nullptr) {
        check_cuda(cudaMalloc(&device_, bytes), "cudaMalloc");
    }
    if (initialize) {
        try {
            write_device(device_, numel_, fill);
        } catch (...) {
            cudaFree(device_);
            device_ = nullptr;
            throw;
        }
    }
#else
    if (numel_ == 0) return;

    host_ = take_host(numel_);
    if (host_.size() != numel_) host_.assign(numel_, 0.0F);
    if (initialize) std::fill(host_.begin(), host_.end(), fill);
#endif
}

Storage::~Storage() {
#if TSGRAD_HAS_CUDA
    if (device_ == nullptr) return;
    float* ptr = device_;
    device_ = nullptr;
    give_device(ptr, numel_ * sizeof(float));
#else
    if (!host_.empty()) give_host(std::move(host_));
#endif
}

float* Storage::data() {
#if TSGRAD_HAS_CUDA
    return device_;
#else
    return host_.empty() ? nullptr : host_.data();
#endif
}

void Storage::fill(float value) {
    if (numel_ == 0) return;
#if TSGRAD_HAS_CUDA
    if (device_ == nullptr) throw std::runtime_error("fill on released storage");
    write_device(device_, numel_, value);
#else
    if (host_.size() != numel_) throw std::runtime_error("fill on released storage");
    std::fill(host_.begin(), host_.end(), value);
#endif
}

void Storage::copy_to_host(float* dst) const {
    if (numel_ == 0) return;
#if TSGRAD_HAS_CUDA
    if (device_ == nullptr) throw std::runtime_error("copyToHost on released storage");
    cuda::Context::get().synchronize();
    check_cuda(cudaMemcpy(dst, device_, numel_ * sizeof(float), cudaMemcpyDeviceToHost), "copyToHost");
#else
    if (host_.size() != numel_) throw std::runtime_error("copyToHost on released storage");
    std::memcpy(dst, host_.data(), numel_ * sizeof(float));
#endif
}

void Storage::copy_from_host(const float* src) {
    if (numel_ == 0) return;
#if TSGRAD_HAS_CUDA
    if (device_ == nullptr) throw std::runtime_error("copyFromHost on released storage");
    cuda::Context& ctx = cuda::Context::get();
    check_cuda(cudaMemcpyAsync(device_, src, numel_ * sizeof(float), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(ctx.getStream())), "copyFromHost");
#else
    if (host_.size() != numel_) throw std::runtime_error("copyFromHost on released storage");
    std::memcpy(host_.data(), src, numel_ * sizeof(float));
#endif
}

} // namespace tsgrad
