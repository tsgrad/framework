#include "tsgrad/ops.h"
#include <cuda_runtime.h>

// One process-wide device and stream. elementwise.cu and matmul.cu call
// Context::get() before a launch and Context::synchronize() before returning
// to addon.cc, because the Float32Array must be filled when the JS call ends.

namespace tsgrad::cuda {

Context::Context() : device_(0), stream_(nullptr) {
    cudaGetDevice(&device_);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    stream_ = stream;
}

Context::~Context() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
}

Context& Context::get() {
    static Context ctx;
    return ctx;
}

void* Context::getStream(){
    return stream_;
}

void Context::synchronize() {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
}

int Context::device() const {
    return device_;
}

} // namespace tsgrad::cuda
