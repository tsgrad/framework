#include "tsgrad/ops.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// One process-wide device, stream, and cublas handle. Launches queue on the
// stream and return. copyToHost and synchronize wait, because that is when a
// host read or an explicit flush must see finished work. The cublas handle is
// bound to that stream.

namespace tsgrad::cuda {
namespace {

void check_cublas(cublasStatus_t err, const char* what) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cublasGetStatusString(err));
    }
}

}

Context::Context() : device_(0), stream_(nullptr), cublas_(nullptr) {
    cudaGetDevice(&device_);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    stream_ = stream;

    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "cublasCreate");
    cublas_ = handle;
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
}

Context::~Context() {
    if (cublas_ != nullptr) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_));
    }
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

void* Context::getCublas(){
    return cublas_;
}

void Context::synchronize() {
    cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(err));
    err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err));
}

void Context::checkLastError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA launch: ") + cudaGetErrorString(err));
}

int Context::device() const {
    return device_;
}

} // namespace tsgrad::cuda
