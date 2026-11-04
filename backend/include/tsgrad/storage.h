#pragma once

#ifndef TSGRAD_HAS_CUDA
#define TSGRAD_HAS_CUDA 0
#endif

#include <cstddef>
#include <cstdint>

#if !TSGRAD_HAS_CUDA
#include <vector>
#endif

// Owned float buffer. A CPU build keeps a std::vector. A CUDA build keeps one
// device allocation. View::data points at data() for the duration of an op
// and must not be freed by that op.
//
// Released blocks go back to a free list. cudaFree runs when that list is
// torn down. A recycled block is left dirty unless initialize is set.
// The next memset, fill, copy, or kernel is queued on the same stream, so it
// runs after whatever still uses that pointer.
namespace tsgrad {

class Storage {
public:
    Storage(size_t numel, bool initialize, float fill);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    size_t numel() const { return numel_; }
    float* data();

    void fill(float value);
    void copy_to_host(float* dst) const;
    void copy_from_host(const float* src);

private:
    size_t numel_;
#if TSGRAD_HAS_CUDA
    float* device_;
#else
    std::vector<float> host_;
#endif
};

} // namespace tsgrad
