#include "tsgrad/ops.h"
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <climits>


// A CUDA build defines TSGRAD_HAS_CUDA. The handoff, later will be a
// call with the same Views, for example tsgrad::cuda::zip_add(out, a, b).
// That call copies host memory to the device and back. See src/cuda/elementwise.cu.


namespace tsgrad {


int64_t index_to_position(const int32_t* index, const int32_t* stride, int32_t size){
    int64_t res = 0;
    for (int32_t i = 0; i < size; i++)
        res += index[i] * stride[i];
    
    return res;
}

void position_to_index(int32_t ordinal, const int32_t* shape, int32_t* out_index, int32_t size){
    for (int32_t i = size - 1; i >= 0; i--){
        out_index[i] = ordinal % shape[i];
        ordinal /= shape[i];
    }
}

// writes to small_index, reads from big_index
void broadcast_index(const int32_t* big_index, const int32_t*  big_shape, int32_t big_size, const int32_t*  small_shape, int32_t* small_index, int32_t small_size){
    if (big_size < small_size) throw std::runtime_error("big_size cannot be less than small_size");

    int32_t diff = big_size - small_size;

    for (int32_t i = 0; i < small_size; i++){
        if (small_shape[i] == 1)
            small_index[i] = 0;
        else
            small_index[i] = big_index[i + diff];
    }
}

u_int64_t prod(const int32_t* shape, int32_t size){
    u_int64_t res = 1;
    for (int32_t i = 0; i < size; i++){
        if (shape[i] < 0) throw std::runtime_error("Cannot take product, negative dimension in shape");
        if (shape[i] == 0) return 0;
        if (res > LLONG_MAX  / shape[i]) throw std::runtime_error("Overflow, shape must be less than int 64 max");
        res *= static_cast<uint64_t>(shape[i]);
    }

    return res;
}

bool is_contiguous(const View& a){
    int64_t expected_stride = 1;

    for (int32_t i = a.rank - 1; i >= 0; i--) {
        if (a.shape[i] > 1) {
            if (static_cast<int64_t>(a.stride[i]) != expected_stride) {
                return false;
            }
        }

        expected_stride *= static_cast<int64_t>(a.shape[i]);
    }

    return true;
}

bool shapesAreEqual(const int32_t* a, const int32_t* b, int32_t a_rank, int32_t b_rank){
    if (a_rank != b_rank) return false;
    for (int32_t i = 0; i < a_rank; i++)
        if (a[i] != b[i]) return false;

    return true;
}

template <typename Func>
void map_op(Func operation, const View& out, const View& in){
    auto size = prod(out.shape, out.rank);

    if (is_contiguous(out) && is_contiguous(in) && shapesAreEqual(out.shape, in.shape, out.rank, in.rank)){
        #pragma omp parallel for
        for (int64_t i = 0; i < size; i++)
            out.data[i] = operation(in.data[i]);

        return;
    }

    #pragma omp parallel
    {
        std::vector<int32_t> out_index(out.rank, 0);
        std::vector<int32_t> in_index(in.rank, 0);
        
        #pragma omp for
        for (u_int64_t i = 0; i < size; i++){
            position_to_index(i, out.shape, out_index.data(), out.rank);
            auto outPos = index_to_position(out_index.data(), out.stride, out.rank);

            broadcast_index(out_index.data(), out.shape, out.rank, in.shape, in_index.data(), in.rank);
            auto inPos = index_to_position(in_index.data(), in.stride, in.rank);

            out.data[outPos] = operation(in.data[inPos]);
        }
    }
}


void map_neg(const View& out, const View& in) {
    map_op([](float a) -> float { return -a; }, out, in);
}

void map_inv(const View& out, const View& in) {
    map_op([](float a) -> float { return 1.0F / a; }, out, in);
}

void map_sigmoid(const View& out, const View& in) {
    map_op([](float a) -> float {
        if (a >= 0)
            return 1.0F/(1.0F + std::exp(-a));  // e^a
        else
            return std::exp(a) / (1.0F + std::exp(a));
    }, out, in);
}

void map_relu(const View& out, const View& in) {
    map_op([](float a) -> float { return std::max(0.0F, a); }, out, in);
}

void map_log(const View& out, const View& in) {
    map_op([](float a) -> float { if (a <= 0) throw std::runtime_error("log received value less than or equal to 0"); return std::log(a); }, out, in);
}

void map_exp(const View& out, const View& in) {
    map_op([](float a) -> float { return std::exp(a); }, out, in);
}

void map_id(const View& out, const View& in) {
    map_op([](float a) -> float { return a; }, out, in);
}

void zip_add(const View& out, const View& a, const View& b) {
    (void)out;
    (void)a;
    (void)b;
}

void zip_mul(const View& out, const View& a, const View& b) {
    (void)out;
    (void)a;
    (void)b;
}

void zip_lt(const View& out, const View& a, const View& b) {
    (void)out;
    (void)a;
    (void)b;
}

void zip_eq(const View& out, const View& a, const View& b) {
    (void)out;
    (void)a;
    (void)b;
}

void zip_is_close(const View& out, const View& a, const View& b) {
    (void)out;
    (void)a;
    (void)b;
}

} // namespace tsgrad
