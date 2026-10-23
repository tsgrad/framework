#include "tsgrad/ops.h"
#include "helpers.hpp"
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

template <typename Func>
void map_op(Func operation, const View& out, const View& in){
    auto size = prod(out.shape, out.rank);

    if (is_contiguous(out) && is_contiguous(in) && shapesAreEqual(out.shape, in.shape, out.rank, in.rank)){
        #pragma omp parallel for
        for (long long i = 0; i < size; i++)
            out.data[i] = operation(in.data[i]);

        return;
    }

    #pragma omp parallel
    {
        std::vector<int32_t> out_index(out.rank, 0);
        std::vector<int32_t> in_index(in.rank, 0);
        
        #pragma omp for
        for (long long i = 0; i < size; i++){
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
    map_op([](float a) -> float { return std::log(a); }, out, in);
}

void map_exp(const View& out, const View& in) {
    map_op([](float a) -> float { return std::exp(a); }, out, in);
}

void map_id(const View& out, const View& in) {
    map_op([](float a) -> float { return a; }, out, in);
}

template <typename Func>
void zip_op(Func operation, const View& out, const View& a, const View& b){
    auto size = prod(out.shape, out.rank);

    if (is_contiguous(out) && is_contiguous(a) && is_contiguous(b) && shapesAreEqual(out.shape, a.shape, out.rank, a.rank) && shapesAreEqual(out.shape, b.shape, out.rank, b.rank)){
        #pragma omp parallel for
        for (long long i = 0; i < size; i++)
            out.data[i] = operation(a.data[i], b.data[i]);

        return;
    }

    if (!is_broadcastable(out.shape, a.shape, out.rank, a.rank) || !is_broadcastable(out.shape, b.shape, out.rank, b.rank))
        throw std::runtime_error("zip_op received unbroadcastable shapes");

    #pragma omp parallel
    {
        std::vector<int32_t> out_index(out.rank, 0);
        std::vector<int32_t> a_index(a.rank, 0);
        std::vector<int32_t> b_index(b.rank, 0);
        
        #pragma omp for
        for (long long i = 0; i < size; i++){
            position_to_index(i, out.shape, out_index.data(), out.rank);
            auto outPos = index_to_position(out_index.data(), out.stride, out.rank);

            broadcast_index(out_index.data(), out.shape, out.rank, a.shape, a_index.data(), a.rank);
            auto aPos = index_to_position(a_index.data(), a.stride, a.rank);

            broadcast_index(out_index.data(), out.shape, out.rank, b.shape, b_index.data(), b.rank);
            auto bPos = index_to_position(b_index.data(), b.stride, b.rank);

            out.data[outPos] = operation(a.data[aPos], b.data[bPos]);
        }
    }
}

void zip_add(const View& out, const View& a, const View& b) {
    zip_op([](float x, float y) -> float { return x + y; }, out, a, b);
}

void zip_mul(const View& out, const View& a, const View& b) {
    zip_op([](float x, float y) -> float { return x * y; }, out, a, b);
}

void zip_lt(const View& out, const View& a, const View& b) {
    zip_op([](float x, float y) -> float { return x < y; }, out, a, b);
}

void zip_eq(const View& out, const View& a, const View& b) {
    zip_op([](float x, float y) -> float { return x == y; }, out, a, b);
}

void zip_is_close(const View& out, const View& a, const View& b) {
    zip_op([](float x, float y) -> float { return std::abs(x - y) < 0.01; }, out, a, b);
}

} // namespace tsgrad
