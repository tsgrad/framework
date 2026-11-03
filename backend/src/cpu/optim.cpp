#include "tsgrad/ops.h"
#include "helpers.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace tsgrad {
namespace {

void check_same_shape(const View& a, const View& b, const char* what) {
    if (!shapesAreEqual(a.shape, b.shape, a.rank, b.rank)) {
        throw std::runtime_error(std::string(what) + " received mismatched shapes");
    }
}

} // namespace

void sgd(const View& param, const View& grad, float lr) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::sgd(param, grad, lr);
        return;
    }

    check_same_shape(param, grad, "sgd");
    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    if (is_contiguous(param) && is_contiguous(grad)) {
        for (long long i = 0; i < n; i++)
            param.data[i] -= lr * grad.data[i];
        return;
    }

    std::vector<int32_t> index(param.rank, 0);
    for (long long i = 0; i < n; i++) {
        position_to_index(i, param.shape, index.data(), param.rank);
        const long long pos = index_to_position(index.data(), param.stride, param.rank);
        const long long gpos = index_to_position(index.data(), grad.stride, grad.rank);
        param.data[pos] -= lr * grad.data[gpos];
    }
}

void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::sgd_momentum(param, grad, velocity, lr, beta);
        return;
    }

    check_same_shape(param, grad, "sgd_momentum");
    check_same_shape(param, velocity, "sgd_momentum");
    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    if (is_contiguous(param) && is_contiguous(grad) && is_contiguous(velocity)) {
        for (long long i = 0; i < n; i++) {
            const float updated = beta * velocity.data[i] + (1.0F - beta) * grad.data[i];
            velocity.data[i] = updated;
            param.data[i] -= lr * updated;
        }
        return;
    }

    std::vector<int32_t> index(param.rank, 0);
    for (long long i = 0; i < n; i++) {
        position_to_index(i, param.shape, index.data(), param.rank);
        const long long pos = index_to_position(index.data(), param.stride, param.rank);
        const long long gpos = index_to_position(index.data(), grad.stride, grad.rank);
        const long long vpos = index_to_position(index.data(), velocity.stride, velocity.rank);
        const float updated = beta * velocity.data[vpos] + (1.0F - beta) * grad.data[gpos];
        velocity.data[vpos] = updated;
        param.data[pos] -= lr * updated;
    }
}

} // namespace tsgrad
