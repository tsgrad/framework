#include "tsgrad/ops.h"

// Body is intentionally empty.
// tensor_functions.matmul only calls this for rank-2 inputs:
//   out [M, N] = a [M, K] @ b [K, N]
// Strides are not necessarily contiguous, because permute shares storage.
//
// CUDA handoff: tsgrad::cuda::matmul(out, a, b), defined in src/cuda/matmul.cu.

namespace tsgrad {

void matmul(const View& out, const View& a, const View& b) {
    (void)out;
    (void)a;
    (void)b;
}

} // namespace tsgrad
