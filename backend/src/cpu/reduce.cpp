#include "tsgrad/ops.h"

// Bodies are intentionally empty. Port tensorReduce from src/tensor_ops.ts.
// dim is the axis whose length collapses to 1 in out.
// reduce_max starts from -infinity, reduce_sum from 0, reduce_prod from 1.
//
// CUDA handoff: tsgrad::cuda::reduce_sum(out, in, dim), defined in
// src/cuda/elementwise.cu. Same borrowed host Views as the CPU function.

namespace tsgrad {

void reduce_max(const View& out, const View& in, int32_t dim) {
    (void)out;
    (void)in;
    (void)dim;
}

void reduce_sum(const View& out, const View& in, int32_t dim) {
    (void)out;
    (void)in;
    (void)dim;
}

void reduce_prod(const View& out, const View& in, int32_t dim) {
    (void)out;
    (void)in;
    (void)dim;
}

} // namespace tsgrad
