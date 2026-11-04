#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// tsgrad::cuda::matmul is the device side of tsgrad::matmul in src/cpu/matmul.cpp.
// addon.cc does not call this directly.
//
// Leading dimensions are a batch: out[..., M, N] = a[..., M, K] @ b[..., K, N].
// Strides still apply, because a permute() result shares its Storage and only
// changes stride.
//
// A uniform stride goes to cuBLAS. Every other layout uses matmul_kernel.
//
// Sequence, same as elementwise.cu:
//   Context::get()
//   cublas or matmul_kernel on View::data
//   Context::synchronize()

namespace tsgrad::cuda {

__global__ void matmul_kernel(float* out, int32_t out_stride_m, int32_t out_stride_n,
    const float* a, int32_t a_stride_m, int32_t a_stride_k,
    const float* b, int32_t b_stride_k, int32_t b_stride_n,
    int32_t M, int32_t N, int32_t K) {
    int32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || column >= N) return;

    int32_t a_pos = row * a_stride_m;
    int32_t b_pos = column * b_stride_n;
    float cur = 0;

    for (long long j = 0; j < K; j++){
        cur += a[a_pos] * b[b_pos];
        a_pos += a_stride_k;
        b_pos += b_stride_k;
    }

    out[row * out_stride_m + column * out_stride_n] = cur;
}

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t err, const char* what) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cublasGetStatusString(err));
    }
}

long long checked_mul(long long a, long long b) {
    if (b < 0) throw std::runtime_error("Cannot take product, negative dimension in shape");
    if (a == 0 || b == 0) return 0;
    if (a > 0) {
        if (a > LLONG_MAX / b) throw std::runtime_error("Overflow, shape must be less than int 64 max");
    } else if (a < LLONG_MIN / b) {
        throw std::runtime_error("Overflow, shape must be less than int 64 max");
    }
    return a * b;
}

long long checked_add(long long a, long long b) {
    if (b < 0 || a > LLONG_MAX - b) throw std::runtime_error("Overflow, shape must be less than int 64 max");
    return a + b;
}

long long abs_stride(int32_t stride) {
    return stride < 0 ? -static_cast<long long>(stride) : static_cast<long long>(stride);
}

long long abs_step(long long step) {
    return step < 0 ? -step : step;
}

void check_nonnegative(const View& v) {
    for (int32_t i = 0; i < v.rank; i++)
        if (v.shape[i] < 0)
            throw std::runtime_error("Cannot take product, negative dimension in shape");
}

void throw_incompatible(const View& a, const View& b) {
    std::span<const int32_t> aview{a.shape, static_cast<size_t>(a.rank)};
    std::span<const int32_t> bview{b.shape, static_cast<size_t>(b.rank)};
    throw std::runtime_error(std::format("Incompatible shapes for matrix mul, a: {:n}, b: {:n}", aview, bview));
}

void throw_incompatible(const View& out, const View& a, const View& b) {
    std::span<const int32_t> oview{out.shape, static_cast<size_t>(out.rank)};
    std::span<const int32_t> aview{a.shape, static_cast<size_t>(a.rank)};
    std::span<const int32_t> bview{b.shape, static_cast<size_t>(b.rank)};
    throw std::runtime_error(std::format("Incompatible shapes for matrix mul, out: {:n}, a: {:n}, b: {:n}", oview, aview, bview));
}

bool out_matches(const View& out, const View& a, int32_t n) {
    if (out.rank != a.rank) return false;
    for (int32_t i = 0; i < a.rank - 2; i++)
        if (out.shape[i] != a.shape[i]) return false;
    return out.shape[a.rank - 2] == a.shape[a.rank - 2] && out.shape[a.rank - 1] == n;
}

int batch_count(const int32_t* shape, int32_t rank) {
    long long acc = 1;
    for (int32_t i = 0; i < rank - 2; i++)
        acc = checked_mul(acc, shape[i]);
    if (acc > static_cast<long long>(INT_MAX))
        throw std::runtime_error("Overflow, batch count must fit in int");
    return static_cast<int>(acc);
}

// Leading dims, last index fastest. step is the distance between consecutive batches.
bool batch_stride(const int32_t* shape, const int32_t* stride, int32_t rank, long long& step) {
    step = 0;
    long long inner = 1;
    bool found = false;
    for (int32_t i = rank - 3; i >= 0; i--) {
        if (shape[i] == 1) continue;
        if (!found) {
            step = stride[i];
            found = true;
        } else if (static_cast<long long>(stride[i]) != checked_mul(step, inner)) {
            return false;
        }
        inner = checked_mul(inner, shape[i]);
    }
    return true;
}

// Logical (rows, cols) at i * stride_row + j * stride_col.
// OP_T is row-major. OP_N is column-major. A length-1 axis does not use its stride.
bool gemm_layout(int32_t rows, int32_t cols, int32_t stride_row, int32_t stride_col, cublasOperation_t& trans, int& ld) {
    if ((stride_col == 1 || cols <= 1) && stride_row >= cols) {
        trans = CUBLAS_OP_T;
        ld = stride_row;
        return true;
    }
    if ((stride_row == 1 || rows <= 1) && stride_col >= rows) {
        trans = CUBLAS_OP_N;
        ld = stride_col;
        return true;
    }
    return false;
}

cublasOperation_t flip_op(cublasOperation_t op) {
    return op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
}

bool gap_less_than(long long lo, long long hi, long long span) {
    if (lo >= 0 || hi < 0) return hi - lo < span;
    return hi - span < lo;
}

long long output_span(int32_t m, int32_t n, int32_t stride_row, int32_t stride_col) {
    long long span = 1;
    span = checked_add(span, checked_mul(static_cast<long long>(m - 1), abs_stride(stride_row)));
    span = checked_add(span, checked_mul(static_cast<long long>(n - 1), abs_stride(stride_col)));
    return span;
}

void check_output_disjoint(const View& out, int32_t m, int32_t n, int batch) {
    if (batch <= 1) return;

    const int32_t rank = out.rank;
    const long long span = output_span(m, n, out.stride[rank - 2], out.stride[rank - 1]);
    long long step = 0;
    if (batch_stride(out.shape, out.stride, rank, step)) {
        if (abs_step(step) < span)
            throw std::runtime_error("Overlapping output batches for matrix mul");
        return;
    }

    std::vector<long long> offsets(static_cast<size_t>(batch));
    int32_t index[16];
    const int32_t batch_rank = rank - 2;
    for (long long ordinal = 0; ordinal < batch; ordinal++) {
        position_to_index(ordinal, out.shape, index, batch_rank);
        offsets[static_cast<size_t>(ordinal)] = index_to_position(index, out.stride, batch_rank);
    }
    std::sort(offsets.begin(), offsets.end());
    for (size_t i = 1; i < offsets.size(); i++) {
        if (gap_less_than(offsets[i - 1], offsets[i], span))
            throw std::runtime_error("Overlapping output batches for matrix mul");
    }
}

void launch_matmul(const View& out, const View& a, const View& b, int32_t m, int32_t n, int32_t k, int batch, cudaStream_t stream) {
    dim3 threads(32, 8);
    dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    const int32_t rank = a.rank;

    if (rank == 2) {
        matmul_kernel<<<blocks, threads, 0, stream>>>(
            out.data, out.stride[0], out.stride[1],
            a.data, a.stride[0], a.stride[1],
            b.data, b.stride[0], b.stride[1],
            m, n, k
        );
        return;
    }

    int32_t index[16];
    const int32_t batch_rank = rank - 2;
    for (long long ordinal = 0; ordinal < batch; ordinal++) {
        position_to_index(ordinal, a.shape, index, batch_rank);
        const long long a_off = index_to_position(index, a.stride, batch_rank);
        const long long b_off = index_to_position(index, b.stride, batch_rank);
        const long long out_off = index_to_position(index, out.stride, batch_rank);
        matmul_kernel<<<blocks, threads, 0, stream>>>(
            out.data + out_off, out.stride[rank - 2], out.stride[rank - 1],
            a.data + a_off, a.stride[rank - 2], a.stride[rank - 1],
            b.data + b_off, b.stride[rank - 2], b.stride[rank - 1],
            m, n, k
        );
    }
}

void gemm(cublasHandle_t handle, cublasOperation_t trans_a, cublasOperation_t trans_b, cublasOperation_t trans_c,
    int32_t m, int32_t n, int32_t k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
    const float alpha = 1;
    const float beta = 0;
    if (trans_c == CUBLAS_OP_N) {
        check_cublas(cublasSgemm(
            handle, trans_a, trans_b, m, n, k,
            &alpha, a, lda, b, ldb,
            &beta, c, ldc
        ), "cublasSgemm");
        return;
    }

    check_cublas(cublasSgemm(
        handle, flip_op(trans_b), flip_op(trans_a), n, m, k,
        &alpha, b, ldb, a, lda,
        &beta, c, ldc
    ), "cublasSgemm");
}

void gemm_batched(cublasHandle_t handle, cublasOperation_t trans_a, cublasOperation_t trans_b, cublasOperation_t trans_c,
    int32_t m, int32_t n, int32_t k,
    const float* a, int lda, long long stride_a,
    const float* b, int ldb, long long stride_b,
    float* c, int ldc, long long stride_c, int batch) {
    const float alpha = 1;
    const float beta = 0;
    if (trans_c == CUBLAS_OP_N) {
        check_cublas(cublasSgemmStridedBatched(
            handle, trans_a, trans_b, m, n, k,
            &alpha, a, lda, stride_a,
            b, ldb, stride_b,
            &beta, c, ldc, stride_c,
            batch
        ), "cublasSgemmStridedBatched");
        return;
    }

    check_cublas(cublasSgemmStridedBatched(
        handle, flip_op(trans_b), flip_op(trans_a), n, m, k,
        &alpha, b, ldb, stride_b,
        a, lda, stride_a,
        &beta, c, ldc, stride_c,
        batch
    ), "cublasSgemmStridedBatched");
}

} // namespace

void matmul(const View& out, const View& a, const View& b) {
    if (out.rank > 16 || a.rank > 16 || b.rank > 16)
        throw std::runtime_error(std::format("CUDA matmul received exceptionally large shape: {}", std::max(std::max(out.rank, a.rank), b.rank)));

    if (!shapes_can_be_mul(a.shape, b.shape, a.rank, b.rank))
        throw_incompatible(a, b);

    check_nonnegative(a);
    check_nonnegative(b);
    check_nonnegative(out);

    const int32_t rank = a.rank;
    const int32_t m = a.shape[rank - 2];
    const int32_t k = a.shape[rank - 1];
    const int32_t n = b.shape[rank - 1];
    if (!out_matches(out, a, n))
        throw_incompatible(out, a, b);

    const int batch = batch_count(a.shape, rank);
    if (m == 0 || n == 0 || batch == 0) return;

    Context& ctx = Context::get();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.getStream());

    if (k == 0) {
        const long long numel = checked_mul(checked_mul(static_cast<long long>(batch), m), n);
        if (numel > static_cast<long long>(SIZE_MAX / sizeof(float)))
            throw std::runtime_error("Overflow, shape must be less than int 64 max");
        if (is_contiguous(out)) {
            check_cuda(cudaMemsetAsync(out.data, 0, static_cast<size_t>(numel) * sizeof(float), stream), "cudaMemsetAsync");
        } else {
            launch_matmul(out, a, b, m, n, k, batch, stream);
        }
        ctx.synchronize();
        return;
    }

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasOperation_t trans_c = CUBLAS_OP_N;
    int lda = 0, ldb = 0, ldc = 0;
    const bool a_ok = gemm_layout(m, k, a.stride[rank - 2], a.stride[rank - 1], trans_a, lda);
    const bool b_ok = gemm_layout(k, n, b.stride[rank - 2], b.stride[rank - 1], trans_b, ldb);
    const bool c_ok = gemm_layout(m, n, out.stride[rank - 2], out.stride[rank - 1], trans_c, ldc);

    long long stride_a = 0, stride_b = 0, stride_c = 0;
    bool strides_ok = true;
    if (batch > 1) {
        strides_ok = batch_stride(a.shape, a.stride, rank, stride_a)
            && batch_stride(b.shape, b.stride, rank, stride_b)
            && batch_stride(out.shape, out.stride, rank, stride_c);
    }

    if (a_ok && b_ok && c_ok && strides_ok) {
        if (batch > 1) {
            const long long cols = trans_c == CUBLAS_OP_T ? m : n;
            const long long footprint = checked_mul(ldc, cols);
            if (abs_step(stride_c) < footprint)
                throw std::runtime_error("Overlapping output batches for matrix mul");
        }

        cublasHandle_t handle = static_cast<cublasHandle_t>(ctx.getCublas());
        if (batch == 1) {
            gemm(handle, trans_a, trans_b, trans_c, m, n, k, a.data, lda, b.data, ldb, out.data, ldc);
        } else {
            gemm_batched(
                handle, trans_a, trans_b, trans_c, m, n, k,
                a.data, lda, stride_a,
                b.data, ldb, stride_b,
                out.data, ldc, stride_c, batch
            );
        }
        ctx.synchronize();
        return;
    }

    check_output_disjoint(out, m, n, batch);
    launch_matmul(out, a, b, m, n, k, batch, stream);
    ctx.synchronize();
}

} // namespace tsgrad::cuda
