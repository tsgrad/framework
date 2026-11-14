#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <vector>
#include <format>
#include <span>

namespace tsgrad{
void gather(const View& out, const View& indicies, const View& values, const int32_t dim){
    if (dim < 0 || dim >= out.rank){
        std::span<const int32_t> outview{out.shape, static_cast<size_t>(out.rank)};
        throw std::runtime_error(std::format("gather received dim outside dimensions of out, received dim: {} and out.shape: {:n}", dim,  outview));
    }

    if (indicies.rank != values.rank) throw std::runtime_error(std::format("indicies, and values have differing ranks inside gather, received indicies.rank: {}, and values.rank: {}", indicies.rank, values.rank));
    for (int i = 0; i < indicies.rank; i++){
        if (i != dim){
            if (indicies.shape[i] < 0 || indicies.shape[i] > values.shape[i]){
                std::span<const int32_t> indiciesview{indicies.shape, static_cast<size_t>(indicies.rank)};
                std::span<const int32_t> valuesview{values.shape, static_cast<size_t>(values.rank)};
                throw std::runtime_error(std::format("gather received incompatible shapes for indicies and values, received indicies.shape: {:n}, values.shape: {:n}", indiciesview, valuesview));
            }
        }
    }

    if constexpr (TSGRAD_HAS_CUDA){
        tsgrad::cuda::gather(out, indicies, values, dim);
        return;
    }

    long long size = prod(out.shape, out.rank);
    #pragma omp parallel if (size >= THRESHOLD)
    {
        std::vector<int32_t> out_index(out.rank, 0);

        #pragma omp for
        for (long long ordinal = 0; ordinal < size; ordinal++){
            // Ordinal represents an index in out.data
            // Get the out/indicies positions for this ordinal
            position_to_index(ordinal, out.shape, out_index.data(), out.rank);
            const long long out_pos = index_to_position(out_index.data(), out.stride, out.rank);
            out_index[dim] = indicies.data[index_to_position(out_index.data(), indicies.stride, indicies.rank)];
            out.data[out_pos] = values.data[index_to_position(out_index.data(), values.stride, values.rank)];
        }
    }
}

} //namespace