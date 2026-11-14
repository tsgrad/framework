#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <vector>
#include <format>
#include <span>

namespace tsgrad{
    void scatter(const View& out, const View& indicies, const View& grad, const int32_t dim){
        if (dim < 0 || dim >= grad.rank){
            std::span<const int32_t> gradview{grad.shape, static_cast<size_t>(grad.rank)};
            throw std::runtime_error(std::format("scatter received dim gradside dimensions of grad, received dim: {} and grad.shape: {:n}", dim,  gradview));
        }
    
        if (indicies.rank != out.rank || indicies.rank != grad.rank) throw std::runtime_error(std::format("indicies, out, and grad have differing ranks inside scatter, received indicies.rank: {}, out.rank: {}, and grad.rank: {}", indicies.rank, out.rank, grad.rank));
        for (int i = 0; i < indicies.rank; i++){
            if (i != dim){
                if (indicies.shape[i] < 0 || indicies.shape[i] > out.shape[i]){
                    std::span<const int32_t> indiciesview{indicies.shape, static_cast<size_t>(indicies.rank)};
                    std::span<const int32_t> outview{out.shape, static_cast<size_t>(out.rank)};
                    throw std::runtime_error(std::format("scatter received incompatible shapes for indicies and out, received indicies.shape: {:n}, out.shape: {:n}", indiciesview, outview));
                }
            }
        }
    
        if constexpr (TSGRAD_HAS_CUDA){
            tsgrad::cuda::scatter(out, indicies, grad, dim);
            return;
        }
    
        long long size = prod(grad.shape, grad.rank);
        #pragma omp parallel if (size >= THRESHOLD)
        {
            std::vector<int32_t> grad_index(grad.rank, 0);
    
            #pragma omp for
            for (long long ordinal = 0; ordinal < size; ordinal++){
                position_to_index(ordinal, grad.shape, grad_index.data(), grad.rank);
                const long long grad_pos = index_to_position(grad_index.data(), grad.stride, grad.rank);
                grad_index[dim] = indicies.data[index_to_position(grad_index.data(), indicies.stride, indicies.rank)];
                
                #pragma omp atomic
                out.data[index_to_position(grad_index.data(), out.stride, out.rank)] += grad.data[grad_pos];
            }
        }
    }
    
    } //namespace