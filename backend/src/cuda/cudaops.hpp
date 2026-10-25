#pragma once
#include <cuda_runtime.h>

namespace tsgrad::cuda{
    __device__ float id(float a){ return a; }
    __device__ float neg(float a){ return -a; }
    __device__ float sigmoid(float a){ 
        if (a >= 0)
            return 1.0F/(1.0F + expf(-a));  // e^a
        else
            return expf(a) / (1.0F + expf(a));
    }
    __device__ float relu(float a){ if (a < 0) a = 0; return a; }
    __device__ float log(float a){ return logf(a); }
    __device__ float exp(float a){ return expf(a); }
    __device__ float inv(float a){ return 1/a; }
    
    __device__ float add(float a, float b){ return a + b; }
    __device__ float mul(float a, float b){ return a * b; }
    __device__ float lt(float a, float b){ return a < b ? 1 : 0; }
    __device__ float eq(float a, float b){ return a == b; }
    __device__ float max(float a, float b){ return a > b ? a : b; }
    __device__ float is_close(float a, float b){ return fabsf(a - b) < 0.01; }
};