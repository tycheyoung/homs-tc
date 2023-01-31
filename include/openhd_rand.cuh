#ifndef OPENHD_RAND_CUH_
#define OPENHD_RAND_CUH_

#include <curand_kernel.h>


__global__ void init_rand(curandState* states, unsigned int D, int seed) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < D) {
        curandState_t* s = new curandState_t;
        if (s != 0) {
            curand_init(seed, tidx, 0, &states[tidx]);
        }
        __syncthreads();
    }
}

__device__ float __draw_random_hypervector__(curandState* states, const int d) {
    curandState_t s = states[d];
    float val = curand_uniform(&s);
    states[d] = s;

    //0.2 * 2 = 0.4 => 0
    //0.6 * 2 = 1.2 => 1

    return (int(val * 2) - 1)? -1 : 1;
}

__device__ float __draw_gaussian_hypervector__(curandState* states, const int d) {
    curandState_t s = states[d];
    float val = curand_normal(&s);
    states[d] = s;
    
    return val;
}

__global__ void generate_large_hvs(float* id_hvs, curandState* states, unsigned int F, unsigned int D) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= D)
        return;
    
    #pragma unroll 1
    for(int f=0; f<F; ++f) {
        *__hxdim__(id_hvs, f, idx, D) = __draw_gaussian_hypervector__(states, idx);
    }
}

__global__ void generate_large_hvs_binary(float* id_hvs, curandState* states, unsigned int F, unsigned int D) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= D)
        return;
    
    #pragma unroll 1
    for(int f=0; f<F; ++f) {
        *__hxdim__(id_hvs, f, idx, D) = __draw_random_hypervector__(states, idx);
    }
}

#endif
