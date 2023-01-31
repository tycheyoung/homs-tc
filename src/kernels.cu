#include "../include/kernels.cuh"


__global__ void packing(unsigned int* output, float* arr, int origLength, int packLength, int numVec) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= origLength)
        return;


    for (int sample_idx = blockIdx.y; sample_idx < numVec; sample_idx += blockDim.y * gridDim.y) 
    {
        int tid = threadIdx.x;
        int lane = tid % warpSize;
        int bitPattern=0;

        if (i < origLength)
            bitPattern = __brev(__ballot_sync(0xFFFFFFFF, arr[sample_idx*origLength+i] > 0));

        if (lane == 0) {
            output[sample_idx*packLength+ (i / warpSize)] = bitPattern;
        }
    }
}


